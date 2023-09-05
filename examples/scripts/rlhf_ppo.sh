source ./utils/env.sh
set -x

ps -ef | grep accelerate | grep -v grep | tee /dev/stderr | awk '{print $2}' | xargs kill -9
#export NCCL_DEBUG=INFO
run_name=ppo
policy_model_name_or_path=${STEP3_ACTOR_DIR:-"output_sft_alpaca_instructions_open_llama_3b"}
reward_model_name_or_path=${STEP3_CRITIC_DIR:-"output_rw_sft_alpaca_noisy_multi_preference_output_sft_sft_alpaca_instructions_open_llama_3b"}
kl_coef=${5:-0.0067}
output_dir=${STEP3_OUTPUT_DIR:-"./output_ppo/"}
dataset_name=${STEP3_DATASET_NAME:-"alpaca_noisy_multi_preference"}

prompt_dict_path=${PROMPT_DICT_PATH:-"./examples/prompts/v0_inputs_noinputs.json"}
config_file=${STEP3_CONFIG_FILE:-"./examples/accelerate_configs/rlhf_ppo_fsdp_llama_8gpu.yaml"}
#config_file="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_16gpu.yaml"
rollout_batch_size=${STEP3_ROLLOUT_BATCH_SIZE:-512}
rollout_per_device_batch_size=${STEP3_ROLLOUT_PER_DEVICE_BATCH_SIZE:-16}
step_batch_size=${STEP3_STEP_BATCH_SIZE:-256}
step_per_device_batch_size=${STEP3_STEP_PER_DEVICE_BATCH_SIZE:-1}
total_epochs=${STEP3_TOTAL_EPOCHS:-2}
save_steps=${STEP3_SAVE_STEPS:-30}

echo $output_dir
mkdir -p $output_dir
set -x
setup_train_env

log_fn=$output_dir/training_`get_distributed_rank`.log
if [[ "$STEP3_EXTRA_ARGS" == *"export"* ]]; then
    log_fn=$output_dir/export_`get_distributed_rank`.log
fi
$launch_prefix_cmd_accelerate \
  --fsdp_offload_params true \
  --config_file "${config_file}" \
  examples/rlhf_ppo.py \
  --run_name "${run_name}" \
  --dataset_name $dataset_name \
  --step_per_device_batch_size $step_per_device_batch_size \
  --rollout_per_device_batch_size $rollout_per_device_batch_size \
  --per_device_eval_batch_size 1 \
  --output_dir "${output_dir}" \
  --reward_model_name_or_path "${reward_model_name_or_path}" \
  --policy_model_name_or_path "${policy_model_name_or_path}" \
  --init_value_with_reward True \
  --rollout_batch_size $rollout_batch_size \
  --step_batch_size $step_batch_size \
  --learning_rate 1e-5 \
  --logging_dir $output_dir \
  --warmup_steps 5 \
  --kl_coef "${kl_coef}" \
  --total_epochs 10 \
  --flash_attn False \
  --prompt_dict_path $prompt_dict_path \
  --save_steps_extra 1 \
  --save_steps $save_steps \
  $STEP3_EXTRA_ARGS \
   2>&1 | \
   tee $log_fn
