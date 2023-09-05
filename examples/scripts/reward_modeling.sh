source ./utils/env.sh
set -x
run_name=${2:-"test"}
model_name_or_path=$3
#model_name_or_path=output_sft_alpaca_instructions_llama-3b
model_name_or_path=${STEP2_MODEL_DIR:-"llama-500m_sft_1plus1_fifth_ws8_bs2_acc8_seq2048"}
dataset_name=${STEP2_DATASET_NAME:-"alpaca_noisy_multi_preference"}
output_dir_def=output_rw_`basename $dataset_name`_`basename $model_name_or_path`
output_dir=${STEP2_OUTPUT_DIR:-"$output_dir_def"}
gradient_accumulation_steps=${STEP2_GRADIENT_ACCUMULATION_STEPS:-8}
prompt_dict_path=${PROMPT_DICT_PATH:-"./examples/prompts/v0_inputs_noinputs.json"}
model_max_length=${STEP2_MODEL_MAX_LENGTH:-512}
per_device_train_batch_size=${STEP2_PER_DEVICE_TRAIN_BATCH_SIZE:-2}
fsdp_config=${STEP2_FSDP_CONFIG:-"./fsdp_config.json"}
mkdir -p $output_dir

set -x
$launch_prefix_cmd_torch \
    examples/reward_modeling.py \
  --fp16 True \
  --bf16 False \
  --seed 42 \
  --model_name_or_path "${model_name_or_path}" \
  --dataset_name "${dataset_name}" \
  --output_dir "${output_dir}" \
  --model_max_length $model_max_length \
  --num_train_epochs 1 \
  --per_device_train_batch_size $per_device_train_batch_size \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --eval_steps 10 \
  --wandb_project "alpaca_farm" \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 3e-6 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "no" \
  --logging_steps 1 \
  --logging_dir $output_dir \
  --run_name "${run_name}" \
  --fsdp "full_shard auto_wrap offload" \
  --fsdp_config $fsdp_config \
  --tf32 False \
  --flash_attn False \
  --ddp_timeout 1800 \
  --prompt_dict_path $prompt_dict_path $STEP2_EXTRA_ARGS \
   2>&1 | \
   tee $output_dir/training_`get_distributed_rank`.log

