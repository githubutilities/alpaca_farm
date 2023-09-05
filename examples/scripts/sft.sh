source ./utils/env.sh

set -x
run_name=${2:-"sft"}
#model_name_or_path=${1:-"`readlink -f ~/model/pretrain_chatgpt/llama-500m`"}
#model_name_or_path=${1:-"`readlink -f ~/model/pretrain_chatgpt/llama-7b`"}
model_name_or_path=${STEP1_MODEL_DIR:-"`readlink -f ~/model/pretrain_chatgpt/llama-500m`"}
dataset_name=${STEP1_DATASET_NAME:-"alpaca_instructions"}
output_dir_def=output_sft_`basename $dataset_name`_`basename $model_name_or_path`
output_dir=${STEP1_OUTPUT_DIR:-"$output_dir_def"}
gradient_accumulation_steps=${STEP1_GRADIENT_ACCUMULATION_STEPS:-16}
prompt_dict_path=${PROMPT_DICT_PATH:-"./examples/prompts/v0_inputs_noinputs.json"}
model_max_length=${STEP1_MODEL_MAX_LENGTH:-512}
fsdp_config=${STEP1_FSDP_CONFIG:-"./fsdp_config.json"}
padding_side=${STEP1_PADDING_SIDE:-"right"}

mkdir -p $output_dir
#torchrun --nproc_per_node=8 --master_port=1234 \

$launch_prefix_cmd_torch \
    examples/supervised.py \
  --dataset_name $dataset_name \
  --model_name_or_path "${model_name_or_path}" \
  --fp16 True \
  --bf16 False \
  --seed 42 \
  --output_dir "${output_dir}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --eval_steps 100 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "no" \
  --logging_steps 1 \
  --logging_dir $output_dir \
  --wandb_project "alpaca_farm" \
  --run_name "${run_name}" \
  --tf32 False \
  --flash_attn False \
  --model_max_length $model_max_length \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config $fsdp_config \
  --train_splits "sft" \
  --padding_side $padding_side \
  --prompt_dict_path $prompt_dict_path \
   2>&1 | \
   tee $output_dir/training_`get_distributed_rank`.log

