export STEP1_MODEL_DIR=`readlink -f ~/model/pretrain_chatgpt/chatglm2-6b`
FSDP_CONFIG="./fsdp_chatglm_config.json"
export STEP1_DATASET_NAME=alpaca_instructions
export STEP1_OUTPUT_DIR=output_sft_`basename $STEP1_DATASET_NAME`_`basename $STEP1_MODEL_DIR`
export STEP1_OUTPUT_DIR=`readlink -f $STEP1_OUTPUT_DIR`
export STEP1_FSDP_CONFIG=$FSDP_CONFIG
export STEP1_PADDING_SIDE="left"

export STEP2_MODEL_DIR=$STEP1_OUTPUT_DIR
export STEP2_DATASET_NAME=alpaca_noisy_multi_preference
export STEP2_OUTPUT_DIR=output_rw_`basename $STEP2_DATASET_NAME`_`basename $STEP2_MODEL_DIR`
export STEP2_OUTPUT_DIR=`readlink -f $STEP2_OUTPUT_DIR`
export STEP2_FSDP_CONFIG=$FSDP_CONFIG

export STEP3_ACTOR_DIR=$STEP1_OUTPUT_DIR
export STEP3_CRITIC_DIR=$STEP2_OUTPUT_DIR
export STEP3_DATASET_NAME=alpaca_instructions
export STEP3_DS_DATASET_NAME=`readlink -f ~/data/chatgpt/sft_alpaca_instructions_unlabeled`
export STEP3_OUTPUT_DIR=output_ppo_`basename $STEP3_DATASET_NAME`_act-`basename $STEP3_ACTOR_DIR`_crit-`basename $STEP3_CRITIC_DIR`
export STEP3_OUTPUT_DIR=`readlink -f $STEP3_OUTPUT_DIR`
export STEP3_EXTRA_ARGS="--save_format shard_full"
export STEP3_CONFIG_FILE="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_8gpu.yaml"

STEP3_EXTRA_ARGS+=" --cpu_offload"
STEP3_EXTRA_ARGS+=" --checkpoint_layer_name transformers.models.llama.modeling_llama.LlamaModel"
STEP3_EXTRA_ARGS+=" --wrap_layer_name transformers.models.llama.modeling_llama.LlamaDecoderLayer"
STEP3_EXTRA_ARGS+=" --gradient_checkpointing"
export STEP3_EXTRA_ARGS=$STEP3_EXTRA_ARGS


