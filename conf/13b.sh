export STEP1_MODEL_DIR=`readlink -f ~/model/pretrain_chatgpt/llama-13b-hf`
export STEP1_DATASET_NAME=alpaca_instructions
export STEP1_OUTPUT_DIR=output_sft_`basename $STEP1_DATASET_NAME`_`basename $STEP1_MODEL_DIR`
export STEP1_OUTPUT_DIR=`readlink -f $STEP1_OUTPUT_DIR`
export STEP1_GRADIENT_ACCUMULATION_STEPS=2

export STEP2_MODEL_DIR=$STEP1_OUTPUT_DIR
export STEP2_DATASET_NAME=alpaca_noisy_multi_preference
export STEP2_OUTPUT_DIR=output_rw_`basename $STEP2_DATASET_NAME`_`basename $STEP2_MODEL_DIR`
export STEP2_OUTPUT_DIR=`readlink -f $STEP2_OUTPUT_DIR`
export STEP2_GRADIENT_ACCUMULATION_STEPS=2

export STEP3_ACTOR_DIR=$STEP1_OUTPUT_DIR
export STEP3_CRITIC_DIR=$STEP2_OUTPUT_DIR
export STEP3_DATASET_NAME=alpaca_instructions
export STEP3_OUTPUT_DIR=output_ppo_`basename $STEP3_DATASET_NAME`_act-`basename $STEP3_ACTOR_DIR`_crit-`basename $STEP3_CRITIC_DIR`
export STEP3_OUTPUT_DIR=`readlink -f $STEP3_OUTPUT_DIR`
export STEP3_CONFIG_FILE="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_8gpu.yaml"
export STEP3_ROLLOUT_BATCH_SIZE=16
export STEP3_ROLLOUT_PER_DEVICE_BATCH_SIZE=1
export STEP3_STEP_BATCH_SIZE=16
STEP3_EXTRA_ARGS+=" --save_format local"
STEP3_EXTRA_ARGS+=" --cpu_offload"
#STEP3_EXTRA_ARGS+=" --wrap_layer_name transformers.models.llama.modeling_llama.LlamaMLP,transformers.models.llama.modeling_llama.LlamaAttention,transformers.models.llama.modeling_llama.LlamaDecoderLayer,torch.nn.Embedding"
STEP3_EXTRA_ARGS+=" --wrap_layer_name transformers.models.llama.modeling_llama.LlamaMLP,transformers.models.llama.modeling_llama.LlamaAttention,torch.nn.Embedding"
STEP3_EXTRA_ARGS+=" --query_len 100 --response_len 100"
export STEP3_EXTRA_ARGS=$STEP3_EXTRA_ARGS
#export STEP3_CONFIG_FILE="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_16gpu.yaml"


