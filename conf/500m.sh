DATASET_NAME=`readlink -f ~/data/chatgpt/alpaca_crime`

export STEP1_MODEL_DIR=`readlink -f ~/model/pretrain_chatgpt/llama-500m`
#export STEP1_MODEL_DIR=`readlink -f ~/model/pretrain_chatgpt/chatglm2-6b`
#FSDP_CONFIG="./fsdp_chatglm_config.json"
#export STEP1_MODEL_DIR=`readlink -f ~/model/pretrain_chatgpt/baichuan-7b`
#FSDP_CONFIG="./fsdp_baichuan_config.json"
export STEP1_DATASET_NAME=$DATASET_NAME
export STEP1_OUTPUT_DIR=output_sft_`basename $STEP1_DATASET_NAME`_`basename $STEP1_MODEL_DIR`
export STEP1_GRADIENT_ACCUMULATION_STEPS=1
export STEP1_MODEL_MAX_LENGTH=4096
export STEP1_FSDP_CONFIG=$FSDP_CONFIG

export STEP2_MODEL_DIR=$STEP1_OUTPUT_DIR
export STEP2_DATASET_NAME=$DATASET_NAME
export STEP2_OUTPUT_DIR=output_rw_`basename $STEP2_DATASET_NAME`_`basename $STEP2_MODEL_DIR`
export STEP2_GRADIENT_ACCUMULATION_STEPS=1
#export STEP2_EXTRA_ARGS=" --use_sft_loss"
export STEP2_MODEL_MAX_LENGTH=2048
export STEP2_PER_DEVICE_TRAIN_BATCH_SIZE=1
export STEP2_FSDP_CONFIG=$FSDP_CONFIG

export STEP3_ACTOR_DIR=$STEP1_OUTPUT_DIR
export STEP3_CRITIC_DIR=$STEP2_OUTPUT_DIR
export STEP3_DATASET_NAME=alpaca_instructions
export STEP3_OUTPUT_DIR=output_ppo_`basename $STEP3_DATASET_NAME`_act-`basename $STEP3_ACTOR_DIR`_crit-`basename $STEP3_CRITIC_DIR`
export STEP3_CONFIG_FILE="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_8gpu.yaml"
#export STEP3_EXTRA_ARGS=" --wrap_layer_name transformers.models.llama.modeling_llama.LlamaMLP,transformers.models.llama.modeling_llama.LlamaAttention,transformers.models.llama.modeling_llama.LlamaDecoderLayer,torch.nn.Embedding"
export STEP3_EXTRA_ARGS=" --wrap_layer_name transformers.models.llama.modeling_llama.LlamaDecoderLayer"
#export STEP3_EXTRA_ARGS=" --wrap_layer_name torch.nn.Embedding,torch.nn.Linear,torch.nn.modules.sparse.Embedding"
#export STEP3_CONFIG_FILE="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_16gpu.yaml"



