DATASET_NAME=`readlink -f ~/data/chatgpt/alpaca_comment`
export PROMPT_DICT_PATH=`readlink -f ./examples/prompts/concat.json`

export STEP1_MODEL_DIR=`readlink -f ~/model/pretrain_chatgpt/llama-7b`
export STEP1_DATASET_NAME=$DATASET_NAME
export STEP1_OUTPUT_DIR=output_sft_`basename $STEP1_DATASET_NAME`_`basename $STEP1_MODEL_DIR`
export STEP1_OUTPUT_DIR=`readlink -f $STEP1_OUTPUT_DIR`
export STEP1_GRADIENT_ACCUMULATION_STEPS=2

export STEP2_MODEL_DIR=$STEP1_OUTPUT_DIR
export STEP2_DATASET_NAME=$DATASET_NAME
export STEP2_OUTPUT_DIR=output_rw_`basename $STEP2_DATASET_NAME`_`basename $STEP2_MODEL_DIR`
export STEP2_OUTPUT_DIR=`readlink -f $STEP2_OUTPUT_DIR`
export STEP2_GRADIENT_ACCUMULATION_STEPS=2

export STEP3_ACTOR_DIR=$STEP1_OUTPUT_DIR
export STEP3_CRITIC_DIR=$STEP2_OUTPUT_DIR
export STEP3_DATASET_NAME=$DATASET_NAME
export STEP3_TOTAL_EPOCHS=10
export STEP3_OUTPUT_DIR=~/wfs/output_chatgpt_ppo/output_ppo_`basename $STEP3_DATASET_NAME`_act-`basename $STEP3_ACTOR_DIR`_crit-`basename $STEP3_CRITIC_DIR`_2std_v2
export STEP3_OUTPUT_DIR=`readlink -f $STEP3_OUTPUT_DIR`
export STEP3_CONFIG_FILE="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_8gpu.yaml"
export STEP3_ROLLOUT_BATCH_SIZE=128
export STEP3_ROLLOUT_PER_DEVICE_BATCH_SIZE=1
export STEP3_STEP_BATCH_SIZE=256
export STEP3_STEP_PER_DEVICE_BATCH_SIZE=1
export STEP3_SAVE_STEPS=15
#export STEP3_CONFIG_FILE="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_16gpu.yaml"

#export STEP3_DS_EXTRA_ARGS+=" --use_scale_reward --scale_reward_std 22.0 --scale_reward_mean 13.9"
export STEP3_DS_EXTRA_ARGS+=" --use_scale_reward --scale_reward_std 22.0 --scale_reward_mean 18.0"
export STEP3_DS_DATASET_NAME=$DATASET_NAME



