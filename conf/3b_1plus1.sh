DATASET_NAME=`readlink -f ~/data/chatgpt/alpaca_1plus1_fifth`
export PROMPT_DICT_PATH=`readlink -f ./examples/prompts/raw.json`

export STEP1_MODEL_DIR=`readlink -f ~/model/pretrain_chatgpt/open_llama_3b`
export STEP1_DATASET_NAME=$DATASET_NAME
export STEP1_OUTPUT_DIR=output_sft_`basename $STEP1_DATASET_NAME`_`basename $STEP1_MODEL_DIR`
export STEP1_OUTPUT_DIR=`readlink -f $STEP1_OUTPUT_DIR`

export STEP2_MODEL_DIR=$STEP1_OUTPUT_DIR
export STEP2_DATASET_NAME=$DATASET_NAME
#export STEP2_OUTPUT_DIR=output_rwsft_`basename $STEP2_DATASET_NAME`_`basename $STEP2_MODEL_DIR`
export STEP2_OUTPUT_DIR=output_rw_`basename $STEP2_DATASET_NAME`_`basename $STEP2_MODEL_DIR`
export STEP2_OUTPUT_DIR=`readlink -f $STEP2_OUTPUT_DIR`
#export STEP2_EXTRA_ARGS=" --use_sft_loss"

export STEP3_ACTOR_DIR=$STEP1_OUTPUT_DIR
export STEP3_CRITIC_DIR=$STEP2_OUTPUT_DIR
export STEP3_DATASET_NAME=$DATASET_NAME
#export STEP3_DS_DATASET_NAME=`readlink -f ~/data/chatgpt/sft_1plus1_fifth_reward`
export STEP3_DS_DATASET_NAME=$DATASET_NAME
export STEP3_OUTPUT_DIR=output_ppo_5w_`basename $STEP3_DATASET_NAME`_act-`basename $STEP3_ACTOR_DIR`_crit-`basename $STEP3_CRITIC_DIR`
export STEP3_OUTPUT_DIR=`readlink -f $STEP3_OUTPUT_DIR`
export STEP3_EXTRA_ARGS="--save_format full"
export STEP3_EXTRA_ARGS+=" --cpu_offload"
export STEP3_EXTRA_ARGS+=" --scale_reward True --scale_reward_std 20.0 --scale_reward_mean 10.8"
export STEP3_EXTRA_ARGS=$STEP3_EXTRA_ARGS
echo $STEP3_EXTRA_ARGS
export STEP3_CONFIG_FILE="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_8gpu.yaml"
#export STEP3_CONFIG_FILE="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_16gpu.yaml"

export STEP3_ROLLOUT_BATCH_SIZE=128
export STEP3_ROLLOUT_PER_DEVICE_BATCH_SIZE=1
export STEP3_STEP_PER_DEVICE_BATCH_SIZE=1
export STEP3_STEP_BATCH_SIZE=256
export STEP3_DS_EXTRA_ARGS+=" --use_scale_reward --scale_reward_std 5.0 --scale_reward_mean 10.0"
#export STEP3_DS_EXTRA_ARGS+=" --use_scale_value --scale_value_std 2.2 --scale_value_mean 6.02"
#export STEP3_DS_EXTRA_ARGS+=" --use_manual_reward"
#export STEP3_DS_EXTRA_ARGS+=" --use_whiten"
export STEP3_DS_EXTRA_ARGS+=" --prompt_dict_path `readlink -f examples/prompts/instruction.json`"
#export STEP3_DS_EXTRA_ARGS+=" --debug"


