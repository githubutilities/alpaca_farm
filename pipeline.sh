set -x
run_step=${2:-"123"}

source ./utils/env.sh
source $1
echo $STEP1_OUTPUT_DIR
echo $STEP2_OUTPUT_DIR
echo $STEP3_OUTPUT_DIR

if [[ "$run_step" == *"1"* ]]; then
    echo "Beginning step1..."
    bash examples/scripts/sft.sh 2>&1 | tee ${1}.step1.log
    #if [ `get_distributed_rank` == 1 ]; then
    #    bash examples/scripts/sft.sh 2>&1 | tee ${1}.step1.log
    #fi
fi
if [[ "$run_step" == *"2"* ]]; then
    echo "Beginning step2..."
    bash examples/scripts/reward_modeling.sh 2>&1 | tee ${1}.step2.log
    #if [ `get_distributed_rank` == 1 ]; then
    #    bash examples/scripts/reward_modeling.sh 2>&1 | tee ${1}.step2.log
    #fi
fi
if [[ "$run_step" == *"3"* ]]; then
    echo "Beginning step3..."
    bash examples/scripts/rlhf_ppo.sh 2>&1 | tee ${1}.step3.log
fi
if [[ "$run_step" == *"export"* ]]; then
    echo "Beginning step3 export..."
    export STEP3_EXTRA_ARGS=" --do_export"
    bash examples/scripts/rlhf_ppo.sh 2>&1 | tee ${1}.export.log
fi


if [[ "$run_step" == *"ds"* ]]; then
    export STEP3_ACTOR_DIR=`readlink -f $STEP1_OUTPUT_DIR`
    export STEP3_CRITIC_DIR=`readlink -f $STEP2_OUTPUT_DIR`
    if [ "$STEP3_DS_DATASET_NAME" == "" ]; then
        echo 'ds_dataset_name not set, exiting...'
        exit
    fi
    export STEP3_DATASET_NAME=$STEP3_DS_DATASET_NAME
    if [[ "$3" != *"--"* ]]; then
        export STEP3_OUTPUT_DIR=${3:-"output_ds_ppo_`basename $STEP3_DATASET_NAME`_act-`basename $STEP3_ACTOR_DIR`_crit-`basename $STEP3_CRITIC_DIR`"}
        export STEP3_OUTPUT_DIR=`readlink -f $STEP3_OUTPUT_DIR`
    fi
    STEP3_EXTRA_ARGS=$STEP3_DS_EXTRA_ARGS
    STEP3_EXTRA_ARGS+=" --load_alpaca_reward_model --prompt_answer"
    export STEP3_EXTRA_ARGS=$STEP3_EXTRA_ARGS
    echo "Beginning ds step3..."
    cp $STEP3_ACTOR_DIR/config.json $STEP3_CRITIC_DIR/config.json.alpaca
    if [ -d ${STEP3_CRITIC_DIR}_split ]; then
        export STEP3_CRITIC_DIR=${STEP3_CRITIC_DIR}_split
        echo $STEP3_CRITIC_DIR
    fi

    bash ~/ss/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/train/step3_rlhf_finetuning/scripts/run.sh
fi
