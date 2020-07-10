#!/bin/bash

NGPU=8
TEST_BATCH_SIZE=64

MODEL_NAME=packdet_R_50_FPN_2x_fe-128-12-2_m4_sep
CONFIG_PATH=configs/packdet/$MODEL_NAME.yaml
WEIGHT_PATH=training_dir/$MODEL_NAME

conda activate packdet

for i in $(seq -f "%07g" 10000 180000)
do
  #echo `pwd`/$WEIGHT_PATH/model_$i.pth
  if [ -f `pwd`/$WEIGHT_PATH/model_$i.pth ] ; then
    echo `pwd`/$WEIGHT_PATH/model_$i.pth
    python -m torch.distributed.launch \
        --nproc_per_node=$NGPU \
        --master_port=$((RANDOM + 10000)) \
        tools/test_net.py \
        --config-file $CONFIG_PATH \
        MODEL.WEIGHT $WEIGHT_PATH/model_$i.pth \
        OUTPUT_DIR $WEIGHT_PATH \
        TEST.IMS_PER_BATCH $TEST_BATCH_SIZE
  fi
done