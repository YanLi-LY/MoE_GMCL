#!/bin/bash 
export CUDA_VISIBLE_DEVICES="3"
for((i=0;i<=4;i++));  
do   
    # echo $i
    TXT_PATH='./minigpt4/output/imagenet_R/ours+loramoe+calib+sharedexpert/5/42/results/test_'"$i"'.txt'
    CKPT_PATH='./minigpt4/output/imagenet_R/ours+loramoe+calib+sharedexpert/5/42/'"$i"'/checkpoint_4.pth'
    python batch_eval.py --cfg-path eval_configs/minigpt4_imagenet_R_10tasks.yaml \
    --gpu-id 0 --task-id $i --txt-path $TXT_PATH \
    --ckpt-path $CKPT_PATH
done  
# python get_score_all.py