#!/bin/bash 
export CUDA_VISIBLE_DEVICES="0"
for((i=0;i<=9;i++));  
do   
    # echo $i
    TXT_PATH='./minigpt4/output/tiny/router_methods/10/42/results/test_'"$i"'.txt'
    CKPT_PATH='./minigpt4/output/tiny/router_methods/10/42/'"$i"'/checkpoint_1.pth'
    python batch_eval.py --cfg-path eval_configs/minigpt4_imagenet100_10tasks.yaml \
    --gpu-id 0 --task-id $i --txt-path $TXT_PATH \
    --ckpt-path $CKPT_PATH
done  
# python get_score_all.py


