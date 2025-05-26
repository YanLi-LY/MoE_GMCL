#!/bin/bash 
export CUDA_VISIBLE_DEVICES="1"
for((i=0;i<=9;i++));  
do   
    # echo $i
    TXT_PATH='./minigpt4/output/tinyimagenet/lora_r16_0.1sharedexperts2_0.9/10/42/results/test_'"$i"'.txt'
    CKPT_PATH='./minigpt4/output/tinyimagenet/lora_r16_0.1sharedexperts2_0.9/10/42/'"$i"'/checkpoint_4.pth'
    python batch_eval.py --cfg-path eval_configs/minigpt4_tinyimagenet_10tasks.yaml \
    --gpu-id 0 --task-id $i --txt-path $TXT_PATH \
    --ckpt-path $CKPT_PATH
done  
# python get_score_all.py