#!/bin/bash 
export CUDA_VISIBLE_DEVICES="3"
for((i=0;i<=19;i++));  
do   
    # echo $i
    TXT_PATH='./minigpt4/output/tinyimagenet/ours_r16_sharedexperts2_0.9/20/42/results/test_'"$i"'.txt'
    CKPT_PATH='./minigpt4/output/tinyimagenet/ours_r16_sharedexperts2_0.9/20/42/'"$i"'/checkpoint_4.pth'
    python batch_eval.py --cfg-path eval_configs/minigpt4_tinyimagenet_20tasks.yaml \
    --gpu-id 0 --task-id $i --txt-path $TXT_PATH \
    --ckpt-path $CKPT_PATH
done  
# python get_score_all.py