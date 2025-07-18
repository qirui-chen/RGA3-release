#!/bin/bash
export NCCL_ALGO=Auto # Tree
export NCCL_TIMEOUT=1800
export OMP_NUM_THREADS=4

DISTRIBUTED_ARGS="
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$GPU_NUM \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT
"

EXP_NAME="7B-8k-f8-4-lr4e-5-b2g8-p4-18-5-all-r128a256"
RUN_NAME="mixed"
LOG_BASE_DIR="./all_runs/${RUN_NAME}"
mkdir -p "${LOG_BASE_DIR}/${EXP_NAME}"

video_max_pixels=$((320*28*28))
image_max_pixels=$((1280*28*28))


torchrun $DISTRIBUTED_ARGS train_joint.py \
    --version="/PATH/TO/Qwen2.5-VL-7B-Instruct" \
    --dataset_dir='/PATH/TO/DATASET_ROOT' \
    --sam_pretrained="/PATH/TO/sam2_hiera_large.pt" \
    --exp_name=${EXP_NAME} \
    --num_frames_mllm=8 \
    --num_frames_sam=4 \
    --lora_r 128 \
    --lora_alpha 256 \
    --train_mask_decoder \
    --num_classes_per_sample=1 \
    --precision="bf16" \
    --epochs=80 \
    --steps_per_epoch=100 \
    --print_freq=20 \
    --batch_size=2 \
    --lr=0.00004 \
    --grad_accumulation_steps=8 \
    --video_max_pixels=$video_max_pixels \
    --image_max_pixels=$image_max_pixels \
    --dataset="vqa,ref_vqa,videoqa,ref_videoqa,sem_seg,refer_seg,reason_seg,vos,ref_vos,reason_vos" \
    --sem_seg_data="ade20k,cocostuff,pascal_part,paco_lvis" \
    --refer_seg_data="refclef,refcoco,refcoco+,refcocog" \
    --vos_data="ytvos" \
    --ref_vos_data="refer_youtube_vos,mevis" \
    --ref_vqa_data="vip_llava_2|3" \
    --sample_rates="4,18,4,4,8,4,3,4,6,5" \
    --log_base_dir=${LOG_BASE_DIR} \
    --workers=8 |& tee -a ${LOG_BASE_DIR}/${EXP_NAME}/debug.txt