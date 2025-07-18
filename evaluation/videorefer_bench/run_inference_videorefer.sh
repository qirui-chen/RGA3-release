#!/bin/bash

EXP_NAME="UniGR-7B"
RUN_NAME="tmp"
DATA_ROOT="/PATH/TO/VideoRefer-Bench/VideoRefer-Bench-Q/"
SHAPE="ellipse"
NUM_FRAMES=16

run_experiment() {
    local USE_STOM=$1
    local GPU_ID=$2

    OUT_PUT_FILE="${VIS_SAVE_PATH}videorefer-q_${SHAPE}_${USE_STOM}_${NUM_FRAMES}"
    VERSION="SurplusDeficit/UniGR-7B"
    VIS_SAVE_PATH="results/VideoRefer/${VERSION}/" 


    CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/videorefer_bench/inference_videorefer.py \
        --version=${VERSION} \
        --video-folder ${DATA_ROOT} \
        --question-file ${DATA_ROOT}"VideoRefer-Bench-Q.json" \
        --output-file ${OUT_PUT_FILE}".json" \
        --shape ${SHAPE} \
        --use_stom ${USE_STOM} \
        --num_frames ${NUM_FRAMES}

    python evaluation/videorefer_bench/eval_videorefer_bench_q.py \
        --pred-path ${OUT_PUT_FILE}".json" | tee -a ${OUT_PUT_FILE}".txt"
}

run_experiment "False" 0 &
