SUBSET_NUM=8
DATA_ROOT="/PATH/TO/VideoInfer-Release"

SCRIPT="evaluation/videoinfer/baselines/inference_videorefer.py"
VIS_SAVE_PATH="results/RefVideoQA/VideoRefer-7B"


# SCRIPT="evaluation/videoinfer/baselines/inference_videollama3.py"
# VIS_SAVE_PATH="results/RefVideoQA/VideoLLaMA3"


# SCRIPT="evaluation/videoinfer/baselines/inference_osprey.py"
# VIS_SAVE_PATH="results/RefVideoQA/Osprey-7B"


# CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
#   --version=$VERSION \
#   --vis_save_path=$VIS_SAVE_PATH \
#   --num_frames=8 \
#   --subset_idx=0 \
#   --subset_num=$SUBSET_NUM


for i in $(seq 0 $((SUBSET_NUM - 1))); do
  CUDA_VISIBLE_DEVICES=$i python $SCRIPT \
    --version=$VERSION \
    --data_root=$DATA_ROOT \
    --vis_save_path=$VIS_SAVE_PATH \
    --num_frames=8 \
    --subset_idx=$i \
    --subset_num=$SUBSET_NUM &
done

wait

python evaluation/videoinfer/merge.py \
  --folder_path=${VIS_SAVE_PATH} \
  --subset_num=${SUBSET_NUM}

wait 

PRED_ROOT=$VIS_SAVE_PATH
python evaluation/videoinfer/eval.py \
    --pred_file="${PRED_ROOT}/merged_result.json" \
    --gt_file=$DATA_ROOT"/test.json" \
    --results_file="${PRED_ROOT}/eval_result.json" |& tee -a "${PRED_ROOT}/eval_result.txt"