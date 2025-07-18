SCRIPT="evaluation/videoinfer/inference_videoinfer.py"

VERSION="SurplusDeficit/UniGR-3B"
VIS_SAVE_PATH="results/RefVideoQA/${VERSION}/"
DATA_ROOT="/PATH/TO/VideoInfer-Release"

SUBSET_NUM=8


# CUDA_VISIBLE_DEVICES=1 python $SCRIPT \
#   --version=$VERSION \
#   --data_root=$DATA_ROOT \
#   --vis_save_path=$VIS_SAVE_PATH"vis" \
#   --num_frames=8 \
#   --subset_idx=0 \
#   --subset_num=$SUBSET_NUM \
#   --vis \
#   --use_stom


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


python evaluation/videoinfer/eval.py \
    --pred_file="${VIS_SAVE_PATH}/merged_result.json" \
    --gt_file=$DATA_ROOT"/test.json" \
    --results_file="${VIS_SAVE_PATH}/eval_result.json" |& tee -a "${VIS_SAVE_PATH}/eval_result.txt"
