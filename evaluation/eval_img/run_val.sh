VERSION="SurplusDeficit/UniGR-3B"
VIS_SAVE_PATH="results/ImageSeg/${VERSION}/"   # remember to add /
DATASET_DIR='/mnt/ali-sh-1/dataset/zeus/chenqr/datasets/qrchen_dataset/other-datasets/'

datasets=(
    "refcoco|unc|testA"
    "refcoco|unc|testB"
    "refcoco|unc|val"
    "refcoco+|unc|testA"
    "refcoco+|unc|testB"
    "refcoco+|unc|val"
    "refcocog|umd|val"
    "refcocog|umd|test"
    "ReasonSeg|val"
    "ReasonSeg|test|short"
    "ReasonSeg|test|long"
    "ReasonSeg|test|all"
)

for i in {0..7}; do
  deepspeed --include "localhost:${i}" --master_port=$((25000 + i)) evaluation/eval_img/val.py \
    --version="$VERSION" \
    --dataset_dir="$DATASET_DIR" \
    --vis_save_path="$VIS_SAVE_PATH" \
    --num_frames_mllm=8 \
    --num_frames_sam=1 \
    --eval_only \
    --val_dataset="${datasets[$i]}" &
done
wait

for i in {8..11}; do
  gpu_id=$((i-8))
  deepspeed --include "localhost:${gpu_id}" --master_port=$((25000 + i)) evaluation/eval_img/val.py \
    --version="$VERSION" \
    --dataset_dir="$DATASET_DIR" \
    --vis_save_path="$VIS_SAVE_PATH" \
    --num_frames_mllm=8 \
    --num_frames_sam=1 \
    --eval_only \
    --val_dataset="${datasets[$i]}" &
done
wait

echo "All evaluations are done!"