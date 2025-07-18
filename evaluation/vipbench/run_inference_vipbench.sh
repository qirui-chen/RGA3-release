SCRIPT="evaluation/vipbench/inference_vipbench.py"
VERSION="SurplusDeficit/UniGR-3B"
VIS_SAVE_PATH="results/ViPBench/${VERSION}"

# VERSION="/mnt/ali-sh-1/usr/chenqirui/data/models/Qwen2.5-VL-7B-Instruct"
# VIS_SAVE_PATH="results/ViPBench/Qwen2.5-VL-7B-Instruct"


DATA_ROOT="/mnt/ali-sh-1/usr/chenqirui/qrchen_dataset/ViP-Bench"
SPLITS=(
    'bbox'
    'human'
)

for split in "${SPLITS[@]}"; do
    CUDA_VISIBLE_DEVICES=7 python $SCRIPT \
        --version=$VERSION \
        --question-file="${DATA_ROOT}/${split}/questions.jsonl" \
        --image-folder="${DATA_ROOT}/${split}/images" \
        --answers-file="${VIS_SAVE_PATH}/${split}-answers.json"
done

wait 

for split in "${SPLITS[@]}"; do
    python evaluation/vipbench/evaluator.py \
        --result_path="${VIS_SAVE_PATH}" \
        --vipbench_split="${split}" &
done