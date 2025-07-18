VERSION="SurplusDeficit/UniGR-3B"
VIS_SAVE_PATH="results/RefDAVIS/${VERSION}/"

OUTPUT_DIR=$VIS_SAVE_PATH"_processed"


python evaluation/refdavis/post_process_davis.py --src_dir $VIS_SAVE_PATH

ANNO0_DIR=${OUTPUT_DIR}/"anno_0"
ANNO1_DIR=${OUTPUT_DIR}/"anno_1"
ANNO2_DIR=${OUTPUT_DIR}/"anno_2"
ANNO3_DIR=${OUTPUT_DIR}/"anno_3"
python3 evaluation/refdavis/eval_davis.py --results_path=${ANNO0_DIR}
python3 evaluation/refdavis/eval_davis.py --results_path=${ANNO1_DIR}
python3 evaluation/refdavis/eval_davis.py --results_path=${ANNO2_DIR}
python3 evaluation/refdavis/eval_davis.py --results_path=${ANNO3_DIR}
