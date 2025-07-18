VERSION="SurplusDeficit/UniGR-3B"
VIS_SAVE_PATH="results/ReasonVOS/${VERSION}/" 

python evaluation/reason_vos/eval_reason_vos.py \
  --pred_path $VIS_SAVE_PATH \
  --save_name $VIS_SAVE_PATH"result.json" | tee -a $VIS_SAVE_PATH"result.txt"

