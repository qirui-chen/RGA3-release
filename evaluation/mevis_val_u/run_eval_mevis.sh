VERSION="SurplusDeficit/UniGR-3B"
VIS_SAVE_PATH="results/MeViS-valid_u/${VERSION}/" 


# Step-2: run evaluation
python evaluation/mevis_val_u/eval_mevis.py \
  --mevis_pred_path $VIS_SAVE_PATH"Annotations/" \
  --save_name $VIS_SAVE_PATH"result.json" | tee -a $VIS_SAVE_PATH"result.txt"

