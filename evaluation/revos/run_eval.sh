VERSION="SurplusDeficit/UniGR-3B"
VIS_SAVE_PATH="results/ReVOS/${VERSION}/" 

# Step-2: run evaluation
python evaluation/revos/eval_revos.py \
    $VIS_SAVE_PATH \
    --visa_exp_path /mnt/ali-sh-1/dataset/zeus/chenqr/datasets/qrchen_dataset/other-datasets/ReVOS/meta_expressions_valid_.json \
    --visa_mask_path /mnt/ali-sh-1/dataset/zeus/chenqr/datasets/qrchen_dataset/other-datasets/ReVOS/mask_dict.json \
    --visa_foreground_mask_path /mnt/ali-sh-1/dataset/zeus/chenqr/datasets/qrchen_dataset/other-datasets/ReVOS/mask_dict_foreground.json

