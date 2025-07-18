EXP_NAME="7B-8k-f8-4-lr4e-5-b2g8-p4-18-5-all-r128a256"
RUN_NAME="mixed"

RUN_ROOT="./all_runs/${RUN_NAME}/${EXP_NAME}"
# CHOICE="ckpt_best" 
CHOICE="ckpt_latest" 

python $RUN_ROOT/$CHOICE/zero_to_fp32.py $RUN_ROOT/$CHOICE/ $RUN_ROOT/$CHOICE/pytorch_model


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python merge_lora_weights_and_save_hf_model.py \
  --version="/PATH/TO/Qwen2.5-VL-7B-Instruct" \
  --lora_r 128 \
  --lora_alpha 256 \
  --weight="${RUN_ROOT}/$CHOICE/pytorch_model" \
  --save_path="${RUN_ROOT}/merged"
