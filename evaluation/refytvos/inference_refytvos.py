import argparse
import json
import os
import sys
from tqdm import tqdm
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, BitsAndBytesConfig
from PIL import Image
from qwen_vl_utils import process_vision_info

sys.path.append(".")
# from model.segment_anything.utils.transforms import ResizeLongestSide
# from model.qwen_2_5_vl import UniGRConfig, UniGRModel
from utils.utils import DirectResize
from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from utils.utils import get_sparse_indices, dict_to_cuda, preprocess


def parse_args(args):
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--version", default="PATH/TO/MODEL")
    parser.add_argument("--vis_save_path", default="", type=str)
    parser.add_argument("--save_overlay", action="store_true", default=False)
    parser.add_argument("--subset_num", default=8, type=int)
    parser.add_argument("--subset_idx", default=0, type=int)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--num_frames_mllm", default=4, type=int)
    parser.add_argument("--max_pixels", default=384*28*28, type=int)

    return parser.parse_args(args)



def main(args):
    # ---------------------------- config env ------------------------------------
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # ---------------------------- prepare model ------------------------------------
    model_args = {
        "train_mask_decoder": False,
        "seg_token_idx": args.seg_token_idx,
    }
    config = UniGRConfig.from_pretrained(
        args.version,
        **model_args,
    )
    model = UniGRModel.from_pretrained(
        args.version,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,
    )

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    else:
        raise NotImplementedError

    transform = DirectResize(args.image_size)

    model.eval()

    # ---------------------------- read data ------------------------------------
    meta_exp_path = "/mnt/ali-sh-1/dataset/zeus/chenqr/datasets/qrchen_dataset/other-datasets/Ref-Youtube-VOS/meta_expressions/valid/meta_expressions.json"
    meta_exp = json.load(open(meta_exp_path, 'r'))
    video_folder = "/mnt/ali-sh-1/dataset/zeus/chenqr/datasets/qrchen_dataset/other-datasets/Ref-Youtube-VOS/valid/JPEGImages"

    # ------------------ split into subsets for each GPU -------------------------
    job_list = []
    vid_id_list = os.listdir(video_folder)
    for vid_id in vid_id_list:
        for exp_id in list(meta_exp['videos'][vid_id]['expressions'].keys()):
            job_list.append((vid_id, exp_id))

    job_list_subset = [job_list[i] for i in range(len(job_list)) if i % args.subset_num == args.subset_idx]

    total_infer = len(job_list_subset)
    progress_bar = tqdm(total=total_infer, desc='Progress {}'.format(args.subset_idx))
    print(f"RefYTVOS: Total {len(vid_id_list)} videos, {len(job_list)} jobs.")

    # ------------------ start processing -------------------------
    for vid_id, exp_id in job_list_subset:
        # prepare folder
        save_dir_vid = os.path.join(args.vis_save_path, vid_id)
        os.makedirs(save_dir_vid, exist_ok=True)

        save_dir_vid_exp = os.path.join(args.vis_save_path, vid_id, exp_id)
        os.makedirs(save_dir_vid_exp, exist_ok=True)

        # prepare video frames
        image_folder = os.path.join(video_folder, vid_id)
        if not os.path.exists(image_folder):
            print("File not found in {}".format(image_folder))
            raise FileNotFoundError

        image_file_list = sorted(glob(os.path.join(image_folder, '*.jpg')))
        total_frames = len(image_file_list)
        sparse_idxs = get_sparse_indices(total_frames, args.num_frames_mllm)

        if os.path.exists(save_dir_vid_exp) and len(os.listdir(save_dir_vid_exp)) == total_frames:
            continue

        # prepare text query and prompt
        ref_query = meta_exp['videos'][vid_id]['expressions'][exp_id]['exp']
        prompt_template = "Please segment the {class_name} in this image."
        prompt = prompt_template.format(class_name=ref_query.lower())

        # pre-process images
        image_list_sam, frames_list, image_list_np = [], [], []

        for frm_idx in sparse_idxs:
            image_path = image_file_list[frm_idx]
            image_pil = Image.open(image_path).convert("RGB")
            frames_list.append(image_pil)

        for frm_idx in range(total_frames):
            image_path = image_file_list[frm_idx]
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

            image_list_sam.append(image)
            image_list_np.append(image_np)

        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": frames_list, "max_pixels": args.max_pixels},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Sure, [SEG]."}  # teacher forcing
            ]}
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        inputs = dict_to_cuda(inputs)
        input_ids = inputs['input_ids']
    
        attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
        pixel_values = inputs['pixel_values'].bfloat16() if 'pixel_values' in inputs else None
        pixel_values_videos = inputs['pixel_values_videos'].bfloat16() if 'pixel_values_videos' in inputs else None
        image_grid_thw = inputs['image_grid_thw'] if 'image_grid_thw' in inputs else None
        video_grid_thw = inputs['video_grid_thw'] if 'video_grid_thw' in inputs else None
        second_per_grid_ts = inputs['second_per_grid_ts'] if 'second_per_grid_ts' in inputs else None

        # stack as video
        image_sam = torch.stack(image_list_sam, dim=1)

        output_ids, pred_masks = model.evaluate(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            image_sam,
            resize_list,
            original_size_list,
        )

        for i, pred_mask_vid in enumerate(pred_masks):
            if pred_mask_vid.shape[0] == 0:
                continue

            assert total_frames == pred_mask_vid.shape[0]

            for frame_idx in range(total_frames):
                pred_mask = pred_mask_vid.detach().cpu().numpy()[frame_idx]
                pred_mask = pred_mask > 0

                save_path = "{}/{}.png".format(save_dir_vid_exp,
                                               os.path.basename(image_file_list[frame_idx]).split('.')[0])
                binary_mask = np.where(pred_mask > 0, 1, 0)
                cv2.imwrite(save_path, binary_mask * 255)

                if args.save_overlay:
                    save_path = "{}/masked_img_{}.jpg".format(save_dir_vid_exp, frame_idx)
                    save_img = image_list_np[frame_idx].copy()
                    save_img[pred_mask] = (
                            image_list_np[frame_idx] * 0.5
                            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                    )[pred_mask]
                    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, save_img)

        torch.cuda.empty_cache()
        progress_bar.update(1)


if __name__ == "__main__":
    main(sys.argv[1:])
