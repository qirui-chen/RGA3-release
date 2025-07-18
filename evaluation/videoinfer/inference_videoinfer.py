import argparse
import json
import os
import sys
from tqdm import tqdm
from glob import glob

from qwen_vl_utils import process_vision_info
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoProcessor

import numpy as np
from PIL import Image

sys.path.append(".")
from utils.utils import get_sparse_indices, REFERRING_VQA_PROMPT, VISUAL_PROMPT, words_shape, set_seed, show_frames
from utils.visual_prompt_generator import blend_image_from_mask

from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from model.STOM import STOM

from pycocotools import mask as coco_mask

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VideoInferDataset(Dataset):
    def __init__(
        self, qa_data, mask_dict, video_folder, num_frames, subset_idx, subset_num, oracle=False
    ):
        self.qa_data = qa_data
        self.mask_dict = mask_dict
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.oracle = oracle

        # Prepare job list
        self.job_list = []
        for vid_name in qa_data.keys():
            vid = qa_data[vid_name]
            for exp_id in vid["expressions"]:
                for qa_id, qa in vid["expressions"][exp_id]["QA"].items():
                    self.job_list.append((vid_name, exp_id, qa_id))

        # Subset splitting
        self.job_list_subset = [
            self.job_list[i]
            for i in range(len(self.job_list))
            if i % subset_num == subset_idx
        ]

    def __len__(self):
        return len(self.job_list_subset)

    def __getitem__(self, idx):
        vid_id, exp_id, qa_id = self.job_list_subset[idx]
        qa_item = self.qa_data[vid_id]["expressions"][exp_id]["QA"][qa_id]

        # Prepare frames and masks
        image_folder = os.path.join(self.video_folder, "frames", vid_id)
        image_file_list = sorted(glob(os.path.join(image_folder, "*.jpg")))
        total_frames = len(image_file_list)

        # Prepare prompted frame idx and visual prompt
        vip_folder = os.path.join(self.video_folder, "visual_prompts", vid_id, exp_id)
        overlayed_frame_idx = self.qa_data[vid_id]["expressions"][exp_id][
            "overlayed_frame_idx"
        ]
        color = self.qa_data[vid_id]["expressions"][exp_id]["color"]
        shape = self.qa_data[vid_id]["expressions"][exp_id]["shape"]

        vip_img_np = np.load(os.path.join(vip_folder, f"{shape}.npz"))["arr_0"]
        vip_img = Image.fromarray(vip_img_np)

        sparse_idxs = get_sparse_indices(total_frames, self.num_frames - 1)
        sparse_idxs.append(overlayed_frame_idx)
        sparse_idxs = sorted(sparse_idxs)

        original_frames, blended_frames, masks, is_key_frame = [], [], [], []

        for frm_idx in sparse_idxs:
            image_path = image_file_list[frm_idx]
            image_pil = Image.open(image_path).convert("RGB")

            mask = np.zeros(image_pil.size[::-1], dtype=np.float32)
            anno_id = self.qa_data[vid_id]["expressions"][exp_id]["anno_id"]
            frm_anno = self.mask_dict[str(anno_id)][frm_idx]
            if frm_anno is not None:
                mask += coco_mask.decode(frm_anno)

            original_frames.append(image_pil)
            masks.append(mask)

            if frm_idx != overlayed_frame_idx:
                if self.oracle:
                    blended_frames.append(blend_image_from_mask(image_pil, mask, color, shape))
                else:
                    blended_frames.append(image_pil)
                is_key_frame.append(False)
            else:
                # alpha blend
                blended_frame = Image.alpha_composite(
                    image_pil.convert("RGBA"), vip_img
                )
                blended_frame = blended_frame.convert("RGB")
                blended_frames.append(blended_frame)
                is_key_frame.append(True)

        return {
            "vid_id": vid_id,
            "exp_id": exp_id,
            "qa_id": qa_id,
            "Q": qa_item["Q"],
            "A": qa_item["A"],
            "original_frames": original_frames,
            "blended_frames": blended_frames,
            "vip_img": vip_img,
            "masks": masks,
            "is_key_frame": is_key_frame,
            "color": color,
            "shape": shape,
        }


def collate_fn(batch):
    vid_id = [item["vid_id"] for item in batch]
    exp_id = [item["exp_id"] for item in batch]
    qa_id = [item["qa_id"] for item in batch]
    Q = [item["Q"] for item in batch]
    A = [item["A"] for item in batch]
    original_frames = [item["original_frames"] for item in batch]
    blended_frames = [item["blended_frames"] for item in batch]
    vip_img = [item["vip_img"] for item in batch]
    masks = [item["masks"] for item in batch]
    is_key_frame = [item["is_key_frame"] for item in batch]
    color = [item["color"] for item in batch]
    shape = [item["shape"] for item in batch]

    return {
        "vid_id": vid_id,
        "exp_id": exp_id,
        "qa_id": qa_id,
        "Q": Q,
        "A": A,
        "original_frames": original_frames,
        "blended_frames": blended_frames,
        "vip_img": vip_img,
        "masks": masks,
        "is_key_frame": is_key_frame,
        "color": color,
        "shape": shape,
    }


def parse_args(args):
    parser = argparse.ArgumentParser(description="VideoInfer Inference")
    parser.add_argument("--version", default="PATH/TO/MODEL")
    parser.add_argument("--data_root", default="PATH/TO/DATASET")
    parser.add_argument(
        "--vis_save_path", default="./results/VideoLISA-ReferQA/", type=str
    )
    parser.add_argument("--subset_num", default=8, type=int)
    parser.add_argument("--subset_idx", default=0, type=int)
    parser.add_argument("--num_frames", default=50, type=int)
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--use_stom", action="store_true", default=False)

    return parser.parse_args(args)


def main(args):
    # ---------------------------- config env ------------------------------------
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # ---------------------------- prepare model ------------------------------------
    # Create model
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

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
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,
    )
    propagator = STOM(device="cuda:0")

    model = model.cuda()
    model.eval()

    # ---------------------------- read data ------------------------------------
    qa_path = (
        os.path.join(args.data_root, "test.json")
    )
    qa_data = json.load(open(qa_path, "r"))
    mask_dict = json.load(
        open(
            os.path.join(args.data_root, "mask_dict.json"),
            "r",
        )
    )
    video_folder = (
        args.data_root
    )

    # Create dataset and dataloader
    dataset = VideoInferDataset(
        qa_data,
        mask_dict,
        video_folder,
        args.num_frames,
        args.subset_idx,
        args.subset_num,
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=16 if not args.use_stom else 0)

    # ------------------ start processing -------------------------
    os.makedirs(args.vis_save_path, exist_ok=True)
    output_file = f"{args.vis_save_path}/pred_{args.subset_idx}.json"
    if os.path.exists(output_file):
        result = json.load(open(output_file))
    else:
        result = {}

    cnt = 0
    for batch in tqdm(dataloader, desc="Progress {}".format(args.subset_idx)):
        vid_id = batch["vid_id"][0]
        exp_id = batch["exp_id"][0]
        qa_id = batch["qa_id"][0]
        Q = batch["Q"][0]
        A = batch["A"][0]
        original_frames = batch["original_frames"][0]
        blended_frames = batch["blended_frames"][0]
        vip_img = batch["vip_img"][0]
        masks = batch["masks"][0]
        is_key_frame = batch["is_key_frame"][0]
        color = batch["color"][0]
        shape = batch["shape"][0]


        if args.vis:
            sample_save_path = os.path.join(args.vis_save_path, vid_id, exp_id)
            tracking_save_path = os.path.join(sample_save_path, "tracking_frames")
            os.makedirs(tracking_save_path, exist_ok=True)

        if args.use_stom:
            if not np.any(np.array(vip_img)[:, :, 3] > 0):
                cnt += 1
                propagated_frames = blended_frames
            else:
                propagated_frames = propagator.propagate_in_video(
                    original_frames,
                    vip_img,
                    is_key_frame.index(True),
                    shape,
                    tracking_save_path if args.vis else ""
                )

        if vid_id in result:
            if exp_id in result[vid_id]:
                if qa_id in result[vid_id][exp_id]:
                    continue

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": blended_frames if not args.use_stom else propagated_frames,
                    },
                    {
                        "type": "text",
                        "text": REFERRING_VQA_PROMPT.format(text=Q) + " Please answer in one sentence.",
                        # "text": VISUAL_PROMPT.format(prep=words_shape[shape], color=color, shape=shape) + " Please answer in one sentence.",
                    },
                    # {"type": "text", "text": Q},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Run inference
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        torch.cuda.empty_cache()

        # Save the results
        if vid_id not in result:
            result[vid_id] = {}
        if exp_id not in result[vid_id]:
            result[vid_id][exp_id] = {}
        result[vid_id][exp_id][qa_id] = output_text

        if args.vis:
            os.makedirs(os.path.join(sample_save_path, "blended_frames"), exist_ok=True)
            os.makedirs(
                os.path.join(sample_save_path, "propagated_frames"), exist_ok=True
            )
            for i, (blended_frame, propagated_frame) in enumerate(
                zip(blended_frames, propagated_frames)
            ):
                blended_frame.save(
                    os.path.join(sample_save_path, "blended_frames", f"{i}.png")
                )
                propagated_frame.save(
                    os.path.join(sample_save_path, "propagated_frames", f"{i}.png")
                )

            frames = blended_frames + propagated_frames
            show_frames(frames, sample_save_path, cols=4, reset_every_n_rows=2)
            vip_img.convert("RGB")
            vip_img.save(os.path.join(sample_save_path, "vip_img.png"))
            print(is_key_frame.index(True), color, shape)
            exit(0)

    print(f"Skip cnt: {cnt}\n")
    # Saving the results to a file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    set_seed(100)
    main(sys.argv[1:])
