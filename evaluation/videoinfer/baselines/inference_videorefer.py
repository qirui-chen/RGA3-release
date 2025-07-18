import sys

sys.path.append("/PATH/TO/VideoRefer")
from videorefer import model_init, mm_infer
from videorefer.mm_utils import process_video

import os
import argparse
import json

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch

import numpy as np

from PIL import Image
from glob import glob
from pycocotools import mask as coco_mask


class VideoInferDataset(Dataset):
    def __init__(
        self, qa_data, mask_dict, video_folder, num_frames, subset_idx, subset_num
    ):
        self.qa_data = qa_data
        self.mask_dict = mask_dict
        self.video_folder = video_folder
        self.num_frames = num_frames

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
        shape = self.qa_data[vid_id]["expressions"][exp_id]["shape"]

        vip_img_np = np.load(os.path.join(vip_folder, f"{shape}.npz"))["arr_0"]
        vip_img = Image.fromarray(vip_img_np)

        sparse_idxs = get_sparse_indices(total_frames, self.num_frames - 1)
        sparse_idxs.append(overlayed_frame_idx)
        sparse_idxs = sorted(sparse_idxs)
        vip_img_idx = sparse_idxs.index(overlayed_frame_idx)

        original_frames, blended_frames, masks = [], [], []

        for idx, frm_idx in enumerate(sparse_idxs):
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
                blended_frames.append(image_pil)
            else:
                # alpha blend
                blended_frame = Image.alpha_composite(
                    image_pil.convert("RGBA"), vip_img
                )
                blended_frame = blended_frame.convert("RGB")
                blended_frames.append(blended_frame)

        return {
            "vid_id": vid_id,
            "exp_id": exp_id,
            "qa_id": qa_id,
            "Q": qa_item["Q"],
            "A": qa_item["A"],
            "original_frames": original_frames,
            "blended_frames": blended_frames,
            "vip_img": vip_img,
            "vip_img_idx": vip_img_idx,
            "masks": masks,
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
    vip_img_idx = [item["vip_img_idx"] for item in batch]
    masks = [item["masks"] for item in batch]

    return {
        "vid_id": vid_id,
        "exp_id": exp_id,
        "qa_id": qa_id,
        "Q": Q,
        "A": A,
        "original_frames": original_frames,
        "blended_frames": blended_frames,
        "vip_img": vip_img,
        "vip_img_idx": vip_img_idx,
        "masks": masks,
    }


def uniform_sample(total_len, sample_num):
    intervals = np.linspace(start=0, stop=total_len, num=sample_num + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def get_sparse_indices(total_frame_num, num_frames_mllm):
    if total_frame_num > num_frames_mllm:  # video is long, uniformly sample frames
        frame_idxs = uniform_sample(total_frame_num, num_frames_mllm)
        return sorted(frame_idxs)
    else:
        num_repeat = num_frames_mllm // total_frame_num
        num_sample = num_frames_mllm % total_frame_num
        frame_idxs = list(range(total_frame_num)) * num_repeat + uniform_sample(
            total_frame_num, num_sample
        )
        return sorted(frame_idxs)


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

    return parser.parse_args(args)


def main(args):
    # ---------------------------- config env ------------------------------------
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    model_path = "/PATH/TO/VideoRefer-7B"
    model, processor, tokenizer = model_init(model_path)

    for m in model.modules():
        m.tokenizer = tokenizer

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
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=4)

    # ------------------ start processing -------------------------
    os.makedirs(args.vis_save_path, exist_ok=True)
    output_file = f"{args.vis_save_path}/pred_{args.subset_idx}.json"
    if os.path.exists(output_file):
        result = json.load(open(output_file))
    else:
        result = {}

    for batch in tqdm(dataloader, desc="Progress {}".format(args.subset_idx)):
        vid_id = batch["vid_id"][0]
        exp_id = batch["exp_id"][0]
        qa_id = batch["qa_id"][0]
        Q = batch["Q"][0]
        A = batch["A"][0]
        original_frames = batch["original_frames"][0]
        blended_frames = batch["blended_frames"][0]
        vip_img = batch["vip_img"][0]
        vip_img_idx = batch["vip_img_idx"][0]
        masks = batch["masks"][0]

        question = Q
        video_tensor, frame_tensor, height, width = process_video(
            blended_frames, processor=processor, aspect_ratio="square"
        )

        masks = np.array(vip_img)[:, :, 3] > 0
        masks_ = np.array(masks)
        masks_tensor = torch.Tensor(masks_).unsqueeze(0)
        masks_tensor = masks_tensor.unsqueeze(0)

        frame_nums = [len(blended_frames)]
        ann_indices = [[[0]]]

        output = mm_infer(
            video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            masks=masks_tensor.cuda(),
            frame=frame_tensor,
            ann_indices=ann_indices,
            frame_nums=frame_nums,
        )
        # print(output)

        # Save the results
        if vid_id not in result:
            result[vid_id] = {}
        if exp_id not in result[vid_id]:
            result[vid_id][exp_id] = {}
        result[vid_id][exp_id][qa_id] = output

    # Saving the results to a file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    main(sys.argv[1:])
