import torch
import sys
sys.path.append("/PATH/TO/Osprey")
from osprey.utils import disable_torch_init
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from osprey.mm_utils import tokenizer_image_token
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.train.train import DataArguments

from functools import partial
import os
import numpy as np
import cv2
import argparse
import json

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from glob import glob
from pycocotools import mask as coco_mask


data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

def show_mask(mask, image, random_color=True, img_trans=0.9, mask_trans=0.5, return_color=False):
  if random_color:
    color = np.concatenate([np.random.random(3)*255], axis=0)
  else:
    color = np.array([30, 144, 255])
  h,w = mask.shape[-2:]
  mask_image = mask.reshape(h,w,1)*color.reshape(1,1,-1)

  image = cv2.addWeighted(image, img_trans, mask_image.astype('uint8'), mask_trans , 0)
  if return_color:
    return image, mask_image
  else:
    return image

class Osprey():
    def __init__(self, model_path, device='cuda'):
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=True
        )
        self.model = OspreyLlamaForCausalLM.from_pretrained(
                                                model_path,
                                                torch_dtype=torch.bfloat16,
                                                ).to(device)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
                                                    do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                    image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        
        spi_tokens = ['<mask>', '<pos>']
        self.tokenizer.add_tokens(spi_tokens, special_tokens=True)
        
        for m in self.model.modules():
            m.tokenizer = self.tokenizer

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device=device)

        begin_str = """<image>\n\nThis provides an overview of the picture.\n"""

        short_question = 'Please give me a short description of <mask><pos>. Using a short phrase.'

        conv = conv_templates['osprey_v1'].copy()
        qs = begin_str+short_question
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        self.input_ids_short = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        detailed_question = 'Can you give me a detailed description of <mask><pos>?'
        
        conv = conv_templates['osprey_v1'].copy()
        qs = begin_str+detailed_question
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        self.input_ids_detailed = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2



    def osprey_predict(self, img, mask, question=None):
        image = self.image_processor.preprocess(img,
                                do_center_crop=False,
                                return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(512, 512),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)

        masks = torch.Tensor(mask).unsqueeze(0).to(self.model.device)

        
        conv = conv_templates['osprey_v1'].copy()
        begin_str = """<image>\n\nThis provides an overview of the picture.\n"""
        qs = begin_str + "Answer the question of <mask><pos>: " + question
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        # self.model.model.tokenizer = self.tokenizer

        with torch.inference_mode():

            self.model.orig_forward = self.model.forward
            self.model.forward = partial(self.model.orig_forward,
                                        img_metas=[None],
                                        masks=[masks.half()])
            
            output_ids = self.model.generate(
                input_ids,
                images=image.unsqueeze(0).half().to(self.model.device),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                num_beams=1,
                # stopping_criteria=[stopping_criteria]
                )

            self.model.forward = self.model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                            skip_special_tokens=True)[0]
    
        outputs = outputs.strip()
        if outputs.endswith(self.stop_str):
            outputs = outputs[:-len(self.stop_str)]
        outputs = outputs.strip()
        if ':' in outputs:
            outputs = outputs.split(':')[1]

        outputs_list = outputs.split('.')
        outputs_list_final = []
        outputs_str = ''
        for output in outputs_list:
            if output not in outputs_list_final:
                if output=='':
                    continue
                outputs_list_final.append(output)
                outputs_str+=output+'.'
            else:
                break
        return outputs_str


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

    model_path = "/PATH/TO/Osprey-7b"
    model = Osprey(model_path=model_path)

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

        vip_img_mask = np.array(vip_img)[:, :, 3] > 0
        output = model.osprey_predict(blended_frames[vip_img_idx], vip_img_mask, question=Q)

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
