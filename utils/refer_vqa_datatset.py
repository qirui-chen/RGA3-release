import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image
from skimage.draw import polygon

import sys
sys.path.append(".")

import re
from copy import deepcopy
from argparse import Namespace

# from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import rank0_print, VISUAL_PROMPT
from .visual_prompt_generator import blend_image
from .visual_prompt_organizer import vip_processor
from .utils import preprocess, DirectResize, REFERRING_VQA_PROMPT



class ReferVQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        ref_vqa_dataset="vip_llava",
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        num_frames_mllm=50,
        num_frames_sam=4,
        max_pixels=1280*28*28
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size

        self.precision = precision
        self.transform = DirectResize(image_size) # ResizeLongestSide(image_size)
        self.preprocess = preprocess

        if ref_vqa_dataset == "osprey":
            self.ref_vqa_train = NewOspreyDataset(
                                            img_folder=os.path.join(base_image_dir, "coco/train2014"),
                                            ann_root=os.path.join(base_image_dir, "../Osprey-724K"),
                                            num_frames=num_frames_mllm,
                                            overlay=True,
                                        )
        elif "vip_llava" in ref_vqa_dataset:
            stage = ref_vqa_dataset.split('_')[-1]
            self.ref_vqa_train = ViPLLaVADataset(
                                            img_folder=os.path.join(base_image_dir, "../ViP-LLaVA-Instruct"),
                                            ann_root=os.path.join(base_image_dir, "../ViP-LLaVA-Instruct"),
                                            num_frames=num_frames_mllm,
                                            stage=stage,
                                        )

        rank0_print('Using RefVQA-{} dataset, {} from train'.format(ref_vqa_dataset, len(self.ref_vqa_train)))

        self.num_frames_mllm = num_frames_mllm
        self.num_frames_sam = num_frames_sam
        self.max_pixels = max_pixels

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.ref_vqa_train) - 1)
        image, target = self.ref_vqa_train.__getitem__(idx)

        source = target["conversations"]
        roles = {"human": "user", "gpt": "assistant"}
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]

        messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role in ["user", "assistant"], f"{j}"
            text = sentence["value"].replace('<image>', '').strip()

            content = []
            if role == "user" and j == 0:
                content.append({"type": "image", "image": image, "max_pixels": self.max_pixels})
                text = REFERRING_VQA_PROMPT.format(text=text)
            
            content.append({"type": "text", "text": text})
            messages.append({"role": role, "content": content})

        # pre-process for SAM
        image_np = np.array(image)
        image = self.transform.apply_image(image_np)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        image_sam_tsr = torch.stack([image] * self.num_frames_sam, dim=0)

        ori_size = target['orig_size']
        masks = torch.rand(0, *ori_size) #NOTE: 0 for the dense dimension to avoid segmentation loss
        label = torch.ones(ori_size) * self.ignore_label

        masks = torch.cat([masks] * self.num_frames_sam, dim=0)

        assert image_sam_tsr.shape[0] == self.num_frames_sam
        assert masks.shape[0] == self.num_frames_sam or masks.shape[0] == 0

        return (
            None,
            image_sam_tsr,
            messages,
            masks,
            label,
            resize,
        )



class ViPLLaVADataset(Dataset):

    def __init__(self,
                 img_folder,
                 ann_root,
                 num_frames,
                 stage='stage2-3',
                 ):
        self.img_folder = img_folder
        self.ann_root = ann_root
        self.num_frames = num_frames
        self.stage = stage

        self.prepare_metas()


    def prepare_metas(self):

        self.metas = []
        stage2, stage3 = [], []

        if '2' in self.stage:
            with open(os.path.join(str(self.ann_root), 'vip-llava_stage2_mix.json'), 'r') as f:
                samples = json.load(f)
            for sample in samples:
                if 'image' not in sample or 'conversations' not in sample:
                    continue
                # if 'bboxes' not in sample and 'segmentations' not in sample and 'ocr_vqa' not in sample['image']:
                #     continue
                if 'vg' not in sample['image'] and 'ocr_vqa' not in sample['image'] and 'gqa' not in sample['image'] and 'refcoco' not in sample['id']:
                    continue
                meta = {
                    'image': sample['image'],
                    'line': sample,
                    'visual_prompt': False if 'bboxes' not in sample and 'segmentations' not in sample else True
                }
                stage2.append(meta)

            print(f"ViP-LLaVA Stage2: {len(stage2)} samples")
            self.metas.extend(stage2)

        if '3' in self.stage:
            with open(os.path.join(str(self.ann_root), 'vip-llava_stage3_mix.json'), 'r') as f:
                samples = json.load(f)
            for sample in samples:
                if 'image' not in sample or 'conversations' not in sample:
                    continue
                # if 'bboxes' not in sample and 'segmentations' not in sample and 'ocr_vqa' not in sample['image']:
                #     continue
                #NOTE: vg, ocr_vqa, RefCOCO, gqa
                if 'vg' not in sample['image'] and 'ocr_vqa' not in sample['image'] and 'gqa' not in sample['image'] and 'refcoco' not in sample['id']:
                    continue

                meta = {
                    'image': sample['image'],
                    'line': sample,
                    'visual_prompt': False if 'bboxes' not in sample and 'segmentations' not in sample else True
                }
                stage3.append(meta)
            
            self.metas.extend(stage3)
            print(f"ViP-LLaVA Stage3: {len(stage3)} samples")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]  # dict

        img = Image.open(os.path.join(self.img_folder, meta['image'])).convert("RGB")
        if meta['visual_prompt']:
            try:
                blended_img, conversations = vip_processor(deepcopy(meta['line']), img, min(img.size), data_args = Namespace(**{"image_folder": self.img_folder, "visual_prompt_style": None}))
            except IndexError:
                return self.__getitem__(random.choice(list(range(len(self.metas)))))
        else:
            blended_img, conversations = img, meta['line']['conversations']
        
        ori_size = blended_img.size[::-1]
        target = {
            'file_name': meta['image'],
            'orig_size': ori_size,
            'conversations': conversations,
        }

        return blended_img, target



class NewOspreyDataset(Dataset):

    def __init__(self,
                 img_folder,
                 ann_root,
                 num_frames,
                 overlay=True,
                 ):
        self.img_folder = img_folder
        self.ann_root = ann_root
        self.num_frames = num_frames
        self.overlay = overlay

        self.prepare_metas()


    def prepare_metas(self):
        # read expression data
        with open(os.path.join(str(self.ann_root), 'osprey_conversation.json'), 'r') as f:
            osprey_conversation = json.load(f)

        self.metas = []
        cnt_conv, cnt_desc = 0, 0
        

        for idx, sample in enumerate(osprey_conversation):
            sample['id'] = f'osprey-conv-{idx}'
            sample['segmentations'] = [region['segmentation'] for region in sample['annotation']]
            sample['bboxes'] = []
            for region in sample['annotation']:
                x_min, y_min, width, height = region['bbox']
                x_max = x_min + width
                y_max = y_min + height
                bbox = [x_min, y_min, x_max, y_max]
                sample['bboxes'].append(bbox)
            del sample['annotation']

            meta = {
                'image': sample['file_name'],
                'line': sample,
                'visual_prompt': bool(len(sample['bboxes']))
            }
            self.metas.append(meta)
            cnt_conv += 1


        # with open(os.path.join(str(self.ann_root), 'osprey_detail_description.json'), 'r') as f:
        #     osprey_description = json.load(f)
        # for sample in osprey_description:
        #     for description, annotation in zip(sample['description'], sample['annotation']):
        #         conversations = []
        #         conversations.append({'from': 'human', 'value': 'Describe the region within <region1>.'}) # to be replaced
        #         # remove <region1>: 
        #         conversations.append({'from': 'gpt', 'value': re.sub(r'^<.*?>:\s*', '', description)})

        #         line = {}
        #         x_min, y_min, width, height = annotation['bbox']
        #         x_max = x_min + width
        #         y_max = y_min + height
        #         bbox = [x_min, y_min, x_max, y_max]

        #         line['id'] = f"osprey-desc-{cnt_desc}"
        #         line['conversations'] = conversations
        #         line['segmentations'] = [annotation['segmentation']]
        #         line['bboxes'] = [bbox]

        #         meta = {
        #             'image': sample['file_name'],
        #             'line': line,
        #             'visual_prompt': True
        #         }
        #         self.metas.append(meta)
        #         cnt_desc += 1

        rank0_print(f"Osprey-conv: {len(osprey_conversation)}, Osprey-desc: {cnt_desc}")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]  # dict

        img = Image.open(os.path.join(self.img_folder, meta['image'])).convert("RGB")
        if meta['visual_prompt']:
            blended_img, conversations = vip_processor(deepcopy(meta['line']), img, min(img.size), data_args = Namespace(**{"image_folder": self.img_folder, "visual_prompt_style": None}))
        else:
            blended_img, conversations = img, meta['line']['conversations']
        
        ori_size = blended_img.size[::-1]
        target = {
            'file_name': meta['image'],
            'orig_size': ori_size,
            'conversations': conversations,
        }

        return blended_img, target


if __name__ == '__main__':
    data_dir = "/mnt/ali-sh-1/usr/chenqirui/qrchen_dataset/ViP-LLaVA-Instruct"
    dataset = ViPLLaVADataset(
        img_folder=data_dir,
        ann_root=data_dir,
        num_frames=4,
        stage="stage2",
    )
    # dataset = NewOspreyDataset(
    #     img_folder="/mnt/ali-sh-1/dataset/zeus/chenqr/datasets/qrchen_dataset/other-datasets/coco/train2014",
    #     ann_root="/mnt/ali-sh-1/usr/chenqirui/qrchen_dataset/Osprey-724K",
    #     num_frames=4,
    #     overlay=True,
    # )
    print(len(dataset))
    idx = random.choice(range(len(dataset)))
    img, target = dataset[idx]
    print(target['file_name'])
    from pprint import pprint
    pprint(target['conversations'])
    img.save('0.png')