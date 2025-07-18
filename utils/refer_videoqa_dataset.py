import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask
from PIL import Image

import sys
sys.path.append(".")

from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import rank0_print, VISUAL_PROMPT, uniform_random_sample
from .utils import preprocess, DirectResize
from .visual_prompt_generator import video_blending_keyframes, color_pool, words_shape



class ReferVideoQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val: bool = False,
        num_frames_mllm: int = 50,
        num_frames_sam: int = 4,
        max_pixels: int = 384 * 28 * 28,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.precision = precision
        self.transform = DirectResize(image_size) # ResizeLongestSide(image_size)
        self.preprocess = preprocess

        DATA_DIR = os.path.join(base_image_dir, "VideoInfer-Release")

        self.ref_videoqa_train = ReVOSPlusDataset(img_folder=DATA_DIR,
                                        ann_file=os.path.join(DATA_DIR, "train.json"),
                                        num_frames=num_frames_mllm,
                                        max_pixels=max_pixels,
                                        overlay=True,
                                        )

        rank0_print('Using RefVideoQA dataset, {} from train'.format(len(self.ref_videoqa_train)))


        self.num_frames_mllm = num_frames_mllm
        self.num_frames_sam = num_frames_sam
        self.max_pixels = max_pixels

    def __len__(self):
        return self.samples_per_epoch

    def get_dense_indices(self):
        sequence = np.arange(self.num_frames_mllm)
        random_numbers = np.random.choice(sequence, size=self.num_frames_sam, replace=False)

        return sorted(random_numbers.tolist())

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.ref_videoqa_train) - 1)
        image_list, target = self.ref_videoqa_train.__getitem__(idx)

        dense_indices = self.get_dense_indices()

        # pre-process for SAM
        image_sam_list = []
        for idx, image in enumerate(image_list):
            if idx in dense_indices:
                image_sam = self.transform.apply_image(image)
                image_sam_list.append(image_sam)
        resize = image_sam_list[0].shape[:2]

        image_sam_list_proc = []
        for image in image_sam_list:
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            image_sam_list_proc.append(image)
        image_sam_tsr = torch.stack(image_sam_list_proc, dim=0)

        ori_size = target['orig_size']
        masks = torch.rand(0, *ori_size) #NOTE: 0 for the dense dimension to avoid segmentation loss
        label = torch.ones(ori_size) * self.ignore_label

        masks = torch.cat([masks] * self.num_frames_sam, dim=0)

        assert image_sam_tsr.shape[0] == self.num_frames_sam
        assert masks.shape[0] == self.num_frames_sam or masks.shape[0] == 0

        return (
            None,
            image_sam_tsr,
            target['messages'],
            masks,
            label,
            resize,
        )




class ReVOSPlusDataset(Dataset):

    def __init__(self,
                 img_folder,
                 ann_file,
                 num_frames,
                 max_pixels,
                 overlay=False,
                 ):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.num_frames = num_frames
        self.max_pixels = max_pixels
        self.overlay = overlay

        self.prepare_metas()

        mask_json = os.path.join(str(self.img_folder), 'mask_dict.json')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)


    def prepare_metas(self):
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            video_qa_data = json.load(f)

        self.metas = []
        for vid in video_qa_data.keys():
            vid_data = video_qa_data[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            for exp_id in vid_data['expressions']:
                QA = vid_data['expressions'][exp_id]['QA']

                for qa_id in QA:

                    meta = {
                        'vid': vid,
                        'exp': vid_data['expressions'][exp_id]['exp'],
                        'anno_id': vid_data['expressions'][exp_id]['anno_id'],
                        'QA': QA[qa_id],
                        'frames': vid_data['frames'],
                        'frame_id': random.choice(list(range(vid_len)))
                    }
                    self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]  # dict

        vid, exp, anno_id, QA, frames, frame_id = \
            meta['vid'], meta['exp'], meta['anno_id'], meta['QA'], meta['frames'], meta['frame_id']

        vid_len = len(frames)

        num_frames = self.num_frames
        # random sparse sample
        # sample_indx = [frame_id]
        # if self.num_frames != 1:
        #     # local sample
        #     # sample_id_before = random.randint(1, 3)
        #     # sample_id_after = random.randint(1, 3)
        #     # local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
        #     # sample_indx.extend(local_indx)

        #     # global sampling
        #     if num_frames > len(sample_indx):
        #         all_inds = list(range(vid_len))
        #         global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
        #         global_n = num_frames - len(sample_indx)
        #         if len(global_inds) > global_n:
        #             select_id = random.sample(range(len(global_inds)), global_n)
        #             for s_id in select_id:
        #                 sample_indx.append(global_inds[s_id])
        #         elif vid_len >= global_n:  # sample long range global frames
        #             select_id = random.sample(range(vid_len), global_n)
        #             for s_id in select_id:
        #                 sample_indx.append(all_inds[s_id])
        #         else:
        #             num_repeat = global_n // vid_len
        #             select_id = random.sample(range(vid_len), global_n % vid_len) + list(range(vid_len)) * num_repeat
        #             for s_id in select_id:
        #                 sample_indx.append(all_inds[s_id])
        #             assert len(sample_indx) == self.num_frames

        # sample_indx.sort()          # ensure the video in correct temporal order
        sample_indx = uniform_random_sample(vid_len, num_frames)

        imgs, masks  = [], []
        for j in range(self.num_frames):
            frame_indx = sample_indx[j]
            frame_name = frames[frame_indx]
            img_path = os.path.join(str(self.img_folder), 'frames', vid, frame_name + '.jpg')
            img = Image.open(img_path).convert('RGB')
            
            mask = np.zeros(img.size[::-1], dtype=np.float32)
            for x in anno_id:
                frm_anno = self.mask_dict[str(x)][frame_indx]
                if frm_anno is not None:
                    mask += coco_mask.decode(frm_anno)

            # append
            imgs.append(img)
            masks.append(mask)

        if not self.overlay:
            frames_list = imgs
            prompt = QA['Q']
        else:

            is_key_frame = [False] * self.num_frames
            key_frame_idx = random.randint(0, self.num_frames - 1)
            is_key_frame[key_frame_idx] = True

            color = random.choice(list(color_pool.keys()))
            shape = random.choice(list(words_shape.keys())) #TODO: rectangle for VideoRefer
            prep = words_shape[shape][0]

            blended_frames = video_blending_keyframes(imgs, masks, is_key_frame, color, shape)
            frames_list = blended_frames
            prompt = VISUAL_PROMPT.format(prep=prep, color=color, shape=shape) + QA['Q']

            # blended_frames[key_frame_idx].save(f'{key_frame_idx}.png')
            # blended_frames[0].save('0.png')

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames_list,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": QA['A']},
                ],
            }
        ]

        # transform
        imgs = [np.array(img) for img in imgs]
        h, w = imgs[0].shape[:2]
        target = {
            'exp': exp,
            'orig_size': (int(h), int(w)),
            'messages': messages,
            'prompt': prompt,
            'answer': QA['A'],
        }

        return imgs, target
