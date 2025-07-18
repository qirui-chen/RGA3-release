"""
https://github.com/cilinyan/VISA/blob/main/utils/rvos_dataset.py
"""
import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import cv2
import json
import numpy as np
import random
import torch.nn.functional as F

from pycocotools import mask as coco_mask
# from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, LONG_QUESTION_LIST
from .utils import rank0_print
from .utils import preprocess, DirectResize


class ReasonVOSDataset(torch.utils.data.Dataset):
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
        exclude_val=False,
        num_frames_mllm=50,
        num_frames_sam=4,
        max_pixels=384*28*28,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        # self.base_image_dir = base_image_dir
        self.base_image_dir = "/mnt/ali-sh-1/dataset/zeus/chenqr/datasets/qrchen_dataset/other-datasets/ReVOS"
        self.image_size = image_size

        self.precision = precision
        self.transform = DirectResize(image_size) # ResizeLongestSide(image_size)
        self.preprocess = preprocess

        self.num_frames_mllm = num_frames_mllm
        self.num_frames_sam = num_frames_sam
        self.max_pixels = max_pixels

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.ReVOS = ReVOSDataset(img_folder=self.base_image_dir,
                                  ann_file=os.path.join(self.base_image_dir,
                                                        "meta_expressions_train_.json"),
                                  num_frames=self.num_frames_mllm)
        rank0_print('Using ReVOS dataset')

    def __len__(self):
        return self.samples_per_epoch

    def get_dense_indices(self):
        sequence = np.arange(self.num_frames_mllm)
        random_numbers = np.random.choice(sequence, size=self.num_frames_sam, replace=False)

        return sorted(random_numbers.tolist())

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.ReVOS) - 1)
        image_list, target = self.ReVOS.__getitem__(idx)

        dense_indices = self.get_dense_indices()

        # pre-process for SAM
        image_sam_list = []
        for idx, image in enumerate(image_list):
            if idx in dense_indices:
                image_sam = self.transform.apply_image(image)
                image_sam_list.append(image_sam)
        resize = image_sam_list[0].shape[:2]

        mask_list = []
        for idx in range(target["masks"].shape[0]):
            if idx in dense_indices:
                mask_list.append(target["masks"][idx])
        masks = torch.stack(mask_list, dim=0)

        questions = []
        answers = []
        sampled_classes = [target["exp"]]
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split('||')) == 1
            if text[-1] == "?":
                question = random.choice(self.long_question_list).format(sent=text)
            else:
                if text and text[0].islower() and text.endswith('.'):
                    text = text[:-1]
                question = random.choice(self.short_question_list).format(class_name=text)            
            questions.append(question)
            answers.append(random.choice(self.answer_list))

        assert len(questions) == 1
        frames_list = [Image.fromarray(img) for img in image_list]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames_list,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": questions[0]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answers[0]},
                ],
            }
        ]

        image_sam_list_proc = []
        for image in image_sam_list:
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            image_sam_list_proc.append(image)
        image_sam_tsr = torch.stack(image_sam_list_proc, dim=0)

        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        assert image_sam_tsr.shape[0] == masks.shape[0] == self.num_frames_sam

        return (
            None,
            image_sam_tsr,
            messages,
            masks,
            label,
            resize,
        )


class ReVOSDataset(Dataset):

    def __init__(self,
                 img_folder,
                 ann_file,
                 num_frames,
                 split='train',
                 ):
        self.img_folder = img_folder
        self.ann_file = ann_file

        self.num_frames = num_frames
        mask_json = os.path.join(str(self.img_folder), 'mask_dict.json')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)

        self.prepare_metas()

    def prepare_metas(self):
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            meta_exp = json.load(f)
        vid_id_list = list(meta_exp['videos'].keys())

        self.metas = []
        for vid in vid_id_list:

            vid_data = meta_exp['videos'][vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            for exp_id, exp_dict in meta_exp['videos'][vid]['expressions'].items():
                
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['vid'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = exp_dict['obj_id']
                    meta['anno_id'] = exp_dict['anno_id']
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id

                    self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]  # dict

        vid, exp, obj_id, anno_id, frames, frame_id = \
            meta['vid'], meta['exp'], meta['obj_id'], meta['anno_id'], meta['frames'], meta['frame_id']
        
        vid_len = len(frames)

        num_frames = self.num_frames
        # random sparse sample
        sample_indx = [frame_id]
        if self.num_frames != 1:
            # # local sample
            # sample_id_before = random.randint(1, 3)
            # sample_id_after = random.randint(1, 3)
            # local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
            # sample_indx.extend(local_indx)

            # global sampling
            if num_frames > len(sample_indx):
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)), global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >= global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len), global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    num_repeat = global_n // vid_len
                    select_id = random.sample(range(vid_len), global_n % vid_len) + list(range(vid_len)) * num_repeat
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                    assert len(sample_indx) == self.num_frames
        sample_indx.sort()      # ensure the video in correct temporal order

        # read frames and masks
        imgs, masks = [], []
        for j in range(self.num_frames):
            frame_indx = sample_indx[j]
            frame_name = frames[frame_indx]
            img_path = os.path.join(str(self.img_folder), 'JPEGImages', vid, frame_name + '.jpg')
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = np.zeros(img.shape[:2], dtype=np.float32)
            for x in anno_id:
                frm_anno = self.mask_dict[str(x)][frame_indx]
                if frm_anno is not None:
                    mask += coco_mask.decode(frm_anno)

            # append
            mask = torch.from_numpy(mask)
            imgs.append(img)
            masks.append(mask)

        # transform
        h, w = imgs[0].shape[:2]
        masks = torch.stack(masks, dim=0)
        target = {
            'masks': masks,  # [T, H, W]
            'exp': exp,
            'orig_size': torch.as_tensor([int(h), int(w)]),
        }

        return imgs, target


if __name__ == '__main__':
    pass