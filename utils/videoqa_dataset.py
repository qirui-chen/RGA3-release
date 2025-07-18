import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


import glob
from decord import VideoReader, cpu
import sys
# sys.path.append("/PATH/TO/utils")


from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import rank0_print
from .utils import preprocess, DirectResize



class GeneralVideoQADataset(torch.utils.data.Dataset):
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

        self.base_image_dir = base_image_dir
        self.image_size = image_size

        self.precision = precision
        self.transform = DirectResize(image_size) # ResizeLongestSide(image_size)
        self.preprocess = preprocess

        DATA_DIR = os.path.join(base_image_dir, "../LLaVA-Video-178K")

        self.general_videoqa_train = LLaVAVideoDataset(
                                        root=DATA_DIR,
                                        num_frames=num_frames_mllm,
                                        )

        rank0_print('Using General VideoQA dataset, {} from train'.format(len(self.general_videoqa_train)))


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
        idx = random.randint(0, len(self.general_videoqa_train) - 1)
        image_list, target = self.general_videoqa_train.__getitem__(idx)

        dense_indices = self.get_dense_indices()

        # pre-process for SAM
        image_sam_list = []
        for idx, image in enumerate(image_list):
            if idx in dense_indices:
                image_sam = self.transform.apply_image(image)
                image_sam_list.append(image_sam)
        resize = image_sam_list[0].shape[:2]


        source = target["conversations"]
        roles = {"human": "user", "gpt": "assistant"}
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]

        frames_list = [Image.fromarray(img) for img in image_list]
        messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role in ["user", "assistant"], f"{j}"
            text = sentence["value"].replace('<image>', '').strip()

            content = []
            if role == "user" and j == 0:
                content.append({"type": "video", "video": frames_list, "max_pixels": self.max_pixels})
            
            content.append({"type": "text", "text": text})
            messages.append({"role": role, "content": content})


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
            messages,
            masks,
            label,
            resize,
        )



def read_uniform_frames(video_path, num_frames=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    frame_indices = []
    
    for i in range(num_frames):
        
        start = (i * total_frames) // num_frames
        end = ((i + 1) * total_frames // num_frames) - 1
        
        if end < start:
            end = start
        
        start = max(0, start)
        end = min(end, total_frames - 1)
        
        idx = random.randint(start, end)
        frame_indices.append(idx)
    
    frames = [vr[i].asnumpy() for i in frame_indices]
    return frames


class LLaVAVideoDataset(Dataset):

    def __init__(self,
                 root,
                 num_frames,
                 ):
        self.root = root
        self.num_frames = num_frames

        self.prepare_metas()


    def find_json_files(self, directory):

        patterns = [
            os.path.join(directory, '**', '*activitynetqa*.json'), #0_30_s_academic_oe_*
            os.path.join(directory, '**', '*nextqa*.json'),
            os.path.join(directory, '**', '*perceptiontest*.json'), 
            # os.path.join(directory, '**', '0_30_s_academic*qa*.json'), 
            # os.path.join(directory, '**', '30_60_s_academic*qa*.json'), 
        ]

        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))

        return files

    def prepare_metas(self):

        paths = self.find_json_files(self.root)

        self.metas = []
        for path in paths:
            data = json.load(open(path, "r"))
            if isinstance(data, list):
                self.metas.extend(data)


    def __len__(self):
        return len(self.metas)


    def __getitem__(self, idx):
        meta = self.metas[idx]  # dict

        conversations = meta['conversations']
        video_path = os.path.join(self.root, meta['data_source'], meta['video'])

        try:
            imgs = read_uniform_frames(video_path)
        except Exception:
                return self.__getitem__(min(idx + 1, len(self.metas)))

        # transform
        h, w = imgs[0].shape[:2]
        target = {
            'orig_size': (int(h), int(w)),
            'conversations': conversations,
        }

        return imgs, target


if __name__ == '__main__':


    dataset = LLaVAVideoDataset("/mnt/ali-sh-1/dataset/zeus/chenqr/data/qrchen_data/set/LLaVA-Video-178K", 32)

    imgs, target = dataset[20000]
    
    print(imgs[0].shape, target['conversations'])