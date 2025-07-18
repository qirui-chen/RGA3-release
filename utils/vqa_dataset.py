import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import rank0_print
from .utils import preprocess, DirectResize


class VQADataset(torch.utils.data.Dataset):
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
        vqa_data="llava_instruct_150k",
        num_frames_mllm=50,
        num_frames_sam=4,
        max_pixels=1280*28*28,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size

        self.precision = precision
        self.transform = DirectResize(image_size) # ResizeLongestSide(image_size)
        self.preprocess = preprocess

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        rank0_print("vqa_data: ", len(self.vqa_data))

        self.num_frames_mllm = num_frames_mllm
        self.num_frames_sam = num_frames_sam
        self.max_pixels=max_pixels

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        img_pil = Image.open(image_path).convert('RGB')
        source = item["conversations"]
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
                content.append({"type": "image", "image": img_pil, "max_pixels": self.max_pixels})
            
            content.append({"type": "text", "text": text})
            messages.append({"role": role, "content": content})

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size) #NOTE: 0 for the dense dimension
        label = torch.ones(ori_size) * self.ignore_label

        # Repeat image into video
        image = torch.stack([image] * self.num_frames_sam, dim=0)
        masks = torch.cat([masks] * self.num_frames_sam, dim=0)

        assert image.shape[0] == self.num_frames_sam
        assert masks.shape[0] == self.num_frames_sam or masks.shape[0] == 0

        return (
            image_path,
            image,
            messages,
            masks,
            label,
            resize,
        )

