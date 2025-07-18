import glob
import os
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from PIL import Image

from qwen_vl_utils import process_vision_info

from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import preprocess, DirectResize, rank0_print

# image datasets
from .data_processing import get_mask_from_json
from .sem_seg_dataset import SemSegPseudoVidDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegPseudoVidDataset
from .reason_seg_dataset import ReasonSegPseudoVidDataset
from .vqa_dataset import VQADataset
from .refer_vqa_datatset import ReferVQADataset

# video datasets
from .vos_dataset import YTVOS_Dataset
from .refer_vos_dataset import ReferYTVOSDataset
from .ref_davis_dataset import RefDAVIS_Dataset
from .mevis_dataset import MeViS_Dataset
from .revos_dataset import ReasonVOSDataset
from .videoqa_dataset import GeneralVideoQADataset
from .refer_videoqa_dataset import ReferVideoQADataset

import tokenizers
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

IGNORE_INDEX = -100

def collate_fn(
    batch, processor=None, local_rank=-1
):
    image_path_list = []
    images_list = []
    messages_list = []
    masks_list = []
    label_list = []
    resize_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        messages,
        masks,
        label,
        resize,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        messages_list.append(messages)
        masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        cnt += 1
        offset_list.append(cnt)
        inferences.append(inference)


    text = processor.apply_chat_template(
        messages_list, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages_list, return_video_kwargs=True)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    input_ids = inputs['input_ids']
    
    labels = input_ids.clone()
    masks = torch.ones_like(labels).bool()
    
    for batch_idx in range(input_ids.shape[0]):
        im_start = torch.where(input_ids[batch_idx] == processor.tokenizer.vocab['<|im_start|>'])[0]
        im_end = torch.where(input_ids[batch_idx] == processor.tokenizer.vocab['<|im_end|>'])[0]

        # skip <|im_start|>system 
        for start, end in zip(im_start[1:], im_end[1:]):
            # <|im_start|>user\n
            if input_ids[batch_idx][start + 1] == processor.tokenizer.vocab['user']:
                continue
            elif input_ids[batch_idx][start + 1] == processor.tokenizer.vocab['assistant']:
                # <|im_start|>assistant\n ...... <|im_end|>\n
                masks[batch_idx][start + 3 : end + 1] = False

    labels[masks] = IGNORE_INDEX
    labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX

    attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None

    pixel_values = inputs['pixel_values'] if 'pixel_values' in inputs else None
    image_grid_thw = inputs['image_grid_thw'] if 'image_grid_thw' in inputs else None

    pixel_values_videos = inputs['pixel_values_videos'] if 'pixel_values_videos' in inputs else None
    video_grid_thw = inputs['video_grid_thw'] if 'video_grid_thw' in inputs else None

    second_per_grid_ts = inputs['second_per_grid_ts'] if 'second_per_grid_ts' in inputs else None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "pixel_values_videos": pixel_values_videos,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "second_per_grid_ts": second_per_grid_ts,

        "images_sam": torch.stack(images_list, dim=0),
        "offset": torch.LongTensor(offset_list),
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "inference": inferences[0],
        "messages_list": messages_list,
    }


class ImgVidHybridDataset(torch.utils.data.Dataset):
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
        dataset="refer_seg_video||vid_qa",
        sample_rate=[1, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        reason_seg_data="ReasonSeg|train",
        vqa_data="llava_instruct_150k",
        ref_vos_data="refer_youtube_vos||mevis",
        vos_data="ytvos||mose",
        ref_vqa_data="osprey||vip_llava",
        explanatory=0.1,
        num_frames_mllm=50,
        num_frames_sam=4,
        video_max_pixels=336*28*28,
        image_max_pixels=1280*28*28,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.precision = precision

        self.datasets = dataset.split(",")
        assert len(self.datasets) == len(sample_rate), (len(self.datasets), len(sample_rate), sample_rate)
        sample_rate_expand = []

        self.all_datasets = []
        for idx, dataset in enumerate(self.datasets):
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegPseudoVidDataset(
                        base_image_dir,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        num_frames_mllm,
                        num_frames_sam,
                        image_max_pixels,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegPseudoVidDataset(
                        base_image_dir,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        num_frames_mllm,
                        num_frames_sam,
                        image_max_pixels,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegPseudoVidDataset(
                        base_image_dir,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        num_frames_mllm,
                        num_frames_sam,
                        image_max_pixels,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        num_frames_mllm,
                        num_frames_sam,
                        image_max_pixels,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "ref_vos":
                vid_datasets = ref_vos_data.split(",")
                for vid_data in vid_datasets:
                    if vid_data == "refer_youtube_vos":
                        self.all_datasets.append(
                            ReferYTVOSDataset(
                                base_image_dir,
                                samples_per_epoch,
                                precision,
                                image_size,
                                num_classes_per_sample,
                                exclude_val,
                                num_frames_mllm,
                                num_frames_sam,
                                video_max_pixels,
                            )
                        )
                        sample_rate_expand.append(sample_rate[idx])
                    elif vid_data == "mevis":
                        self.all_datasets.append(
                            MeViS_Dataset(
                                base_image_dir,
                                samples_per_epoch,
                                precision,
                                image_size,
                                num_classes_per_sample,
                                exclude_val,
                                num_frames_mllm,
                                num_frames_sam,
                                video_max_pixels,
                            )
                        )
                        sample_rate_expand.append(sample_rate[idx])
                    else:
                        raise NotImplementedError
            elif dataset == "davis":
                self.all_datasets.append(
                    RefDAVIS_Dataset(
                        base_image_dir,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        num_frames_mllm,
                        num_frames_sam,
                        video_max_pixels,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "vos":
                vid_datasets = vos_data.split(",")
                for vid_data in vid_datasets:
                    if vid_data == "ytvos":
                        self.all_datasets.append(
                            YTVOS_Dataset(
                                base_image_dir,
                                samples_per_epoch,
                                precision,
                                image_size,
                                num_classes_per_sample,
                                exclude_val,
                                num_frames_mllm,
                                num_frames_sam,
                                video_max_pixels,
                            )
                        )
                        sample_rate_expand.append(sample_rate[idx])
                    else:
                        raise NotImplementedError
            elif dataset == "reason_vos":
                self.all_datasets.append(
                    ReasonVOSDataset(
                        base_image_dir,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        num_frames_mllm,
                        num_frames_sam,
                        video_max_pixels,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "ref_videoqa":
                self.all_datasets.append(
                    ReferVideoQADataset(
                        base_image_dir,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        num_frames_mllm,
                        num_frames_sam,
                        video_max_pixels,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "videoqa":
                self.all_datasets.append(
                    GeneralVideoQADataset(
                        base_image_dir,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        num_frames_mllm,
                        num_frames_sam,
                        video_max_pixels,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "ref_vqa":
                ref_vqa_datasets = ref_vqa_data.split(",")
                for ref_vqa_dataset in ref_vqa_datasets:
                    self.all_datasets.append(
                        ReferVQADataset(
                            base_image_dir,
                            ref_vqa_dataset,
                            samples_per_epoch,
                            precision,
                            image_size,
                            num_classes_per_sample,
                            exclude_val,
                            num_frames_mllm,
                            num_frames_sam,
                            image_max_pixels,
                        )
                    )
                    if ref_vqa_dataset == 'osprey':
                        sample_rate_expand.append(sample_rate[idx]/2)
                    else:
                        sample_rate_expand.append(sample_rate[idx])
            else:
                raise NotImplementedError

        assert len(self.all_datasets) == len(sample_rate_expand)
        sample_rate = np.array(sample_rate_expand)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            for idx in range(len(sample_rate)):
                print("Dataset: {}, sample rate: {}".format(self.all_datasets[idx], sample_rate[idx]))
        self.sample_rate = sample_rate / sample_rate.sum()

        if local_rank == 0:
            print("="*100)
            print(f"{self.samples_per_epoch} samples per epoch.")
            max_len = max(len(type(dataset).__name__) for dataset in self.all_datasets)
            print('\n'.join([f'{type(dataset).__name__: <{max_len}} = {rate*100:.2f}%' for dataset, rate in zip(self.all_datasets, self.sample_rate)]))
            print("="*100)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.all_datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class VideoValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        val_dataset,
        image_size=1024,
        num_frames_mllm=50,
        num_frames_sam=4,
        max_pixels=1280*28*28
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg"))
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        # "images/mscoco/images/train2014",
                        "refer_seg/images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.transform = DirectResize(image_size) # ResizeLongestSide(image_size)
        self.preprocess = preprocess

        self.num_frames_mllm = num_frames_mllm
        self.num_frames_sam = num_frames_sam
        self.max_pixels = max_pixels

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]


        assert len(sampled_sents) == 1
        text = sampled_sents[0].strip()
        img_pil = Image.open(image_path).convert('RGB')
        if is_sentence:
            prompt = "{} Please output segmentation mask.".format(text)
        else:
            prompt = "What is {} in this video? Please output segmentation mask.".format(
                        text
                    ),
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_pil,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[SEG]."},
                ],
            }
        ]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        # Repeat image into video
        image = torch.stack([image] * self.num_frames_sam, dim=0)
        masks = torch.cat([masks] * self.num_frames_sam, dim=0)

        assert image.shape[0] == self.num_frames_sam
        assert masks.shape[0] == self.num_frames_sam or masks.shape[0] == 0, masks.shape

        return (
            image_path,
            image,
            messages,
            masks,
            labels,
            resize,
            inference,
        )


class ReasonSegTestDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        val_dataset,
        image_size=1024,
        num_frames_mllm=50,
        num_frames_sam=4,
        max_pixels=1280*28*28
    ):
        # self.base_image_dir = base_image_dir
        self.base_image_dir = "/mnt/ali-sh-1/dataset/zeus/chenqr/datasets/qrchen_dataset/other-datasets/reason_seg/ReasonSeg"
        splits = val_dataset.split("|")
        assert len(splits) == 3

        ds, split, query_type = splits
        images = glob.glob(
            os.path.join(self.base_image_dir, split, "*.jpg")
        )

        if query_type == "all":
            images_query_type = images
        elif query_type == "long":
            images_query_type = []
            for image_path in images:
                json_path = image_path.replace(".jpg", ".json")
                try:
                    with open(json_path, "r") as r:
                        anno = json.loads(r.read())
                except:
                    with open(json_path, "r", encoding="cp1252") as r:
                        anno = json.loads(r.read())
                is_sentence = anno["is_sentence"]
                if is_sentence:
                    images_query_type.append(image_path)
        else:
            assert query_type == "short"
            images_query_type = []
            for image_path in images:
                json_path = image_path.replace(".jpg", ".json")
                try:
                    with open(json_path, "r") as r:
                        anno = json.loads(r.read())
                except:
                    with open(json_path, "r", encoding="cp1252") as r:
                        anno = json.loads(r.read())
                is_sentence = anno["is_sentence"]
                if not is_sentence:
                    images_query_type.append(image_path)

        self.images = images_query_type
        self.data_type = "reason_seg"
        self.query_type = query_type

        self.ds = ds
        self.image_size = image_size
        self.transform = DirectResize(image_size) # ResizeLongestSide(image_size)
        self.preprocess = preprocess

        self.num_frames_mllm = num_frames_mllm
        self.num_frames_sam = num_frames_sam
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_path = image_path.replace(".jpg", ".json")
        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
        sampled_sents = [sampled_sents[0]]

        assert len(sampled_sents) == 1
        text = sampled_sents[0].strip()
        img_pil = Image.open(image_path).convert('RGB')
        if is_sentence:
            prompt = "{} Please output segmentation mask.".format(text)
        else:
            prompt = "What is {} in this video? Please output segmentation mask.".format(
                        text
                    ),
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_pil,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[SEG]."},
                ],
            }
        ]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        # Repeat image into video
        image = torch.stack([image] * self.num_frames_sam, dim=0)
        masks = torch.cat([masks] * self.num_frames_sam, dim=0)

        assert image.shape[0] == self.num_frames_sam
        assert masks.shape[0] == self.num_frames_sam or masks.shape[0] == 0, masks.shape

        return (
            image_path,
            image,
            messages,
            masks,
            labels,
            resize,
            inference,
        )


class RefImgValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        val_dataset,
        image_size=1024,
        num_frames_mllm=50,
        num_frames_sam=4,
        max_pixels=1280*28*28
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        assert len(splits) == 3

        ds, splitBy, split = splits
        refer_api = REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds, splitBy)
        ref_ids_val = refer_api.getRefIds(split=split)
        images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
        refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
        refer_seg_ds = {}
        refer_seg_ds["images"] = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
        for item in loaded_images:
            item = item.copy()
            if ds == "refclef":
                item["file_name"] = os.path.join(
                    base_image_dir, "images/saiapr_tc-12", item["file_name"]
                )
            elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                item["file_name"] = os.path.join(
                    base_image_dir,
                    # "images/mscoco/images/train2014",
                    "refer_seg/images/mscoco/images/train2014",
                    item["file_name"],
                )
            refer_seg_ds["images"].append(item)
        refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

        img2refs = {}
        for ref in refs_val:
            image_id = ref["image_id"]
            img2refs[image_id] = img2refs.get(image_id, []) + [
                ref,
            ]
        refer_seg_ds["img2refs"] = img2refs
        self.refer_seg_ds = refer_seg_ds
        self.data_type = "refer_seg"

        data_samples = []
        for idx in range(len(refer_seg_ds["images"])):
            image_info = refer_seg_ds["images"][idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]
            refs = img2refs[image_id]
            for ref in refs:
                for sent in ref["sentences"]:
                    one_sample = [image_path, image_id, image_info, sent["sent"].strip().lower(), ref["ann_id"]]
                    data_samples.append(one_sample)
        self.data_samples = data_samples

        self.ds = ds
        self.image_size = image_size
        self.transform = DirectResize(image_size) # ResizeLongestSide(image_size)
        self.preprocess = preprocess

        self.num_frames_mllm = num_frames_mllm
        self.num_frames_sam = num_frames_sam
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        image_path, image_id, image_info, sampled_sent, sampled_ann_id = self.data_samples[idx]
        annotations = self.refer_seg_ds["annotations"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_pil = Image.open(image_path).convert('RGB')
        text = sampled_sent.strip()
        prompt = "What is {} in this video? Please output segmentation mask.".format(
                        text
                    ),
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_pil,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[SEG]."},
                ],
            }
        ]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = []
        ann = annotations[sampled_ann_id]
        if len(ann["segmentation"]) == 0 and sampled_sent != "":
            m = np.zeros((image_info["height"], image_info["width"], 1))
        else:
            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"],
                    image_info["height"],
                    image_info["width"],
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
        m = np.sum(
            m, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        masks.append(m)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        # Repeat image into video
        image = torch.stack([image] * self.num_frames_sam, dim=0)
        masks = torch.cat([masks] * self.num_frames_sam, dim=0)

        assert image.shape[0] == self.num_frames_sam
        assert masks.shape[0] == self.num_frames_sam or masks.shape[0] == 0, masks.shape

        return (
            image_path,
            image,
            messages,
            masks,
            labels,
            resize,
            inference,
        )