import json
import os
import sys
from tqdm import tqdm
from glob import glob

from openai import AzureOpenAI, BadRequestError
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

sys.path.append(".")
from utils.utils import get_sparse_indices
from pycocotools import mask as coco_mask



import base64
from io import BytesIO

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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

        original_frames, blended_frames, masks = [], [], []

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
            "masks": masks,
        }



result_root = "results/RefVideoQA/GPT-4o-high-8frames"
os.makedirs(result_root, exist_ok=True)

data_root = "/PATH/TO/VideoInfer-Release"
qa_path = (os.path.join(data_root, "test.json"))
qa_data = json.load(open(qa_path, "r"))
mask_dict = json.load(
    open(
        os.path.join(data_root, "mask_dict.json"),
        "r",
    )
)
video_folder = (data_root)


# Create dataset and dataloader
dataset = VideoInferDataset(
    qa_data,
    mask_dict,
    video_folder,
    num_frames=8,
    subset_idx=0,
    subset_num=1,
)

client = AzureOpenAI(
                api_key="",
                base_url="https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-10-01-preview",
                api_version = '2024-05-01-preview'
)


count = 0

save_path = os.path.join(result_root, "merged_result.json")
if os.path.exists(save_path):
    gpt_answers = json.load(open(save_path))
else:
    gpt_answers = {}

progress_bar = tqdm(range(len(dataset)), desc=f"Success-{count}")
for idx in progress_bar:

    sample = dataset[idx]
    vid_id = sample["vid_id"]
    exp_id = sample["exp_id"]
    qa_id = sample["qa_id"]
    Q = sample["Q"]
    A = sample["A"]
    original_frames = sample["original_frames"]
    blended_frames = sample["blended_frames"]
    vip_img = sample["vip_img"]
    masks = sample["masks"]

    if vid_id in gpt_answers and exp_id in gpt_answers[vid_id] and qa_id in gpt_answers[vid_id][exp_id]:
        count += 1
        progress_bar.set_description(f"Success-{count}")
        continue

    blended_frames_encoded = [image_to_base64(image) for image in blended_frames]


    task_id = f"{vid_id}____{exp_id}____{qa_id}"
    
    video_question = [{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{encoded_frame}", "detail": "high"}} for encoded_frame in blended_frames_encoded]
    video_question.append({"type": "text", "text": Q})

    messages = [
                {
                    "role": "system",
                    "content": 
                    "You are a helpful assistant." 
                    "Your task is to watch the video and answer the question."
                    "You should response in JSON format, like: {'answer': '...'} directly. Remember to use escape characters."
                },
                {
                    "role": "user",
                    "content": video_question
                },
            ]

    temperature = 0.0
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=64,
            temperature=temperature,
            messages=messages)
        content = response.choices[0].message.content
    except BadRequestError:
        content = "{'answer': 'None'}"

    flag = True
    try_time = 1
    while flag:
        try:
            result = eval(content)
            if 'answer' in result:
                count += 1
                answer = result['answer']
                flag = False
        except:
            print(content)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=64,
                    temperature=temperature,
                    messages=messages)
                content = response.choices[0].message.content
            except BadRequestError:
                content = "{'answer': 'None'}"
            
            try_time += 1
            temperature += 0.5
            print(f"{task_id} try {try_time} times")
            if try_time > 3:
                answer = "I don't know."
                flag = False


    progress_bar.set_description(f"Success-{count}")

    if vid_id not in gpt_answers:
        gpt_answers[vid_id] = {}
    if exp_id not in gpt_answers[vid_id]:
        gpt_answers[vid_id][exp_id] = {}
    gpt_answers[vid_id][exp_id][qa_id] = answer

    if idx % 20 == 0:
        with open(os.path.join(result_root, "merged_result.json"), "w", encoding="utf-8") as file:
            json.dump(gpt_answers, file, indent=2, ensure_ascii=False)

with open(os.path.join(result_root, "merged_result.json"), "w", encoding="utf-8") as file:
    json.dump(gpt_answers, file, indent=2, ensure_ascii=False)
print(f"{count} | {len(dataset)}")