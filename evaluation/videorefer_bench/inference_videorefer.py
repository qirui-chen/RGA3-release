import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re
from PIL import Image
from pycocotools import mask
import random
import numpy as np
import time
import sys
sys.path.append(".")
from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from utils.visual_prompt_generator import video_blending_keyframes, color_pool
from utils.utils import set_seed
from model.STOM import STOM


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VideoRefer_Bench_Q_general(Dataset):
    def __init__(self, video_folder, data_list, shape='ellipse', use_stom=True, num_frames=16):
        self.video_folder = video_folder
        self.data_list = data_list
        self.propagator = STOM(device="cuda:0") if use_stom else None
        self.shape = shape
        self.num_frames = num_frames
        self.total_frames = 0
        self.total_time = 0
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]

        key_frame_idxs = [int(line["frame_idx"])]
        frames_root = os.path.join(self.video_folder, line['video'])
        frames_name = os.listdir(frames_root)

        indices = np.linspace(0, len(frames_name) - 1, self.num_frames - 1, dtype=int)
        selected_frame_idx = indices.tolist()

        selected_frame_idx.extend(key_frame_idxs)
        selected_frame_idx = sorted(selected_frame_idx)

        frames, is_key_frame = [], []
        for frame_idx in selected_frame_idx:

            frames.append(Image.open(os.path.join(frames_root, frames_name[frame_idx])))
            is_key_frame.append(frame_idx in key_frame_idxs)
            # is_key_frame.append(True) #NOTE: oracle propagator


        cur_frames = frames        
        colors = random.sample(list(color_pool.keys()), len(line['annotation'])) # Each color for each object
        shape = self.shape
        idx_list = [int(idx) for idx in re.findall(r'<object(\d+)><region>', line['Question'])]
        
        for idx, anno in enumerate(line['annotation']):
            masks = []
            for frame_idx in selected_frame_idx:
                if str(frame_idx) in anno and anno[str(frame_idx)]['segmentation'] is not None:
                    masks.append(mask.decode(anno[str(frame_idx)]['segmentation']))
                else:
                    masks.append(np.zeros(1))
            color = colors[idx]
            # blended_frames = video_blending_keyframes(cur_frames, masks, is_key_frame, color, shape)
            # cur_frames = blended_frames

            blended_frames, vip_img = video_blending_keyframes(cur_frames, masks, is_key_frame, color, shape, return_vip_img=True)
            if vip_img is not None and (np.array(vip_img)[:, :, 3] > 0).any() and self.propagator is not None:
                # start = time.time()
                # vip_img.save('tmp.png')
                propagated_frames = self.propagator.propagate_in_video(cur_frames, vip_img, is_key_frame.index(True), shape=shape)
                # end = time.time()
                # self.total_time += end - start
                # self.total_frames += len(cur_frames)
                # avg_fps = self.total_frames/self.total_time
                # print(f"FPS: {avg_fps:.2f}={self.total_frames}/{self.total_time}")
                cur_frames = propagated_frames
            else:
                cur_frames = blended_frames

            line['Question'] = re.sub(fr'<object{idx_list[idx]}?><region>', f"the object within {color} {shape}", line['Question'])
            line['options'] = [re.sub(fr'<object{idx_list[idx]}?>', f"the object within {color} {shape}", option) for option in line['options']]

        prompt = line['Question'] + '\n' + ' '.join(line['options']) + '\n' + "Answer with the option\'s letter from the given choices directly."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": cur_frames,
                        # "max_pixels": 2560 * 28 * 28,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        return {
            'video': line['video'],
            'messages': messages,
            'is_key_frame': is_key_frame,
            'answer': line['Answer'],
            'type': line['type']
        }

def collate_fn(batch):
    video = [x['video'] for x in batch]
    messages = [x['messages'] for x in batch]
    is_key_frame = [x['is_key_frame'] for x in batch]
    ans = [x['answer'] for x in batch]
    tp = [x['type'] for x in batch]
    return video, messages, ans, tp, is_key_frame

def build_videorefer_bench_q_eval(args):
    questions = json.load(open(args.question_file))
    dataset = VideoRefer_Bench_Q_general(args.video_folder, questions, args.shape, args.use_stom, args.num_frames)
    num_workers = 0 if args.use_stom else 16
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    return dataloader

def run_inference(args):

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

    model = model.cuda()
    model.eval()

    val_loader = build_videorefer_bench_q_eval(args)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(args.output_file, "w")
    
    for i, (videos, messages, answers, types, is_key_frame) in enumerate(tqdm(val_loader)):
        video = videos[0]
        messages = messages[0]
        answer = answers[0]
        type_ = types[0]
        is_key_frame = is_key_frame[0]

            
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

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
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


        record = {
            'video': video,
            'Answer': answer,
            'pred': output_text[0],
            'type': type_,
        }
        ans_file.write(json.dumps(record) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', help='Model checkpoint.', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--shape', help='Visual prompt shape.', required=True)
    parser.add_argument('--use_stom', help='Whether using STOM.', required=True)
    parser.add_argument('--num_frames', type=int, help='The number of input frames', required=True)

    args = parser.parse_args()

    set_seed(100)

    run_inference(args)