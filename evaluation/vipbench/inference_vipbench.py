import argparse
import os
import sys
from tqdm import tqdm
import json
import torch
from torch.utils.data import Dataset, DataLoader
import os.path as osp
from PIL import Image

sys.path.append(".")
from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from utils.utils import SuppressTokenProcessor, REFERRING_VQA_PROMPT

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, LogitsProcessorList

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def parse_args(args):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--version', type=str, default="")
    parser.add_argument('--question-file', type=str, default="")
    parser.add_argument('--image-folder', type=str, default="")
    parser.add_argument('--answers-file', type=str, default="")

    # 解析参数
    args = parser.parse_args(args)
    return args

class QADataset(Dataset):
    def __init__(self, questions_file, image_folder):
        self.questions = self.read_jsonl(questions_file)
        self.image_folder = image_folder

    def read_jsonl(self, file_path: str):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip())) 
        return data

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx: int):
        sample = self.questions[idx]

        image_path = osp.join(self.image_folder, sample["image"])
        img_pil = Image.open(image_path).convert('RGB')

        text = sample["text"]
        question_id = sample["question_id"]
        
        return {
            "image_path": image_path,
            "img_pil": img_pil,
            "question": text,
            "question_id": question_id
        }

def collate_fn(batch):

    image_path = [item["image_path"] for item in batch]
    img_pil = [item["img_pil"] for item in batch]
    question = [item["question"] for item in batch]
    question_id = [item["question_id"] for item in batch]

    return {
        "image_path": image_path,
        "img_pil": img_pil,
        "question": question,
        "question_id": question_id,
    }


def create_dataloader(question_file, image_folder):
    dataset = QADataset(question_file, image_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, collate_fn=collate_fn)
    return dataloader


def main(args):
    
    args = parse_args(args)

    # Create model
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]
    logits_processor = LogitsProcessorList()
    logits_processor.append(SuppressTokenProcessor(tokenizer.encode(["[SEG]", "segmentation"])))

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

    dataloader = create_dataloader(args.question_file, args.image_folder)

    results = {}
    for batch in tqdm(dataloader, leave=False):

        question_id = batch['question_id'][0]
        question = batch['question'][0]
        img_pil = batch['img_pil'][0]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_pil,
                    },
                    {"type": "text", "text": REFERRING_VQA_PROMPT.format(text=question)},
                ],
            }
        ]

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
                max_new_tokens=1024,
                do_sample=False,
                # do_sample=True,
                # num_beams=5,
                # temperature=0.2,
                # top_p=0.9,
                # top_k=None,
                # logits_processor=logits_processor,
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
        print(question_id)
        print(question)
        print(output_text)
        results[f'v1_{question_id}'] = output_text


    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    with open(args.answers_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":

    # import debugpy
    # debugpy.listen(("10.148.254.82", 7850))
    # debugpy.wait_for_client()
    main(sys.argv[1:])
