import argparse
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
import json

from transformers import (
    AutoProcessor,
)
from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel



def parse_args(args):
    parser = argparse.ArgumentParser(description="merge lora weights and save model with hf format")
    parser.add_argument("--version", default="")

    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--out_dim", default=256, type=int)

    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)

    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./lisa_model", type=str, required=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    # Create model
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer

    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "seg_token_idx": args.seg_token_idx,
        "sam_pretrained": None,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    config = UniGRConfig.from_pretrained(
        args.version,
        **model_args,
    )
    model = UniGRModel.from_pretrained(
        args.version,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False
    )
    model.initialize_sam_modules(config)

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "sam_model",
                                "grounding_encoder"
                                "visual",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))


    # state_dict = torch.load(args.weight, map_location="cpu")
    # model.load_state_dict(state_dict, strict=True)

    index_file = os.path.join(args.weight, "pytorch_model.bin.index.json")
    with open(index_file, "r") as f:
        weight_map = json.load(f)["weight_map"]
    shard_files = list(set(weight_map.values()))
    state_dict = {}
    for shard in shard_files:
        state_dict.update(torch.load(os.path.join(args.weight, shard), map_location="cpu"))
    model.load_state_dict(state_dict, strict=True)

    model = model.merge_and_unload()
    model.save_pretrained(args.save_path, state_dict=model.state_dict())
    tokenizer.save_pretrained(args.save_path)
    processor.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
