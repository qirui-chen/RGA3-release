# -*- coding: utf-8 -*-
import argparse
from functools import partial
import deepspeed
import torch
import tqdm
from transformers import AutoProcessor
import torch.distributed as dist

import os
import sys
sys.path.append(".")

from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from utils.dataset import collate_fn, VideoValDataset, ReasonSegTestDataset, RefImgValDataset
from utils.utils import (AverageMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def initialize_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def parse_args(args):
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--version", default="PATH/TO/MODEL")
    parser.add_argument("--vis_save_path", default="", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)

    parser.add_argument("--num_frames_mllm", default=50, type=int)
    parser.add_argument("--num_frames_sam", default=4, type=int)

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.local_rank = initialize_distributed()
    save_name = str(args.val_dataset).replace('|', '_')
    writer = None

    # Create model
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer

    # num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    model_args = {
        "train_mask_decoder": False,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
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
        attn_implementation="flash_attention_2", #"sdpa"
        low_cpu_mem_usage=False,
    )

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    if not args.eval_only:
        model.initialize_sam_modules(config)

    for p in model.visual.parameters():
        p.requires_grad = False

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    # for n, p in model.named_parameters():
    #     if any(
    #         [
    #             x in n
    #             for x in ["lm_head", "embed_tokens", "mask_decoder", "sam_mask_decoder", "text_hidden_fcs"]
    #         ]
    #     ):
    #         p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    if str(args.val_dataset).lower().startswith("reason"):
        split = str(args.val_dataset).strip().split("|")
        if len(split) == 2:
            val_dataset_reason = VideoValDataset(
                args.dataset_dir,
                args.val_dataset,
                args.image_size,
                args.num_frames_mllm,
                args.num_frames_sam,
            )
        else:
            val_dataset_reason = ReasonSegTestDataset(
                args.dataset_dir,
                args.val_dataset,
                args.image_size,
                args.num_frames_mllm,
                args.num_frames_sam,
            )
    else:
        val_dataset_reason = RefImgValDataset(
            args.dataset_dir,
            args.val_dataset,
            args.image_size,
            args.num_frames_mllm,
            args.num_frames_sam,
        )

    for p in model.parameters():
        p.requires_grad = False
    ds_config = {
    }
    model_engine = deepspeed.init_inference(
        model=model,
        config=ds_config,
        dtype=torch.bfloat16 if args.precision == "bf16" else torch.half,
        replace_with_kernel_inject=True,
    )

    # validation dataset
    assert args.val_batch_size == 1
    val_sampler_reason = torch.utils.data.distributed.DistributedSampler(
        val_dataset_reason, shuffle=False, drop_last=False
    )
    val_loader_reason = torch.utils.data.DataLoader(
        val_dataset_reason,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        sampler=val_sampler_reason,
        collate_fn=partial(
            collate_fn,
            processor=processor,
            local_rank=args.local_rank,
        ),
    )

    giou_reason, ciou_reason = validate(val_loader_reason, model_engine, 0, writer, args, save_name)
    os.makedirs(args.vis_save_path, exist_ok=True)
    if args.local_rank == 0:
        with open(os.path.join(args.vis_save_path, "{}.txt".format(save_name)), 'w') as f:
            output = "{}samples, {} eval: giou: {:.4f}, ciou: {:.4f}".format(len(val_dataset_reason), save_name, giou_reason, ciou_reason)
            f.write(output)


def validate(val_loader, model_engine, epoch, writer, args, dataset_name):
    print("Evaluating {}".format(dataset_name))
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            for k in ['images_sam', 'pixel_values', 'pixel_values_videos']:
                if input_dict[k] is not None:
                    input_dict[k] = input_dict[k].half()
        elif args.precision == "bf16":
            for k in ['images_sam', 'pixel_values', 'pixel_values_videos']:
                if input_dict[k] is not None:
                    input_dict[k] = input_dict[k].bfloat16()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.int().contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    torch.distributed.barrier()
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        print("{} eval: giou: {:.4f}, ciou: {:.4f}".format(dataset_name, giou, ciou))
        if writer is not None:
            writer.add_scalar("val_{}/giou".format(dataset_name), giou, epoch)
            writer.add_scalar("val_{}/ciou".format(dataset_name), ciou, epoch)

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
