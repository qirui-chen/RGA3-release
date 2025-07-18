# -*- coding: utf-8 -*-
import os
import argparse
import shutil
import sys
import time
from functools import partial

import deepspeed
import torch
import datetime
import torch.distributed as dist
import tqdm
import json
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint

from transformers import (
    AutoProcessor,
)

from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel

from utils.dataset import collate_fn, ImgVidHybridDataset, VideoValDataset
from utils.utils import rank0_print
from utils.utils import (
    AverageMeter,
    ProgressMeter,
    Summary,
    dict_to_cuda,
    intersectionAndUnionGPU,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args(args):
    parser = argparse.ArgumentParser(description="UniGR Model Training")
    # Env
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--global_rank", default=0, type=int, help="global rank")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # Model
    parser.add_argument("--version", default="")
    parser.add_argument("--sam_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--num_frames_mllm", default=2, type=int)
    parser.add_argument("--num_frames_sam", default=1, type=int)
    parser.add_argument("--out_dim", default=256, type=int)

    # Lora
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    # Data - image
    parser.add_argument(
        "--dataset",
        default="sem_seg,refer_seg,vqa,reason_seg,vid_qa,refer_seg_video",
        type=str,
    )
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k,cocostuff,pascal_part,paco_lvis,mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef,refcoco,refcoco+,refcocog", type=str
    )
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--ref_vqa_data", default="osprey,vip_llava", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)

    # Data - video
    parser.add_argument("--vos_data", default="ytvos", type=str)
    parser.add_argument("--ref_vos_data", default="refer_youtube_vos", type=str)

    # Data - hyper-param
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--video_max_pixels", default=336*28*28, type=int)
    parser.add_argument("--image_max_pixels", default=1280*28*28, type=int)

    # Training
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(f"Running on local rank {args.local_rank}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=1800)
    )
    args.global_rank = dist.get_rank()
    # deepspeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=1800))
    # args.global_rank = deepspeed.comm.get_rank()

    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        pprint(vars(args))
    else:
        writer = None
    # Create model

    # min_pixels = 256*28*28
    # max_pixels = 640*28*28
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer

    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "sam_pretrained": args.sam_pretrained,
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
        # device_map="auto",
        attn_implementation="flash_attention_2", #"sdpa"
        # use_cache=False,
        low_cpu_mem_usage=False,
    )
    model.initialize_sam_modules(config)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    for p in model.visual.parameters():
        p.requires_grad = False

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

    # Make following parameters trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in [
                    "lm_head",
                    "embed_tokens",
                    "mask_decoder",
                    "sam_mask_decoder",
                    "text_hidden_fcs",
                ]
            ]
        ):
            # print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = ImgVidHybridDataset(
        args.dataset_dir,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        reason_seg_data=args.reason_seg_data,
        vqa_data=args.vqa_data,
        ref_vos_data=args.ref_vos_data,
        vos_data=args.vos_data,
        ref_vqa_data=args.ref_vqa_data,
        explanatory=args.explanatory,
        num_frames_sam=args.num_frames_sam,
        num_frames_mllm=args.num_frames_mllm,
        video_max_pixels=args.video_max_pixels,
        image_max_pixels=args.image_max_pixels,
    )

    if args.no_eval == False:
        val_dataset_reason = VideoValDataset(
            args.dataset_dir,
            args.val_dataset,
            args.image_size,
            args.num_frames_mllm,
            args.num_frames_sam,
            args.image_max_pixels,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset_reason)} examples."
        )
    else:
        val_dataset_reason = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_ratio": 0,
                "cos_min_ratio": 0.03,
                "warmup_num_steps": int(0.03 * args.epochs * args.steps_per_epoch),
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            # "allgather_partitions": True,
            "reduce_bucket_size": 5e9,
            "allgather_bucket_size": 5e9,
            # "sub_group_size": 1e9,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            processor=processor,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # args.global_rank = model_engine.global_rank

    best_score_reason, cur_giou_reason = 0.0, 0.0
    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_latest")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        if args.auto_resume:
            load_path, client_state = model_engine.load_checkpoint(args.resume)
            args.start_epoch = (
                int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
            )
        else:
            load_path, client_state = model_engine.load_checkpoint(args.resume, load_optimizer_states=False, load_lr_scheduler_states=False)
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )
        json_filename = os.path.join(args.log_dir, "meta_log_info.json")
        if os.path.exists(json_filename):
            with open(json_filename, 'r') as json_file:
                all_info = json.load(json_file)
            best_score_reason = max([info['ciou_reason'] for info in all_info])

    # validation dataset
    if val_dataset_reason is not None:
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

    train_iter = iter(train_loader)

    if args.eval_only:
        giou_reason, ciou_reason = validate(
            val_loader_reason, model_engine, 0, writer, args, args.val_dataset
        )
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou_reason, ciou_reason = validate(
                val_loader_reason, model_engine, epoch, writer, args, args.val_dataset
            )
            is_best_reason = ciou_reason > best_score_reason
            best_score_reason = max(ciou_reason, best_score_reason)
            cur_giou_reason = giou_reason if is_best_reason else cur_giou_reason

        save_dir_latest = os.path.join(args.log_dir, "ckpt_latest")  
        torch.distributed.barrier()
        if args.global_rank == 0:
            if os.path.exists(save_dir_latest):
                shutil.rmtree(save_dir_latest)
        torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir_latest)

        if (not args.no_eval) or is_best_reason:
            save_dir_best = os.path.join(args.log_dir, "ckpt_best")
            if args.global_rank == 0:
                info = {
                    "epoch": epoch,
                    "giou_reason": float(giou_reason),
                    "ciou_reason": float(ciou_reason)
                }
                json_filename = os.path.join(
                    args.log_dir,
                    "meta_log_info.json"
                )
                if os.path.exists(json_filename):
                    with open(json_filename, 'r') as json_file:
                        all_info = json.load(json_file)
                else:
                    all_info = []

                all_info.append(info)
                with open(json_filename, 'w') as json_file:
                    json.dump(all_info, json_file, indent=4)
                    
                if is_best_reason and os.path.exists(save_dir_best):
                    shutil.rmtree(save_dir_best)

            torch.distributed.barrier()
            if (not args.no_eval) and is_best_reason:
                model_engine.save_checkpoint(save_dir_best)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.2f")
    data_time = AverageMeter("Data", ":6.2f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()

    # for global_step in range(args.steps_per_epoch):
    for local_step in range(args.steps_per_epoch):
        global_step = local_step + epoch * args.steps_per_epoch
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                for k in ['images_sam', 'pixel_values', 'pixel_values_videos']:
                    if input_dict[k] is not None:
                        input_dict[k] = input_dict[k].half()
            elif args.precision == "bf16":
                for k in ['images_sam', 'pixel_values', 'pixel_values_videos']:
                    if input_dict[k] is not None:
                        input_dict[k] = input_dict[k].bfloat16()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images_sam"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images_sam"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images_sam"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images_sam"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images_sam"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


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
                output_i.contiguous().clone(),
                mask_i.int().contiguous(),
                2,
                ignore_index=255,
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        (
            intersection_meter.update(intersection),
            union_meter.update(union),
            acc_iou_meter.update(acc_iou, n=masks_list.shape[0]),
        )

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val_{}/giou".format(dataset_name), giou, epoch)
        writer.add_scalar("val_{}/ciou".format(dataset_name), ciou, epoch)
        print("{} eval: giou: {:.4f}, ciou: {:.4f}".format(dataset_name, giou, ciou))

    return giou, ciou


if __name__ == "__main__":

    main(sys.argv[1:])
