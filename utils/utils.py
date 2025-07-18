from enum import Enum

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
import random

from transformers import LogitsProcessor
from typing import List
import matplotlib.pyplot as plt
import os
# import deepspeed

IGNORE_INDEX = -100
# IMAGE_TOKEN_INDEX = -200
# DEFAULT_IMAGE_TOKEN = "<image>"

# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"

SEG_TEMPLATE = "You are good at segmentation. "

SHORT_QUESTION_LIST = [
    "Can you segment the {class_name} in this image?",
    "Please segment the {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",
]


LONG_QUESTION_LIST = [
    "{sent} Please respond with segmentation mask.",
    "{sent} Please output segmentation mask.",
]


EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

VISUAL_PROMPT = "Look at the marked region {prep} the {color} {shape} in the video and then answer the question. "
REFERRING_VQA_PROMPT = "Look at the marked region and then answer the question. {text}"
OLD_REFERRING_VQA_PROMPT = "Look at the highligted region in the image and then answer the question. {text}"


words_shape ={
    "rectangle": ["within", "rectangle"], 
    "ellipse": ["within", "ellipse"],
    "triangle": ["with", "triangle"],
    "point": ["at", "point"], 
    "scribble" : ["with", "scribble"], 
    "mask contour": ["with", "mask contour"],
    "mask": ["with", "mask"],
    "arrow": ["pointed to by", "arrow"],
 }

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


# def rank0_print(*args):
#     if deepspeed.comm.get_rank() == 0:
#         print(*args)


def uniform_sample(total_len, sample_num):
    intervals = np.linspace(start=0, stop=total_len, num=sample_num + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def get_sparse_indices(total_frame_num, num_frames_mllm):
    if total_frame_num > num_frames_mllm:       # video is long, uniformly sample frames
        frame_idxs = uniform_sample(total_frame_num, num_frames_mllm)
        return sorted(frame_idxs)
    else:
        num_repeat = num_frames_mllm // total_frame_num
        num_sample = num_frames_mllm % total_frame_num
        frame_idxs = list(range(total_frame_num)) * num_repeat + uniform_sample(total_frame_num, num_sample)
        return sorted(frame_idxs)


def get_dense_indices(num_frames_mllm, num_frames_sam):
    intervals = np.linspace(start=0, stop=num_frames_mllm - 1, num=num_frames_sam + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    # h, w = x.shape[-2:]
    # padh = img_size - h
    # padw = img_size - w
    # x = F.pad(x, (0, padw, 0, padh))
    return x


class DirectResize:
    def __init__(self, target_length: int):
        self.target_length = target_length

    def apply_image(self, image: np.ndarray):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


def uniform_random_sample(vid_len, num_frames):
    if vid_len == 0 or num_frames == 0:
        return []

    if num_frames > vid_len:
        sample_indx = sorted(random.choices(range(vid_len), k=num_frames))

    step = vid_len / num_frames

    sample_indx = []
    for i in range(num_frames):

        start = int(i * step)
        end = int((i + 1) * step)

        sample_indx.append(random.randint(start, min(end, vid_len - 1)))
    
    sample_indx.sort()
    return sample_indx


class SuppressTokenProcessor(LogitsProcessor):
    def __init__(self, forbid_token_id_list: List[int] = None):
        self.forbid_token_id_list = forbid_token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for id_ in self.forbid_token_id_list:
            scores[:, id_] = -float('inf')
        return scores



def show_frames(frames, save_path='.', cols=4, reset_every_n_rows=2):
    num_frames = len(frames)
    
    rows = num_frames // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows / 2), dpi=250)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, ax in enumerate(axes.flat):
        if i < num_frames:
            ax.imshow(frames[i])
            ax.axis('off')
            
            row = i // cols
            
            index_in_row = (i % (reset_every_n_rows * cols)) + 1
            
            ax.text(
                0.95, 0.95,
                f'{index_in_row}',
                transform=ax.transAxes,
                fontsize=12,
                color='white',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
            )
            if reset_every_n_rows > 0 and row % reset_every_n_rows == 0 and row > 0:
                for j in range(cols):
                    axes[row, j].axhline(y=0, color='blue', linewidth=3, linestyle='--')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_path, 'vis.png'))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)