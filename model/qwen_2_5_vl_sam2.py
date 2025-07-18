import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from typing import List, Optional, Tuple


from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
)
from qwen_vl_utils import process_vision_info
from .sam2 import SAM2


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
):
    masks = F.interpolate(
        masks.float(),
        (1024, 1024),
        mode="bilinear",
        align_corners=False,
    )

    masks = masks[..., : input_size[0], : input_size[1]] #(768, 1024)
    masks = F.interpolate(
        masks, original_size, mode="bilinear", align_corners=False
    ) #(480, 640)
    return masks


class UniGRConfig(Qwen2_5_VLConfig):
    def __init__(self, 
        train_mask_decoder=False,
        out_dim=256,
        ce_loss_weight=.0,
        dice_loss_weight=.0,
        bce_loss_weight=.0,
        seg_token_idx=0,
        sam_pretrained=None, 
        **kwargs):

        self.train_mask_decoder=train_mask_decoder
        self.out_dim=out_dim
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.seg_token_idx = seg_token_idx
        self.sam_pretrained = sam_pretrained

        super().__init__(**kwargs)


class UniGRModel(Qwen2_5_VLForConditionalGeneration):
    config_class = UniGRConfig

    def __init__(self, config):
        super().__init__(config)

        # print(config.train_mask_decoder)
        if not config.train_mask_decoder:
            # inference mode
            self.initialize_sam_modules(
                config
            )

    def initialize_sam_modules(self, config):
        # SAM
        self.grounding_encoder = SAM2(ckpt_path=config.sam_pretrained) #config.sam_pretrained, None
        # self.sam_model.image_encoder.pos_embed.data = self.sam_model.image_encoder.pos_embed.data.contiguous()
        self.grounding_encoder.sam2_model.requires_grad_(False)
        if config.train_mask_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.train()
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)
        else:
            self.grounding_encoder.sam2_model.sam_mask_decoder.eval()

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)


    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        images_sam: Optional[torch.FloatTensor] = None,
        offset: Optional[torch.LongTensor] = None,
        masks_list: Optional[List[torch.FloatTensor]] = None,
        label_list: Optional[List[torch.Tensor]] = None,
        resize_list: Optional[List[tuple]] = None,
        inference: bool = False,
        **kwargs,
    ):
        batch_size, num_frames_sam = images_sam.shape[:2]
        device = images_sam.device
        assert batch_size == len(offset) - 1

        images_reshape = rearrange(images_sam, 'b t c h w -> (b t) c h w')

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )
        output_hidden_states = output.hidden_states

        model_output = output
        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.config.ce_loss_weight

        # find seg token in hidden states
        seg_token_mask = (labels == self.config.seg_token_idx)
        seg_token_mask = torch.cat([seg_token_mask[:, 1:], torch.zeros_like(seg_token_mask)[:, 0].unsqueeze(1)], dim=1)

        hidden_states = []

        assert len(self.text_hidden_fcs) == 1
        hidden_states.append(self.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask] # [3, 256]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ] [1, 1, 0, 1]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), seg_token_offset], dim=0
        )
        seg_token_offset = seg_token_offset[offset] #[0, 1, 2, 2, 3]

        language_embeddings = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            if start_i == end_i:
                language_embeddings = language_embeddings + [torch.zeros((1, 256), device=device, dtype=images_sam.dtype)] * num_frames_sam
            else:
                language_embeddings = language_embeddings + [pred_embeddings[start_i:end_i]] * num_frames_sam
        language_embeddings = torch.cat(language_embeddings, dim=0).unsqueeze(1) # (N, 1, D)


        if inference:
            
            language_embeddings = language_embeddings.reshape(batch_size, num_frames_sam, 1, 256)
            pred_masks = []
            for i in range(batch_size):
                language_embedding = language_embeddings[i] # (num_frames, 1, 256)

                sam_states = self.grounding_encoder.get_sam2_embeddings(images_sam.squeeze(0)) # here all frames
                masks = self.grounding_encoder.language_embd_inference(sam_states, language_embedding) # (num_frames, 1, 1024, 1024)

                h, w = label_list[i].shape
                masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False)
                masks = masks[:, 0] #(num_frames, 1, h, w)
                masks = masks.sigmoid() > 0.5

                pred_masks.append(masks)

            return {
                "pred_masks": pred_masks,
                "gt_masks": masks_list,
            }


        has_seg_token = seg_token_counts.bool()
        g_pixel_values = images_reshape # (B*T, 3, 1024, 1024)

        num_objs = 1
        sam_states = self.grounding_encoder.get_sam2_embeddings_train(g_pixel_values, expand_size=num_objs)
        low_res_masks, high_res_masks = self.grounding_encoder.inject_language_embd_train(sam_states, language_embeddings, nf_nobj=None)

        low_res_masks = low_res_masks.reshape(batch_size, num_frames_sam, 256, 256) #(B*T, 256, 256)
        high_res_masks = high_res_masks.reshape(batch_size, num_frames_sam, 1024, 1024) #(B*T, 1024, 1024)

        valid_pred_masks, valid_gt_masks = [], []
        for i in range(batch_size):
            pred_mask_cur_vid = F.interpolate(high_res_masks[i].unsqueeze(1), size=label_list[i].shape, mode='bilinear', align_corners=False)[:, 0]
            gt_mask_cur_vid = masks_list[i]
            
            # pred_mask_cur_vid = low_res_masks[i]
            # gt_mask_cur_vid = F.interpolate(masks_list[i].unsqueeze(1), size=(256, 256), mode='nearest')[:, 0]
            
            valid_pred_masks.append(pred_mask_cur_vid)
            valid_gt_masks.append(gt_mask_cur_vid)


        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(batch_size):
            gt_mask = valid_gt_masks[batch_idx]
            pred_mask = valid_pred_masks[batch_idx]

            if not has_seg_token[batch_idx]:
                pred_mask = pred_mask[0: 0]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0] # maybe zero
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0] # maybe zero
            )
            num_masks += gt_mask.shape[0] # maybe zero

        mask_bce_loss = self.config.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.config.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        torch.cuda.empty_cache()
        # print('='*80)
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }


    # resize: (768, 1024), original_size: (480, 640)
    def evaluate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,

        images_sam: Optional[torch.FloatTensor] = None,
        resize_list: Optional[List[tuple]] = None,
        original_size_list: Optional[List[tuple]] = None, 
    ):

        with torch.no_grad():
            assert images_sam.shape[0] == 1

            seg_token_mask = (input_ids == self.config.seg_token_idx)
            seg_token_mask = torch.cat([seg_token_mask[:, 1:], torch.zeros_like(seg_token_mask)[:, 0].unsqueeze(1)], dim=1)

            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

            hidden_states = []
            assert len(self.text_hidden_fcs) == 1
            hidden_states.append(self.text_hidden_fcs[0](output_hidden_states[-1]))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ] [1]

            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1, device=images_sam.device, dtype=torch.long), seg_token_offset], dim=0
            ) #[0, 1]

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_


            pred_masks = []
            for i in range(len(pred_embeddings)):

                language_embeddings = pred_embeddings[i] # (1, 256)

                sam_states = self.grounding_encoder.get_sam2_embeddings(images_sam.squeeze(0)) # here all frames
                masks = self.grounding_encoder.language_embd_inference(sam_states, [language_embeddings] * images_sam.shape[1]) # (num_frames, 1, 1024, 1024)

                h, w = original_size_list[i] 
                masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False)
                masks = masks[:, 0] #(num_frames, 1, h, w)
                masks = masks.sigmoid() > 0.5

                pred_masks.append(masks)

        return output, pred_masks





if __name__ == "__main__":

    model_dir = "/mnt/ali-sh-1/usr/chenqirui/data/models/Qwen2.5-VL-3B-Instruct"

    # import debugpy
    # debugpy.listen(("10.148.246.76", 7850)) # 10.148.254.81
    # debugpy.wait_for_client() 
