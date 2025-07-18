import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision.utils import save_image


from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor


class STOM:
    def __init__(
        self,
        device='cpu',
        ckpt="/PATH/TO/co-tracker/checkpoints/scaled_offline.pth",
    ):
        self.device = device
        self.model = CoTrackerPredictor(checkpoint=ckpt).to(device)
        # self.grid_size = 100


    def track_in_video(self, frames, vip_frame, vip_frame_idx, save_path):
        video = torch.from_numpy(np.stack([np.array(frame) for frame in frames])).to(self.device)
        video = video.permute(0, 3, 1, 2).unsqueeze(0).float()

        vip_mask = (np.array(vip_frame)[:, :, 3] > 0).astype(np.uint8) * 255

        # contours, hierarchy = cv2.findContours(
        #     vip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        # vip_closed_mask = np.zeros_like(vip_mask)
        # cv2.drawContours(vip_closed_mask, contours, -1, (255), thickness=cv2.FILLED)


        coords = np.argwhere(vip_mask == 255)
        # Find the bounding rectangle (min and max coordinates)
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        # Calculate the center of the bounding rectangle
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2

        # Calculate the radius as 10% of the minimum dimension of the bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        radius = int(min(width, height) * 0.3)
        
        # Create a new mask and draw a circle at the center
        vip_closed_mask = np.zeros_like(vip_mask)
        cv2.circle(vip_closed_mask, (cx, cy), radius, 255, cv2.FILLED)


        segm_mask = vip_closed_mask

        pred_tracks, pred_visibility = self.model(
            video,
            grid_size=100, #TODO: determined by mask size, larger for more points
            segm_mask=torch.from_numpy(segm_mask)[None, None].to(self.device),
            grid_query_frame=vip_frame_idx,
            backward_tracking=True,
        )
        if save_path:
            self.show(video, pred_tracks, pred_visibility, save_path)

        return pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy()


    def propagate_in_video(
        self, frames, src_frame_vip, vip_frame_idx, shape="rectangle", save_path=""
    ):

        pred_tracks, pred_visibility = self.track_in_video(frames, src_frame_vip, vip_frame_idx, save_path)
        
        
        vip_frame_track = pred_tracks[0, vip_frame_idx]

        propagated_frames = []
        for idx, tgt_frame in enumerate(frames):

            if idx == vip_frame_idx:
                tgt_frame_pil = tgt_frame.convert("RGBA")
                tgt_frame_pil = Image.alpha_composite(tgt_frame_pil, src_frame_vip).convert('RGB')
                propagated_frames.append(tgt_frame_pil)
                continue

            tgt_frame_track = pred_tracks[0, idx]  # (N, 2)
            tgt_frame_visibility = pred_visibility[0, idx]

            if shape in ["mask", "mask contour"]:
                try:
                    tgt_frame_pil, warped_vip = self.warp_point(
                        np.array(src_frame_vip), np.array(tgt_frame), tgt_frame_track, tgt_frame_visibility
                    )
                    propagated_frames.append(tgt_frame_pil)
                except:
                    propagated_frames.append(tgt_frame)

            elif shape not in ["mask", "mask contour"]:

                vip_visible = vip_frame_track[tgt_frame_visibility]
                tgt_visible = tgt_frame_track[tgt_frame_visibility]
                flows = tgt_visible - vip_visible
                
                if len(flows) == 0: 
                    propagated_frames.append(tgt_frame)
                    continue
                else:
                    flow_magnitudes = np.linalg.norm(flows, axis=1)
                    median = np.median(flow_magnitudes)
                    mad = np.median(np.abs(flow_magnitudes - median))
                    threshold = 3 * mad
                    mask = (flow_magnitudes >= median - threshold) & (
                        flow_magnitudes <= median + threshold
                    )
                    filtered_flows = flows[mask]

                    # print(idx, tgt_frame_track.shape[0], tgt_frame_visibility.sum(), len(filtered_flows))
                    if len(filtered_flows) < tgt_frame_visibility.shape[0] // 2:
                        propagated_frames.append(tgt_frame)
                        continue

                    avg_flow_x = np.mean(filtered_flows[:, 1]) if filtered_flows.size > 0 else 0.0
                    avg_flow_y = np.mean(filtered_flows[:, 0]) if filtered_flows.size > 0 else 0.0

                    if np.isnan(avg_flow_x) or np.isnan(avg_flow_y):
                        propagated_frames.append(tgt_frame)
                        continue

                    tgt_frame_pil, warped_vip = self.warp(
                        np.array(src_frame_vip), np.array(tgt_frame), avg_flow_y, avg_flow_x
                    )
                    propagated_frames.append(tgt_frame_pil)

            else:
                propagated_frames.append(tgt_frame)

        return propagated_frames



    def warp(self, src_frame_vip, tgt_frame, avg_flow_x, avg_flow_y):
        src_frame_vip_mask = (src_frame_vip[:, :, 3] > 0).astype(np.uint8) * 255

        warped_src_frame_vip = np.zeros_like(src_frame_vip)
        for y, x in np.argwhere(src_frame_vip_mask > 0):
            new_x = int(x + avg_flow_x)
            new_y = int(y + avg_flow_y)

            if 0 <= new_x < tgt_frame.shape[1] and 0 <= new_y < tgt_frame.shape[0]:
                warped_src_frame_vip[new_y, new_x, :] = src_frame_vip[y, x, :]

        warped_vip = Image.fromarray(warped_src_frame_vip, "RGBA")
        tgt_frame_pil = Image.fromarray(tgt_frame, "RGB").convert("RGBA")
        tgt_frame_pil = Image.alpha_composite(tgt_frame_pil, warped_vip)

        return tgt_frame_pil.convert("RGB"), warped_vip


    def warp_point(self, src_frame_vip, tgt_frame, pred_tracks, pred_visibility):

        if pred_visibility.sum() < len(pred_tracks)//2:
            return Image.fromarray(tgt_frame, "RGB"), None

        src_frame_vip_mask = (src_frame_vip[:, :, 3] > 0).astype(np.uint8) * 255
        if np.any(src_frame_vip_mask > 0):
            color_rgba = src_frame_vip[src_frame_vip_mask > 0][0]
        else:
            color_rgba = np.array([0, 0, 0, 0], dtype=np.uint8)

        color_rgba[3] = max(min(color_rgba[3], 148), 96)

        h, w = src_frame_vip.shape[:2]
        warped_src_frame_vip = np.zeros_like(src_frame_vip)

        mask = np.zeros((h, w), dtype=np.uint8)
        for idx, point in enumerate(pred_tracks):
            if pred_visibility[idx]:
                x = int(point[1].item())
                y = int(point[0].item())
                if 0 <= x < h and 0 <= y < w:
                    mask[x, y] = 255

        kernel_size = min(h, w) // 15
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # warped_src_frame_vip[closed_mask > 0] = color_rgba
        M = cv2.moments(closed_mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = M["m00"]
            # radius = int(round((area / np.pi) ** 0.5))
            radius = min(h, w) // 20
            circle_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(circle_mask, (cx, cy), radius, 255, -1)
            warped_src_frame_vip[circle_mask > 0] = color_rgba

        warped_vip = Image.fromarray(warped_src_frame_vip, "RGBA")
        tgt_frame_pil = Image.fromarray(tgt_frame, "RGB").convert("RGBA")
        tgt_frame_pil = Image.alpha_composite(tgt_frame_pil, warped_vip)

        return tgt_frame_pil.convert("RGB"), warped_vip



    def show(self, propagated_video, pred_tracks, pred_visibility, save_dir="./videos"):
        vis = Visualizer(
            save_dir=save_dir,
            show_first_frame=0,
            linewidth=2,
            pad_value=0,
            fps=1,
            tracks_leave_trace=0,
        )
        try:
            res_video = vis.visualize(
                video=propagated_video,
                tracks=pred_tracks,
                visibility=pred_visibility,
                filename="propagated_video",
                save_video=False,
            )

            res_video = res_video.squeeze().float() / 255.0


            for idx in range(len(res_video)):
                save_image(res_video[idx], os.path.join(save_dir, f'{idx}.png'))
        
        except ValueError:
            return
