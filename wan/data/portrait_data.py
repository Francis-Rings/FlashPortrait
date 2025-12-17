import os
import random
import warnings
import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from einops import rearrange
import torch.nn.functional as F

from wan.models.pdf import det_landmarks_without_tqdm, get_drive_expression_pd_fgc_training

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05])
        else:
            mask_index = np.random.choice([0, 1, 7, 8], p = [0.2, 0.7, 0.05, 0.05])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask


class LargeScalePortraitVideos(Dataset):
    def __init__(self, txt_path, width, height, n_sample_frames, sample_frame_rate, enable_inpaint=True, face_aligner=None):
        self.txt_path = txt_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.enable_inpaint = enable_inpaint
        self.video_files = self._read_txt_file_images()
        self.face_aligner = face_aligner

    def _read_txt_file_images(self):
        with open(self.txt_path, 'r') as file:
            lines = file.readlines()
            video_files = []
            for line in lines:
                video_file = line.strip()
                video_files.append(video_file)
        return video_files

    def __len__(self):
        return len(self.video_files)

    def frame_count(self, frames_path):
        files = os.listdir(frames_path)
        png_files = [file for file in files if (file.startswith('frame_') and (file.endswith('.png') or file.endswith('.jpg')))]
        png_files_count = len(png_files)
        return png_files_count

    def find_frames_list(self, frames_path):
        files = os.listdir(frames_path)
        image_files = [file for file in files if (file.startswith('frame_') and (file.endswith('.png') or file.endswith('.jpg')))]
        if image_files[0].startswith('frame_'):
            image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            image_files.sort(key=lambda x: int(x.split('.')[0]))
        return image_files

    def __getitem__(self, idx):
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        frames_path = os.path.join(self.video_files[idx], "images")
        face_masks_path = os.path.join(self.video_files[idx], "face_masks")
        lip_masks_path = os.path.join(self.video_files[idx], "lip_masks")

        video_length = self.frame_count(frames_path)
        frames_list = self.find_frames_list(frames_path)
        all_indices = list(range(0, video_length))
        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_frame_rate + 1)

        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int).tolist()

        tgt_pil_image_list = []
        tgt_bgr_image_list = []
        tgt_face_masks_list = []
        tgt_lip_masks_list = []

        reference_frame_path = os.path.join(frames_path, frames_list[start_idx])
        reference_pil_image = Image.open(reference_frame_path).convert('RGB')
        reference_pil_image = reference_pil_image.resize([self.width, self.height], Image.LANCZOS)
        reference_pil_image = torch.from_numpy(np.array(reference_pil_image)).float()
        reference_pil_image = reference_pil_image / 127.5 - 1

        for index in batch_index:
            tgt_img_path = os.path.join(frames_path, frames_list[index])
            file_name = os.path.basename(tgt_img_path)
            face_mask_path = os.path.join(face_masks_path, file_name)
            lip_mask_path = os.path.join(lip_masks_path, file_name)
            try:
                tgt_img_pil = Image.open(tgt_img_path).convert('RGB')
                tgt_img_pil = tgt_img_pil.resize([self.width, self.height], Image.LANCZOS)
                tgt_img_tensor = torch.from_numpy(np.array(tgt_img_pil)).float()
                tgt_img_normalized = tgt_img_tensor / 127.5 - 1
                tgt_pil_image_list.append(tgt_img_normalized)
            except Exception as e:
                print(f"Fail loading the image: {tgt_img_path}")
                print(1/0)

            try:
                bgr_img = cv2.imread(tgt_img_path)
                tgt_bgr_image_list.append(bgr_img.copy())
            except Exception as e:
                print(f"Fail loading the bgr image: {tgt_img_path}")
                print(1/0)

            try:
                tgt_lip_mask = Image.open(lip_mask_path)
                tgt_lip_mask = tgt_lip_mask.resize([self.width, self.height], Image.LANCZOS)
                tgt_lip_mask = torch.from_numpy(np.array(tgt_lip_mask)).float()
                tgt_lip_mask = tgt_lip_mask / 255
            except Exception as e:
                print(f"Fail loading the lip masks: {lip_mask_path}")
                tgt_lip_mask = torch.ones(self.height, self.width)
            tgt_lip_masks_list.append(tgt_lip_mask)

            try:
                tgt_face_mask = Image.open(face_mask_path)
                tgt_face_mask = tgt_face_mask.resize([self.width, self.height], Image.LANCZOS)
                tgt_face_mask = torch.from_numpy(np.array(tgt_face_mask)).float()
                tgt_face_mask = tgt_face_mask / 255
            except Exception as e:
                print(f"Fail loading the face masks: {face_mask_path}")
                tgt_face_mask = torch.ones(self.height, self.width)
            tgt_face_masks_list.append(tgt_face_mask)

        tgt_bgr_image_list = tgt_bgr_image_list[:self.n_sample_frames]
        with torch.no_grad():
            try:
                landmark_list = det_landmarks_without_tqdm(self.face_aligner, tgt_bgr_image_list)[1]
                emo_list = get_drive_expression_pd_fgc_training(tgt_bgr_image_list, landmark_list)
            except Exception as e:
                print(f"Fail detecting the faces")
                emo_list = torch.zeros((self.n_sample_frames, 3, 224, 224)).float()

        tgt_pil_image_list = torch.stack(tgt_pil_image_list, dim=0)
        tgt_pil_image_list = rearrange(tgt_pil_image_list, "f h w c -> f c h w")
        reference_pil_image = rearrange(reference_pil_image, "h w c -> c h w")

        tgt_face_masks_list = torch.stack(tgt_face_masks_list, dim=0)
        tgt_face_masks_list = torch.unsqueeze(tgt_face_masks_list, dim=-1)
        tgt_face_masks_list = rearrange(tgt_face_masks_list, "f h w c -> c f h w")
        tgt_lip_masks_list = torch.stack(tgt_lip_masks_list, dim=0)
        tgt_lip_masks_list = torch.unsqueeze(tgt_lip_masks_list, dim=-1)
        tgt_lip_masks_list = rearrange(tgt_lip_masks_list, "f h w c -> c f h w")


        clip_pixel_values = reference_pil_image.permute(1, 2, 0).contiguous()
        clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255

        if "singing" in self.video_files[idx]:
            text_prompt = "The protagonist is singing"
        elif "speech" in self.video_files[idx]:
            text_prompt = "The protagonist is talking"
        elif "dancing" in self.video_files[idx]:
            text_prompt = "The protagonist is simultaneously dancing and singing"
        else:
            text_prompt = ""
            print(1 / 0)

        sample = dict(
            pixel_values=tgt_pil_image_list,
            reference_image=reference_pil_image,
            clip_pixel_values=clip_pixel_values,
            tgt_face_masks=tgt_face_masks_list,
            text_prompt=text_prompt,
            tgt_lip_masks=tgt_lip_masks_list,
            emo_list=emo_list,
        )

        if self.enable_inpaint:
            pixel_value_masks = get_random_mask(tgt_pil_image_list.size(), image_start_only=True)
            masked_pixel_values = tgt_pil_image_list * (1-pixel_value_masks)
            sample["masked_pixel_values"] = masked_pixel_values
            sample["pixel_value_masks"] = pixel_value_masks


        return sample