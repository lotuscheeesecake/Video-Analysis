# src/datasets.py
import os
import math
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torchvision import transforms
from PIL import Image

class VideoFrameDataset(Dataset):
    """
    Manifest CSV must have columns: video_path, label_indices
    where label_indices is comma-separated integer indices into labels.json
    """
    def __init__(self, manifest_csv, num_frames=16, image_size=224, transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def _sample_frame_indices(self, video_len):
        # uniform sampling of num_frames across the video
        if video_len <= self.num_frames:
            # pad by repeating last frame index
            idxs = list(range(video_len)) + [video_len - 1] * (self.num_frames - video_len)
            return [min(i, video_len-1) for i in idxs]
        else:
            interval = video_len / self.num_frames
            return [int(i * interval) for i in range(self.num_frames)]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['video_path']
        label_str = str(row['label_indices']) if not pd.isna(row['label_indices']) else ""
        label_indices = [int(x) for x in label_str.split(',')] if label_str else []
        # read video using decord
        try:
            vr = VideoReader(path, ctx=cpu(0))
        except Exception as e:
            # fallback: create black frames
            frames = [torch.zeros(3, self.image_size, self.image_size) for _ in range(self.num_frames)]
            vid = torch.stack(frames)  # T, C, H, W
            label = torch.zeros(self.get_num_labels(), dtype=torch.float32)
            label[label_indices] = 1.0
            return vid, label
        vlen = len(vr)
        indices = self._sample_frame_indices(vlen)
        # decord returns HWC uint8 frame
        frames = vr.get_batch(indices).asnumpy()  # shape (T, H, W, C)
        # convert to CHW tensors and apply transforms per frame
        processed = []
        for f in frames:
            pil = Image.fromarray(f)
            t = self.transform(pil)  # C,H,W
            processed.append(t)
        vid = torch.stack(processed)  # T, C, H, W
        label = torch.zeros(self.get_num_labels(), dtype=torch.float32)
        label[label_indices] = 1.0
        return vid, label

    @staticmethod
    def get_num_labels():
        # override by caller (monkey patch after loading vocab), or use env var.
        # We'll set after reading labels in training script.
        raise NotImplementedError("Set dataset.get_num_labels() after loading labels.")
