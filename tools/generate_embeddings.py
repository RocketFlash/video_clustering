import sys
sys.path.append('../')

import multiprocessing
import random
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass
from src.model import (get_model,
                       generate_video_embeddings)
from src.dataset import load_MSRVTT_dataset


@dataclass
class Config:
    random_seed: int = 28
    vis_n_samples: int = 9
    vis_n_cols: int = 3
    vis_width: int = 320
    img_h: int = 224
    img_w: int = 224
    frame_interval: int = 10
    norm_mean = [0.48145466, 0.4578275, 0.40821073]
    norm_std = [0.26862954, 0.26130258, 0.27577711]
    model_type = 'openclip_b32'
    batch_size: int = 128
    n_workers: int = multiprocessing.cpu_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    config = Config()
    random.seed(config.random_seed)

    dataset_path = Path('/datasets/video_clustering/MSRVTT/')
    msrvtt_dataset = load_MSRVTT_dataset(dataset_path)

    model = get_model(
        model_type=config.model_type,
        device=config.device
    )

    transform = A.Compose([
        A.Resize(
            height=config.img_h, 
            width=config.img_w
        ),
        A.Normalize(
            mean=config.norm_mean,
            std=config.norm_std,
        ),
        ToTensorV2()
    ])

    embeddings_save_path = Path('/datasets/video_clustering/MSRVTT/embeddings/')
    embeddings_save_path.mkdir(exist_ok=True)

    for video_path in tqdm(msrvtt_dataset.train_video_paths):
        video_embeddings, video_frame_indexes = generate_video_embeddings(
            video_path,
            model,
            transform,
            batch_size=config.batch_size,
            n_workers=config.n_workers,
            frame_interval=config.frame_interval,
            device=config.device
        )
            
        save_name = video_path.stem + '.npz'
        save_path = embeddings_save_path / save_name
        np.savez(
            save_path, 
            embeddings=video_embeddings, 
            frame_indexes=video_frame_indexes
        )