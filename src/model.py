import torch
import open_clip
from .dataset import VideoFrameDataset
from torch.utils.data import DataLoader


def get_model(
        model_type='openclip_b32',
        device='cpu'
    ):

    model = None
    if model_type=='openclip_b32':
        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='laion2b_s34b_b79k'
        )
        model = model.visual.to(device).eval()

    return model


def generate_video_embeddings(
    video_path,
    model,
    transform,
    batch_size=8,
    n_workers=4,
    frame_interval=10,
    device='cpu'
):
    video_dataset = VideoFrameDataset(
        video_path=video_path, 
        frame_interval=frame_interval, 
        transform=transform
    )
    
    video_dataloader = DataLoader(
        dataset=video_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
    )

    video_embeddings = []
    video_frame_indexes = []
    
    with torch.no_grad():
        for batch_frames, batch_frame_indexes in video_dataloader:
            batch_frames = batch_frames.to(device)
            batch_features = model(batch_frames)
            video_embeddings.append(batch_features.cpu())
            video_frame_indexes.append(batch_frame_indexes)
    
    video_embeddings = torch.cat(video_embeddings).numpy()
    video_frame_indexes = torch.cat(video_frame_indexes).numpy()
    
    return video_embeddings, video_frame_indexes