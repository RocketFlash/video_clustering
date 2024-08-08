import cv2
import json
from collections import defaultdict
from torch.utils.data import Dataset
from easydict import EasyDict as edict


def load_MSRVTT_dataset(dataset_path):
    video_dir = dataset_path / 'videos/all'
    annos_file = dataset_path / 'annotation/MSR_VTT.json'

    annos = defaultdict(list)
    with open(annos_file, 'r') as f:
        annos_temp = json.load(f)['annotations']
        for anno in annos_temp:
            image_name = anno['image_id'] + '.mp4'
            caption = anno['caption']
            annos[image_name].append(caption)

    train_set_list_file = dataset_path / 'videos/train_list_new.txt'
    test_set_list_file = dataset_path / 'videos/test_list_new.txt'

    with open(train_set_list_file) as f:
        file_names = [line.rstrip() + '.mp4' for line in f]
        train_video_paths = [video_dir / file_name for file_name in file_names]
        train_annos = {file_name : annos[file_name] for file_name in file_names}

    with open(test_set_list_file) as f:
        file_names = [line.rstrip() + '.mp4' for line in f]
        test_video_paths = [video_dir / file_name for file_name in file_names]
        test_annos = {file_name : annos[file_name] for file_name in file_names}

    msrvtt_dataset = dict(
        train_video_paths=train_video_paths,
        train_annos=train_annos,
        test_video_paths=test_video_paths,
        test_annos=test_annos
    )
    return edict(msrvtt_dataset)


class VideoFrameDataset(Dataset):
    def __init__(
        self, 
        video_path, 
        frame_interval=1, 
        transform=None
    ):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.transform = transform
        self.frames, self.frame_indexes = self.extract_frames()

    def extract_frames(self):
        frames = []
        frame_indexes = []
        video = cv2.VideoCapture(str(self.video_path))
        if not video.isOpened(): 
            return frames
            
        frame_count = 0
        while True:
            success = video.grab()
            if not success:
                break
            if frame_count % self.frame_interval == 0:
                success, frame = video.retrieve()
                if not success:
                    break
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(image=frame)
                    frame = frame['image']
                
                frame_indexes.append(frame_count)
                frames.append(frame)
            frame_count += 1
        video.release()

        return frames, frame_indexes

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]   
        frame_idx = self.frame_indexes[idx]     
        return frame, frame_idx
