import cv2
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict


def get_random_files(files_list, n):
    if len(files_list)>n:
        return random.sample(files_list, n)
    else:
        return files_list


def get_video_info(filename):
    video = cv2.VideoCapture(filename)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = round(frame_count / fps, 3)

    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    video_info = dict(
        duration=duration,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height
    )

    return video_info


def get_video_infos(video_paths):
    video_infos = {}

    for video_path in tqdm(video_paths):
        video_file_name = video_path.name 
        video_infos[video_file_name] = get_video_info(video_path)

    return video_infos


def get_stats_from_array(array):
    stats = dict(
        mean=np.round(array.mean(), 3),
        std=np.round(array.std(), 3),
        min=np.round(array.min(), 3),
        max=np.round(array.max(), 3),
    )
    return stats
    

def get_dataset_stats(video_infos):
    min_max_mean_std_props = [
        'duration',
        'frame_count',
        'fps',
        'width',
        'height'
    ]

    stats_temp = defaultdict(list) 
    for video_name, video_info in video_infos.items():
        for prop_name, prop_value in video_info.items():
            stats_temp[prop_name].append(prop_value)
    
    stats = {}
    for prop_name in min_max_mean_std_props:
        prop_array = np.array(stats_temp[prop_name])
        stats[prop_name] = get_stats_from_array(prop_array)
    
    return stats


def show_dataset_stats(video_infos):
    dataset_stats = get_dataset_stats(video_infos)
    print(pd.DataFrame.from_dict(dataset_stats))
    # for prop_name, prop_stats in dataset_stats.items():
    #     print(prop_name)
    #     for k, v in prop_stats.items():
    #         print(f'{k} : {v}')
    #     print('='*10)


def read_embeddings_and_aggregate(
    file_path,
    agg_strategy='mean'
):
    embeddings_data = np.load(file_path)
    embeddings = embeddings_data['embeddings']
    
    if agg_strategy == 'mean':
        embeddings_agg = embeddings.mean(axis=0)
    else:
        embeddings_agg = embeddings.max(axis=0)

    return embeddings_agg.reshape(1, -1)