import os
import torch
import numpy as np
from tqdm import tqdm
import subprocess
import argparse

def save_video_from_frames(frame_dir: str, fps: int = 16) -> str:
    video_save_path = f"{frame_dir}/movie.gif"
    ffmpeg_command = (
        f"ffmpeg -framerate {fps} -i {frame_dir}/%05d.jpg " 
        f" -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" " 
        f"-r {fps} {video_save_path}"
    )
    subprocess.run(ffmpeg_command,shell=True)
    return video_save_path

def video_to_frames(video_path, save_dir, fps=16):
    os.makedirs(save_dir, exist_ok=True)
    ffmpeg_command = f"ffmpeg -i {video_path} -vf fps={fps} -start_number 0 {save_dir}/%05d.jpg"
    subprocess.run(ffmpeg_command,shell=True)

def process_video(video_path):
    video_name = os.path.basename(video_path)
    dir_name = os.path.dirname(video_path)
    index = video_name.split(".")[0].split("_")[-1]
    save_dir = os.path.join(dir_name, index)
    os.makedirs(save_dir, exist_ok=True)
    video_to_frames(video_path, save_dir)
    save_video_from_frames(save_dir)
    os.remove(video_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    files = sorted(os.listdir(args.root_path))
    for i in tqdm(range(len(files))):
        video_path = os.path.join(args.root_path, files[i])
        process_video(video_path)