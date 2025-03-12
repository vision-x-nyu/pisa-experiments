import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
from scipy.ndimage import zoom


def process_single_video(config, is_real):
    src_video = config["src_video"]
    dst_video = config["dst_video"]
    resolution = config["resolution"]

    file_names = [f for f in sorted(os.listdir(src_video)) if f.endswith(".jpg")]
    src_paths = [os.path.join(src_video, f) for f in file_names]
    src_frames = [Image.open(f) for f in src_paths]
    dst_paths = [os.path.join(dst_video, f) for f in file_names]
    width, height = src_frames[0].size
    ratio = resolution / width

    os.makedirs(dst_video, exist_ok=True)
    for src_path, src_frame, dst_path in zip(src_paths, src_frames, dst_paths):
        dst_frame = src_frame.resize((resolution, resolution))
        dst_frame.save(dst_path)

    src_clip_info = os.path.join(src_video, "clip_info.json")
    dst_clip_info = os.path.join(dst_video, "clip_info.json")
    if is_real:
        # For real world videos, point annotations are stored in clip_info.json.
        clip_info = json.load(open(src_clip_info))
        for i in range(len(clip_info["points"])):
            for j in range(len(clip_info["points"][i]["positive"])):
                clip_info["points"][i]["positive"][j][0] = int(clip_info["points"][i]["positive"][j][0] * ratio)
                clip_info["points"][i]["positive"][j][1] = int(clip_info["points"][i]["positive"][j][1] * ratio)
            for j in range(len(clip_info["points"][i]["negative"])):
                clip_info["points"][i]["negative"][j][0] = int(clip_info["points"][i]["negative"][j][0] * ratio)
                clip_info["points"][i]["negative"][j][1] = int(clip_info["points"][i]["negative"][j][1] * ratio)
        with open(dst_clip_info, "w") as f:
            json.dump(clip_info, f)
    else:
        # For simulated videos, point annotations are stored in mask.npz.
        os.system(f"cp {src_clip_info} {dst_clip_info}")
        mask = np.load(os.path.join(src_video, "mask.npz"))["mask"] # [num_frames, num_objects, height, width]
        mask = zoom(mask, (1, 1, ratio, ratio), order=0) # [num_frames, num_objects, resolution, resolution]
        np.savez_compressed(os.path.join(dst_video, "mask.npz"), mask=mask)


def run(args) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config_file = args.config_file
    logger.info(f"Loaded resolution configs from {config_file}")
    config_list = []
    with open(config_file, "r") as f:
        for line in f:
            config_list.append(json.loads(line))
    num_configs = len(config_list)
    logger.info(f"Loaded {num_configs} resolution configs from {config_file}")
    for i in tqdm(range(num_configs)):
        try:
            config = config_list[i]
            process_single_video(config, args.is_real)
        except Exception as e:
            logger.error(f"Error in processing config {i + 1}/{num_configs}")
            logger.error(e)
            continue
    logger.info("Resolution conversion completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Resolution config file path.", required=True)
    parser.add_argument("--is_real", action="store_true", help="Whether the video is real world or not.")
    args = parser.parse_args()
    run(args)