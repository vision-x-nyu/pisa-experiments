import logging
import argparse
import os
import numpy as np
from tqdm import tqdm
import json
import torch
from PIL import Image
import sys; sys.path = ["sam2"] + sys.path
from sam2.sam2_video_predictor import SAM2VideoPredictor

def load_model(pretrained_model_name_or_path, device):
    video_predictor = SAM2VideoPredictor.from_pretrained(pretrained_model_name_or_path)
    video_predictor.to(device)
    return video_predictor

def process_single_video(config, model):
    video_path = config["video"]
    annotation_path = config["annotation"]
    mask_path = config["mask"]
    
    points = json.load(open(annotation_path, "r"))["points"]
    num_objects = len(points)
    inference_state = model.init_state(video_path=video_path)
    model.reset_state(inference_state)
    
    for ann_obj_id in range(num_objects):
        positive_points = np.array(points[ann_obj_id]["positive"]).reshape(-1, 2)
        negative_points = np.array(points[ann_obj_id]["negative"]).reshape(-1, 2)
        labels = np.concatenate([np.ones(len(positive_points)), np.zeros(len(negative_points))])
        ann_points = np.concatenate([positive_points, negative_points])
        _, out_obj_id, out_mask_logis = model.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=ann_obj_id,
            points=ann_points,
            labels=labels,
        )
    masks = []
    for frame_id, obj_id, mask in model.propagate_in_video(inference_state):
        masks.append(mask.squeeze())
    masks = torch.stack(masks)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    masks = (masks > 0.0).cpu().numpy() # [num_frames, num_objects, height, width]
    np.savez_compressed(mask_path, mask=masks)

def run(args) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    model = load_model(args.pretrained_model_name_or_path, args.device)
    logger.info("Loaded SAM 2 model.")
    
    config_file = args.config_file
    logging.info(f"Loaded mask generation config from {config_file}")
    config_list = []
    with open(config_file, "r") as f:
        for line in f:
            config_list.append(json.loads(line))
    num_configs = len(config_list)
    logger.info(f"Loaded {num_configs} mask generation configs from {config_file}")
    for i in tqdm(range(num_configs)):
        try:
            config = config_list[i]
            process_single_video(config, model)
        except Exception as e:
            logger.error(f"Error in processing config {i + 1}/{num_configs}")
            logger.error(e)
            continue
    logger.info("Mask generation completed.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Mask generation config file path.", required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="SAM 2 model path.", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", type=str, help="Device to run the model on.", default="cuda")
    args = parser.parse_args()
    run(args)