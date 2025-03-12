import json
import os
import numpy as np
from tqdm import tqdm
import logging
import argparse
from utils.metrics import binary_mask_IOU, scaled_l2_distance, chamfer_distance


def get_dropping_obj_index(points):
    result = 0
    min_y = 10000
    for i in range(len(points)):
        positive_points = points[i]["positive"]
        min_y_i = min([p[1] for p in positive_points])
        if min_y_i < min_y:
            min_y = min_y_i
            result = i
    return result


def find_centroid(mask):
    centroids = []
    for i in range(mask.shape[0]):
        frame_centroids = []
        for j in range(mask.shape[1]):
            mask_slice = mask[i, j]
            if mask_slice.ndim != 2:
                raise ValueError(f"Unexpected shape for mask_slice: {mask_slice.shape}")
            y_indices, x_indices = np.where(mask_slice == 1)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x = np.mean(x_indices)
                y = np.mean(y_indices)
                frame_centroids.append([x, y])
            else:
                frame_centroids.append([np.nan, np.nan])
        centroids.append(frame_centroids)
    return np.array(centroids)


def process_single_video(config):
    gt_mask = np.load(config["gt_mask"])["mask"]
    gt_fps = config["gt_fps"]
    pd_mask = np.load(config["pd_mask"])["mask"]
    pd_fps = config["pd_fps"]

    target_frame_index = [
        int(pd_frame_index * (gt_fps / pd_fps))
        for pd_frame_index in range(len(pd_mask))
        if int(pd_frame_index * (gt_fps / pd_fps)) < len(gt_mask)
    ]
    gt_mask = gt_mask[target_frame_index]
    pd_mask = pd_mask[:min(len(gt_mask), len(pd_mask))]

    if pd_mask.ndim != 4:
        pd_mask = np.expand_dims(pd_mask, axis=1)
    if gt_mask.ndim != 4:
        gt_mask = np.expand_dims(gt_mask, axis=1)
        
    if "gt_annotation" in config:
        # For real world videos, we need to find the dropping object index.
        gt_annotation = config["gt_annotation"]
        points = json.load(open(gt_annotation))["points"]
        dropping_obj_idx = get_dropping_obj_index(points)
    else:
        # For simulated videos, the dropping object is the last object.
        dropping_obj_idx = -1
    # The metrics are calculated for the dropping object only.
    gt_mask = np.expand_dims(gt_mask[:, dropping_obj_idx], axis=1)  # (f, 1, h, w)
    pd_mask = np.expand_dims(pd_mask[:, dropping_obj_idx], axis=1)  # (f, 1, h, w)
    gt_centroids = find_centroid(gt_mask)
    pd_centroids = find_centroid(pd_mask)

    result = {
        "l2": scaled_l2_distance(gt_centroids, pd_centroids, gt_mask.shape[2]),
        "chamfer_distance": chamfer_distance(gt_mask, pd_mask, gt_mask.shape[2]),
        "iou": binary_mask_IOU(gt_mask, pd_mask),
        "config": config
    }

    return result


def run(args) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config_file = args.config_file
    logger.info(f"Loaded metrics configs from {config_file}")
    config_list = []
    with open(config_file, "r") as f:
        for line in f:
            config_list.append(json.loads(line))
    num_configs = len(config_list)
    logger.info(f"Loaded {num_configs} metrics configs from {config_file}")
    results = []
    for i in tqdm(range(num_configs)):
        try:
            config = config_list[i]
            results.append(process_single_video(config))
        except Exception as e:
            logger.error(f"Error in processing config {i + 1}/{num_configs}")
            logger.error(e)
            continue
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")

    # Get average results
    iou_list = [result["iou"] for result in results]
    l2_list = [result["l2"] for result in results]
    chamfer_distance_list = [result["chamfer_distance"] for result in results]
    avg_result = {
        "iou": np.nanmean(iou_list),
        "l2": np.nanmean(l2_list),
        "chamfer_distance": np.nanmean(chamfer_distance_list)
    }
    logger.info(f"Average results: {avg_result}")
    # Save average results
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "a") as f:
        f.write(f"Task: {args.task_name}\n")
        f.write(f"L2: {avg_result['l2']}\n")
        f.write(f"Chamfer Distance: {avg_result['chamfer_distance']}\n")
        f.write(f"IoU: {avg_result['iou']}\n")
        f.write("\n")
    logger.info(f"Results saved to {args.results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Metrics config file path.", required=True)
    parser.add_argument("--output_path", type=str, help="Output path.", required=False, default="./results/default_task.json")
    parser.add_argument("--task_name", type=str, help="Task name.", required=False, default="default_task")
    parser.add_argument("--results_path", type=str, help="Result path.", required=False, default="./results/results.txt")
    args = parser.parse_args()
    run(args)