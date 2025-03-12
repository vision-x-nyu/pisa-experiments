import numpy as np
from scipy.spatial import cKDTree

def binary_mask_IOU(mask1, mask2):
    # mask1, mask2: (f, n, h, w)
    f, n = mask1.shape[:2]
    iou_list = []
    for i in range(f):
        for j in range(n):
            if mask1[i, j].ndim != 2 or mask2[i, j].ndim != 2:
                raise ValueError(f"Unexpected shape for mask_slice: {mask1[i, j].shape}")
            mask1_area = np.count_nonzero(mask1[i] == 1)
            mask2_area = np.count_nonzero(mask2[i] == 1)
            intersection = np.count_nonzero(np.logical_and(mask1[i] == 1, mask2[i] == 1))
            if mask1_area == mask2_area == intersection:
                iou_list.append(1)
            else:
                iou = intersection / (mask1_area + mask2_area - intersection)
                iou_list.append(iou)
    return np.nanmean(iou_list)

def scaled_l2_distance(centroid1, centroid2, resolution, scale_factor=1/np.sqrt(2)):
    res = np.nanmean(np.sqrt((centroid1 - centroid2)**2)) * scale_factor / resolution
    return res

def mask_to_points(mask):
    points = np.column_stack(np.where(mask == True)).astype(np.float32)
    return points


def chamfer_distance(mask1, mask2, resolution, scale_factor=1/np.sqrt(2)):
    f, n = mask1.shape[:2]
    chamfer_distance_list = []
    for i in range(f):
        for j in range(n):
            mask1_points = mask_to_points(mask1[i, j])
            mask2_points = mask_to_points(mask2[i, j])
            tree1 = cKDTree(mask1_points)
            tree2 = cKDTree(mask2_points)
            dist1_to_2, _ = tree1.query(mask2_points)
            dist2_to_1, _ = tree2.query(mask1_points)
            chamfer_dist = np.nanmean(dist1_to_2) + np.nanmean(dist2_to_1)
            # Normalize by resolution
            chamfer_dist = chamfer_dist * scale_factor / resolution
            chamfer_distance_list.append(chamfer_dist)
    return np.nanmean(chamfer_distance_list)

