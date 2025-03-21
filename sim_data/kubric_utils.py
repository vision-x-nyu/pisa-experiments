import kubric as kb
import numpy as np
import bpy
from typing import List, Tuple
import os
import subprocess

def obj_drop_visible_from_camera(obj, scene):
    min_corner, max_corner = get_world_object_bounds(obj) 

    drop_bounds = np.array([ # top left and bottom right of plane near camera
        [min_corner[0], max_corner[1], max_corner[2]],
        [min_corner[0], min_corner[1], 0],
    ])
    projected_points = project_points_to_image(drop_bounds, scene)
    return all(
        0 <= p[0] < scene.resolution[0] and 0 <= p[1] < scene.resolution[1]
        for p in projected_points
    )

def project_points_to_image(points, scene):
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    w, h = scene.resolution

    cam_info = kb.get_camera_info(scene.camera)
    intrinsics = cam_info["K"]
    extrinsic_matrix = np.linalg.inv(cam_info["R"])

    points_camera_h = (extrinsic_matrix @ points_h.T).T
    points_image_h = (intrinsics @ points_camera_h[:, :3].T).T

    points_image = (points_image_h[:, :2] / points_image_h[:, 2][:, np.newaxis]) * np.array([w, h])
    return points_image.astype(int)

def get_world_object_bounds(obj: kb.Object3D) -> Tuple[np.ndarray, np.ndarray]:
    position = obj.position
    local_min_corner, local_max_corner = get_local_object_bounds(obj)
    min_corner = position + local_min_corner
    max_corner = position + local_max_corner
    return min_corner, max_corner

def get_local_object_bounds(obj: kb.Object3D) -> Tuple[np.ndarray, np.ndarray]:
    min_bound, max_bound = obj.bounds
    scale = obj.scale
    min_corner = min_bound*scale
    max_corner = max_bound*scale
    return min_corner, max_corner   

def get_object_metadata(obj: kb.Object3D) -> dict:
    metadata = obj.metadata
    metadata["asset_id"] = obj.asset_id
    metadata["position"] = obj.position.tolist()
    metadata["quaternion"] = obj.quaternion.tolist()
    metadata["scale"] = obj.scale.tolist()
    metadata["mass"] = obj.mass
    metadata["friction"] = obj.friction
    return metadata

def get_camera_metadata(camera: kb.Camera) -> dict:
    return {
        "position": camera.position.tolist(),
        "quaternion": camera.quaternion.tolist(),
        "fov": camera.field_of_view,
        "focal_length": camera.focal_length,
        "intrinsics": camera.intrinsics.tolist(),
        "sensor_width": camera.sensor_width,
        "sensor_height": camera.sensor_height,
        "matrix_world": camera.matrix_world.tolist(),
    }

def sample_obj_dropping_position(
    obj: kb.Object3D, 
    placement_region: Tuple[np.ndarray, np.ndarray], 
    min_drop_height: float
) -> Tuple[float, float, float]:

    object_min_corner, object_max_corner = get_local_object_bounds(obj)
    placement_min_corner, placement_max_corner = placement_region
    placement_min_corner[2] += min_drop_height

    assert np.all(placement_max_corner - placement_min_corner >= object_max_corner - object_min_corner), \
        "Object cannot fit inside the placement region"
    
    object_h, object_w, object_d = object_max_corner - object_min_corner

    placement_min_corner += np.array([object_w/2, object_h/2, object_d/2])
    placement_max_corner -= np.array([object_w/2, object_h/2, object_d/2])

    return np.random.uniform(placement_min_corner, placement_max_corner)

def sample_position_in_region(
        obj: kb.Object3D, 
        placement_region: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[float, float, float]:
    
    object_min_corner, object_max_corner = get_local_object_bounds(obj)
    placement_min_corner, placement_max_corner = placement_region
    
    assert np.all(placement_max_corner - placement_min_corner >= object_max_corner - object_min_corner), \
        "Object cannot fit inside the placement region"
    
    object_h, object_w, object_d = object_max_corner - object_min_corner
    placement_min_corner += np.array([object_w/2, object_h/2, object_d/2])
    placement_max_corner -= np.array([object_w/2, object_h/2, object_d/2])

    return np.random.uniform(placement_min_corner, placement_max_corner)
    


def save_state(
    engine: kb.renderer.Blender,
    save_path: str,
):
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)

    engine.save_state(save_path)


def get_static_placement_bounds(
    obj: kb.Object3D,
    offset: float = 0.1,
):
    min_corner, max_corner = get_world_object_bounds(obj)
    new_min_corner = np.array([min_corner[0], min_corner[1], 0])
    new_max_corner = np.array([max_corner[0], max_corner[1], min_corner[2]])
    new_min_corner -= np.array([offset, offset, 0])
    new_max_corner += np.array([offset, offset, 0])
    return (new_min_corner, new_max_corner)

def get_object_height(obj: kb.Object3D):
    min_corner, max_corner = get_world_object_bounds(obj)
    return max_corner[2] - min_corner[2]

def get_object_width(obj: kb.Object3D):
    min_corner, max_corner = get_world_object_bounds(obj)
    return max_corner[1] - min_corner[1]

def get_object_length(obj: kb.Object3D):
    min_corner, max_corner = get_world_object_bounds(obj)
    return max_corner[0] - min_corner[0]

def get_object_height_from_center(obj: kb.Object3D):
    min_corner, _ = get_world_object_bounds(obj)
    return obj.position[2] - min_corner[2]

def get_object_img_area(obj: kb.Object3D, scene: kb.Scene):
    min_corner, max_corner = get_world_object_bounds(obj)
    projected_points = project_points_to_image(np.array([min_corner, max_corner]), scene)
    return np.prod(np.abs(projected_points[0] - projected_points[1])) / (scene.resolution[0] * scene.resolution[1])

def save_video(output_dir, vid_name, key, fps=16):
    _, ext = os.path.splitext(key)
    
    if ext == ".jpg":
      ffmpeg_cmd = (
          f"ffmpeg -y -framerate {fps} -i {os.path.join(output_dir, key)} "
          f"-pix_fmt yuv420p {os.path.join(output_dir, vid_name)}"
      )
    elif ext == ".png":
      ffmpeg_cmd = (
        f"ffmpeg -y -framerate {fps} -i {os.path.join(output_dir, key)} "
        f"-vcodec png -pix_fmt rgba {os.path.join(output_dir, vid_name)}"
      )
    subprocess.run(ffmpeg_cmd, shell=True)