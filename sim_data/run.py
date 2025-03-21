
import os 
import sys
import pyquaternion as pyquat
import bpy
from mathutils import Vector, Matrix, Quaternion, Euler
import hashlib
import string
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import sys
import sys; sys.path = ["kubric"] + sys.path
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from kubric.core import materials
import numpy as np
from loguru import logger
import logging
from skimage import io 
import json 
import scipy.spatial.transform as transform
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation as R
from config import Configuration
import uuid
from math import radians
from kubric_utils import *
import copy
import signal
import tarfile
import shutil
# import moviepy.editor as mp
import subprocess
from PIL import Image

os.environ["KUBRIC_USE_GPU"] = "0"

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def time_limit(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Reset the alarm
            return result
        return wrapper
    return decorator

parser = kb.ArgumentParser()
parser.add_argument(
    "--output_dir", 
    type=str,    
    default="output"
)
parser.add_argument(
    "--scenario", 
    type=str, 
    default="fall"
)
parser.add_argument(
    "--video_id",
    type=str,
    default=str(uuid.uuid4())
)
parser.add_argument(
    "--scene_save_path",
    type=str,
    default=""
)
parser.add_argument(
    "--save_mp4",
    action="store_true",
    default=False
)
parser.add_argument(
    "--save_gif",
    action="store_true",
    default=False
)
parser.add_argument(
    "--tar",
    action="store_true",
    default=False
)
parser.add_argument(
    "--num_threads",
    type=int,
    default=None
)
parser.add_argument(
    "--objects_split",
    type=str,
    default="train",
    choices=["train", "test"]
)
parser.add_argument(
    "--backgrounds_split",
    type=str,
    default="train",
    choices=["train", "test"]
)
parser.add_argument(
    "--max_motion_blur",
    type=float,
    default=0.0
)
parser.add_argument(
    "--kubasic_assets",
    type=str,
    default="gs://kubric-public/assets/KuBasic/KuBasic.json"
)
parser.add_argument(
    "--hdri_assets", 
    type=str,
    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"
)
parser.add_argument(
    "--gso_assets", 
    type=str,
    default="gs://kubric-public/assets/GSO/GSO.json"
)
parser.add_argument(
    "--save_state", 
    action="store_true"
)
parser.add_argument(
    "--focal_length", 
    type=float, 
    default=35.0
)
parser.add_argument(
    "--sensor_width", 
    type=float, 
    default=32
)
parser.set_defaults(resolution="1024x1024",frame_rate=16,frame_end=32)
args = parser.parse_args()

class Intuitive:
    def __init__(self, args):
        (
            self.scene, 
            self.rng, 
            _, 
            self.scratch_dir, 
            self.renderer, 
            self.simulator
        ) = self.configure(args)

        self.video_id = args.video_id
        self.args = args
        self.base_dir = args.output_dir
        self.output_dir = os.path.join(args.output_dir, self.video_id)
        (
            self.kubasic, 
            self.gso, 
            self.hdri_source
        ) = self.load_scene(args)
        self.cfg = Configuration.get_config_for_scenario(self.args.scenario)
        
        if self.cfg.min_num_dynamic_objects == self.cfg.max_num_dynamic_objects:
            self.num_dynamic_objects = self.cfg.min_num_dynamic_objects
        else:
            self.num_dynamic_objects = self.rng.randint(
                self.cfg.min_num_dynamic_objects,
                self.cfg.max_num_dynamic_objects,
            )
        self.cfg.dynamic_spawn_region[0][2] += self.cfg.min_drop_height
        self.metadata = {
            "object_split": args.objects_split,
            "background_split": args.backgrounds_split,
        }
        self.num_threads = args.num_threads
                
    
    def configure(self, args):
        scene, rng, output_dir, scratch_dir = kb.setup(args)
        simulator = PyBullet(
            scene, scratch_dir
        )
        renderer = Blender(
            scene, 
            scratch_dir,
            use_denoising=True, 
            samples_per_pixel=64,
            motion_blur=args.max_motion_blur
        )
        return scene, rng, output_dir, scratch_dir, renderer, simulator

    def load_scene(self, args):
        hdri_source = kb.AssetSource.from_manifest(args.hdri_assets)
        kubasic = kb.AssetSource.from_manifest(args.kubasic_assets)
        gso = kb.AssetSource.from_manifest(args.gso_assets)
        return kubasic, gso, hdri_source

    def configure_split(self):
        train_object_split, test_object_split = self.gso.get_test_split(fraction=0.1)
        train_background_split, test_background_split = self.hdri_source.get_test_split(fraction=0.1)
        if args.objects_split == "train":
            active_object_split = train_object_split
            active_background_split = train_background_split
        else:
            active_object_split = test_object_split
            active_background_split = test_background_split
        self.active_object_split = active_object_split
        self.active_background_split = active_background_split

        
    def setup_scene_background(self):
        hdri_id = self.rng.choice(self.active_background_split)
        background_hdri = self.hdri_source.create(asset_id=hdri_id)
        self.metadata["hdri_id"] = hdri_id
        assert isinstance(background_hdri, kb.Texture)
        logging.info("Using background %s", hdri_id)
        self.scene.metadata["background"] = hdri_id
        self.renderer._set_ambient_light_hdri(background_hdri.filename)

        # Dome
        dome = self.kubasic.create(asset_id="dome", name="dome",
                friction=1.0,
                restitution=0.0,
                static=True, background=True)
        assert isinstance(dome, kb.FileBasedObject)
        self.scene += dome
        dome_blender = dome.linked_objects[self.renderer]
        texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
        texture_node.image = bpy.data.images.load(background_hdri.filename)
        
    def setup_camera(self):
        height = self.rng.uniform(0.4, 0.6)
        # tilt = random.uniform(-15, 15)
        position = (-2,0,height)
        self.scene.camera = kb.PerspectiveCamera(
            focal_length=self.args.focal_length, 
            sensor_width=self.args.sensor_width,
            position=position,
            euler=(radians(90), 0, radians(270))
        )
     
    def place_dynamic_objects(self):
        objs = []
        for i in range(self.num_dynamic_objects):
            attempts = 0
            while attempts < 10:
                obj_id = self.rng.choice(self.active_object_split)
                obj = self.gso.create(asset_id=obj_id)
                self.scene.add(obj)
                samplers = [
                    kb.randomness.position_sampler(self.cfg.dynamic_spawn_region), 
                    kb.randomness.rotation_sampler(axis=None), 
                ]
                logging.info("Sampling dynamic object pose...")
                kb.resample_while(obj,samplers,self.simulator.check_overlap,rng=self.rng)
                min_corner, max_corner = get_world_object_bounds(obj) 
                drop_bounds = np.array([ # top left and bottom right of plane near camera
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], min_corner[1], 0],
                ])
                projected_points = project_points_to_image(drop_bounds, self.scene)
                area = abs((projected_points[0][0] - projected_points[1][0]) * (projected_points[0][1] - projected_points[1][1]))
                self.scene.remove(obj)
                if area / (self.scene.resolution[0] * self.scene.resolution[1]) > 0.025:
                    break
                attempts += 1
                if attempts < 10:
                    del obj

            objs.append(obj)
            

        return objs
       
      
    def place_static_objects_under_dynamic_obj(self, dynamic_obj):
        tower = []
        tower_size = self.rng.randint(self.cfg.min_tower_size, self.cfg.max_tower_size)
        if tower_size == 0:
            return tower
        obj_ids = self.rng.choice(self.active_object_split, tower_size, replace=False)
        z,offset = 0.0, 0.001
        logging.info(f"Making a tower of {tower_size} objects...")
        for obj_id in obj_ids:
            obj = self.gso.create(asset_id=obj_id)
            obj.position = (dynamic_obj.position[0], dynamic_obj.position[1], z + offset +get_object_height_from_center(obj))
            _, max_corner = get_world_object_bounds(obj)
            z = max_corner[2]
            tower.append(obj)
            self.scene.add(obj)
            self.static_objects.append(obj)
        return tower
    
    def move_camera_back(self):
        while not all(obj_drop_visible_from_camera(obj,self.scene) for obj in self.dynamic_objects):
            logging.info("Moving camera back...")
            position = (
                self.scene.camera.position[0] - abs(self.rng.normal(0, 0.05)), 
                self.scene.camera.position[1], 
                self.scene.camera.position[2]
            )
            self.scene.camera.position = position

    def setup_lights(
        self, 
    ):
        # this code is borrowed from TaskMeAnything (https://github.com/JieyuZ2/TaskMeAnything/blob/906f7f6ed5ddfd5ec5e94f3c4a5eb6e9670e2e72/tma/imageqa/tabletop_3d/run_blender.py#L223)
        orientation = self.rng.choice([-1, 1])
        key_light_horizontal_angle = orientation * radians(self.rng.uniform(15, 45))
        fill_light_horizontal_angle = - orientation * radians(self.rng.uniform(15, 60))
        key_light_vertical_angle = -radians(self.rng.uniform(15, 45))
        fill_light_vertical_angle = -radians(self.rng.uniform(0, 30))

        sun_x, sun_y = radians(self.rng.uniform(0, 45)), radians(self.rng.uniform(0, 45))
        sun_energy = self.rng.uniform(1.0, 6.0)

        
        # for setting up the three point lighting, we mostly follow https://courses.cs.washington.edu/courses/cse458/05au/reading/3point_lighting.pdf
        # in order to keep lights and camera on the hemisphere pointing to origin, we use a hierarchy of empties

        # create the sun

        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.active_object
        sun.rotation_euler = Euler((sun_x, sun_y, 0), "XYZ")
        sun.data.energy = sun_energy

        # create global empty

        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        x_rot, y_rot, z_rot = radians(90), radians(0), radians(-90)
        empty = bpy.context.scene.objects.get("Empty")


        # radius = random.uniform(1.8,2.2)
        radius = 2.5

        # bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(-radius, 0, 0), rotation=Euler((x_rot, y_rot, z_rot), "XYZ"), scale=(1, 1, 1))
        cam = bpy.context.scene.objects.get("PerspectiveCamera")

        # create camera empty

        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=cam.location, scale=(1, 1, 1))
        x_rot, y_rot, z_rot = radians(90), radians(0), radians(270)
        cam_empty = bpy.context.scene.objects.get("Empty.001")
        cam_empty.name = "camera_empty"

        # make camera empty parent of camera

        bpy.ops.object.select_all(action='DESELECT')
        cam.select_set(True)
        cam_empty.select_set(True)
        bpy.context.view_layer.objects.active = cam_empty
        bpy.ops.object.parent_set()
        bpy.ops.object.select_all(action='DESELECT')

        # make camera empty parent of global empty

        bpy.ops.object.select_all(action='DESELECT')
        cam_empty.select_set(True)
        empty.select_set(True)
        bpy.context.view_layer.objects.active = empty
        bpy.ops.object.parent_set()
        bpy.ops.object.select_all(action='DESELECT')

        light_names = ["key_light", "fill_light", "back_light"]
        light_energies = [100., 75., 75.]

        for light_name, light_energy in zip(light_names, light_energies):
            # create light empty

            empty_name = light_name + "_empty"
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
            x_rot, y_rot, z_rot = radians(90), radians(0), radians(-90)
            light_empty = bpy.context.scene.objects.get("Empty.001")
            light_empty.name = empty_name

            # parent light empty to main (camera) empty

            bpy.ops.object.select_all(action='DESELECT')
            light_empty.select_set(True)
            empty.select_set(True)
            bpy.context.view_layer.objects.active = empty
            bpy.ops.object.parent_set()
            bpy.ops.object.select_all(action='DESELECT')

            # create light

            x_loc, y_loc, z_loc = -radius, 0, 0
            bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(x_loc, y_loc, z_loc), rotation=Euler((x_rot, y_rot, z_rot), "XYZ"), scale=(1, 1, 1))
            bpy.data.objects["Point"].name = light_name
            light = bpy.data.objects[light_name]
            light.data.energy = light_energy
            # light.data.size = 0.5

            # parent light empty to light

            bpy.ops.object.select_all(action='DESELECT')
            light.select_set(True)
            light_empty.select_set(True)
            bpy.context.view_layer.objects.active = light_empty
            bpy.ops.object.parent_set()
            bpy.ops.object.select_all(action='DESELECT')

       

        bpy.context.view_layer.update()

        back_light_horizontal_angle = radians(180)
        light_horizontal_angles = [key_light_horizontal_angle, fill_light_horizontal_angle, back_light_horizontal_angle]
        for light_angle, light_name in zip(light_horizontal_angles, light_names):
            light_empty = bpy.data.objects[light_name + "_empty"]
            global_z = (light_empty.matrix_world.inverted() @ Vector((0.0, 0.0, 1.0, 0.0)))[:3]
            quat = Quaternion(global_z, light_angle)
            light_empty.rotation_euler = quat.to_euler()

        back_light_vertical_angle = 0
        light_vertical_angles = [key_light_vertical_angle, fill_light_vertical_angle, back_light_vertical_angle]

        for light_angle, light_name in zip(light_vertical_angles, light_names):
            light_empty = bpy.data.objects[light_name + "_empty"]
            global_x = (light_empty.matrix_world.inverted() @ Vector((1.0, 0.0, 0.0, 0.0)))[:3]
            quat = Quaternion(global_x, light_angle)
            euler_add = quat.to_euler()
            euler_current = light_empty.rotation_euler
            new_euler = Euler((euler_add[0] + euler_current[0], euler_add[1] + euler_current[1], euler_add[2] + euler_current[2]))
            light_empty.rotation_euler = new_euler
            light = bpy.data.objects[light_name]
            light.location = [light.location[0], light.location[1], 1.5]


        return cam, empty

 
    @time_limit(1200) # 20 minute timeout
    def render_a_scene(self):
        self.configure_split()
        self.setup_scene_background()
        self.setup_camera()
        self.dynamic_objects = self.place_dynamic_objects()
        self.static_objects = []
        self.metadata["static_objects"] = []
        self.metadata["dynamic_objects"] = []
        towers = []
        for dynamic_obj in self.dynamic_objects:
            towers.append(self.place_static_objects_under_dynamic_obj(dynamic_obj))
        base, ext = os.path.splitext(args.scene_save_path)
        if args.scene_save_path:
            save_state(self.renderer, f"{base}_tower_setup{ext}")
        self.simulator.run(frame_start=-200, frame_end=0)
        if args.scene_save_path:
            save_state(self.renderer, f"{base}_tower_settled{ext}")
        
        for i,obj in enumerate(self.dynamic_objects):
            obj = copy.deepcopy(obj)
            self.dynamic_objects[i] = obj
            self.scene.add(obj) # Workaround

        for dyn_obj, tower in zip(self.dynamic_objects, towers):
            if tower:
                max_height = max(
                    get_world_object_bounds(obj)[1][2] for obj in tower
                )
                min_corner, _ = get_world_object_bounds(dyn_obj)
                if min_corner[2] < max_height:
                    logging.info("Raising dynamic object...")
                    dyn_obj.position = (dyn_obj.position[0], dyn_obj.position[1], max_height + abs(self.rng.normal(0.005, 0.005)))
                
        
        self.move_camera_back()
        self.metadata["camera"] = get_camera_metadata(self.scene.camera)
        
        for obj in self.dynamic_objects:
            self.metadata["dynamic_objects"].append(get_object_metadata(obj))
        for obj in self.static_objects:
            self.metadata["static_objects"].append(get_object_metadata(obj))
        
        self.setup_lights()
        if args.scene_save_path:
            save_state(self.renderer,f"{base}_init{ext}")

        logging.info("Running simulation...")
        self.simulator.run(frame_start=0, frame_end=self.scene.frame_end+1)

        logging.info("Rendering frames...")
        if self.num_threads is not None:
            bpy.context.scene.render.threads_mode = 'FIXED'
            bpy.context.scene.render.threads = self.num_threads
        

        data_stack = self.renderer.render(return_layers=["rgba","segmentation","depth"]) 
    
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info("Saving metadata...")
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
        
        for i in range(data_stack['rgba'].shape[0]):
            img_path = os.path.join(self.output_dir, f"rgba_{i:05d}.jpg")
            io.imsave(img_path, 
                    data_stack["rgba"][i][..., :3])
            
            np.save(os.path.join(self.output_dir, f"segmentation_{i:05d}.npy"), data_stack["segmentation"][i])
            np.save(os.path.join(self.output_dir, f"depth_{i:05d}.npy"), data_stack["depth"][i])
        
        logging.info("Deleting scratch directory...")
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)
        if args.save_mp4 or args.save_gif:
            save_video(self.output_dir, "rgba.mp4", "rgba_%05d.jpg", fps=args.frame_rate)
        if args.save_gif:
            video_path = os.path.join(self.output_dir, "rgba.mp4")
            gif_path = os.path.join(self.output_dir, "rgba.gif")
            subprocess.run(['ffmpeg', '-i', video_path, '-vf', 'fps={}'.format(args.frame_rate), gif_path])
        if args.tar:
            tar_path = os.path.join(self.base_dir, f"{self.video_id}.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(self.output_dir, arcname=os.path.basename(self.output_dir))            
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)



if __name__ == "__main__":
    intuit = Intuitive(args)
    try:
        intuit.render_a_scene()
    except TimeoutException:
        logging.error("Timed out")
        sys.exit(1)
