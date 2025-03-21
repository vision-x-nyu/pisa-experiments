from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class ScenarioConfig:
    static_spawn_region: np.ndarray
    dynamic_spawn_region: np.ndarray
    obj_scale_range: np.ndarray
    min_drop_height: float
    inner_radius: int
    outer_radius: int
    theta_min: float
    theta_max: float
    min_num_dynamic_objects: int
    max_num_dynamic_objects: int
    plane_asset_dir: str
    min_tower_size: int
    max_tower_size: int
    num_trials: int = 1

class Configuration:
  @staticmethod
  def get_config_for_scenario(scenario):
    config_methods = {
      "fall": Configuration._fall_config,
    }
    return config_methods.get(scenario, lambda: "Invalid scenario")()
  
  @staticmethod
  def _fall_config():
    return ScenarioConfig(
        static_spawn_region=np.array([(-1, -1, 0), (1, 1, 1)],dtype=np.float64),
        dynamic_spawn_region=np.array([(-1, -1, 0), (1, 1, 1.5)],dtype=np.float64),
        obj_scale_range=np.array([0.15, 0.5],dtype=np.float64),
        min_drop_height=0.5,
        inner_radius=10,
        outer_radius=12,
        theta_min=10.0,
        theta_max=45.0,
        min_num_dynamic_objects=1,
        max_num_dynamic_objects=6,
        plane_asset_dir="/data/oscar/assets/plane_glbs",
        min_tower_size=0, 
        max_tower_size=4
    )
  