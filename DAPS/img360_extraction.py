import os
import argparse
import numpy as np

import pickle

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="/home/directory/daps", type=str)
args = parser.parse_args()

rgb_sensor = True  # rgb image
depth_sensor = True  # depth image
semantic_sensor = True  # semantic, semantic_cat image

with open('scene_pos_coord_2d.pkl', 'rb') as g:
    scene_position_coord = pickle.load(g)
scene_position_coord = scene_position_coord[0]
scenes = list(scene_position_coord.keys())

# for semantic_cat image > each category label is pixel value 1: void, 2: wall, ..., 41: unlabeled
mp3d_cat = [
    'void', 'wall', 'floor', 'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window',
    'sofa', 'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs', 'ceiling', 'toilet', 'stool',
    'towel', 'mirror', 'tv_monitor', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting', 'beam',
    'railing', 'shelving', 'blinds', 'gym_equipment', 'seating', 'board_panel', 'furniture', 'appliances', 'clothes', 'objects',
    'misc', 'unlabeled'
]
MP_TO_COMP = {
    0:0, 8:0, 14:0, 18:0, 20:0, 30:0, 32:0, 33:0, 35:0, 38:0, 39:0, 40:0, 41:0,
    1:1, 6:1, 12:1, 21:1, 22:1, 23:1, 27:1, 31:1,
    5:6, 7:6, 13:6, 15:6, 25:6, 26:6, 36:6, 37:6,
    4:5, 3:4, 10:4, 19:4, 34:4,
    17:3, 28:3, 29:3,
    2:2, 16:2,
    9:7,
    11:8,
    24:9
}
mp3d_cat_dict = {}
for i in range(len(mp3d_cat)):
    mp3d_cat_dict[mp3d_cat[i]] = i

save_path = args.data_path
for scene in tqdm(scenes):
    save_img_path = os.path.join(save_path, 'img', scene)
    save_img_path_depth = os.path.join(save_img_path, 'depth')
    save_img_path_rgb = os.path.join(save_img_path, 'rgb')
    save_img_path_semantic = os.path.join(save_img_path, 'semantic')
    save_img_path_semantic_cat = os.path.join(save_img_path, 'semantic_cat')

    os.makedirs(save_img_path_rgb)
    os.makedirs(save_img_path_depth)
    os.makedirs(save_img_path_semantic)
    os.makedirs(save_img_path_semantic_cat)

    scene_path = os.path.join(save_path, "v1/tasks/mp3d/{:s}/{:s}.glb".format(scene, scene))
    scene_config = os.path.join(save_path, "v1/tasks/mp3d/mp3d.scene_dataset_config.json")
    h = 1.5
    width = 384
    height = 192
    sim_settings = {
        "width": width,  # Spatial resolution of the observations
        "height": height,
        "scene": scene_path,  # Scene path
        "scene_dataset": scene_config,  # the scene dataset configuration files
        "default_agent": 0,
        "sensor_height": h,  # Height of sensors in meters
        "color_sensor": rgb_sensor,  # RGB sensor
        "depth_sensor": depth_sensor,  # Depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }
    threshold = 0.01 * width * height

    def make_cfg(settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
        sim_cfg.enable_physics = settings["enable_physics"]

        # Note: all sensors must have the same resolution
        sensor_specs = []

        color_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        sensor_specs.append(color_sensor_spec)

        depth_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        sensor_specs.append(depth_sensor_spec)

        semantic_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        sensor_specs.append(semantic_sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=90.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=90.0)
            ),
            "look_left": habitat_sim.agent.ActionSpec(
                "look_left", habitat_sim.agent.ActuationSpec(amount=90.0)
            ),
            "look_up": habitat_sim.agent.ActionSpec(
                "look_up", habitat_sim.agent.ActuationSpec(amount=90.0)
            ),
            "look_down": habitat_sim.agent.ActionSpec(
                "look_down", habitat_sim.agent.ActuationSpec(amount=90.0)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    # semantic_cat image settings
    scene_sem = sim.semantic_scene
    instance_id_to_name = {}
    for obj in scene_sem.objects:
        if obj and obj.category:
            obj_id = int(obj.id.split("_")[-1])
            instance_id_to_name[obj_id] = obj.category.name()

    for k, v in instance_id_to_name.items():
        if v == '':
            v = 'unlabeled'
        instance_id_to_name[k] = mp3d_cat_dict[v]

    # move around each positions in scene and capture image
    positions = scene_position_coord[scene]
    for pos in positions.keys():
        coord = positions[pos]
        current = [coord[0], coord[2]-h, -coord[1]]

        # Set agent state
        agent = sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(current)  # in world space
        agent.set_state(agent_state)

        # Get agent state
        agent_state = agent.get_state()
        print(pos, "agent_state: position", agent_state.position, "rotation", agent_state.rotation)

        action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

        observations = sim.get_sensor_observations()

        # rgb image
        rgb_obs = observations["color_sensor"]
        rgb_img = Image.fromarray(rgb_obs[:,:,:3])
        rgb_img.save(os.path.join(save_img_path_rgb, '{:d}.jpg'.format(pos)))

        # depth image
        depth_obs = observations["depth_sensor"]
        depth_img = Image.fromarray((depth_obs / 10 * 65536).astype(np.uint16))
        depth_img.save(os.path.join(save_img_path_depth, '{:d}.png'.format(pos)))

        semantic_obs = observations["semantic_sensor"]
        # semantic image
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img.save(os.path.join(save_img_path_semantic, '{:d}.jpg'.format(pos)))
        # semantic_cat_image
        sem_unique, sem_idx = np.unique(semantic_obs, return_inverse=True)
        new_cat = sem_unique.copy()
        for idx, uni in enumerate(sem_unique):
            new_cat[idx] = instance_id_to_name[uni]
        semantic_obs_cat = new_cat[sem_idx].reshape(semantic_obs.shape).astype('uint8')

        # wrapper for compact labeling
        for i in range(semantic_obs_cat.shape[0]):
            for j in range(semantic_obs_cat.shape[1]):
                if np.count_nonzero(semantic_obs_cat == semantic_obs_cat[i][j]) < threshold:
                    semantic_obs_cat[i][j] = 0
                if semantic_obs_cat[i][j] in MP_TO_COMP:
                    semantic_obs_cat[i][j] = MP_TO_COMP[semantic_obs_cat[i][j]]

        semantic_img_cat = Image.fromarray(semantic_obs_cat)
        semantic_img_cat.save(os.path.join(save_img_path_semantic_cat, '{:d}.png'.format(pos)))

    sim.close()
