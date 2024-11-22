import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg
import numpy as np
import os
from typing import Dict, List
import quaternion
from habitat_sim.utils.common import quat_from_angle_axis
import open3d as o3d
import pickle
from multiprocessing import Pool
from utils.get_predictor import Wrapped_Predictor
from utils.map_tools import update_map, merge_sem_map, dump_3dmap
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
import time
from utils.constants import stuff_colors
import warnings
warnings.filterwarnings("ignore")

num_process = 2
ISLAND_RADIUS_LIMIT = 1.5
RECORD_FREQ  = 5
MAX_SAMPLE_ROUND = 100
DRAW_VOXEL=True

parser = argparse.ArgumentParser()
# optional arguments
parser.add_argument(
    "--scene_root",
    default="../data/scene_datasets",
    type=str,
    help='scenedataset root',
)
parser.add_argument(
    "--dataset",
    default="./ui_data/mp3d.scene_dataset_config.json",
    type=str,
    metavar="DATASET",
    help='dataset configuration file to use',
)
parser.add_argument(
    "--output_dir",
    default="../out",
    type=str,
    help='output dir',
)
parser.add_argument(
    "--l",
    default=0,
    type=int,
)
parser.add_argument(
    "--h",
    default=90,
    type=int,
)
parser.add_argument(
    "--vis",
    default=False,
    action='store_true'
)
args = parser.parse_args()

class Viewer:
    def __init__(self, sim_settings):
        # configure our sim_settings but then set the agent to our default
        self.sim_settings = sim_settings
        self.cfg = make_cfg(self.sim_settings)
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        self.sim = habitat_sim.Simulator(self.cfg)
        self.default_agent = self.sim.get_agent(self.agent_id)
        
        self.step = -1

        self.total_frames = 0
        self.scene_id = sim_settings["scene"].split('/')[-2]

        self.cached_points = [] # record trajectories

        # init segmentation network
        self.predictor = Wrapped_Predictor()

        # init semantic map
        self.sem_map = defaultdict(Counter)
        self.resolution = 0.1

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """
        Set up our own agent and agent controls
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        # MOVE, LOOK = 0.07, 1.5
        MOVE, LOOK = 0.5, 30

        # all of our possible actions' names
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "move_forward",
        ]

        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

        # build our action space map
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[
            self.agent_id
        ].sensor_specifications

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.2,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config

    def map_by_dict(self, arr, mapping_dict):
        # NOTE: check missing meta
        missing_key = set(np.unique(arr))-mapping_dict.keys()
        for k in missing_key:
            mapping_dict[k] = -100

        return np.vectorize(mapping_dict.get)(arr)

    def get_rand_action_seq(self, rand_pos=False):
        if rand_pos:
            new_agent_state = habitat_sim.AgentState()
            new_agent_state.position = (
                self.sim.pathfinder.get_random_navigable_point()
            )
            new_agent_state.rotation = quat_from_angle_axis(
                self.sim.random.uniform_float(0, 2.0 * np.pi),
                np.array([0, 1, 0]),
            )
            self.default_agent.set_state(new_agent_state)

        start_pos = self.sim.get_agent(0).get_state().position
        point_sample = self.sim.pathfinder.get_random_navigable_point()
        if np.all(start_pos == point_sample):
            return None
        if self.sim.pathfinder.island_radius(point_sample) < ISLAND_RADIUS_LIMIT:
            return None
        try:
            greedy_follower = self.sim.make_greedy_follower(goal_radius=0.5)
            action_seq = greedy_follower.find_path(point_sample)
        except:
            # sometimes meet unexpected error
            return None
        return action_seq

    def execute(self) -> None:
        for i in tqdm(range(MAX_SAMPLE_ROUND)):
            tmp_pos = []
            acts = self.get_rand_action_seq(rand_pos=True)
            if acts is None:
                continue
            if acts[-1] is None:
                acts.pop()
            
            if len(acts)>500: # some error case
                continue

            for x in acts:
                observations = self.sim.step(x)
                if self.total_frames % RECORD_FREQ ==0:
                    img = observations['color_sensor']
                    depth = observations['depth_sensor']
                    sem = self.predictor.predict(img)
                    
                    state = self.sim.get_agent(0).get_state()
                    sensor_state = state.sensor_states['color_sensor']
                    sim_pos = sensor_state.position
                    sim_rot = sensor_state.rotation
                    
                    update_map(depth, sem, sim_pos, sim_rot, self.sem_map, resolution=self.resolution)

                    tmp_pos.append((state.position, quaternion.as_float_array(state.rotation)))
                self.total_frames += 1

            self.cached_points.append(tmp_pos)
        
        pts, labels = merge_sem_map(self.sem_map)
        os.makedirs(os.path.join(args.output_dir, f'out_obs/{self.scene_id}'), exist_ok=True)
        
        dump_3dmap(pts, labels, dump_dir=os.path.join(args.output_dir, f'out_obs/{self.scene_id}/sem_map.pkl'), resolution=self.resolution)
        
        with open(os.path.join(args.output_dir,f'out_obs/{self.scene_id}/sim_trajs.pkl'), 'wb') as f:
            pickle.dump(self.cached_points, f, protocol=pickle.HIGHEST_PROTOCOL)
        return pts, labels

def run(scene_path):
    tic = time.time()
    # Setting up sim_settings
    scene_id = scene_path.split('/')[-2]
    print(f"Running scene {scene_id}")
    sim_settings = default_sim_settings
    sim_settings["scene"] = scene_path
    sim_settings["scene_dataset_config_file"] = args.dataset
    
    sim_settings['color_sensor'] = True
    sim_settings['depth_sensor'] = True
    sim_settings["enable_physics"] = False

    sim_settings["seed"] = 199

    # start the application
    runner = Viewer(sim_settings)
    pts, labels = runner.execute()
    print(f"Dump trajectory of scene {scene_id} info successfully!")
    
    if args.vis:
        # NOTE: verifying procedure [draw sampled points]
        nav_maps = set()
        resolution = 0.1
        for i in range(10000):
            pt = runner.sim.pathfinder.get_random_navigable_point()
            if runner.sim.pathfinder.island_radius(pt) < ISLAND_RADIUS_LIMIT:
                # SKIP island point
                continue
            pt = tuple((pt/resolution).astype(int))
            nav_maps.add(pt)
        
        # NOTE: open3d and habitat have conflicts, seems to use some shared resource
        #       MUST CLOSE simulator first
        runner.sim.close() 

        nav_maps = list(nav_maps)
        nav_map = np.array(nav_maps).astype(int)
        pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(nav_map)
        pcd_o3d.paint_uniform_color([0.5, 0.5, 0.5])
        for traj in runner.cached_points:
            for (pt, rot) in traj:
                try:
                    pt = tuple((np.array(pt)/resolution).astype(int))
                    index = nav_maps.index(pt)
                    pcd_o3d.colors[index] = [1.,0.,0.]
                except:
                    pass
        o3d.io.write_point_cloud(os.path.join(args.output_dir, f'out_obs/vis/{scene_id}_landcover.ply'), pcd_o3d)
        print(f"Dump visualization of scene {scene_id} successfully!")

        # NOTE: visualization sem map
        vis_path = os.path.join(args.output_dir, f'out_obs/vis/{scene_id}_sem_vis.ply')
        pcd_nps = np.vstack(pts) # color become more red while z value increase
        pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_nps)
        color_list = stuff_colors[labels]
        if color_list[0] is not None:
            color_list = np.vstack(color_list)
            pcd_o3d.colors = o3d.utility.Vector3dVector(color_list/255.0)

        if DRAW_VOXEL:
            pcd_o3d_vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d,
                                                                voxel_size=1)
            o3d.visualization.draw_geometries([pcd_o3d_vg])
        else:
            o3d.visualization.draw_geometries([pcd_o3d])

        o3d.io.write_point_cloud(vis_path, pcd_o3d)
    print(f"Finish scene {scene_id} with {int(time.time()-tic)/60} min")

def gen(l=0, h=30):
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.makedirs(os.path.join(args.output_dir,'out_obs/vis'), exist_ok=True)
    scene_list = sorted(os.listdir(args.scene_root))[l:h]
    # pool = Pool(num_process)
    for scene_name in scene_list:
        scene_path = os.path.join(args.scene_root, scene_name, f"{scene_name}.glb")
        run(scene_path)
        # pool.apply_async(run, args=(scene_path, ))
    # pool.close()
    # pool.join()

if __name__ == "__main__":
    gen(args.l, args.h)

