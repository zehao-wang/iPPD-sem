from multiprocessing import Pool
import os
import numpy as np
from sim_utils import set_up_habitat, get_rand_action_seq_near, euclidean_distance, get_action_seq_by_ref_path
import quaternion
from fastdtw import fastdtw
import jsonlines
from tqdm import tqdm
from utils.io import load_meta
from vis_tools.map_tools import load_maps
num_process = 10
# MAX_SAMPLES = 10000
# MAX_HITS = 10

class dset_generator():
    """
        each datum contains {"id": , "scene": , "instruction": ,"cam_poses": (position, rot), "actions": ,"split": }
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.args = config
        self.split = config.split
        self.out_dir = config.out_root
        os.makedirs(self.out_dir.format(split=self.split), exist_ok=True)

        # init annotations of dataset
        if config.split == 'trainaug':
            splits = ['train_aug', 'train']
        else:
            splits = [config.split]
        data_dict, ep_num = load_meta(splits, config.annt_root)
        self.episodes = data_dict

    def gen(self):
        pool = Pool(num_process)
        scene_names = sorted(self.episodes.keys())
        print(len(scene_names))
        for scene_name in scene_names:
            if os.environ.get('SMALL', False):
                if scene_name not in ['8WUmhLawc2A', 'zsNo4HB9uLZ']:
                    continue
            self.process_scene(scene_name)
            # pool.apply_async(self.process_scene, args=(scene_name, ))
        pool.close()
        pool.join()

    def init_scene(self, scene_name):
        # Init simulator
        scene_path = os.path.join(self.args.scene_datasets, scene_name, f"{scene_name}.glb")
        # sim, _ = set_up_habitat(scene_path, no_vis=True)
        sim, _ = set_up_habitat(scene_path, no_vis=False) #NOTE we should get sensor position

        return sim, self.episodes[scene_name]

    def process_scene(self, scene_name):
        sim, episodes = self.init_scene(scene_name)
        # nav_map = get_navmaps(scene_name, self.args.navmap_root)

        map_path_format='../data/preprocessed_navmap/{scene_id}.h5'
        map_resolution, lower_bound, nav_map = load_maps(scene_name, map_path_format=map_path_format)

        num_samples = int(np.sum(nav_map)* self.args.sample_ratio)
        print(f'Sampled {num_samples} trajs')

        for ep in episodes:
            ep_id = ep['ep_id']
            counter = 0
            rotation = np.array([ep['start_rotation'][3], *ep['start_rotation'][:3]]) 
            start_rot = quaternion.from_float_array(rotation)
            start_pos = ep['start_position']
            goal_pos= ep['goal_position']
            gt_locations = ep['gt_locations']

            num_hits = 0
            for i in range(0, num_samples):
                sim.reset()
                sim.set_agent_state(start_pos, start_rot)
                if i==0:
                    if self.split in ['val_seen', 'val_unseen']:
                        continue
                    if 'gt_actions' not in ep:
                        action_seq = ep['gt_actions']
                    else:
                        action_seq = get_action_seq_by_ref_path(sim, ep['reference_path'], start_pos, start_rot)
                        if action_seq is None:
                            import ipdb;ipdb.set_trace() # breakpoint 81
                            print()
                            continue
                else:
                    action_seq = get_rand_action_seq_near(sim, start_pos)
                    if action_seq is None:
                        continue
                    action_seq[-1] = 0
                    if len(action_seq) > 200:
                        print('\033[1;31m [WANGING]\033[0m TOO long action sequence')
                        continue

                agent_states = []
                # record path
                action_seq = [-1] + action_seq # to delay the first action
                for j in range(len(action_seq)-1):
                    if action_seq[j] != -1: 
                        sim.step(action_seq[j])
          
                    state = sim.get_agent_state()
                    sensor_state = state.sensor_states['rgb']
                    agent_states.append({
                        'floor_loc': state.position.tolist(),
                        'sim_loc': sensor_state.position.tolist(),
                        'sim_rot': quaternion.as_float_array(sensor_state.rotation).tolist(),
                    })

                # Calculate DTW and NDTW
                dtw_distance = fastdtw(
                    [s['floor_loc'] for s in agent_states], gt_locations, dist=euclidean_distance
                )[0]
                nDTW = np.exp(
                    -dtw_distance
                    / (len(gt_locations) * ep['goal_radius'])
                )

                distance_to_goal = sim.geodesic_distance(agent_states[-1]['floor_loc'], goal_pos)
                
                if i == 0:
                    print(distance_to_goal)
                    # if os.environ.get('DEBUG', False): # NOTE: check gt path is correct
                        # import ipdb;ipdb.set_trace() # breakpoint 123    
                        
                if distance_to_goal < 3:
                    num_hits += 1

                if distance_to_goal >= 5: # scoring region radius 5m
                    score = 0
                else:
                    score = 10 - int(distance_to_goal / 0.5) # 10 level score 10 is the highest, 1 is the lowest

                out_dict = {
                    'scene': scene_name,
                    "ep_id": ep_id,
                    "traj_id": f"{ep_id}-{i}",
                    "annt_traj_idx": ep['trajectory_id'],
                    'instruction': ep['instruction'],
                    'target': score,
                    "geodist_to_goal": distance_to_goal,
                    "ndtw": nDTW,
                    "dtw": dtw_distance,
                    "cam_poses": agent_states,
                    "actions": action_seq[1:],
                    "split": self.split

                }

                counter +=1
                with jsonlines.open(os.path.join(self.out_dir.format(split=self.split), f'{scene_name}.jsonl'), mode='a') as writer:
                    writer.write(out_dict)
                
            if counter == 0:
                print("[WARNING] no trajectory sample for episode ", ep_id)
        sim.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_datasets", type=str)
    parser.add_argument("--split", choices=["train", "val_seen", "val_unseen", "trainaug"])
    parser.add_argument("--out_root", type=str)
    parser.add_argument('--annt_root', type=str)
    parser.add_argument('--sample_ratio', type=float, default=0.002)
    args = parser.parse_args()

    generator = dset_generator(args)
    generator.gen()