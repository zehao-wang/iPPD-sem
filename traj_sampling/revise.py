import numpy as np
from vis_tools.map_tools import load_maps
import h5py
import pickle
import os
import gzip
import json
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--revise_dir", type=str)
parser.add_argument("--split", choices=["train", "val_seen", "val_unseen"], required=True)
args = parser.parse_args()

split = args.split
annt_root = args.revise_dir.format(split=split)
map_root = '../data/scene_datasets/preprocessed_navmap'
ori_annt =  '../data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'.format(split=split)

def fix_cam_pose(rel_scaled_pose, lower_bound, resolution, start_pt):
    rel_pose = (np.array(rel_scaled_pose) - np.array(lower_bound))/resolution
    corrected_pose = rel_pose + np.array(start_pt)

    return corrected_pose
# ========== load maps ============
sem_maps = dict()
for map_file in os.listdir(map_root):
    scene_id = map_file.split('.')[0]
    map_resolution, lower_bound, nav_map = load_maps(f"{map_root}/{map_file}")
    sem_maps[scene_id] = {
        "lower_bound": lower_bound,
        "resolution": map_resolution
    }

# ========= original annotations ===========
epid2startpt = dict()
with gzip.open(ori_annt, 'rt') as f:
    meta = json.load(f)['episodes']
    for ep in meta:
        epid2startpt[ep['episode_id']] = ep['start_position']

h5_path = annt_root + '.h5'
with open(annt_root + '.pkl', 'rb') as f:
    data_keys = pickle.load(f)

data = h5py.File(h5_path, 'r+')
for traj_id in tqdm(data_keys):
    h5_datum = data[traj_id]
    scene_id = h5_datum.attrs['scene']
    sem_map_info = sem_maps[scene_id]

    lower_bound, resolution = sem_map_info['lower_bound'], sem_map_info['resolution']
    start_pt = epid2startpt[int(traj_id.split('-')[0])]
    error_poses = h5_datum['cam_pos'][()]
    fixed_poses = []
    for error_pose in error_poses:
        fixed_poses.append(fix_cam_pose(error_pose, lower_bound, resolution, start_pt))
    
    if "cam_pos_fixed" in h5_datum.keys():
        del h5_datum['cam_pos_fixed']
    h5_datum.create_dataset("cam_pos_fixed", data=fixed_poses)

data.close()