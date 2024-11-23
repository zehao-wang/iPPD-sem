import torch
from torch.utils.data import Dataset
import jsonlines
from utils.file_process_tools import find_all_ext
import open3d as o3d
import numpy as np
from torch.utils.data._utils.collate import default_collate
from utils.parse_wordmap import load_embeddings
from utils.pos_emb import get_embedder
import h5py
import gzip
import json
from torch.nn.utils.rnn import pad_sequence 
import pickle
import random
import os
from utils.layeridx2wordidx import gt_obj_id2word_idx, seg_obj_id2word_idx
import math

def load_map(map_path): # skip floor sem
    with open(map_path, 'rb') as handle:
        sem_map = pickle.load(handle) # map, labels, resolution
    
    labels = np.asarray(sem_map['labels'])
    msk = (labels != 11) & (labels != 0) # skip floor sem, and none sem
    labels = labels[msk]
    map = np.asarray(sem_map['map'])[msk]

    resolution = sem_map['resolution']
    pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(map)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_o3d)
    return pcd_o3d, pcd_tree, labels, resolution

def load_gtmap(map_path, exemption_list=[1, 15, 16, 38, 39]): # skip floor sem
    resolution = 0.1
    sem_map = json.load(open(map_path))
    labels = np.array([obj['obj_index']-1 for obj in sem_map])
    
    msk = (~np.isin(labels, exemption_list)) & (labels >= 0)
    labels = labels[msk]
    map = (np.array([obj['obj_bbox']['center'] for obj in sem_map])[msk] / resolution).astype(int)

    pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(map)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_o3d)
    return pcd_o3d, pcd_tree, labels, resolution

def get_local_obs(query_pt, search_tree, distance):
    " get nearby points within radius distance (distance scale in map scale)"
    [k, idx, _] = search_tree.search_radius_vector_3d(query_pt, distance)
    return np.asarray(idx[1:])

def unit_vector(vector):
    axis=None
    if len(vector.shape) == 2:
        axis = 1
    return vector / (np.linalg.norm(vector, axis=axis, keepdims=True) + 1e-7)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip( np.sum(v1_u * v2_u, axis=1), -1.0, 1.0)).astype(int)
    
def get_dis_dir(rot1, rot2, chunck_size=30):
    dis_dir = angle_between(rot1, rot2) / np.pi * 360
    dis_dir = (dis_dir // chunck_size)
    dis_dir[dis_dir==0] = int(360/chunck_size)
    # middle_pt = 360 / (2*chunck_size)
    # dis_dir = dis_dir - middle_pt
    assert np.all(dis_dir > 0) and np.all(dis_dir <= int(360/chunck_size)), f"[ERROR] dis_dir=={dis_dir}"
    return dis_dir

def scaling(bound, target_range, value):
    """
    Scaling a value from a range (bound) to a target_range
    """
    value = np.clip(value, bound[0], bound[1])
    v_std = (value-bound[0]) / (bound[1]-bound[0])
    return v_std * (target_range[1] - target_range[0]) + target_range[0]

def rotate(origin, points, angle):
    """
    Rotate a point counter-clockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin

    px = ox + np.cos(angle) * (points[:, 0] - ox) - np.sin(angle) * (points[:, 2] - oy)
    py = oy + np.sin(angle) * (points[:, 0] - ox) + np.cos(angle) * (points[:, 2] - oy)
    points[:, 0] = px
    points[:, 2] = py
    return points

def change_agent_rot(agent_dir):
    rot = np.arctan2(agent_dir[:, 0], agent_dir[:, 2])
    direction_r = 2 * np.cos(rot) # [-2,2]
    direction_c = 2 * np.sin(rot) # [-2,2]
    agent_dir[:, 0] = direction_r
    agent_dir[:, 1] = direction_c
    agent_dir[:, 2] = 0
    return agent_dir

class MLNv3_Dataset(Dataset):
    def __init__(self, config, split):
        # setup config
        self.target_type = config.target_type
        self.obj_3dpos = config.obj_3dpos
        self.compass_radius = config.compass_radius
        self.obs_sample_size = config.obs_sample_size
        self.discrete_dir = config.discrete_dir
        self.agent_rel_pos = config.agent_rel_pos
        self.use_gt_obs = config.gt_obj

        annt_root = config.annt_root.format(split=split)
        ori_annt = config.original_annt_path.format(split=split)
        # Load path annotations
        self.data = []
        self.h5_path = annt_root + '.h5'

        with open(annt_root + '.pkl', 'rb') as f:
            self.data_keys = pickle.load(f)

        print(f"Successfully load {len(self.data_keys)} data for split {split} from {self.h5_path}\n")
        
        self.epid2tokens = dict()
        print(f"Load instruction annotations form {ori_annt}\n")
        scene_set = set()
        with gzip.open(ori_annt, 'rt') as f:
            meta = json.load(f)['episodes']
            for ep in meta:
                scene_id = ep['scene_id'].split('/')[-1].split('.')[0]
                scene_set.add(scene_id)
                self.epid2tokens[ep['episode_id']] = ep['instruction']['instruction_tokens']

        # embeddings
        # self.embedding_layer = load_embeddings()
        self.agent_loc_emb, _ = get_embedder(10)

        # Load map
        self.sem_maps = {}
        if self.use_gt_obs:
            gt_root = './dataset/gt_obj_info'
            for file in os.listdir(gt_root):
                scene_id = file.split('.')[0]
                sem_pcd, search_tree, sem_labels, resolution = load_gtmap(os.path.join(gt_root, file))
                self.sem_maps[scene_id] = {
                    "sem_pcd": sem_pcd, "search_tree": search_tree, 
                    "sem_labels": sem_labels, "resolution": resolution
                }
        else:
            for scene_id in scene_set:
                map_dir = os.path.join(config.map_root, scene_id, f"sem_map.pkl")
                if os.environ.get('DEBUG', False):
                    if not os.path.exists(map_dir): continue

                sem_pcd, search_tree, sem_labels, resolution = load_map(map_dir)
                self.sem_maps[scene_id] = {
                    "sem_pcd": sem_pcd, "search_tree": search_tree, 
                    "sem_labels": sem_labels, "resolution": resolution
                }

    def __len__(self):
        return len(self.data_keys)

    def open_hdf5(self):
        self.data = h5py.File(self.h5_path, 'r')

    def extract_path_obs(self, sem_map_info, cam_pos, cam_dir, actions, step_size, agent_rot=None):
        resolution = sem_map_info['resolution']
        if agent_rot is not None:
            index = [i for i in range(len(cam_pos))]
        else:
            index = [i for i in range(0, len(cam_pos), step_size)] + [len(cam_pos)-1]
        poses_ori = (np.array(cam_pos[index])/resolution).astype(int)
        rots  = np.array(cam_dir[index]) 

        # calculate relative pos
        if self.agent_rel_pos:
            poses = poses_ori[1:] - poses_ori[:-1]
            poses = np.vstack([np.array([[0,0,0]]), poses])
            poses = scaling((-5/resolution, 5/resolution), (-2,2), poses)
        else:
            poses = (poses_ori - poses_ori[0])*resolution    
            # poses = scaling((-20/resolution, 20/resolution), (-2,2), poses)
    
        if self.discrete_dir: 
            dis_rots = get_dis_dir(rots[:-1, [0,2]], rots[1:, [0,2]])
            dis_rots = np.hstack([np.array([0]), dis_rots])
            rots = dis_rots
        else:
            # rots = change_agent_rot(rots) # original version
            rots = np.array(rots)

        assert len(rots) == len(poses)
        
        obs_pos_embs = []
        obs_rots = []
        labels = []
        for i, (pos, rot) in enumerate(zip(poses_ori, cam_dir[index])):
            indices = get_local_obs(pos, sem_map_info['search_tree'], self.compass_radius/resolution)
            if len(indices) == 0:
                rel_coords = np.zeros((self.obs_sample_size, 3), dtype=float)
                sem_labels = np.ones((self.obs_sample_size), dtype=int) * 597 # use word 'current' if no objects
            else:
                sampled_indices = np.random.choice(len(indices), self.obs_sample_size, replace=True)
                selected = indices[sampled_indices]
                
                sem_labels = sem_map_info['sem_labels'][selected]
                if self.use_gt_obs:
                    sem_labels = gt_obj_id2word_idx[sem_labels]
                else:
                    sem_labels = seg_obj_id2word_idx[sem_labels]

                rel_coords = np.asarray(sem_map_info['sem_pcd'].points, dtype=float)[selected] - pos

                # NOTE: egocentric view
                if agent_rot is not None:
                    agent_orientation_angle = agent_rot[index[i]]
                else:
                    agent_orientation_angle = math.atan2(rot[0], rot[2])
                # TODO from visualization is check_compass.py should apply +agent_orientation_angle
                #      if meet any strange problem, probably use -agent_orientation_angle
                rel_coords = rotate((0, 0), rel_coords, agent_orientation_angle)

                # Discrete direction
                # if self.discrete_dir: 
                #     obs_dis_rots = get_dis_dir(np.array([[rot[0], rot[2]]]), rel_coords[:, [0,2]])
                #     obs_rots.append(obs_dis_rots)
                # else:
                #     obs_rots.append(unit_vector(unit_vector(rel_coords)-rot))
                
            rel_coords = scaling((-self.compass_radius/resolution, self.compass_radius/resolution), (-2,2), rel_coords)
            rel_coords = torch.tensor(rel_coords).float()
            rel_coords_emb = self.agent_loc_emb(rel_coords)
            
            obs_pos_embs.append(rel_coords_emb)
            labels.append(sem_labels)

        poses_emb = self.agent_loc_emb(torch.tensor(poses).float())
        if self.discrete_dir:
            rots = torch.tensor(np.array(rots)).long()
            # obs_rots = torch.tensor(np.array(obs_rots)).long()
        else:
            rots = torch.tensor(np.array(rots)).float()
            # obs_rots = torch.tensor(np.array(obs_rots)).float() 
        
        obs_pos_embs = torch.stack(obs_pos_embs)
        return obs_pos_embs, obs_rots, torch.tensor(np.vstack(labels)), poses_emb, rots

    def __getitem__(self, idx):
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()

        traj_id = self.data_keys[idx]
        h5_datum = self.data[traj_id]
        datum = {}
        for k,v in h5_datum.attrs.items():
            datum[k] = v
        for k,v in h5_datum.items():
            datum[k] = v[()]

        actions = datum['actions']
        instruction_raw = self.epid2tokens[int(traj_id.split('-')[0])]
        instruction = dict()
        inupt_ids = instruction_raw[:instruction_raw.index(0)]
        token_type = [1] * len(inupt_ids)
        attn_mask  = [1] * len(inupt_ids)
        instruction = {"input_ids": inupt_ids, "token_type": token_type, "attn_mask": attn_mask}

        sem_map = self.sem_maps[datum['scene']]
        if 'agent_rot' in datum:
           obs_pos_embs, obs_rots, sem_labels, poses_emb, rots = \
                self.extract_path_obs(sem_map, datum['cam_pos_fixed'], datum['cam_dir'], 
                    actions, step_size=5, agent_rot = datum['agent_rot']
                )
        else:
            obs_pos_embs, obs_rots, sem_labels, poses_emb, rots = \
                self.extract_path_obs(sem_map, datum['cam_pos'], datum['cam_dir'], 
                    actions, step_size=5
                )
        # ==> obs_pose_embs [num_obs, num_samples, feat_size]
        # ==> sem_labels [num_obs, num_samples]
        # ==> agent_pos_emb [num_obs, pos_emb_size]
 
        target_ndtw =  datum['ndtw']
        target_score = datum['target']/10.0 # distance to goal score, high the better
        
        info={
            "scene": datum['scene'],
            "traj_id": traj_id,
            "ndtw": float(datum['ndtw']),
            "dis_score": int(datum["target"]),
        }

        return instruction, (poses_emb, rots), (obs_pos_embs, obs_rots, sem_labels), target_ndtw, target_score, info

    def collate_fc(self, batch):
        # use word 'separate' as start token: 1908
        input_ids = pad_sequence([  torch.tensor([1908] + b[0]['input_ids']).long() for b in batch], batch_first=True)
        token_type = pad_sequence([ torch.tensor([1] + b[0]['token_type']).long() for b in batch], batch_first=True)
        attn_mask = pad_sequence([  torch.tensor([1] + b[0]['attn_mask']).long() for b in batch], batch_first=True)

        instructions = {"input_ids":input_ids, "token_type_ids": token_type, "attention_mask": attn_mask }
        agent_pos_embs = [b[1][0] for b in batch]
        agent_rots = [b[1][1] for b in batch]

        seq_len = []
        for b in batch:
            seq_len.append(len(b[1][0]))

        # obs_pos_embs = [b[2][0] for b in batch]
        # obs_rots = [b[2][1] for b in batch]
        obs_rots = []
        # sem_labels = [b[2][2] for b in batch]

        obs_pos_embs = torch.cat([b[2][0] for b in batch], dim=0)
        sem_labels = torch.cat([b[2][2] for b in batch], dim=0)
        # sem_labels = self.embedding_layer(sem_labels)
        
        target_ndtw = torch.tensor([b[3] for b in batch]).float()
        target_score = torch.tensor([b[4] for b in batch]).float()
        
        infos = [ b[5] for b in batch]
        return instructions, (agent_pos_embs, agent_rots), (obs_pos_embs, obs_rots, sem_labels), target_ndtw, target_score, infos, seq_len

