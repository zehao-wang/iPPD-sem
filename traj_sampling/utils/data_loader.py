import os
import gzip
import pickle
import json
import numpy as np
import open3d as o3d 
import h5py
from nltk.wsd import lesk 
from .constants import cat2synset, word_mapping, word_mapping_built_sem, semantic_sensor_40cat, built_sem_labels, cat2synset_built_sem
from .editDistance import levenshteinDistance
import jsonlines
import spacy
nlp = spacy.load("en_core_web_sm")
def map_by_dict(arr, mapping_dict):
    # NOTE: check missing meta
    # missing_key = set(np.unique(arr))-mapping_dict.keys()
    # for k in missing_key:
    #     mapping_dict[k] = -100

    return np.vectorize(mapping_dict.get)(arr)

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

class MLNDset(object):
    def __init__(self, annt_root, map_root, original_annt_path, split='train', only_instruction=False, radius=None):
        self.use_gt_obs = False
        self.only_instruction = only_instruction
        annt_root = annt_root.format(split=split)
        ori_annt = original_annt_path.format(split=split)

        self.epid2tokens = dict()
        print(f"Load instruction annotations form {ori_annt}\n")
        scene_set = set()
        with gzip.open(ori_annt, 'rt') as f:
            meta = json.load(f)['episodes']
            for ep in meta:
                scene_id = ep['scene_id'].split('/')[-1].split('.')[0]
                scene_set.add(scene_id)
                self.epid2tokens[ep['episode_id']] = ep['instruction']

        self.start_idx = 0
        if self.use_gt_obs:
            self.word_mapping = word_mapping
            self.cat2synset = cat2synset
            self.label_dict = semantic_sensor_40cat
        else:
            self.word_mapping = word_mapping_built_sem
            self.cat2synset = cat2synset_built_sem
            self.label_dict = built_sem_labels
        
        if self.only_instruction:
            self.data = [{"ep_id":k, "instruction": v['instruction_text']} for k,v in self.epid2tokens.items()]
            return
    
        # Load path annotations
        self.data = []
        self.h5_path = annt_root + '.h5'
        with open(annt_root + '.pkl', 'rb') as f:
            self.data_keys = pickle.load(f)
        
        print(f"Successfully load {len(self.data_keys)} data for split {split} from {self.h5_path}\n")
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
                map_dir = os.path.join(map_root, scene_id, f"sem_map.pkl")
                if os.environ.get('DEBUG', False):
                    if not os.path.exists(map_dir): continue

                sem_pcd, search_tree, sem_labels, resolution = load_map(map_dir)
                self.sem_maps[scene_id] = {
                    "sem_pcd": sem_pcd, "search_tree": search_tree, 
                    "sem_labels": sem_labels, "resolution": resolution
                }
        
        self.data = h5py.File(self.h5_path, 'r')
        self.compass_radius = radius

    def __len__(self):
        if self.only_instruction:
            return len(self.data)
        else:
            return len(self.data_keys)

    def get_path_obs(self, sem_map_info, cam_pos, cam_dir):
        resolution = sem_map_info['resolution']
        poses_ori = (np.array(cam_pos)/resolution).astype(int)
     
        label_list = []
        for i, (pos, rot) in enumerate(zip(poses_ori, cam_dir)):
            indices = get_local_obs(pos, sem_map_info['search_tree'], self.compass_radius/resolution)
                
            sem_labels = sem_map_info['sem_labels'][indices]
            sem_labels = np.unique(sem_labels)
            if len(sem_labels) == 0:
                continue
            sem_labels = map_by_dict(sem_labels, self.label_dict).tolist()
            
            label_list.append(sem_labels)

        return label_list
    
    def extract_obj_words(self, instruction):
        res  = {
            'instruction': instruction,
            'obj_list': []
        }
        doc = nlp(instruction.lower())
        ns = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
        ns_raw = [token.text for token in doc if token.pos_ == 'NOUN']
        
        for n, n_txt in zip(ns, ns_raw):
            if n in ['left', 'right', 'straight', 'turn']:
                continue
            n_raw = '%s' % n
            if n in self.word_mapping:
                n = self.word_mapping[n]
            
            if n not in self.cat2synset:
                sim_list = []
                obj_synset = lesk(instruction, n, 'n')
                if obj_synset is None:
                    continue
                    
                for k,v in self.cat2synset.items():
                    sim_score = obj_synset.wup_similarity(v)
                    # sim_list.append((sim_score, n_raw, k))
                    sim_list.append((sim_score, n_txt, k))
                selected_obj = sorted(sim_list)[-1]
                
                if selected_obj[0] > 0.85:
                    res['obj_list'].append((selected_obj[1], selected_obj[2]))
                else:
                    continue
            else:
                res['obj_list'].append((n_raw, n))
        return res

    def get_next(self, dump_dir=None, exemption_set={"wall"}, instruction_obj_dict=None):
        if self.only_instruction:
            datum = self.data[self.start_idx]
            self.start_idx += 1
            instruction = datum['instruction']
            label_list=[]
            traj_id = None
        else:
            traj_id = self.data_keys[self.start_idx]
            self.start_idx += 1
            h5_datum = self.data[traj_id]
            datum = {}
            for k,v in h5_datum.attrs.items():
                datum[k] = v
            for k,v in h5_datum.items():
                datum[k] = v[()]
            
            instruction_raw = self.epid2tokens[int(traj_id.split('-')[0])]
            sem_map = self.sem_maps[datum['scene']]

            label_list = self.get_path_obs(sem_map, datum['cam_pos_fixed'], datum['cam_dir'])

            # TODO: process language
            instruction = instruction_raw['instruction_text']

        if instruction_obj_dict is None:
            res = self.extract_obj_words(instruction)
            ins_objs = []
            for i, p in enumerate(res['obj_list']):
                if p[1] in exemption_set:
                    continue

                if i>0 and len(ins_objs) > 0:
                    if ins_objs[-1] == p[1]:
                        continue
                ins_objs.append(p[1])
        else:
            ins_objs = instruction_obj_dict[int(datum['ep_id'])]['object_list']

        if dump_dir is not None:
            with jsonlines.open(dump_dir, 'a') as f:
                f.write({
                    "ep_id": datum['ep_id'], "instruction": datum['instruction'], 
                    "object_list": ins_objs
                    }
                )

        return ins_objs, label_list, traj_id
    
    def check_matching(self, ins_objs, label_list):
        if len(ins_objs) ==0 or len(label_list) == 0:
            return -1
        dist, dist_matrix = levenshteinDistance(ins_objs, label_list)
        if os.environ.get('DEBUG', False):
            dd = np.zeros((len(ins_objs)+1, len(label_list)+1))
            for k,v in dist_matrix.items():
                dd[k[0], k[1]] = v
            print(dd)
        
        num_hits = np.max(list(dist_matrix.values())) - dist
        hit_ratio = num_hits / len(ins_objs) 
        assert 0<= hit_ratio <=1, f"{hit_ratio}, {len(ins_objs)}, {num_hits}"
        return hit_ratio