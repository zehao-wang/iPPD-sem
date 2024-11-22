import networkx as nx
import json
import numpy as np
from collections import defaultdict
import os
import gzip

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    
    paths = {}
    for scan,G in graphs.items(): # compute all shortest paths
        paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    distances = {}
    for scan,G in graphs.items(): # compute all shortest paths
        distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    return graphs, paths, distances


def load_meta(splits, data_root):
    ep_num = 0
    data_dict = defaultdict(list)
    for split in splits:
        if split in ['train', 'val_seen', 'val_unseen']:
            with gzip.open(os.path.join(data_root, split,f"{split}.json.gz"), 'r') as f:
                meta = json.load(f)

            with gzip.open(os.path.join(data_root, split, f"{split}_gt.json.gz"), 'r') as f:
                gt_actions_data = json.load(f) 
        else:
            raise

        for v in meta['episodes']:
            scene_id = v['scene_id'].split('/')[-1].split('.')[0]
            ep_num += 1
            datum = {
                'trajectory_id': v['trajectory_id'],
                'ep_id': v['episode_id'],
                'scene_id': scene_id,
                'start_position': v['start_position'],
                'goal_position': v['goals'][0]['position'],
                'end_position': v['goals'][0]["position"],
                'start_rotation': v['start_rotation'],
                'reference_path': v['reference_path'],
                'goal_radius': 3.0, 
                'goals': v['goals'],
                'distance': v['info']['geodesic_distance'],
                'instruction': v['instruction'],
            }
           
            if split in ['train', 'val_seen', 'val_unseen']:
                datum.update({
                    'gt_actions': gt_actions_data[str(v['episode_id'])]['actions'],
                    'gt_locations': gt_actions_data[str(v['episode_id'])]['locations'],
                })

            data_dict[scene_id].append(datum)

    return data_dict, ep_num
