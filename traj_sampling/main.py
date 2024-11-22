from vis_tools.map_visualizer import visualizer
from vis_tools.map_tools import load_maps, get_agent_orientation, euclidean_distance, simloc2maploc
import gzip
import json
from collections import defaultdict
import numpy as np
import quaternion
from vis_tools.navigator import RobotState, Point
from copy import copy
import re
from collections import deque
import os
from tqdm import tqdm
import random
import jsonlines
import open3d as o3d
from multiprocessing import Queue


# traj start end distance range
MIN_DIST=2 # in meter
MAX_DIST=15 # in meter
FILTER_RATIO = 0.5 # density filter ratio
MIN_PROPOSALS=10

def load_annts(annt_file_path, annt_gt_actions_path):
    ep_num = 0
    with gzip.open(annt_file_path, 'r') as f:
        eps_data = json.load(f)["episodes"]
    with gzip.open(annt_gt_actions_path, 'r') as f:
        gt_actions_data = json.load(f)

    data_dict = defaultdict(list)
    for v in eps_data:
        scene_id = v['scene_id'].split('/')[-1].split('.')[0]
        ep_num += 1
        data_dict[scene_id].append( {
            'trajectory_id': v['trajectory_id'],
            'ep_id': v['episode_id'],
            'scene_id': scene_id,
            'start_position': v['start_position'],
            'start_rotation': v['start_rotation'],
            'reference_path': v['reference_path'],
            'goals': v['goals'],
            'distance': v['info']['geodesic_distance'],
            'instruction': v['instruction'],
            'gt_actions': gt_actions_data[str(v['episode_id'])]['actions'],
            'gt_locations': gt_actions_data[str(v['episode_id'])]['locations'],
        }
        )
    return data_dict, ep_num

# Rule based
def extract_act_seq_rule(instruction):
    key_words = ['right', 'left', 'turn around', "up", "turn", 'forward', "pass", 'straight', "go", "walk", "exit", "out", "leave", "enter"]

    action_seq = []
    for word in key_words:
        idx_list = [m.start() for m in re.finditer(word, instruction.lower())]
        if len(idx_list) > 0:
            if word == 'left':
                action_seq += [(idx, 'left') for idx in idx_list]
            elif word == 'right':
                action_seq += [(idx, 'right') for idx in idx_list]
            elif word == 'turn around':
                action_seq += [(idx, 'around') for idx in idx_list]
            # else:
            #     action_seq += [(idx, 'forward') for idx in idx_list]
    
    if len(action_seq) == 0:
        return []

    action_seq = sorted(action_seq, key=lambda x: x[0])
    action_seq = [pair[1] for pair in action_seq]

    # NOTE: find the init agent position sometimes not correct, we do not trust init agent dir,
    #       make every agent at the start point has a noisy orientation with a forward command
    if action_seq[0] not in ["forward", "around"]:
        action_seq = ["forward"] + action_seq
    
    return action_seq

# GPT3.5
from utils.llm_parser import KeyComponentExtractor
from utils.constants import cat2synset
llm = KeyComponentExtractor(obj_labels=list(cat2synset.keys()), action_labels=['turn right', 'turn left', 'turn around'])

def extract_act_seq_llm(instruction):
    msg_dict, response_full = llm.run(instruction)

    action_seq=[]
    for (k,v) in msg_dict:
        if k == 'action':
            if 'right' in v.lower():
                action_seq.append('right')
            elif 'left' in v.lower():
                action_seq.append('left')
            elif 'around' in v.lower():
                action_seq.append('around')

    if len(action_seq) == 0:
        action_seq = ["forward"]

    if action_seq[0] not in ["forward", "around"]:
        action_seq = ["forward"] + action_seq
    return action_seq


def filter_agents(agent_list, map_resolution):
    if len(agent_list) == 0:
        return []

    filter_lists = []
    for agent in agent_list:
        # NOTE start end distance constraints
        dist_in_meter = euclidean_distance(agent.traj[0], agent.traj[-1]) * map_resolution
        if os.environ.get('DEBUG', False):
            filter_lists.append(agent)
        else:
            if dist_in_meter > MIN_DIST and dist_in_meter < MAX_DIST:
                filter_lists.append(agent)
    
    # NOTE sparsify the proposals
    end_pts = []
    for agent in filter_lists:
        end_pts.append(agent.traj[-1])

    if len(end_pts) == 0:
        return []

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.array(end_pts))
    labels = np.array(pcd_o3d.cluster_dbscan(eps=1/map_resolution, min_points=1))
    label_max = np.amax(labels) + 1
    kept_indicies = []
    for i in range(label_max):
        indicies = np.where(labels == i)[0].tolist()
        tot_pts = len(indicies)
        if tot_pts > 1:
            sample_size = np.ceil(tot_pts * FILTER_RATIO).astype(int)
            if sample_size > 200:
                sample_size = 200
            kept_indicies += random.sample(indicies, sample_size)
        else:
            kept_indicies += indicies

    filter_lists = [filter_lists[idx] for idx in kept_indicies]
    print(len(agent_list), len(filter_lists))
    return filter_lists

def process_ep(start_pt, start_dir, map_resolution, act_seq, nav_map, 
               max_samples=10000):
    rob_init_stat = RobotState(
            point=Point(*start_pt), 
            angle = start_dir,
            map_resolution=map_resolution,
            left_turn_range = [np.pi / 6, np.pi * 5 / 6], # in radian
            right_turn_range= [np.pi / 6, np.pi * 5 / 6], # in radian
            forward_range= [0, 10], # in meter
        )
    
    queue = []
    for i in range(max_samples):
        rob_init = copy(rob_init_stat)
        rob_init.add_noise()
        queue.append(rob_init)
    queue = deque(queue)

    for act in act_seq:
        que_len = len(queue)
        for i in range(que_len):
            rob_stat = queue.popleft()
            if rob_stat.move(act, nav_map): 
                queue.append(rob_stat)
    
    agents = filter_agents(list(queue), map_resolution)
    return agents

def process_scene(scene_id, scene_to_eps, split, dump_root, instr_paser):
    eps = scene_to_eps[scene_id]
    # load maps
    map_path_format='../data/preprocessed_navmap/{scene_id}.h5'
    map_resolution, lower_bound, nav_map = load_maps(scene_id, map_path_format=map_path_format)

    # setup visualizer
    vis = visualizer(nav_map)

    scores = []
    for ep in tqdm(eps):
        start_pos = ep['start_position']
        rotation = np.array([ep['start_rotation'][3], *ep['start_rotation'][:3]]) 
        start_rot = quaternion.from_float_array(rotation)
        goal_pos = ep['goals'][0]['position']
        traj_sim = ep['reference_path']
        instruction = ep['instruction']['instruction_text']
        if instr_paser=='rule':
            act_seq = extract_act_seq_rule(instruction)
        else:
            act_seq = extract_act_seq_llm(instruction)

        if len(act_seq) == 0:
            continue
        
        start_map_pt = simloc2maploc(start_pos, lower_bound, map_resolution)
        start_dir, agent_orientation_vec = get_agent_orientation(start_rot, sim_rot=True)
        stat_objs = process_ep(start_map_pt, start_dir, map_resolution, act_seq, nav_map)
        if len(stat_objs) < MIN_PROPOSALS:
            continue

        # if os.environ.get('DEBUG', False):
        #     traj_map_pts = []
        #     for sim_pt in traj_sim:
        #         traj_map_pts.append(simloc2maploc(sim_pt, lower_bound, map_resolution))

        #     # draw gt trajectory        
        #     vis.draw_traj(traj_map_pts, f'out/ep{ep["ep_id"]}_gt.ply', agent_dir=agent_orientation_vec)

        #     vis.draw_trajs([x.traj for x in stat_objs], dump_path=f'out/ep{ep["ep_id"]}_test.ply')
        #     # for i, stat_obj in enumerate(stat_objs):
        #     #     traj = stat_obj.traj
        #     #     vis.draw_traj(traj, f'./out/{i}.ply')

        # TODO replace the sample by density filter
        # if len(stat_objs) > 20:
        #     stat_objs = random.sample(stat_objs, 20)

        ep_stats = []
        for x in stat_objs:
            traj = x.traj
            end_pt = traj[-1]
            dis = euclidean_distance(end_pt, simloc2maploc(goal_pos, lower_bound, map_resolution))
            # NOTE: set within 1meter as success find matching
            if dis < 1 / map_resolution:
                ep_stats.append(True)
            else:
                ep_stats.append(False)

        if sum(ep_stats) == 0:
            if os.environ.get('DEBUG', False):
                print(ep['ep_id'], instruction, act_seq)
                traj_map_pts = []
                for sim_pt in traj_sim:
                    traj_map_pts.append(simloc2maploc(sim_pt, lower_bound, map_resolution))

                # draw gt trajectory        
                vis.draw_traj(traj_map_pts, f'out/debug/ep{ep["ep_id"]}_gt.ply', agent_dir=agent_orientation_vec)
                vis.draw_trajs([x.traj for x in stat_objs[:5]], dump_path=f'out/debug/ep{ep["ep_id"]}_test.ply')

            scores.append(False)
        else:
            if os.environ.get('DEBUG', False):
                print(ep['ep_id'], instruction, act_seq)
                traj_map_pts = []
                for sim_pt in traj_sim:
                    traj_map_pts.append(simloc2maploc(sim_pt, lower_bound, map_resolution))

                # draw gt trajectory        
                vis.draw_traj(traj_map_pts, f'out/debug/True-ep{ep["ep_id"]}_gt.ply', agent_dir=agent_orientation_vec)
                vis.draw_trajs([[x.traj[0], x.traj[-1]] for x in stat_objs], dump_path=f'out/debug/True-ep{ep["ep_id"]}_test.ply')

            scores.append(True)

        # dump trajectories
        for i in range(len(stat_objs)):
            traj = stat_objs[i].traj

            out_dict = {
                'scene': scene_id,
                "ep_id": ep['ep_id'],
                "traj_id": f"{ep['ep_id']}-act-{i}",
                "annt_traj_idx": ep['trajectory_id'],
                'instruction': instruction,
                'cam_poses_raw': traj,  # map coordinate
                "split": split
            }
            with jsonlines.open(os.path.join(dump_root, split, f"{ep['ep_id']}.jsonl"), mode='a') as writer:
                writer.write(out_dict)

    return scores

if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool
    queue = Queue()
    num_process = 30
    pool = Pool(num_process)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./out", type=str)
    parser.add_argument("--split", choices=["train", "val_seen", "val_unseen"], required=True)
    parser.add_argument("--instr_paser", choices=['rule', 'gpt35'], default='gpt35')
    args = parser.parse_args()

    split=args.split
    annt_path = f'../data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'
    annt_gt_path = f'../data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz'
    dump_root = args.out_dir
    os.makedirs(os.path.join(dump_root, split), exist_ok=True)

    scene_to_eps, ep_num = load_annts(annt_path, annt_gt_path)
    processes = []

    for scene_id in scene_to_eps:
        if os.environ.get('SMALL', False):
            if scene_id not in ['8WUmhLawc2A', 'zsNo4HB9uLZ']:
                continue
        
        # NOTE: try non-parallel version for debugging
        # process_scene(scene_id, scene_to_eps, split, dump_root, args.instr_paser)
        p = pool.apply_async(process_scene, (scene_id, scene_to_eps, split, dump_root, args.instr_paser,))
        processes.append(p)

    pool.close()
    pool.join()

    scores = []
    for p in processes:
        score_list = p.get()
        print(len(score_list))
        scores += score_list

    print(f"Episode cover ratio {len(scores)}/{ep_num} = {len(scores)/ep_num}")
    print(f"Episode propose precision {sum(scores)} / {len(scores)} = {sum(scores)/len(scores)}")