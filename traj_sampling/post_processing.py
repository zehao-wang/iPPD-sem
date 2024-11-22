import os
from habitat_sim import ShortestPath
import timeit
import argparse
import gzip
import json
from collections import defaultdict
import numpy as np
from astar.astar_planner_weighted import astar_planner_weighted
from vis_tools.map_tools import load_maps
from fastdtw import fastdtw
from multiprocessing import Pool
import jsonlines
import os.path as osp
import habitat_sim
from constants import roomidx2name, semantic_sensor_40cat
from compass import create_compass
name2roomidx = {v: k for k, v in roomidx2name.items()}
name2objidx = {v: k for k, v in semantic_sensor_40cat.items()}

GRAVITY = (1.5, 2)  # default (1.1, 4)
flag = False
PATH_STEP_SIZE = 20
SUB_TRAJ_DIST_THRESH = 10

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir", type=str)
parser.add_argument(
    "--input_dir", type=str)
parser.add_argument(
    "--split", choices=["train", "val_seen", "val_unseen"], required=True)
parser.add_argument("--num_process", type=int, required=True)
args = parser.parse_args()
split = args.split
num_process = args.num_process

original_annt_path = '../data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'
ori_annt = original_annt_path.format(split=split)
with gzip.open(ori_annt, 'rt') as f:
    meta = json.load(f)['episodes']

# group by scene
scene2eps = defaultdict(list)
for ep in meta:
    scene_id = ep['scene_id'].split('/')[-1].split('.')[0]
    ep_dict = {
        'ep_id': str(ep['episode_id']),
        'start_pt': ep['start_position'],
        'start_rot': ep['start_rotation'],
        'end_pt': ep['goals'][0]['position'],
        'ref_path': ep['reference_path'],
        'trajectory_id': ep['trajectory_id'],
        'instruction': ep['instruction']['instruction_text'],
    }
    scene2eps[scene_id].append(ep_dict)

def execution_time(method):
    """ decorator style """

    def time_measure(*args, **kwargs):
        ts = timeit.default_timer()
        result = method(*args, **kwargs)
        te = timeit.default_timer()

        print(
            f'Excution time of method {method.__qualname__} is {te - ts} seconds.')
        #print(f'Excution time of method {method.__name__} is {te - ts} seconds.')
        return result

    return time_measure


def to_map_pt(sim_pt, lower_bound, map_resolution):
    return tuple(np.round((np.array(sim_pt) - np.array(lower_bound))/map_resolution).astype(int).tolist())


objword_map = {'chest_of_drawers': 'drawer'}


def process_obj_dict(obj_info, lower_bound, map_resolution):
    objs = []
    obj_dict = defaultdict(list)
    for obj in obj_info:
        obj_mp_center = tuple(np.round((np.array(
            obj['obj_bbox']['center']) - np.array(lower_bound))/map_resolution).astype(int).tolist())
        if obj['obj_cat'] in objword_map:
            obj['obj_cat'] = objword_map[obj['obj_cat']]

        if obj['obj_cat'] not in name2objidx.keys():
            if obj['obj_cat'] == '' or obj['obj_cat'] == 'void':
                continue
            print(obj['obj_cat'])
            continue
        obj_dict[obj['obj_cat']].append(obj_mp_center)
        objs.append([name2objidx[obj['obj_cat']]]+list(obj_mp_center))
    return obj_dict, np.array(objs)


roomword_map = {
    'entryway/foyer/lobby': 'lobby', 'familyroom/lounge': 'familyroom', 'laundryroom/mudroom': 'laundryroom', 'porch/terrace/deck': 'terrace',
    'meetingroom/conferenceroom': 'meetingroom', 'other room': 'room', 'utilityroom/toolroom': 'toolroom', 'workout/gym/exercise': 'gym',
}


def process_room_dict(room_info, lower_bound, map_resolution):
    obj_dict = defaultdict(list)
    rooms = []
    for obj in room_info:
        obj_mp_center = tuple(np.round((np.array(
            obj['room_bbox']['center']) - np.array(lower_bound))/map_resolution).astype(int).tolist())
        if obj['room_cat'] in roomword_map:
            obj['room_cat'] = roomword_map[obj['room_cat']]

        if obj['room_cat'] not in name2roomidx.keys():
            print(obj['room_cat'])
            continue

        obj_dict[obj['room_cat']].append(obj_mp_center)
        rooms.append([name2roomidx[obj['room_cat']]]+list(obj_mp_center))
    return obj_dict, np.array(rooms)


def euclidean_distance(
    pos_a, pos_b
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


def ndtw(path1, path2, map_resolution, success_dis=3):
    dtw_distance = fastdtw(
        path1, path2, dist=euclidean_distance
    )[0]

    nDTW = np.exp(
        -dtw_distance
        / (len(path1) * (success_dis/map_resolution))
    )
    return nDTW, dtw_distance


def shorten_path(path, step_size=20):
    tmp_path = []
    tmp_path.append(path[0])
    for pt in path[1:-1:step_size]:
        tmp_path.append(pt)
    tmp_path.append(path[-1])
    return tmp_path


def get_geodesic_dist_sim(pathfinder, start_pos, goal_pos):
    path = ShortestPath()
    path.requested_start = start_pos
    path.requested_end = goal_pos
    pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    return geodesic_distance


def check_navigable_on_sim(pf, mp_pt, map_resolution, lower_bound, sim_start):
    if not pf.is_navigable(sim_start):
        import ipdb; ipdb.set_trace()  # breakpoint 163

    changable_pt = mp_pt*map_resolution + np.array(lower_bound)
    geodesic_distance = None

    spath = ShortestPath()
    spath.requested_start = sim_start
    spath.requested_end = changable_pt
    if pf.find_path(spath):
        geodesic_distance = spath.geodesic_distance
    else:
        changable_pt = pf.get_random_navigable_point_near(
            changable_pt, radius=0.5)

        spath = ShortestPath()
        spath.requested_start = sim_start
        spath.requested_end = changable_pt
        if pf.find_path(spath):
            geodesic_distance = spath.geodesic_distance

    if not pf.is_navigable(changable_pt):
        return None, None

    if geodesic_distance is None:
        return None, None
    return to_map_pt(changable_pt, lower_bound, map_resolution), changable_pt


def cal_geo_dist(pf, changable_pt, fix_pt):
    if not pf.is_navigable(fix_pt):
        import ipdb
        ipdb.set_trace()  # breakpoint 163

    changable_pt = np.copy(changable_pt)
    geodesic_distance = None

    spath = ShortestPath()
    spath.requested_start = fix_pt
    spath.requested_end = changable_pt
    if pf.find_path(spath):
        geodesic_distance = spath.geodesic_distance
    else:
        changable_pt = pf.get_random_navigable_point_near(
            changable_pt, radius=0.5)

        spath = ShortestPath()
        spath.requested_start = fix_pt
        spath.requested_end = changable_pt
        if pf.find_path(spath):
            geodesic_distance = spath.geodesic_distance

    return changable_pt, geodesic_distance

# @execution_time


def run_ep(scene_id, ep, nav_map, lower_bound, map_resolution, objs, rooms):
    try:
        pf = habitat_sim.PathFinder()
        nav_mesh_file = "../data/scene_datasets/mp3d/{scene_id}/{scene_id}.navmesh"
        pf.load_nav_mesh(nav_mesh_file.format(scene_id=scene_id))

        traj_raw_list = []
        input_path = osp.join(args.input_dir.format(
            split=split), f"{ep['ep_id']}.jsonl")
        if not os.path.exists(input_path):
            return

        with jsonlines.open(osp.join(args.input_dir.format(split=split), f"{ep['ep_id']}.jsonl"), 'r') as reader:
            for line in reader:
                traj_raw_list.append(line)

        path_generated = _run_ep(pf, ep, nav_map, lower_bound,
                                 map_resolution, objs, rooms, scene_id, traj_raw_list)
    #     if path_generated == 0:
    #         raise Exception('path_generated == 0')
    except Exception as e:
        print("Error with ep", ep['ep_id'], e)


def _run_ep(pf, ep, nav_map, lower_bound, map_resolution, objs, rooms, scene_id, traj_raw_list):
    # print(ep['ep_id'])
    os.makedirs(args.out_dir.format(split=split), exist_ok=True)
    path_generated = 0
    if osp.exists(osp.join(args.out_dir.format(split=split), f"{ep['ep_id']}.jsonl")):
        return path_generated

    # NOTE: object sequence
    weight_map = np.copy(nav_map).astype(float)
    start_pt = to_map_pt(ep['start_pt'], lower_bound, map_resolution)
    goal_pt = to_map_pt(ep['end_pt'], lower_bound, map_resolution)

    gt_path = [to_map_pt(pt, lower_bound, map_resolution)
               for pt in ep['ref_path']]
    solver = astar_planner_weighted(nav_map, weight_map)
    path_merged = []
    # point valid check
    # print([solver.is_free(*pt) for pt in gt_path])
    for i, pt in enumerate(gt_path[1:]):
        path = solver.solve(gt_path[i], pt)
        path_merged += path
    path_gt = np.array(path_merged)

    # print(f"GT map path time cost: {time.time()-tic:.4f} seconds")
    for traj_info in traj_raw_list:
        path_merged = []
        traj_raw = traj_info['cam_poses_raw']

        # ===================================================

        traj_end = traj_raw[-1]
        if len(traj_raw) == 0:
            continue
        path_merged = solver.solve(gt_path[0], tuple(traj_end))
        if path_merged is None:
            continue
        path_merged = list(path_merged)
        # ===================================================

        traj_propose = np.array(path_merged)

        end_pt, end_pt_sim = check_navigable_on_sim(
            pf, traj_propose[-1], map_resolution, lower_bound, ep['start_pt'])
        if end_pt is None:
            print(scene_id, ep['start_pt'], traj_propose[-1]
                  * map_resolution + np.array(lower_bound))
            continue

        _, geodesic_distance_path = cal_geo_dist(
            pf, end_pt_sim, ep['start_pt'])  # to start
        if geodesic_distance_path is None or geodesic_distance_path <= 3 or geodesic_distance_path > 20:
            continue

        _, geodesic_distance = cal_geo_dist(
            pf, end_pt_sim, ep['end_pt'])  # to goal

        distance_to_goal = geodesic_distance
        if distance_to_goal >= 5:  # scoring region radius 5m
            score = 0
        else:
            # 10 level score 10 is the highest, 1 is the lowest
            score = 10 - int(distance_to_goal / 0.5)
        nDTW, dtw_distance = ndtw(traj_propose, path_gt, map_resolution)
        discrete_path = shorten_path(traj_propose, PATH_STEP_SIZE)

        trajectory, points_compass = \
            create_compass(discrete_path, rooms, objs, nav_map,
                           ep['start_rot'], map_resolution)

        agent_states = []
        for pt in trajectory:
            pt = np.array(pt)
            sim_loc = pt[:3]*map_resolution + np.array(lower_bound)
            agent_dir = pt[3:6]
            agent_rot = pt[6]

            agent_states.append({
                'floor_loc': sim_loc.tolist(),
                'sim_loc': sim_loc.tolist(),
                'sim_rot': agent_dir.tolist(),
                'agent_rot': agent_rot.tolist()
            }
            )

        traj_info.update({
            'target': score,
            "geodist_to_goal": distance_to_goal,
            "ndtw": nDTW,
            "dtw": dtw_distance,
            'points_list': points_compass,
            "cam_poses": agent_states,
            "actions": [1, 1, 1, 1, 1],
        })

        with jsonlines.open(osp.join(args.out_dir.format(split=split), f"{ep['ep_id']}.jsonl"), mode='a') as writer:
            writer.write(traj_info)

        path_generated += 1
    return path_generated


def main():
    pool = Pool(num_process)
    for scene_id in scene2eps.keys():
        if os.environ.get('SMALL', False):
            if scene_id not in ['8WUmhLawc2A', 'zsNo4HB9uLZ']:
                continue

        eps = scene2eps[scene_id]
        map_path_format='../data/preprocessed_navmap/{scene_id}.h5'
        map_resolution, lower_bound, nav_map = load_maps(scene_id, map_path_format=map_path_format)

        obj_info = json.load(
            open(f"../data/scene_datasets/mp3d_info/objs_info/{scene_id}.json"))
        obj_dict, objs = process_obj_dict(
            obj_info, lower_bound, map_resolution)

        room_info = json.load(
            open(f"../data/scene_datasets/mp3d_info/room_info/{scene_id}.json"))
        room_dict, rooms = process_room_dict(
            room_info, lower_bound, map_resolution)

        for ep in eps:
            # run_ep(scene_id, ep, nav_map, lower_bound, map_resolution, objs, rooms)
            pool.apply_async(run_ep, (scene_id, ep, nav_map,
                             lower_bound, map_resolution, objs, rooms, ))

    pool.close()
    pool.join()
    # print(np.mean(nDTWs_st), np.mean(nDTWs_obj))
    # 0.7928479638481266 0.796010933330721


if __name__ == "__main__":
    main()
