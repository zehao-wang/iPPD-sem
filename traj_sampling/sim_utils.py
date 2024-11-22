import numpy as np
from habitat.sims import make_sim
from habitat import get_config # v0.2.0 and v0.2.3 has very large change, use version>0.2.3
from habitat_sim import ShortestPath
ISLAND_RADIUS_LIMIT = 1.5
MAX_SAMPLE_GEODIST=20
MIN_SAMPLE_GEODIST=3
from omegaconf import read_write

# def set_up_habitat(scene, no_vis=False): # v0.2.0
#     config = get_config()
#     config.defrost()
#     config.SIMULATOR.SCENE = scene
#     if no_vis:
#         config.SIMULATOR.AGENT_0.SENSORS = []
#     else:
#         config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
#         config.SIMULATOR.RGB_SENSOR.HFOV = 120
#         config.SIMULATOR.DEPTH_SENSOR.HFOV = 120
#     config.SIMULATOR.TURN_ANGLE = 15
#     config.freeze()

#     sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
#     pathfinder = sim.pathfinder
#     return sim, pathfinder

def set_up_habitat(scene, no_vis=False, cfg_path='config/benchmark/nav/vln_r2r.yaml'): # v0.2.3
    config = get_config(cfg_path)
    with read_write(config):
        config.habitat.simulator.scene=scene
    sim = make_sim(id_sim=config.habitat.simulator.type, config=config.habitat.simulator)
    pathfinder = sim.pathfinder
    return sim, pathfinder

def get_geodesic_dist(pathfinder, start_pos, goal_pos):
    path = ShortestPath()
    path.requested_start = start_pos
    path.requested_end = goal_pos
    pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    return geodesic_distance


def get_num_steps(sim, start_pos, start_rot, goal_pos):
    try:
        assert type(start_rot) == np.quaternion
    except:
        raise RuntimeError("rotation was not a quaternion")
    sim.set_agent_state(start_pos, start_rot)
    greedy_follower = sim.make_greedy_follower(goal_radius=0.25)
    try:
        steps = greedy_follower.find_path(goal_pos)
        total_steps = len(steps) - 1
    except:
        print("Error: greedy follower could not find path!")
        total_steps = 20
    if total_steps > 50:
        total_steps = 20
    return total_steps


def get_steps(sim, start_pos, start_rot, goal_pos, radius=1):
    next_step = -1
    steps = []
    try:
        assert type(start_rot) == np.quaternion
    except:
        raise RuntimeError("rotation was not a quaternion")
    sim.set_agent_state(start_pos, start_rot)
    greedy_follower = sim.make_greedy_follower(goal_radius=radius)
    try:
        steps = greedy_follower.find_path(goal_pos)
        next_step = steps[0]
    except:
        print("Error: greedy follower could not find path!")
    return steps, next_step

def get_rand_action_seq(sim, start_pos):
    point_sample = sim.sample_navigable_point()
    if start_pos == point_sample:
        return None
    if sim.island_radius(point_sample) < ISLAND_RADIUS_LIMIT:
        return None
    try:
        geo_dist = sim.geodesic_distance(start_pos, point_sample)
        if geo_dist==np.inf or geo_dist>MAX_SAMPLE_GEODIST or geo_dist<MIN_SAMPLE_GEODIST:
            return None
    
        greedy_follower = sim.make_greedy_follower(goal_radius=0.25)
        action_seq = greedy_follower.find_path(point_sample)
    except:
        # sometimes meet unexpected error
        return None
    return action_seq

def get_rand_action_seq_near(sim, start_pos):
    point_sample = sim.sample_navigable_point()
    if start_pos == point_sample:
        return None
    if sim.island_radius(point_sample) < ISLAND_RADIUS_LIMIT:
        return None
    try:
        geo_dist = sim.geodesic_distance(start_pos, point_sample)
        if geo_dist==np.inf or geo_dist>MAX_SAMPLE_GEODIST or geo_dist<MIN_SAMPLE_GEODIST:
            return None
    
        greedy_follower = sim.make_greedy_follower(goal_radius=0.25)
        action_seq = greedy_follower.find_path(point_sample)
    except:
        # sometimes meet unexpected error
        return None
    return action_seq

def get_action_seq_by_ref_path(sim, ref_path, start_pos, start_rot):
    action_seq = []
    # curr_pose = start_pos
    # for i in range(1, len(ref_path)):
    #     geo_dist = sim.geodesic_distance(curr_pose, ref_path[i])
    #     if geo_dist==np.inf or geo_dist>MAX_SAMPLE_GEODIST or geo_dist<MIN_SAMPLE_GEODIST:
    #         return None
    #     greedy_follower = sim.make_greedy_follower(goal_radius=0.25)
    #     actions = greedy_follower.find_path(ref_path[i])

    #     for act in actions:
    #         if act != -1: 
    #             sim.step(act)
    #             action_seq.append(act)
    #     curr_pose = sim.get_agent_state().position

    # action_seq.append(-1)
    # sim.reset()
    # sim.set_agent_state(start_pos, start_rot)

    greedy_follower = sim.make_greedy_follower(goal_radius=1.0)
    action_seq = greedy_follower.find_path(ref_path[-1])
    return action_seq

def euclidean_distance(
    pos_a, pos_b
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


import os.path as osp
import h5py
def get_navmaps(scene_id, raw_nav_map_root):
    with h5py.File(osp.join(raw_nav_map_root, f"{scene_id}_navmap.h5"), 'r') as f:
        nav_map = f['nav_map'][()]
    return nav_map