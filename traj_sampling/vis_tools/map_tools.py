from typing import Tuple
import numpy as np
import math
import quaternion
import h5py
import open3d as o3d

def load_maps(scene_id, map_path_format = "../data/preprocessed_navmap/{scene_id}.h5"):
    with h5py.File(map_path_format.format(scene_id=scene_id), "r") as f:
        map_resolution = f.attrs['map_resolution']
        lower_bound = f['lower_bound'][()]
        navmap = f['nav_map'][()]
    return map_resolution, lower_bound, navmap

def voxmap2points(nav_map):
    xv, yv, zv = np.meshgrid(
        np.arange(nav_map.shape[0]), np.arange(nav_map.shape[1]), 
        np.arange(nav_map.shape[2]), indexing='ij'
    )
    coords = np.transpose(np.vstack([xv.flatten(), yv.flatten(), zv.flatten()]))
    nav_map = nav_map.flatten()
    coords = coords[nav_map]

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(coords)
    pcd_o3d.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd_o3d

def maploc2simloc(mp_pt, map_resolution, lower_bound):
    sim_loc =  mp_pt*map_resolution + np.array(lower_bound)
    return sim_loc

def simloc2maploc(sim_pt, lower_bound, map_resolution):
    return tuple(np.round((np.array(sim_pt) - np.array(lower_bound))/map_resolution).astype(int).tolist())

def euclidean_distance(
    pos_a, pos_b
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)

def get_agent_orientation(arot, sim_rot=True):
    """
    Args:
        arot is quaternion x,y,z,w
        sim_rot: whether the rotation is from simulator, otherwise most probab is the annotation file, which should be adjust the order
    Return:
        angle in radian, rotate from (head to down) counter-clockwise
    """
    if sim_rot:
        arot_q = arot
    else:
        arot_q = quaternion.from_float_array(np.array([arot[3], *arot[:3]]) )
    agent_forward = quaternion.rotate_vectors(arot_q, np.array([0,0,-1.]))
    rot = math.atan2(agent_forward[0], agent_forward[2]) 
    
    rot = -rot + math.pi /2
    return rot, agent_forward