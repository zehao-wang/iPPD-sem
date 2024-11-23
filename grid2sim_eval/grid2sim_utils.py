""" This script is for transferring trajectory predicted on maps to simulation environment """
import numpy as np
import os.path as osp
import h5py
import jsonlines

def grid2sim(path_meta, map_root, dump_dir=None):
    """ mln_v1
    Convert a not discretized path to simulation results, this version only 
    consider the start point and the last few points, simulator path is generated 
    by simulator shortest path planner.
    Args:
        path_meta: trajectory on map (not discretized), scene_name, episode_id
        map_root: directory of map data
        dump_dir: if not None, save the simulation path to file
    """
    episode_id = path_meta['episode_id']
    scene_name = path_meta['scene_name']
    path = path_meta['path_continue']

    grid_resolution, bounds = get_map_info(scene_name, map_root)
    upper_bound, lower_bound = bounds[0], bounds[1]

    sim_loc_path = path_loc_to_sim_loc(path, grid_resolution, lower_bound, upper_bound)
    path_meta.update({"sim_path": sim_loc_path})

    if dump_dir is not None:
        with jsonlines.open(dump_dir, "a") as writer:
            writer.write(path_meta)

####################### utils #########################
def path_loc_to_sim_loc(grid_path, grid_resolution, lower_bound, upper_bound):
    sim_path = []

    for point in grid_path:
        p1, p2 ,p3 = point 
        sim_x, sim_y = grid_to_sim_loc(p1, p2, grid_resolution, lower_bound, upper_bound)
        sim_path.append([sim_y, p3, sim_x])
    
    return sim_path 

def grid_to_sim_loc(grid_x, grid_y, grid_resolution, lower_bound, upper_bound):
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    real_word_x = grid_x * grid_size[0] + lower_bound[2]
    real_word_y = grid_y * grid_size[1] + lower_bound[0]
    return real_word_x, real_word_y

def get_map_info(scene_name, root_path):
    gmap_path = osp.join(root_path, f"{scene_name}_gmap.h5")
    with h5py.File(gmap_path, "r") as f:
        nav_map  = f['nav_map'][()]
        bounds = f['bounds'][()]
     
    grid_dimensions = (nav_map.shape[0], nav_map.shape[1]) # row, column 
    return grid_dimensions, bounds


################## deprecated #########################
def connect2D(point1, point2):
    """ 
    Connecting two points by sampling locations between them
    Args:
        point1: start point
        point2: end point to be connected
    """
    ends = [point1, point2]
    d0, d1 = np.abs(np.diff(ends, axis=0))[0]
    if d0 > d1: 
        return np.c_[np.linspace(ends[0, 0], ends[1, 0], d0+1, dtype=np.int32),
                     np.round(np.linspace(ends[0, 1], ends[1, 1], d0+1))
                     .astype(np.int32)]
    else:
        return np.c_[np.round(np.linspace(ends[0, 0], ends[1, 0], d1+1))
                     .astype(np.int32),
                     np.linspace(ends[0, 1], ends[1, 1], d1+1, dtype=np.int32)]