import os
import os.path as osp
from typing import Tuple
import h5py
import numpy as np
import math
import quaternion
import pickle 

def euclidean_distance(
    pos_a, pos_b
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)

def scaling(bound, target_range, value):
    """
    Scaling a value from a range (bound) to a target_range
    """
    value = np.clip(value, bound[0], bound[1])
    v_std = (value-bound[0]) / (bound[1]-bound[0])
    return v_std * (target_range[1] - target_range[0]) + target_range[0]
    
def surrounding_3Dpos(radius, num_split=None, sorted_type='ascending'):
    """
    Args:
        radius: max radius of point relative to center
        num_split: number of splits on one axis
        sorted_type: from centor to bound use 'ascending'
    """
    if num_split is None:
        num_split = 2*int(radius) + 1
    x = np.linspace(-radius, radius, num_split)
    y = np.linspace(-radius, radius, num_split)
    z = np.linspace(-radius, radius, num_split)
    xv, yv, zv = np.meshgrid(x, y, z)

    coord_pairs = list(zip(xv.flatten(), yv.flatten(), zv.flatten()))

    if sorted_type == 'ascending':
        out_pairs = sorted(coord_pairs, key=lambda k: k[0]**2 + k[1]**2 + k[2]**2 )
        out_pairs = coord_pairs[ : int(len(coord_pairs) * 0.52)+1] # 0.52 ratio of points is inside the sphere
    else:
        raise NotImplementedError()
    return out_pairs

SEARCH_RANGE = surrounding_3Dpos(2, num_split=5, sorted_type="ascending")

def get_raw_maps(scene_id, map_root):
    gmap_path = osp.join(map_root, f"{scene_id}_gmap.h5")
    with h5py.File(gmap_path, "r") as f:
        obj_ins_map  = f['obj_ins_map'][()]
        room_ins_map = f['room_ins_map'][()]
        room_map = f['room_map'][()] 
        obj_map = f['obj_map'][()] 
        bounds = f['bounds'][()]

    grid_dimensions = (obj_ins_map.shape[0], obj_ins_map.shape[1], obj_ins_map.shape[2]) # order x dim, z dim, y dim (height)
    return obj_ins_map, room_ins_map, room_map, obj_map, grid_dimensions, bounds

def get_processed_maps(scene_id, map_root):
    """
    get obstacle maps, room instance list, object instance list
    """
    gmap_path = osp.join(map_root, f"{scene_id}_gmap.pkl")
    with open(gmap_path, 'rb') as f:
        gmaps = pickle.load(f)
    nav_map = gmaps['nav_map'] # 1 for navigable 0 for empty -1 for obstacle
    room_list = gmaps['room_list']
    obj_list = gmaps['obj_list']
    bounds = gmaps['bounds']
    grid_dimensions = nav_map.shape
    return nav_map, room_list, obj_list, grid_dimensions, bounds

def is_legal(point, nav_map):
    x,y,z = point
    return ((x>=0) & (x < nav_map.shape[0])) and ((y>=0) & (y < nav_map.shape[1]))\
             and ((z>=0) & (z < nav_map.shape[2])) and (nav_map[x][y][z] > 0)

def find_nearest_legal(point, nav_map):
    x, y, z = point
    for point_s in SEARCH_RANGE:
        xs, ys, zs = point_s
        if is_legal((x+int(xs), y+int(ys), z+int(zs)), nav_map):
            return (x+int(xs), y+int(ys), z+int(zs))
    return None

# MAP version
# def simloc2maploc(aloc, grid_dimensions, upper_bound, lower_bound, nav_map):
#     agent_grid_pos = to_grid3d(
#         aloc[0], aloc[2], aloc[1], grid_dimensions, lower_bound=lower_bound, upper_bound=upper_bound
#     )
#     # print(agent_grid_pos)

#     if not is_legal(agent_grid_pos, nav_map):
#         agent_grid_pos = find_nearest_legal(agent_grid_pos, nav_map)
#         if agent_grid_pos is None: 
#             print(f"Please Check whether point transfer axis is correct, we got point {agent_grid_pos}, for map shape {nav_map.shape}")
#             agent_grid_pos = (to_grid3d(
#                 aloc[0], aloc[2], aloc[1], grid_dimensions, lower_bound=lower_bound, upper_bound=upper_bound
#             ), None)
#     return agent_grid_pos

# sim version
def simloc2maploc(aloc, grid_dimensions, upper_bound, lower_bound, nav_map):
    agent_grid_pos = to_grid3d(
        aloc[0], aloc[2], aloc[1], grid_dimensions, lower_bound=lower_bound, upper_bound=upper_bound
    )
    return agent_grid_pos

def to_grid3d(
    realworld_x: float,
    realworld_y: float,
    realworld_z: float,
    grid_resolution: Tuple[int, int],
    lower_bound, upper_bound
) -> Tuple[int, int]:
    """
    single point implementation
    """
    grid_size = (
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[0],
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[1], 
        abs(upper_bound[1] - lower_bound[1]) / grid_resolution[2]  # height
    )
    grid_x = int((realworld_x - lower_bound[0]) / grid_size[0]) 
    grid_y = int((realworld_y - lower_bound[2]) / grid_size[1])
    grid_z = int((realworld_z - lower_bound[1]) / grid_size[2])
    return grid_x, grid_y, grid_z

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
    return rot