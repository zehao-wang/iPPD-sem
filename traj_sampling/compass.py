from collections import defaultdict
from map_tools_3D import get_agent_orientation, scaling
import math
import numpy as np
from scipy.spatial import distance
import os
import random
from constants import obj_layeridx2wordidx, room_layeridx2wordidx

def get_surrounding_objs(raw_objs, agent_grid_pos, meter_per_voxel, radius=5, exemption_list=[1,15,16,38,39],  skip_not_valid=False, height_limit=3):
    """
    Args:
        obj_map: raw obj map (denote pixel by instance id)
        radius: in meter
    """
    x, y, z, arot = agent_grid_pos # rot in radian
    rot = arot

    radius_voxel = int(round(radius/meter_per_voxel))
    
    result_dict = defaultdict(list)
    relative_dict = defaultdict(list)

    # add agent point (also for handling no object corner cases)
    result_dict[-1].append((x, y, z))
    relative_dict[-1].append((0, 0, 0, 0, 0))
    
    objs = raw_objs[(~np.isin(raw_objs[:,0], exemption_list)), :] # filter by exemption

    # Current multi-floor object only be limited by height in a range
    # TODO: maybe use better height to identify same floor
    selected_mask = (np.linalg.norm(objs[:,1:4] - (x, y, z), ord=2, axis=1) <= radius_voxel) & \
        ((objs[:,2] - y) >=0 ) & ((objs[:,2] - y) <= height_limit/meter_per_voxel) 
    
    for obj in objs[selected_mask, :]:
        layer_idx = int(obj[0])
        if layer_idx == -1:
            import ipdb;ipdb.set_trace() # breakpoint 37
        cx = obj[1]
        cy = obj[2]
        cz = obj[3]
        result_dict[layer_idx].append((cx, cy, cz, None)) # grid row,  grid h, grid col, instance id

        # rotate to egocentric coordinate (first coord is along agent head, second coord is perpendicular to agent head in clockwise direction, )
        rel_cx, rel_cz = rotate((0, 0), (cx-x, cz-z), -rot)
        rel_cy = cy - y

        # standardize relative position to [-1,1]
        rel_cx = scaling((-radius_voxel, radius_voxel), (-1,1), rel_cx)
        rel_cz = scaling((-radius_voxel, radius_voxel), (-1,1), rel_cz)
        rel_cy = scaling((-radius_voxel, radius_voxel), (-1,1), rel_cy)
        
        # TODO: change 2D rotate to 3D, currently rot is not used for processing objects
        # relative_dict[layer_idx].append((rel_cx, rel_cy, rel_cz, math.atan2(cx-x, cz-z) - rot, None)) 
        relative_dict[layer_idx].append((rel_cx, rel_cy, rel_cz, math.atan2(cx-x, cz-z) - rot, None)) 
    
    if os.environ.get('DEBUG', False): 
        print("object list: ")
        for k,v in relative_dict.items():
            print(f"\tobj Label {k}: {len(v)}")
        print()
    return result_dict, relative_dict

def rotate(origin, point, angle):
    """
    Rotate a point counter-clockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# v4 version
def create_compass(path, rooms, objs, nav_map, start_rot, map_resolution, room_compass_chunks=12):
    scene_shape = nav_map.shape # x, h, y

    points_list = []

    trajectory = [] 
    start_pt = path[0]
    start_rt = get_agent_orientation(start_rot, sim_rot=False)
    for i, agent_pos in enumerate(path):
        if i == 0:
            agent_rot = get_agent_orientation(start_rot, sim_rot=False)
        else:
            agent_rot = angle_between(agent_pos, path[i-1])
        
        # normalize traj path [not really used in the mln icra]
        # scaled_r = scaling((0,scene_shape[0]), (-2, 2), agent_pos[0])
        # scaled_c = scaling((0,scene_shape[1]), (-2, 2), agent_pos[1])
        # # TODO: might change the height relative position to positive
        # scaled_h = scaling((0,scene_shape[2]), (-2, 2), agent_pos[2])
        # scaled_x = agent_pos[0] * map_resolution
        # scaled_y = agent_pos[1] * map_resolution
        # scaled_z = agent_pos[2] * map_resolution

        # NOTE: test relative position at Feb 20 2023 [can also overfitting]
        scaled_x = (agent_pos[0]-start_pt[0]) * map_resolution
        scaled_y = (agent_pos[1]-start_pt[1]) * map_resolution
        scaled_z = (agent_pos[2]-start_pt[2]) * map_resolution

        direction_r = 2 * math.cos(agent_rot) # [-2,2]
        direction_c = 2 * math.sin(agent_rot) # [-2,2]
        trajectory.append((scaled_x, scaled_y, scaled_z, direction_r, direction_c, 0, agent_rot))

        _, relative_dict = get_surrounding_objs(
            objs, (*agent_pos, agent_rot), meter_per_voxel=map_resolution,
            radius=5 # v1
            # radius=3 # v2
        )
        
        points = [] # object points
        for k, v in relative_dict.items():
            for point in v:
                points.append((point[0], point[1], point[2], point[3], obj_layeridx2wordidx[k])) # point: [rel_r, rel_c, ral_h, rel_ang, word_id]
        
        if len(points) > 50:
            points = random.sample(points, 50)
        else:
            choices = np.random.choice(len(points), 50-len(points))
            points.extend(np.array(points)[choices].tolist())
        
        points_list.append(points)

    return trajectory, points_list