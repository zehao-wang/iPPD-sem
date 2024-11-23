from turtle import color
import numpy as np
from typing import Any, Tuple, Sequence
import os.path as osp
import h5py
import numpy as np
import copy
import cv2
import sys
sys.setrecursionlimit(20000)
import math
# import magnum as mn
import quaternion
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("./data/arial.ttf", 8)

from constants import obj_merged_dict, room_merged_dict, roomidx2name, semantic_sensor_40cat, room_set, objs_set, tab10_colors_rgba
from queue import PriorityQueue
import os
action_mapping={
    0:"stop",1:"forward",2:"left",3:"right"
}

def get_contour_points(pos, size):
    x, y, o = pos
    pt1 = (int(x),
           int(y))
    # pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)),
    #        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)))
    # pt3 = (int(x + size * np.cos(o)),
    #        int(y + size * np.sin(o)))
    # pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)),
    #        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)))
    
    pt2 = (int(x + size / 1.5 * np.sin(o + np.pi * 4 / 3)),
           int(y + size / 1.5 * np.cos(o + np.pi * 4 / 3)))
    pt3 = (int(x + size * np.sin(o)),
           int(y + size * np.cos(o)))
    pt4 = (int(x + size / 1.5 * np.sin(o - np.pi * 4 / 3)),
           int(y + size / 1.5 * np.cos(o - np.pi * 4 / 3)))

    return np.array([pt1, pt2, pt3, pt4])

import math
 
def euler_from_quaternion(w,x,y,z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    # t3 = +2.0 * (w * z + x * y)
    # t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw_z = math.atan2(t3, t4)
    
    # return roll_x, pitch_y, yaw_z # in radians
    return pitch_y

# def draw_agent(aloc, arot, np_map):
#     arot = quaternion.from_float_array(np.array([arot[3], *arot[:3]]) )
#     # arot = quaternion.from_float_array(np.array(arot))
#     agent_forward = mn.Quaternion(arot.imag, arot.real).transform_vector(mn.Vector3(0, 0, -1.0))
#     agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
#     agent_arrow = get_contour_points( (aloc[1], aloc[0], agent_orientation), size=15)
#     cv2.drawContours(np_map, [agent_arrow], 0, (0,0,255,255), -1)

def get_agent_orientation(arot):
    """
    Args:
        arot is quaternion x,y,z,w
    Return:
        angle in radian, rotate from (head to down) counter-clockwise
    """
    arot_q = quaternion.from_float_array(np.array([arot[3], *arot[:3]]) )
    agent_forward = quaternion.rotate_vectors(arot_q, np.array([0,0,-1.]))
    rot = math.atan2(agent_forward[0], agent_forward[2])
    return rot

def draw_agent(aloc, arot, np_map, is_radian=False, color=(0,0,255,255)):
    if is_radian:
        agent_orientation = arot
    else:
        agent_orientation = get_agent_orientation(arot)
    agent_arrow = get_contour_points( (aloc[1], aloc[0], agent_orientation), size=15)
    cv2.drawContours(np_map, [agent_arrow], 0, color, -1)

def draw_point(pil_img, x, y, point_size, color, text=None):
    drawer = ImageDraw.Draw(pil_img, 'RGBA')
    drawer.ellipse((x-point_size, y-point_size, x+point_size, y+point_size), fill=color)
    if text is not None:
        drawer.text((x+point_size, y-point_size), text, font=font, fill=color)

def draw_path(np_map, gt_annt, grid_dimensions, upper_bound, lower_bound, is_grid=False, color=(0,128,128,255)):
    if isinstance(gt_annt, dict):
        locations = gt_annt['locations']
        if os.environ.get('DEBUG', False): 
            print('\033[92m'+'DEBUG mode:'+'\033[0m')
            print("GT Action sequence: ", [action_mapping[act] for act in gt_annt['actions']])
    elif isinstance(gt_annt, list):
        locations = gt_annt
    else:
        raise NotImplemented()

    for i in range(1, len(locations)):

        if not is_grid:
            start_grid_pos = simloc2maploc(
                locations[i-1], grid_dimensions, upper_bound, lower_bound
            )
            end_grid_pos = simloc2maploc(
                locations[i], grid_dimensions, upper_bound, lower_bound
            )
        else:
            start_grid_pos = locations[i-1]
            end_grid_pos = locations[i]
        cv2.line(
            np_map,
            (start_grid_pos[1], start_grid_pos[0]), # use x,y coord order
            (end_grid_pos[1], end_grid_pos[0]),
            color=color,
            thickness=2,
        )

def simloc2maploc(aloc, grid_dimensions, upper_bound, lower_bound):
    agent_grid_pos = to_grid(
        aloc[2], aloc[0], grid_dimensions, lower_bound=lower_bound, upper_bound=upper_bound
    )
    return agent_grid_pos

def get_maps(scene_id, root_path, merged):
    gmap_path = osp.join(root_path, f"{scene_id}_gmap.h5")
    with h5py.File(gmap_path, "r") as f:
        nav_map  = f['nav_map'][()]
        room_map = f['room_map'][()] 
        obj_maps = f['obj_maps'][()] 
        # obj_maps[:,:,1] = ((obj_maps[:,:,1]>0) ^ (obj_maps[:,:,15]>0)) * obj_maps[:,:,15] + obj_maps[:,:,1] # merge stairs to floor
        if merged:
            room_map, obj_maps=merge_maps(room_map, obj_maps)
        bounds = f['bounds'][()]

    grid_dimensions = (nav_map.shape[0], nav_map.shape[1])
    return nav_map, room_map, obj_maps, grid_dimensions, bounds

def colorize_nav_map(nav_map):
    recolor_map = np.array(
            [[255, 255, 255, 255], [128, 128, 128, 255], [0, 0, 0, 255]], dtype=np.uint8
    )
    nav_map = recolor_map[nav_map]
    return nav_map

def load_panos(scene_name, pano_path):
    """
    Args:
        scene_name
        pano_path
    Return:
        panos [X * Y, K, K, 3]
    """
    pass

def to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: Tuple[int, int],
    lower_bound, upper_bound
) -> Tuple[int, int]:
    """
    single point implementation
    """
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
    grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
    return grid_x, grid_y

def from_grid(
    grid_x: int,
    grid_y: int,
    grid_resolution: Tuple[int, int],
    lower_bound, upper_bound
) -> Tuple[float, float]:
    """
    single point implementation
    """
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    realworld_x = lower_bound[2] + grid_x * grid_size[0]
    realworld_y = lower_bound[0] + grid_y * grid_size[1]
    return realworld_x, realworld_y


def merge_maps(room_map, obj_maps):
    """
    first merge room maps

    move closet maps from obj to rooms

    Return:
        return merged maps with indexing list
    """
    new_room_map = np.zeros((room_map.shape[0], room_map.shape[1], len(room_set)))
    new_obj_map = np.zeros((obj_maps.shape[0], obj_maps.shape[1], len(objs_set)))

    for idx, room_name in roomidx2name.items():
        new_room_map[:,:,room_set[room_merged_dict[room_name]]] = \
            ((new_room_map[:,:,room_set[room_merged_dict[room_name]]]>0) ^ (room_map[:,:,idx]>0) ) \
            * new_room_map[:,:,room_set[room_merged_dict[room_name]]] + room_map[:,:,idx]
    
    for idx, obj_name in semantic_sensor_40cat.items():
        if obj_name in ["stairs", "ceiling", "misc", "objects"]:
            continue
        if obj_name in room_merged_dict or obj_merged_dict[obj_name] == 'closet':
            new_room_map[:,:,room_set[obj_merged_dict[obj_name]]] = \
                ((new_room_map[:,:,room_set[obj_merged_dict[obj_name]]]>0) ^ (obj_maps[:,:,idx]>0) ) \
                * new_room_map[:,:,room_set[obj_merged_dict[obj_name]]] + obj_maps[:,:,idx]
        else:
            new_obj_map[:,:,objs_set[obj_merged_dict[obj_name]]] = \
                ((new_obj_map[:,:,objs_set[obj_merged_dict[obj_name]]]>0) ^ (obj_maps[:,:,idx]>0) ) \
                * new_obj_map[:,:,objs_set[obj_merged_dict[obj_name]]] + obj_maps[:,:,idx]
    return new_room_map, new_obj_map


def draw_candidates(map, start_grid_pos, size, order):
    """
    Args:
        map: single instance level map
    """
    ins_set = set(map.flatten().tolist())
    ins_set.remove(0)
    overlay = Image.new('RGBA', size, (255,0,0)+(0,))
    for i, ins in enumerate(ins_set):
        coords = np.where(map == ins)
        #coords = tuple(zip(*coords))
        cr, cc = round(np.median(coords[0])), round(np.median(coords[1]))
        draw_point(overlay, cc, cr, 8, color=tuple(tab10_colors_rgba[order%10]), text=str(order))
    return overlay

def create_candidates(valid_map, sample_gap=1.2, floor_idx = 1, meters_per_pixel=0.05):
    """
    Args:
        sample_gap: meter of sampling gap
    """
    step_size = int(sample_gap / meters_per_pixel)
    candidates = []
    # 6/0.05 = 120
    # sqrt(84 ** 2 * 2) < 120
    r = 10
    xs = [0,0, r, -r,r,-r,r,-r]
    ys = [r,-r,0,0,r,-r,-r,r]

    for i in range(step_size//2, valid_map.shape[0], step_size):
        for j in range(step_size//2, valid_map.shape[1], step_size):
            if valid_map[i,j]:
                candidates.append((i, j))
            else:
                for t in zip(xs, ys):
                    newx = np.clip(i+t[0], 0, valid_map.shape[0]-1)
                    newy = np.clip(j+t[1], 0, valid_map.shape[1]-1)
                    if valid_map[newx,newy]:
                        candidates.append((newx, newy))
                        break
    return candidates


def get_possible_paths(nav_map, obj_maps, start_point):
    solver = shortest_path(nav_map, obj_maps, start_point, radius=1, step=1)
    candidate_targets = create_candidates(nav_map, obj_maps)
    # print(candidate_targets)
    candidate_pathes = []
    for target in candidate_targets:
        path = solver.find_path_by_target(target).tolist()
        candidate_pathes.append(path)
    # for i, path in enumerate(candidate_pathes):
    #     if len(path) <3:
    #         continue
    #     candidate_pathes[i] = discretize_path(path, start_point)
    return candidate_targets, candidate_pathes

# def get_rot(point1, point2):
#     return math.atan2(point2[1]-point1[1], point2[0]-point1[0])

# def discretize_path(path, start_rot, rot_smooth_range=2):
#     dis_steps=40
#     dis_path = [path[0]] # start status
#     for i in range(0,len(path), dis_steps):
#         if i==0:
#             continue
#         if i == len(path)-1:
#             break

#         p1 = path[max(i-rot_smooth_range, 0)]
#         p2 = path[i]
#         p3 = path[min(i+rot_smooth_range, len(path)-1)]
#         dis_path.append(path[i])
    
#     dis_path.append(path[-1]) # end status
#     print(len(path))
#     print(len(dis_path))
#     print('=====')
#     return dis_path


import timeit
def execution_time(method):
    """ decorator style """

    def time_measure(*args, **kwargs):
        ts = timeit.default_timer()
        result = method(*args, **kwargs)
        te = timeit.default_timer()

        print(f'Excution time of method {method.__qualname__} is {te - ts} seconds.')
        #print(f'Excution time of method {method.__name__} is {te - ts} seconds.')
        return result

    return time_measure

class shortest_path:
    def __init__(self, nav_map, obj_maps, start_point, radius = 1, step =1, bound=None):
        self.bound = bound
        self.ro, self.co = 0,0
        if self.bound is None:
            self.bound = (0,0,nav_map.shape[0], nav_map.shape[1])
        dims = nav_map.shape

        get_graph_id_fn = lambda point:point[0]*dims[1] + point[1]
        get_coords_fn = lambda graph_id:(graph_id//dims[1], graph_id%dims[1])
        start_vertex=get_graph_id_fn(start_point)
        D = {v:float('inf') for v in range(dims[0]*dims[1])}
        prevs =  {v:-1 for v in range(dims[0]*dims[1])}
        D[start_vertex] = 0

        pq = PriorityQueue()
        pq.put((0, start_vertex))

        graph_visited = set()
        xs = [0,0, 1, -1,1,-1,1,-1]
        ys = [1,-1,0,0,1,-1,-1,1]
        value = [1,1,1,1,1.41421,1.41421,1.41421,1.41421]
        
        def check_valid(point):
            if point[0]<self.bound[0] or point[0]>=self.bound[2] or point[1]<self.bound[1] or point[1]>=self.bound[3]:
                return False
            return True
            
        while not pq.empty():
            (dist, current_vertex) = pq.get()
            graph_visited.add(current_vertex)

            current_point = get_coords_fn(current_vertex)

            neighbors = []
            for i in range(8):
                new_point = (current_point[0]+xs[i]*step, current_point[1]+ys[i]*step)
                if not check_valid(new_point):
                    continue
                flag = False
                for j in range(8):
                    if not check_valid((new_point[0]+xs[j]*radius, new_point[1]+ys[j]*radius)):
                        flag = True
                        break
                    if nav_map[new_point[0]+xs[j]*radius, new_point[1]+ys[j]*radius] <= 0 \
                        and obj_maps[new_point[0]+xs[j]*radius, new_point[1]+ys[j]*radius, 1] <= 0:
                        flag = True
                        break
                    # if nav_map[new_point[0]+xs[j], new_point[1]+ys[j]] <= 0 \
                    #     and obj_maps[new_point[0]+xs[j], new_point[1]+ys[j], 1] <= 0:
                    #     flag = True
                    #     break
                if not flag:
                    neighbors.append((get_graph_id_fn(new_point),value[i]))

            for neighbor in neighbors:
                if neighbor[0] not in graph_visited:
                    dis = neighbor[1]
                    neighbor = neighbor[0]
                    old_cost = D[neighbor]
                    new_cost = D[current_vertex] + dis*step
                    if new_cost < old_cost:
                        pq.put((new_cost, neighbor))
                        D[neighbor] = new_cost
                        prevs[neighbor] = current_vertex
        self.get_graph_id_fn = get_graph_id_fn
        self.get_coords_fn = get_coords_fn
        self.D = D
        self.prevs = prevs
        
    def find_path_by_target(self, tar_point):

        def find_path(v, path):
            if self.prevs[v]>0:
                path.append(self.get_coords_fn(self.prevs[v]))
                find_path(self.prevs[v], path)
        path = [tar_point]
        find_path(self.get_graph_id_fn(tar_point), path)
        return np.array(path[::-1])

