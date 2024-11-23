import numpy as np
import numpy as np
import cv2
import sys
sys.setrecursionlimit(20000)

from queue import PriorityQueue
from utils.math_tools import get_surrounding_point_rel_pos_with_radius
import math
import torch
import random
import os
import matplotlib.pyplot as plt

def gen_valid_map(nav_map, obj_maps, bound=None, floor_idx=1, stairs_idx=15, shower_idx=22, door_idx=3):
    if bound is None:
        bound = (0, 0, nav_map.shape[0], nav_map.shape[1])
    valid_map = np.zeros(nav_map.shape, dtype=bool)
    valid_map[bound[0]:bound[2], bound[1]:bound[3]] = True
    valid_map = ((nav_map>0) | (obj_maps[:,:,floor_idx]>0) | (obj_maps[:,:,stairs_idx]>0) | (obj_maps[:,:,shower_idx]>0)| (obj_maps[:,:,door_idx]>0)) & valid_map

    valid_map_medianblured = cv2.medianBlur(valid_map.astype(np.float32), 3)
    return valid_map, valid_map_medianblured

def gen_valid_map_ori(nav_map, obj_maps, bound=None, floor_idx=1, stairs_idx=15):
    if bound is None:
        bound = (0, 0, nav_map.shape[0], nav_map.shape[1])
    valid_map = np.zeros(nav_map.shape, dtype=bool)
    valid_map[bound[0]:bound[2], bound[1]:bound[3]] = True
    valid_map = ((nav_map>0) | (obj_maps[:,:,floor_idx]>0) | (obj_maps[:,:,stairs_idx]>0)) & valid_map

    valid_map_medianblured = cv2.medianBlur(valid_map.astype(np.float32), 3)
    return valid_map, valid_map_medianblured

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

def create_candidates_by_ratio(valid_map, sample_ratio, agent_radius, meters_per_pixel=0.05):
    """
    Args:
        sample_gap: meter of sampling gap
    """
    agent_radius_in_pixel = int(agent_radius / meters_per_pixel)
    valid_map_pengzhang = np.copy(valid_map)
    for i in range(valid_map_pengzhang.shape[0]):
        for j in range(valid_map_pengzhang.shape[1]):
            if valid_map[i][j] == False:
                for m in range(-agent_radius_in_pixel, agent_radius_in_pixel+1):
                    for n in range(-agent_radius_in_pixel, agent_radius_in_pixel+1):
                        xv = i+m
                        yv = j+n
                        if (0 <= xv < valid_map_pengzhang.shape[0]) and (0 <= yv < valid_map_pengzhang.shape[1]):
                            valid_map_pengzhang[xv][yv] = False
    
    indices = np.nonzero(valid_map_pengzhang)
    valid_points = np.transpose(np.stack((indices[0], indices[1]))).tolist()
    num_samples = int(len(valid_points) * sample_ratio)
    if num_samples < 10:
        import ipdb;ipdb.set_trace() # breakpoint 71

    # if os.environ.get('DEBUG', False): 
    #     plt.matshow(valid_map_pengzhang)
    #     plt.savefig(f'./dset_gen/tmp.png', dpi=500)
    #     plt.clf()
    #     import ipdb;ipdb.set_trace() # breakpoint 79

    candidates = random.sample(valid_points, k=num_samples)
    return candidates

def check_valid(valid_map, center_coord, check_rel_points):
    points = np.array(center_coord, dtype=int) + np.array(check_rel_points, dtype=int)
    points[:,0] = np.clip(points[:,0], 0, valid_map.shape[0]-1)
    points[:,1] = np.clip(points[:,1], 0, valid_map.shape[1]-1)

    points_checked = valid_map[points[:,0], points[:,1]]
    if all(points_checked):
        return True
    else:
        return False

def create_candidates_with_margin(valid_map, sample_gap=1.2,  meters_per_pixel=0.05, agent_radius=0.2):
    """
    Args:
        sample_gap: meter of sampling gap
    """
    margin = int(agent_radius/meters_per_pixel)

    searching_rel_points = get_surrounding_point_rel_pos_with_radius(10, num_split=5*2+1)
    coords_around_agent = get_surrounding_point_rel_pos_with_radius(margin, num_split=margin*2+1)

    step_size = int(sample_gap / meters_per_pixel)
    candidates = []
    # 6/0.05 = 120
    # sqrt(84 ** 2 * 2) < 120
    r = 10

    for i in range(step_size//2, valid_map.shape[0], step_size):
        for j in range(step_size//2, valid_map.shape[1], step_size):
            if check_valid(valid_map, (i,j), coords_around_agent):
                candidates.append((i, j))
            else:
                for t in searching_rel_points:
                    newx = int(np.clip(i+t[0], 0, valid_map.shape[0]-1))
                    newy = int(np.clip(j+t[1], 0, valid_map.shape[1]-1))
                    if check_valid(valid_map, (newx, newy), coords_around_agent):
                        candidates.append((newx, newy))
                        break
    return candidates

def colorize_valid_map(valid_map):
    recolor_map = np.array(
            [[255, 255, 255, 255], [128, 128, 128, 255]], dtype=np.uint8
    )
    nav_map = recolor_map[valid_map.astype(int)]
    return nav_map

def euclidean_distance(
    pos_a, pos_b
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)

class shortest_path2:
    # with downsample2d and valid map
    def __init__(self, valid_map, start_point, radius = 1, step =1, scale_percent=50):
        """
        Args:
            radius: pixel radius
            scale_percent # percent of original size
        """
        self.scale_percent = scale_percent / 100.
        width =  int(valid_map.shape[1] * self.scale_percent )
        height = int(valid_map.shape[0] * self.scale_percent )
        dim = (width, height)

        downsampled_map = cv2.resize(valid_map.astype(np.float32), dim, interpolation=cv2.INTER_NEAREST).astype(bool)
        start = list(start_point)
        start[0] =  int(start[0] * scale_percent/100)
        start[1] =  int(start[1] * scale_percent/100)

        dims = downsampled_map.shape
        get_graph_id_fn = lambda point:point[0]*dims[1] + point[1]
        get_coords_fn = lambda graph_id:(graph_id//dims[1], graph_id%dims[1])
        start_vertex=get_graph_id_fn(start)
        D = {v:float('inf') for v in range(dims[0]*dims[1])}
        prevs =  {v:-1 for v in range(dims[0]*dims[1])}
        D[start_vertex] = 0

        pq = PriorityQueue()
        pq.put((0, start_vertex))

        graph_visited = set()
        xs = [0,  0, 1, -1, 1, -1,  1, -1]
        ys = [1, -1, 0,  0, 1, -1, -1,  1]
        value = [1,1,1,1,1.41421,1.41421,1.41421,1.41421]
        
        def check_nav(point):
            return downsampled_map[point[0], point[1]]
        
        def check_valid(point):
            if (0<=point[0]<dims[0]) and (0<=point[1]<dims[1]):
                return True
            return False
            
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
                    if not check_nav((new_point[0]+xs[j]*radius, new_point[1]+ys[j]*radius)):
                        flag = True
                        break
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
        tar_point = (int(tar_point[0] * self.scale_percent), int(tar_point[1] * self.scale_percent))
        def find_path(v, path):
            if self.prevs[v]>0:
                path.append(self.get_coords_fn(self.prevs[v]))
                find_path(self.prevs[v], path)
        path = [tar_point]
        find_path(self.get_graph_id_fn(tar_point), path)
        return (np.array(path[::-1]) / self.scale_percent).astype(int)