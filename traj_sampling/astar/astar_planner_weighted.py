from .astar_base import AStar
import math
import numpy as np
from scipy.spatial import distance
import copy
from heapq import heappush, heappop, heapify
from scipy.ndimage import label
from scipy import ndimage

def surrounding_3Dpos(radius, num_split=None, sorted_type='ascending', shape='cube'):
    """
    Args:
        radius: max radius of point relative to center
        num_split: number of splits on one axis
        sorted_type: from centor to bound use 'ascending'
    """
    if num_split is None:
        num_split = 2*int(radius) + 1
    x = np.linspace(-radius, radius, num_split).astype(int)
    y = np.linspace(-radius, radius, num_split).astype(int)
    z = np.linspace(-radius, radius, num_split).astype(int)
    xv, yv, zv = np.meshgrid(x, y, z)

    coord_pairs = list(zip(xv.flatten(), yv.flatten(), zv.flatten()))
    
    if sorted_type == 'ascending':
        if shape=='cube':
            out_pairs = sorted(coord_pairs, key=lambda k: k[0]**2 + k[1]**2 + k[2]**2 )
        elif shape == 'sphere':
            out_pairs = sorted(coord_pairs, key=lambda k: k[0]**2 + k[1]**2 + k[2]**2)[ : int(len(coord_pairs) * 0.52)+1] # 0.52 ratio of points is inside the sphere
    else:
        raise NotImplementedError()
    return out_pairs

def process_surrounding(center, rel_sur_pts, shape):
    pts = np.array(center) + np.array(rel_sur_pts)
    pts[:,0] = np.clip(pts[:,0], 0, shape[0]-1)
    pts[:,1] = np.clip(pts[:,1], 0, shape[1]-1)
    pts[:,2] = np.clip(pts[:,2], 0, shape[2]-1)
    return np.unique(pts, axis=0)

def pt_w_with_decay(x, decay=1):
    y = 1 - np.power(1/decay, x)
    return y

def decay_schedular(obj_seq_raw, gravity_range = (1.1, 4), gravity_type='uniform'):
    obj_seq=copy.deepcopy(obj_seq_raw)
    unique_list = [] # apply most gravity => apply least gravity
    while obj_seq != []:
        obj = obj_seq.pop()
        if obj in unique_list:
            continue
        unique_list.append(obj)

    if unique_list == []:
        return []
    
    res_list = [(unique_list[0], gravity_range[1])]
    if len(unique_list) == 1:
        return res_list

    if gravity_type == 'uniform':
        step = (gravity_range[1] - gravity_range[0])/(len(unique_list)-1)
        for obj in unique_list[1:]:
            res_list.append((obj, gravity_range[1]-step))
        return res_list
    else:
        raise NotImplementedError()

def get_geo_dist(path, map_resolution):
    """ return geodistance in meter"""
    dists = np.linalg.norm(path[1:] - path[:-1], ord=2, axis=1) * map_resolution
    return np.sum(dists)

def map_expansion(nav_map, expansion_width=0.5, map_resolution=0.05):
    expansion_step = int(expansion_width/map_resolution)
    one_layer = np.ones((expansion_step*2+1, expansion_step*2+1), dtype=bool)
    stru = np.zeros((expansion_step*2+1, expansion_step*2+1, expansion_step*2+1), dtype=bool)
    stru[:, expansion_step, :] = one_layer
    cspace = ndimage.binary_dilation(nav_map==False, stru)
    return cspace == False

class astar_planner_weighted(AStar):
    """ modified from python-astar lib """

    def __init__(self, nav_map, weight_map, same_start=False, expansion_map=None):
        self.cspace = nav_map
        self.connected_components, self.num_components = label(self.cspace, structure=np.ones((3,3,3)))
        #print("Number of components:", self.num_components)

        self.weight_map = weight_map
        pairs = surrounding_3Dpos(1, num_split=3)
        pairs.remove((0,0,0))
        self.sur_coords = pairs

        if expansion_map is None:
            indices = np.nonzero(self.cspace>0) # -1 is obstacle point 0 is void point
        else:
            indices = np.nonzero(expansion_map) # -1 is obstacle point 0 is void point
        self.valid_points = np.transpose(np.stack((indices[0], indices[1], indices[2])))
        
        self.same_start = same_start
        if self.same_start:
            self.searchNodes = AStar.SearchNodeDict()

    def distance_between(self, n1, n2):
        nx,ny,nz = n2
        return  distance.euclidean(n1, n2) * self.weight_map[nx][ny][nz]

    def heuristic_cost_estimate(self, current, goal):
        dx = abs(current[0]-goal[0])
        dy = abs(current[1]-goal[1])
        dz = abs(current[2]-goal[2])
        dmin, dmid, dmax = sorted([dx, dy, dz])
        # sqrt(3) = 1.73205 sqrt(2) = 1.41421
        return 1.73205 * dmin + 1.41421 * (dmid - dmin) + 1 * (dmax - dmid)

    def closest_navigable(self, pt, use_center_component=False):
        if not use_center_component:
            dists = np.linalg.norm(self.valid_points - np.array(pt), ord=2, axis=1)
            idx = np.argmin(dists, axis=0)
            return tuple(self.valid_points[idx])
        else:
            indices = np.nonzero(self.connected_components == self.center_component)
            valid_points = np.transpose(np.stack((indices[0], indices[1], indices[2])))
            dists = np.linalg.norm(valid_points - np.array(pt), ord=2, axis=1)
            idx = np.argmin(dists, axis=0)
            return tuple(valid_points[idx])

    def is_free(self, x,y,z):
        return (x>=0) and (x < self.cspace.shape[0]) and (y>=0) and (y < self.cspace.shape[1])\
             and (z>=0) and (z < self.cspace.shape[2]) and (self.cspace[x][y][z]>0)

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        neighbor_list = []
        for (i,j,k) in self.sur_coords:
            candidate = (node[0]+i, node[1]+j, node[2]+k)
            if self.is_free(*candidate):
                neighbor_list.append(candidate)
        return neighbor_list

    def check_inside_cc(self, p1, p2):
        return self.connected_components[p1[0], p1[1], p1[2]] == self.connected_components[p2[0], p2[1], p2[2]]
    
    def set_center_component(self, i):
        self.center_component = i

    def solve(self, start, goal, reversePath=False):
        if not self.is_free(*start):
            start = self.closest_navigable(start)

        self.center_component = self.connected_components[start[0], start[1], start[2]]
        if not (self.is_free(*goal) and self.check_inside_cc(start, goal)):
            goal = self.closest_navigable(goal, use_center_component=True) # keep goal position in the same component
        
        if self.num_components > 1:
            if not self.check_inside_cc(start, goal):
                return None

        if self.is_goal_reached(start, goal):
            return [start]
        
        if self.same_start and (start in self.searchNodes): # multiround search, but only support same start
            startNode = self.searchNodes[start]
            if goal in self.searchNodes and self.searchNodes[goal].closed:
                return self.reconstruct_path(self.searchNodes[goal], reversePath)
    
            for v in self.openSet:
                v.fscore = self.heuristic_cost_estimate(v.data, goal)
            heapify(self.openSet)
        else:
            self.openSet = []
            self.searchNodes = AStar.SearchNodeDict()
            startNode = self.searchNodes[start] = AStar.SearchNode(
                start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
            heappush(self.openSet, startNode)

        while self.openSet:
            current = heappop(self.openSet)
            if self.is_goal_reached(current.data, goal):
                heappush(self.openSet, current)
                return self.reconstruct_path(current, reversePath)

            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: self.searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue

                tentative_gscore = current.gscore + \
                    self.distance_between(current.data, neighbor.data)

                if tentative_gscore >= neighbor.gscore:
                    continue
                # tesntaive_gscore < neighbor.gscore, record better came_from
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(self.openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    self.openSet.remove(neighbor)
                    heappush(self.openSet, neighbor)
        