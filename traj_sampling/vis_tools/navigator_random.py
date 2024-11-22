"""
This script only for examing different strategy, separate from main pipeline
"""
import math
import numpy as np
import random

from dataclasses import dataclass
class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def tolist(self):
        return [self.x, self.y, self.z]
        
@dataclass(frozen=True)
class Noise:
    forward: float = 0.
    turn: float = 0.
    sense: float = 0.

def get_line(pt1, pt2, num_pts=50):
    x = np.round(np.linspace(pt1[0], pt2[0], num_pts)).astype(int)
    y = np.round(np.linspace(pt1[1], pt2[1], num_pts)).astype(int)
    z = np.round(np.linspace(pt1[2], pt2[2], num_pts)).astype(int)
    return np.transpose(np.vstack([x,y,z]))

def check_valid(pt, nav_map, height_range=None): 
    """
    Args:
        pt: 3d map coord
        nav_map: vox map
    """
    if (pt.x < 0) or (pt.x >= nav_map.shape[0]) or (pt.y < 0) or (pt.y >= nav_map.shape[1]) or (pt.z < 0) or (pt.z >= nav_map.shape[2]):
        return False

    if not nav_map[pt.x][pt.y][pt.z]:
        search_range = [i-pt.y for i in range(nav_map.shape[1])]
        search_range = sorted(search_range, key=lambda x: abs(x))
        for i in search_range:
            if height_range is not None and abs(i) > height_range:
                continue

            if nav_map[pt.x][pt.y + i][pt.z]:
                # print(f"[WARNING] probably find stairs, change height from {pt.y} to {pt.y + i}")
                pt.y = int(pt.y) + i
                return True
        return False
    return True

# def check_valid(pt, nav_map, height_range=None): 
#     """
#     Args:
#         pt: 3d map coord
#         nav_map: vox map
#     """
#     if (pt.x < 0) or (pt.x >= nav_map.shape[0]) or (pt.y < 0) or (pt.y >= nav_map.shape[1]) or (pt.z < 0) or (pt.z >= nav_map.shape[2]):
#         return False

#     return True

def check_valid_xyz(pt, nav_map, height_range): 
    x,y,z = pt
    if (x < 0) or (x >= nav_map.shape[0]) or (y < 0) or (y >= nav_map.shape[1]) or (z < 0) or (z >= nav_map.shape[2]):
        return False

    if not nav_map[x][y][z]:
        search_range = [i-y for i in range(nav_map.shape[1]) if abs(i-y) > height_range ]
        search_range = sorted(search_range, key=lambda x: abs(x))
        for i in search_range:
            if nav_map[x][y + i][z]:
                return True
        return False
    return True
    

class RobotState: 
    def __init__(self, 
                 point: Point, 
                 angle, 
                 map_resolution, 
                 left_turn_range = [0, np.pi], # in radian
                 right_turn_range=[0, np.pi], # in radian
                 forward_range= [0, 5], # in meter
                 ) -> None:
        self.point = point
        self.left_turn_range = left_turn_range
        self.right_turn_range = right_turn_range
        self.forward_range = forward_range
        
        self.map_resolution = map_resolution
        
        self._angle = angle
        self.traj = [tuple(self.point.tolist())]
        self.full_traj = []
        
        # if self._angle:
        #     if not 0 <= self._angle <= 2 * math.pi:
        #         raise ValueError(f'Angle must be within [{0.}, {2 * math.pi}, '
        #                          f'the given value is {angle}]')

    def __copy__(self) -> 'RobotState':
        return type(self)(self.point, self._angle, self.map_resolution, self.left_turn_range, self.right_turn_range, self.forward_range)
    
    def get_orientation(self):
        turning_dir = np.array([math.cos(self._angle), 0, math.sin(self._angle)])
        return turning_dir/np.linalg.norm(turning_dir)
    
    def add_noise(self):
        # self._angle += random.gauss(0., 0.5)
        # self._angle%= 2 * math.pi
        # self._angle += random.gauss(0., 1.)
        # self._angle%= 2 * math.pi
        pass

    def move(self, action, nav_map) -> None:
        forward = np.random.uniform(0, 10)/self.map_resolution
        if action == 'forward':
            turn = np.random.uniform(0, 2 * math.pi)
        else:
            raise ValueError(f"Invalid action {action}")

        if forward < 0.:
            raise ValueError('RobotState cannot move backwards')
    
        # turn, and add randomness to the turning command
        angle = self._angle + turn
        angle %= 2 * math.pi
    
        # move, and add randomness to the motion command
        x = self.point.x + math.cos(angle) * forward
        z = self.point.z + math.sin(angle) * forward
    
        self.point = Point(int(x), int(self.point.y), int(z))
        self._angle = angle

        if check_valid(self.point, nav_map):
            self.traj.append(tuple(self.point.tolist()))
            return True
        else:
            return False
        

    