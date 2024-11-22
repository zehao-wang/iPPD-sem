import open3d as o3d
import numpy as np
from .map_tools import voxmap2points
import copy

class visualizer(object):
    def __init__(self, nav_map) -> None:
        """
        Args:
            nav_map: bool occupancy map for navigable area
        """
        self.nav_map = (nav_map == 1)
        self.nav_pcd = voxmap2points(self.nav_map)

    def draw_traj(self, traj, dump_path=None, path_color=[1,0,0], agent_dir=None):
        """
        Args:
            traj: list of map pt
        """
        pcd = copy.deepcopy(self.nav_pcd)
        for i, pt in enumerate(traj):
            pcd.points.extend(np.array([[pt[0], pt[1]+2, pt[2]]], dtype=float)) # add height in the y axis
            if i ==0:
                pcd.colors.extend(np.array([[0,0,1]])) # blue for start pt
            else:
                pcd.colors.extend(np.array([path_color])) # red for traj pt
        
        if agent_dir is not None:
            dir_vec = np.vstack([
                        np.linspace(traj[0][0], traj[0][0] + agent_dir[0]*10, 50),
                        np.linspace(traj[0][1]+2, traj[0][1]+2 + agent_dir[1]*10, 50),
                        np.linspace(traj[0][2], traj[0][2] + agent_dir[2]*10, 50),
                    ]
                )
            dir_vec = np.transpose(dir_vec)[2:]
            pcd.points.extend(dir_vec)
            pcd.colors.extend(np.array([[0,1,0] for i in range(dir_vec.shape[0])]))

        if dump_path is None:
            return 
        elif dump_path == 'draw':
            voxmap = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
            o3d.visualization.draw_geometries([voxmap])
        else:
            o3d.io.write_point_cloud(dump_path, pcd)

    def draw_trajs(self, trajs, dump_path=None):
        """
        Args:
            trajs: list of trajs
        """
        pcd = copy.deepcopy(self.nav_pcd)
        for traj in trajs:
            for i, pt in enumerate(traj):
                pcd.points.extend(np.array([[pt[0], pt[1]+2, pt[2]]], dtype=float)) # add height in the y axis
                if i ==0:
                    pcd.colors.extend(np.array([[0,0,1]])) # blue for start pt
                else:
                    pcd.colors.extend(np.random.rand(1,3)) # red for traj pt
            
        if dump_path is None:
            voxmap = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
            o3d.visualization.draw_geometries([voxmap])
        else:
            o3d.io.write_point_cloud(dump_path, pcd)