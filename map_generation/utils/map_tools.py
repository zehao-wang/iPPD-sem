import open3d as o3d
import numpy as np
import quaternion
from collections import Counter
from .constants import stuff_colors
import pandas as pd
import pickle 
dt = 1

hfov = float(90) * np.pi / 180.

def sum_counter(row, sem_map):
    sem_map[(row[0], row[1], row[2])] += row['counter']

def accum_sem(stacked_np, sem_map):
    data = pd.DataFrame(stacked_np)
    count_series = data.groupby([0,1,2])[3].agg(Counter).reset_index(name='counter')
    count_series.apply(lambda row : sum_counter(row, sem_map), axis = 1)

def depth2points(depth_im, sem_info, sim_pos, sim_rot):
    H, W = depth_im.shape
    vfov = 2 * np.arctan(np.tan(hfov/2)*H/W)
    fl_x = W / (2 * np.tan(hfov / 2.)) # 320
    fl_y = H / (2 * np.tan(vfov / 2.)) # 320

    cx=320
    cy=240

    # xs, ys = np.meshgrid(np.arange(W), np.arange(H-1,-1,-1))
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    depth = depth_im.reshape(1,H,W)
    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)
    
    xs = (xs - cx) / fl_x
    ys = (ys - cy) / fl_y

    xys = np.vstack((xs * depth , -ys * depth, -depth, np.ones(depth.shape)))
    msk = depth > 0
    msk = msk.flatten()
    xys = xys.reshape(4, -1)
    xy_c0 = xys[:, msk]

    sems = sem_info[:,:]
    sems = sems.reshape(-1, 1)
    sems = sems[msk, :]

    translation = np.array(sim_pos)
    orientation = sim_rot
    rotation_0 = quaternion.as_rotation_matrix(orientation)
    T_world_camera0 = np.eye(4) # transformation matrix cam -> world, i.e. inv of extrinsic
    T_world_camera0[0:3,0:3] = rotation_0
    T_world_camera0[0:3,3] = translation

    # Finally transform actual points
    pcd = np.matmul(T_world_camera0, xy_c0)
    return np.transpose(pcd)[:,:3], sems

def merge_sem_map(sem_map, exemption_label=[4]): # 32 for gt ceiling, 4 for mask2former ceiling
    pts = []
    labels = []
    for k,v in sem_map.items():
        tmp_cnts = v.most_common(2)
        label, cnt = tmp_cnts[0]
        val_sum = sum(v.values())
        if label == -100:
            if len(tmp_cnts) == 1:
                continue
            val_sum -= cnt
            label, cnt = tmp_cnts[1]

        if label in exemption_label:
            continue
        common_item_ratio = cnt/val_sum
        # NOTE: two manually defined threshold, one for vote ratio one for total voting
        # Majority voting, variation of max-pool
        if common_item_ratio > 0.5 and val_sum > 5: # Counter object
            pts.append(k)
            labels.append(label)
    pts = np.array(pts)
    return pts, labels

def dump_3dmap(pcd_nps, labels, dump_dir=None, resolution=None):
    out_map = {'map': pcd_nps, 'labels': labels, 'resolution': resolution}
    with open(dump_dir, 'wb') as handle:
        pickle.dump(out_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # NOTE: this is an example of how to use tree for fast search nearby points
    # pcd_tree = o3d.geometry.KDTreeFlann(pcd_o3d)
    # [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd_o3d.points[point], 5)
    # points = np.asarray(pcd_o3d.points)[idx[1:], :]



def update_map(depth, sem, sim_pos, sim_rot, sem_map, resolution=0.1):
    """
    Input depth sem sim_pos, sim_rot 
    update sem_map
    return None
    """
    pcd_np, sems = depth2points(depth, sem, sim_pos, sim_rot)
    # [NOTE] simulate if no groud truth start location
    # voxelized_pts = (pcd_np - start_pos)/resolution
    # [NOTE] directly use known gps location
    voxelized_pts = pcd_np/resolution
    voxelized_pts = voxelized_pts.astype(int)
    
    sems = sems.astype(int)
    stacked_np = np.hstack((voxelized_pts, sems))
    accum_sem(stacked_np, sem_map)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_root', type=str, default='/media/zeke/project_data/Projects/torch-ngp/my_local_process/run_sim/test_output-bk')
#     parser.add_argument('--output_root', type=str, default='./')
#     parser.add_argument('--scene_id', type=str, default='TESTSCENE')
#     parser.add_argument('--resolution', type=float, default=0.1, help="resollution in meter")
#     args = parser.parse_args()
#     # input_dir = os.path.join(args.input_dir, args.scene_id)
#     input_dir = args.input_root
#     os.makedirs(args.output_root, exist_ok=True)
#     output_dir = os.path.join(args.output_root, args.scene_id + '_sem.plk')
#     tic = time.time()
#     run(input_dir, output_dir, args.resolution)
#     print(f"The processing time of build map {args.scene_id} is {time.time()-tic} seconds")
