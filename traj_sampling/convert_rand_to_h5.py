from random import random
import os
import os.path as osp
def find_all_ext(root_path, ext):
  paths = []
  for root, dirs, files in os.walk(root_path, topdown=True):
      for name in files:
          file_path = os.path.join(root, name)
          if file_path.endswith(ext):
              paths.append(file_path)
  return paths
  
import h5py
import jsonlines
import argparse
from collections import defaultdict
from tqdm import tqdm 
import pickle
import numpy as np
import quaternion
parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    choices=["train", "val_seen", "val_unseen"],
    required=True
)

parser.add_argument("--out_root", default="../out/rand_{split}", type=str)
parser.add_argument("--input_dir", default="../out/rand_{split}", type=str)
args = parser.parse_args()


root = args.input_dir.format(split=args.split)
out_path = osp.join(args.out_root.format(split=args.split), f'{args.split}.h5')
out_path_ep_id_list = osp.join(args.out_root.format(split=args.split), f'{args.split}.pkl')
ep_id_list = []

counter = defaultdict(int)
cnt = 0
f = h5py.File(out_path, "w")
file_list = find_all_ext(root, 'jsonl')
for file in tqdm(file_list):
    with jsonlines.open(file) as reader:
        for i, b in enumerate(reader):
            if i%10 != 0: # tmp for seen
                continue

            counter[b['target']] +=1

            # NOTE: uncomment if evaluate on less training data
            if args.split == "train" and b['target'] == 0:
                if random() < 0.8:
                    continue

            grp = f.create_group(b['traj_id'])
            ep_id_list.append(b['traj_id'])

            grp.attrs['ep_id'] = b['ep_id']
            # original trajectory id (1 traj corresponding to 3 or more episode)
            grp.attrs['annt_traj_idx'] = b['annt_traj_idx'] 
            grp.attrs['instruction'] = b['instruction']['instruction_text']
            grp.attrs['scene'] = b['scene']
            grp.attrs['target'] = b['target']
            grp.attrs['ndtw'] = b['ndtw']
            grp.attrs['dtw'] = b['dtw']
            cam_pos = [bb['sim_loc'] for bb in b['cam_poses']]
            cam_rot = [bb['sim_rot'] for bb in b['cam_poses']]
            grp.create_dataset("cam_pos", data=cam_pos)

            cam_dir = []
            for rot in cam_rot:
                qrot = quaternion.from_float_array(rot)
                qrot = quaternion.rotate_vectors(qrot, np.array([0,0,-1.]))
                qrot = qrot / np.linalg.norm(qrot) # unit direction vector
                cam_dir.append(qrot)

            grp.create_dataset("cam_dir", data=cam_dir)
            if b['actions'][-1] is None:
                b['actions'] = b['actions'][:-1]
            grp.create_dataset("actions", data=b['actions'])

            cnt += 1


with open(out_path_ep_id_list, 'wb') as handle:
    pickle.dump(ep_id_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(counter)
print(f"{cnt} data processed")
f.close()