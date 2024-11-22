"""
In this script, we only keep the path if objact is detected
for no-objact episode, we keep all random path
"""
import h5py
import jsonlines
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm 
import pickle
from random import random
import os
import os.path as osp
def find_all_ext(root_path, root_path2, ext):
  counter = Counter()
  ep_set = set()
  paths = []
  for root, dirs, files in os.walk(root_path, topdown=True):
      for name in files:
          file_path = os.path.join(root, name)
          if file_path.endswith(ext):
              with jsonlines.open(file_path) as reader:
                local_cnt = 0
                for item in reader:
                    local_cnt+=1
                    if local_cnt>10:
                        break

                if local_cnt < 10:
                    print(f"[WARNING] skip {file_path} since less than 10 proposed paths")
                    continue

              ep_set.add(name)
              paths.append(file_path)
              counter["propose"] += 1

  for root, dirs, files in os.walk(root_path2, topdown=True):
      for name in files:
          file_path = os.path.join(root, name)
          if file_path.endswith(ext):
              if name in ep_set:
                continue
              paths.append(file_path)
              counter["random"] += 1
  print(counter)
  return paths
  
parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    choices=["train", "val_seen", "val_unseen"],
    required=True
)

parser.add_argument("--out_root", default="../out/merged", type=str)
parser.add_argument("--input_dir", default='../out/particle_{split}', type=str, help="traj from particle")
parser.add_argument("--input_dir2", default='../out/rand_{split}', type=str, help="random traj")
args = parser.parse_args()

def list_sub(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3], a[4], a[5]]

root = args.input_dir.format(split=args.split)
root2 = args.input_dir2.format(split=args.split)
out_path = osp.join(args.out_root, f'{args.split}/{args.split}.h5')
print("Output to ", out_path)
out_path_ep_id_list = osp.join(args.out_root, f'{args.split}/{args.split}.pkl')
ep_id_list = []

counter = defaultdict(int)
cnt = 0
f = h5py.File(out_path, "w")
file_list = find_all_ext(root, root2, 'jsonl')
for file in tqdm(file_list):
    data_list = []
    with jsonlines.open(file) as reader:
        local_cnt = 0
        for item in reader:
            data_list.append(item)
            local_cnt+=1

        if local_cnt < 10:
            print(f"[WARNING] {file} as less than 10 proposed paths")
    
    for b in data_list:
        if args.split == "train" and b['target'] == 0:
            if random() < 0.8:
                continue
 
        counter[b['target']] +=1
            
        grp = f.create_group(b['traj_id'])
        ep_id_list.append(b['traj_id'])

        grp.attrs['ep_id'] = b['ep_id']
        # original trajectory id (1 traj corresponding to 3 or more episode)
        grp.attrs['annt_traj_idx'] = b['annt_traj_idx'] 
        try:
            grp.attrs['instruction'] = b['instruction']
        except:
            grp.attrs['instruction'] = b['instruction']['instruction_text']
        grp.attrs['scene'] = b['scene']
        grp.attrs['target'] = b['target']
        grp.attrs['ndtw'] = b['ndtw']
        grp.attrs['dtw'] = b['dtw']
        cam_pos = [bb['sim_loc'] for bb in b['cam_poses']]
        cam_rot = [bb['sim_rot'] for bb in b['cam_poses']]
        grp.create_dataset("cam_pos", data=cam_pos)
        grp.create_dataset("cam_dir", data=cam_rot)

        # grp.create_dataset("agent_rot", data=[bb['agent_rot'] for bb in b['cam_poses']])
        # grp.create_dataset("points_list", data=b['points_list'])

        if b['actions'][-1] is None:
            b['actions'] = b['actions'][:-1]
        grp.create_dataset("actions", data=b['actions'])

        cnt += 1


with open(out_path_ep_id_list, 'wb') as handle:
    pickle.dump(ep_id_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(counter)
print(f"{cnt} data processed")