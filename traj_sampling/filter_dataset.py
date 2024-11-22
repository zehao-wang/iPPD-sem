from tqdm import tqdm
from utils.data_loader import MLNDset
from collections import Counter
import pickle
import jsonlines

if __name__ == '__main__':
    split='val_seen'
    radius=3

    annt_root='../out/merged_act_seq/{split}/{split}'
    out_root='../out/merged_act_seq_pruned/{split}/{split}'
    original_annt_path='../data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'
    map_root = '../out/out_obs'
    instruction_obj_dict = {}
    with jsonlines.open(f'../data/scene_datasets/obj_seq_{split}.jsonl') as f:
        for obj in f:
            instruction_obj_dict[obj['ep_id']] = obj

    dset = MLNDset(annt_root, map_root, original_annt_path, split=split, radius=radius)

    data_keys = []    
    full_set = set()
    set_has_prune = set()
    counter = Counter()
    for i in tqdm(range(len(dset))):
        ins_objs, label_list, traj_id = dset.get_next(instruction_obj_dict=instruction_obj_dict)
        assert traj_id is not None
        ep_id = int(traj_id.split('-')[0])
        full_set.add(ep_id)

        matching_ratio = dset.check_matching(ins_objs, label_list)
        if matching_ratio > 0.5:
            pruned=False
            counter["kept"] += 1
            data_keys.append(traj_id)
            set_has_prune.add(ep_id)
        else:
            pruned=True
            counter["pruned"] += 1

    # Add back those no matching episodes
    add_back_set = set.difference(full_set, set_has_prune)
    for traj_id in dset.data_keys:
        ep_id = int(traj_id.split('-')[0])
        if ep_id in add_back_set:
            data_keys.append(traj_id)

    print(counter)
    try:
        with open(out_root.format(split=split) + '.pkl', 'wb') as f:
            pickle.dump(data_keys, f, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        import ipdb;ipdb.set_trace() # breakpoint 59
        print("Try to create folder")