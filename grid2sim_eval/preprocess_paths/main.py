import h5py
import jsonlines
from tqdm import tqdm
import habitat_sim
from collections import defaultdict, Counter
from habitat_sim import ShortestPath
import gzip
import json

def combine_results(dset_h5_path, route_path):
    """
    {"episode_id": "1723", "traj_id": "1723-act-99", "pred": 0.7312865853309631, 
    "is_correct": false, "ndtw": 0.34734842486082584, "dist_score": 0, 
    "scene_name": "QUCTc6BB5sX"}
    """
    scene2datum = defaultdict(dict)
    data = h5py.File(dset_h5_path, 'r')
    with jsonlines.open(route_path, 'r') as reader:
        for line in tqdm(reader):
            traj_id = line['traj_id']
            ep_id = line['episode_id']
            scene_name = line['scene_name']

            h5_datum = data[traj_id]
            traj = h5_datum['cam_pos_fixed'][()]
            scene2datum[scene_name][ep_id] = {"sim_path" : traj}
    data.close()
    return scene2datum

def load_scene(scene_id):
    pf = habitat_sim.PathFinder()
    nav_mesh_file = "../../data/scene_datasets/mp3d/{scene_id}/{scene_id}.navmesh"
    # nav_mesh_file="/media/zeke/pssd/UbuntuData/3D/habitat/scene_datasets/mp3d/{scene_id}/{scene_id}.navmesh"
    pf.load_nav_mesh(nav_mesh_file.format(scene_id=scene_id))
    return pf

def check_navigable(pf, sim_start, sim_end):
    if not pf.is_navigable(sim_start):
        sim_start2 = pf.get_random_navigable_point_near(sim_start, radius=0.5)
        print(f"Change start pt {sim_start} to {sim_start2}")
        sim_start = sim_start2
    
    if not pf.is_navigable(sim_end):
        sim_end2 = pf.get_random_navigable_point_near(sim_end, radius=0.5)
        print(f"Change end pt {sim_end} to {sim_end2}")
        sim_end = sim_end2

    geodesic_distance = None

    spath = ShortestPath()
    spath.requested_start = sim_start
    spath.requested_end = sim_end
    if pf.find_path(spath):
        geodesic_distance = spath.geodesic_distance

    return geodesic_distance, sim_end.astype(float).tolist()

def load_raw_annt(split='val_unseen'):
    original_annt_path = '../../data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'
    ori_annt = original_annt_path.format(split=split)
    with gzip.open(ori_annt, 'rt') as f:
        meta = json.load(f)['episodes']

    # group by scene
    epid2start = dict()
    for ep in meta:
        epid2start[ep['episode_id']] = ep['start_position']
    return epid2start

if __name__ == '__main__':
    exp_name='std_unseen'
    split='val_unseen'
    dset_h5_path = f'../../data/exps_snap/merged_act_seq_v3/{split}/{split}.h5'
    import os
    if not os.path.exists(dset_h5_path):
        import ipdb;ipdb.set_trace() # breakpoint 75

    # route_path = f'/data/leuven/335/vsc33595/ProjectsVSC/iPPD/mln-trainer-v2/tmp/unseen_as_eval_eval/map/PointNet_Transformer_Glove/{exp_name}/best-routes.jsonl'
    # sub_folder='gtobj_50m'
    # route_path = f'../../../../data/exps_snap/unseen_as_eval_eval/{sub_folder}/PointNet_Transformer_Glove/version_0/best_routes.jsonl'
    route_path = f'./best_routes_{exp_name}_merged_act_seq_v3.jsonl'
    
    
    scene2datum = combine_results(dset_h5_path, route_path)
    epid2start = load_raw_annt(split)
    for scene, eps in scene2datum.items():
        pf = load_scene(scene)
        for ep_id, ep in eps.items():
            start_pt = epid2start[int(ep_id)]
            pts = [start_pt]
            # start_pt = ep['sim_path'][0]
            for pt in ep['sim_path'][1:]:
                geo_dist, pt_end = check_navigable(pf, start_pt, pt)
                if geo_dist is not None:
                    pts.append(pt_end)
            with jsonlines.open(f'./out_{exp_name}.jsonl', mode='a') as writer:
                writer.write(    
                    {"episode_id": ep_id, "scene_name": scene, "pts": pts}
                )
