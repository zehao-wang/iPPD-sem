scene_dataset=../data/scene_datasets/mp3d
out_root="../out/rand_{split}"
out_root2="../out/rand_{split}"
input_root="../data/R2R_VLNCE_v1-3_preprocessed"

python gen_rand_traj.py \
--scene_datasets ${scene_dataset} \
--split train \
--annt_root ${input_root} \
--out_root ${out_root}

python convert_rand_to_h5.py --input_dir ${out_root} \
--out_root ${out_root2} --split train