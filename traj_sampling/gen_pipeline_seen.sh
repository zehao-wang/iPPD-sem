split=val_seen
# export API_KEY=""
# export ORGANIZATION=""

# Gen rand path
rm -rf ../out/rand_${split}
scene_dataset=../data/scene_datasets/mp3d
out_root_rand="../out/rand_{split}"
input_root="../data/R2R_VLNCE_v1-3_preprocessed"

python gen_rand_traj.py \
--scene_datasets ${scene_dataset} \
--split $split \
--annt_root ${input_root} \
--out_root ${out_root_rand}

# Gen particle path
rm -rf ../out/$split
python main.py --split $split --out_dir ../out

rm -rf ../out/particle_${split}
PYTHONPATH=./:$PYTHONPATH python post_processing.py \
--split ${split} \
--input_dir "../out/{split}" \
--out_dir "../out/particle_{split}" \
--num_process 16

mkdir ../out/merged_act_seq
rm -rf ../out/merged_act_seq/${split}
mkdir ../out/merged_act_seq/${split}

# Merge both parts (rand + particle)
PYTHONPATH=./:$PYTHONPATH python convert_to_mergedh5.py --split $split \
--out_root "../out/merged_act_seq" \
--input_dir "../out/{split}_post_processed" \
--input_dir2 ${out_root_rand}

python revise.py --revise_dir "../out/merged_act_seq/{split}/{split}" \
--split $split

# [optional] filter by obj sequence
# python filter_dataset.py