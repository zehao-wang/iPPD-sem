scene_root="../data/scene_datasets/mp3d"
PYTHONPATH=../external_lib/Mask2Former python -u main.py --l 0 --h 90 \
--scene_root $scene_root \
--dataset ./meta_data/mp3d.scene_dataset_config.json 