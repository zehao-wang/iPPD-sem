# Instruction-Guided Path Planning with 3D Semantic Maps for Vision-Language Navigation

The source code for pipeline iPPD of language-guided navigation on semantic map.

## Preparation

### A. Prepare Data

#### Matterport3D Scenes (following [vlnce repo](https://github.com/jacobkrantz/VLN-CE/tree/e41ffc9ea6194655fa13f59e27f0868c4c67207a?tab=readme-ov-file))
Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script (download_mp.py) can be accessed by following the instructions on their project webpage. The 90 scene data can then be downloaded:
```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

#### VLNCE Dataset

Please download from [link](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view) and extract to ```data``` folder

### B. Preparing Env
```bash
conda create -n mp3d python=3.9 cmake=3.14.0
conda activate mp3d
```

#### install necessary packages
```bash
# for old version habitat-sim, recommand to use mamba
conda install -n base -c conda-forge mamba

conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

mamba install habitat-sim=0.2.4 headless -c conda-forge -c aihabitat

pip install open3d==0.14.1

```

#### place Mask2Former
```bash
mkdir external_lib 
cd external_lib
pip install timm
git clone https://github.com/facebookresearch/Mask2Former.git
cd Mask2Former
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

## Full procedure
The intermediate data for different step can be found in [link]() for convinient reproduction.

### 1. Pre-exploration Phase and Semantic Map Constructor 
<details>
<summary>Details</summary>

#### a. download mask2former ckpt
Please place the checkpoint [link](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl) under ```map_generation/meta_data/mask2former_ckpt```
```bash
cd map_generation/meta_data/mask2former_ckpt
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl

```

#### b. run the script

```bash
sh run_recon.sh
```
Here is a sample of reconstructed semantic map
![sem map](./assets/semmap_sample.png)

</details>

### 2. Trajectory sampling

<details>
<summary>Details</summary>

#### place the processed navmap based on semmap

Move the ```preprocessed_navmap``` folder to ```data/preprocessed_navmap```. This is used as a navigation map.

#### set OpenAI keys

```bash
export API_KEY="..."
export ORGANIZATION="..."
```

#### generating data
Random paths are enough for training trajectories. The validation trajectory should be proposed in two parts, first from random trajectories, another part from particle sampled trajectories.

```bash
cd traj_sampling
sh gen_train.sh
sh gen_pipeline_seen.sh
sh gen_pipeline_unseen.sh

```

</details>

### 3. Model training
<details>
<summary>Details</summary>

```bash
cd scoring_trainer

# training and evaluation with merged object list requires less reproduction effort 
sh scripts/train_gtobj_eval_unseen_cat.sh # train
sh scripts/eval_gtobj_eval_unseen_cat.sh  # eval 

```

#### evaluate on habitat-sim

Currently only support evaluation on habitat-sim v0.2.2, the config system has huge change due to the update.

```bash
cd grid2sim_eval/preprocess_paths
# modify the prediction file in main.py
# then
python main.py

```

```bash
# modify the input config RESULT_PATH and evaluate the results
sh exps/eval_nolearning_route.sh

```

TODO:

- [ ]  adapt the config of no learning agent to higher habitat-sim


</details>