work_path=$(dirname $0)
now=$(date +%s)
model_name=unseen_as_eval

TOKENIZERS_PARALLELISM=false PYTHONPATH=./ python -u \
run.py --exp-config configs/mlnv3_gtobj_eval_unseen.yaml \
--run-type train \
LOG_DIR ../out/scoring_model/${model_name}_eval/map \
EVAL_PATH "../data/exps_snap/ckpts/full_model_gt/checkpoints/epoch=7-step=89119.ckpt" \
DATASET.EVAL.NAME "val_unseen" \
DATASET.EVAL.num_workers 6 \
DATASET.EVAL.batch_size 64 \
DATASET.annt_root '../data/exps_snap/merged_act_seq_v3/{split}/{split}'