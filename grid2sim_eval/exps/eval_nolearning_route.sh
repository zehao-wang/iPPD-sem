
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

work_path=$(dirname $0)
now=$(date +%s)
split='val_unseen'

PYTHONPATH=./ python -u run.py --exp-config config/no_learning.yaml \
TASK_CONFIG.DATASET.DATA_PATH "../data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.NDTW.GT_PATH "../data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz" \
TASK_CONFIG.DATASET.SPLIT $split \
EVAL.NONLEARNING.RESULT_PATH ./preprocess_paths/out_std_unseen_gtobj.jsonl \
EVAL.NONLEARNING.DSET_H5 ../data/exps_snap/merged_act_seq_v3/val_unseen/val_unseen.h5 \
EVAL.NONLEARNING.DUMP_DIR $work_path \
EVAL.SPLIT $split