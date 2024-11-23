import argparse
import torch
from pytorch_lightning import loggers as pl_loggers
from configs.default import get_cfg_defaults
import random
import numpy as np
from trainer import BaseTrainer
import pprint 
import logging
import os


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """

    config = get_cfg_defaults()
    config.defrost()
    config.merge_from_file(exp_config)
    if opts:
        config.merge_from_list(opts)
    config.freeze()

    tb_logger = pl_loggers.TensorBoardLogger(config.LOG_DIR, name=config.TRAINER.MODEL_NAME)
    # NOTE: official solution for logging on console, file handler is declared in run.py
    # configure logging at the root level of lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    os.makedirs(tb_logger.log_dir)
    logger.addHandler(logging.FileHandler(f"{tb_logger.log_dir}/core.log"))
    logger.info("CONFIG: ")
    logger.info(pprint.pformat(config, indent=4))
    logger.info("="*20)

    logger.info(f"using seed {config.SEED}")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    trainer = BaseTrainer(config, run_type, logger=tb_logger)
    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        print('\033[92m'+f"Evaluating for dataset split {config.DATASET.EVAL.NAME}"+'\033[0m')
        trainer.eval(config.EVAL_PATH)
    elif run_type == "inference":
        print('\033[92m'+f"Inference on dataset split {config.DATASET.TEST.NAME}"+'\033[0m')
        trainer.inference(config.EVAL_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))

if __name__ == "__main__":
    main()