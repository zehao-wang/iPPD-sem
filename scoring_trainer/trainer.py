import os
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import models
import dataset

class BaseTrainer:
    def __init__(self, config, run_type, logger=None):
        self.model_config = config.TRAINER.MODEL
        self.model = getattr(models, config.TRAINER.MODEL_NAME)(config.TRAINER.MODEL)
        checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="success_rate", mode='max')
        
        if run_type == 'train':
            train_dset = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.TRAIN.NAME)
            self.train_loader = DataLoader(train_dset, batch_size=config.DATASET.TRAIN.batch_size, num_workers=config.DATASET.TRAIN.num_workers, shuffle=True, drop_last=True, collate_fn=train_dset.collate_fc)
            valid_dset = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.EVAL.NAME)
            self.valid_loader = DataLoader(valid_dset, batch_size=config.DATASET.EVAL.batch_size, num_workers=config.DATASET.EVAL.num_workers, shuffle=False, drop_last=False, collate_fn=valid_dset.collate_fc)
            # whole_set = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.TRAIN.NAME)
            # train_len = int(len(whole_set) * 0.9)
            # train_dset, valid_dset = random_split(whole_set, [train_len, len(whole_set)-train_len])
            # self.train_loader = DataLoader(train_dset, batch_size=config.DATASET.TRAIN.batch_size, num_workers=config.DATASET.TRAIN.num_workers, shuffle=True, drop_last=True, collate_fn=whole_set.collate_fc)
            # self.valid_loader = DataLoader(valid_dset, batch_size=config.DATASET.EVAL.batch_size, num_workers=config.DATASET.EVAL.num_workers, shuffle=False, drop_last=False, collate_fn=whole_set.collate_fc)

        if run_type == 'inference':
            test_dset = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.TEST.NAME)
            self.test_loader = DataLoader(test_dset, batch_size=config.DATASET.TEST.batch_size, num_workers=config.DATASET.TEST.num_workers, shuffle=False, drop_last=False, collate_fn=test_dset.collate_fc)

        if run_type == 'eval':
            valid_dset = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.EVAL.NAME)
            self.valid_loader = DataLoader(valid_dset, batch_size=config.DATASET.EVAL.batch_size, num_workers=config.DATASET.EVAL.num_workers, shuffle=False, drop_last=False, collate_fn=valid_dset.collate_fc)
        
        # checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config.LOG_DIR, config.TRAINER.MODEL_NAME, "ckpts"), save_top_k=2, monitor="success_rate", mode='max')
        self.trainer = pl.Trainer(
            max_epochs=config.TRAINER.EPOCHS, progress_bar_refresh_rate=20, 
            default_root_dir=config.LOG_DIR,
            logger=logger,
            callbacks=[checkpoint_callback],
            gpus=1,
            check_val_every_n_epoch = 1
            # val_check_interval=config.TRAINER.EVAL_INTERVAL, # how oftern validation
        )

    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.valid_loader)

    def eval(self, ckpt_path):
        if os.path.isdir(ckpt_path):
            files = os.listdir(ckpt_path)
            for f in files:
                if f.endswith('ckpt'):
                    print(f"Evaluating checkpoint {os.path.join(ckpt_path,f)}, file exists: {os.path.exists(os.path.join(ckpt_path,f))}")
                    self.trainer.validate(
                        self.model, self.valid_loader, 
                        ckpt_path=os.path.join(ckpt_path,f)
                    )
        else:
            self.trainer.validate(self.model, self.valid_loader, ckpt_path=ckpt_path)

    def inference(self, ckpt_path):
        # model = self.model.load_from_checkpoint(ckpt_path, config=self.model_config)
        # print("Load best checkpoint from", self.trainer.checkpoint_callback.best_model_path)
        results = self.trainer.test(self.model, ckpt_path=ckpt_path, dataloaders=self.test_loader, verbose=True)


