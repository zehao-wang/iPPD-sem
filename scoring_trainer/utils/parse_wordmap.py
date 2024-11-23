import h5py
import os
import os.path as osp
import gzip
import torch
import json
import torch.nn as nn
import numpy as np

embedding_dir = '../data/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz'

def load_embeddings():
    with gzip.open(embedding_dir, "rt") as f:
        embeddings = torch.tensor(json.load(f))
    embedding_layer = nn.Embedding.from_pretrained(
        embeddings=embeddings,
        freeze=True,
    )
    return embedding_layer

def load_sem_embeddings():
    """34 labels
    ['none', 'bag', 'bed', 'cabinet', 'ceiling', 'chair', 'clock', 'counter', 'curtain', 
    'door', 'fence', 'floor', 'keyboard', 'laptop', 'light', 'microwave', 'mirror', 'mouse', 
    'oven', 'pillow', 'plant', 'refrigerator', 'shelving', 'sink', 'sofa', 'stairs', 'table',
    'toaster', 'toilet', 'towel', 'tv', 'vase', 'wall', 'window']
    """
    sem_labels = [778, 181, 243, 375, 414, 424, 476, 553, 598, 701, 849, 883, 1198, 1226,
    1266, 1383, 1390, 1198, 1532, 1644, 1667, 1799, 1940, 1972, 2020, 2058, 2159, 2247, 2248,
    2261, 2306, 2348, 2392, 2449]
    with gzip.open(embedding_dir, "rt") as f:
        embeddings = torch.tensor(json.load(f))
    embedding_layer = nn.Embedding.from_pretrained(
        embeddings=embeddings[sem_labels],
        freeze=True,
    )
    return embedding_layer

if __name__ == "__main__":
    meta_file = '../data/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz'
    with gzip.open(meta_file, "rt") as f:
        meta = json.load(f)
    
    word_dict = meta['instruction_vocab']['word2idx_dict']
    import ipdb;ipdb.set_trace() # breakpoint 95
    load_sem_embeddings()
    print()