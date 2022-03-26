from argparse import Namespace
from os import makedirs
import numpy as np
import random
import torch


def get_params() -> dict:
    params = {
        "path data": "../Data",
        "word2vec path": "../Data/word2vec",
        "word2vec file": "word2vec_col.txt",
        "train data": "mex_train.txt",
        "train labels": "mex_train_labels.txt",
        "validation data": "mex_val.txt",
        "validation labels": "mex_val_labels.txt",
        "model path": "../Data/Model_02",
        "file model": "model_best.pt",
    }
    return params


def get_args() -> Namespace:
    args = Namespace()
    args.batch_size = 64
    args.num_workers = 2
    args.N = 4
    # Dimension of word Embeddings
    args.d = 100
    # Dimension for Hidden Layer
    args.d_h = 200
    args.dropout = 0.1
    # Training hyperparameters
    args.lr = 2.3e-1
    args.num_epochs = 100
    args.patience = 20
    # Scheduler hyperparameters
    args.lr_patience = 10
    args.lr_factor = 0.5
    # Save directory
    args.savedir = "model"
    makedirs(args.savedir,
             exist_ok=True)
    return args


def init_seeds() -> None:
    seed = 1111
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
