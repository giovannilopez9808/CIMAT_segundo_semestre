from nltk.tokenize import TweetTokenizer as tokeizer
from argparse import Namespace
from os import makedirs
import numpy as np
import random
import torch


def get_params() -> dict:
    params = {
        "path data": "/content/drive/MyDrive/Lenguaje/Tarea_06/Data",
        "train data": "mex_train.txt",
        "train labels": "mex_train_labels.txt",
        "test data": "mex_val.txt",
        "test labels": "mex_val_labels.txt",
        "path model": "/content/drive/MyDrive/Lenguaje/Tarea_06/Results",
        "file model": "model_best.pt",
        "stadistics  file": "stadistics.csv",
    }
    return params


def get_args() -> Namespace:
    args = Namespace()
    args.seed = 1111
    args.max_vocabulary = 5000
    args.tokenize = tokeizer().tokenize
    args.test_size = 0.1
    args.batch_size = 64
    args.num_workers = 0
    # Maximum sequence length
    args.max_seq_len = 20
    # Model hyperparameters
    args.filter_sizes = [3, 4, 5]
    args.num_filters = 100
    # Training hyperparameters
    args.lr = 1e-2
    args.num_epochs = 100
    args.patience = 10
    # Scheduler hyperparameters
    args.lr_patience = 5
    args.lr_factor = 0.5
    # Dimension of word Embeddings
    args.d = 100
    # Dimension for Hidden Layer
    args.d_h = 200
    args.dropout = 0.1
    # Save directory
    args.savedir = "/content/drive/MyDrive/Lenguaje/Tarea_06/Results"
    makedirs(args.savedir,
             exist_ok=True)
    init_seeds(args)
    return args


def init_seeds(args: Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
