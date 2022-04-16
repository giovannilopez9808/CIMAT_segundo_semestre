from nltk.tokenize import TweetTokenizer as tokeizer
from argparse import Namespace
from os import makedirs
import numpy as np
import random
import torch


def get_params() -> dict:
    """
    Parámetros de las rutas y nombres de los archivos de datos y resultados
    """
    params = {
        "path data": "/content/drive/MyDrive/Lenguaje/Tarea_06/Data",
        "path word2vec": "/content/drive/MyDrive/Lenguaje/Tarea_06/Data/word2vec",
        "file word2vec": "word2vec_col.txt",
        "train data": "mex_train.txt",
        "train labels": "mex_train_labels.txt",
        "test data": "mex_val.txt",
        "test labels": "mex_val_labels.txt",
        "path model": "/content/drive/MyDrive/Lenguaje/Tarea_06/Results",
        # "path results": "/content/drive/MyDrive/Lenguaje/Tarea_06/Results",
        "path results": "../Results",
        "file model": "model_best.pt",
        "stadistics  file": "stadistics.csv",
    }
    return params


def get_args() -> Namespace:
    """
    parámetros para las redes neuronales
    """
    args = Namespace()
    # numero de semilla
    args.seed = 1111
    # maximo vocabulario
    args.max_vocabulary = 5000
    # tokenizer a utilizar
    args.tokenize = tokeizer().tokenize
    # tamaño en porcentaje del test
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
    args.num_epochs = 40
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
    """
    incrustación de las semillas en cada uno de las librerias a utilizar
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
