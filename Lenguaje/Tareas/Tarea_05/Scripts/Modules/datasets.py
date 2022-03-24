from argparse import Namespace


def get_params() -> dict:
    params = {"path data": "../data/",
              "train data": "mex_train.txt",
              "train labels": "mex_train_labels.txt",
              "validation data": "mex_val.txt",
              "validation labels": "mex_val_labels.txt",
              }
    return params


def get_args() -> Namespace:
    return args
