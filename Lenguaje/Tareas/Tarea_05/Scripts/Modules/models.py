import torch.nn.functional as F
from argparse import Namespace
from pandas import read_csv
from os.path import join
import torch.nn as nn
import torch


class Mex_data_class:
    def __init__(self, params: dict) -> None:
        self.params = params

    def read(self) -> None:
        """
        Lectura de los archivos de datos a partir de su ruta y nombre de archivo
        """
        train_filename = join(self.params["path data"],
                              self.params["train data"])
        validation_filename = join(self.params["path data"],
                                   self.params["train data"])
        self.train_data = self.read_file(train_filename)
        self.validation_data = self.read_file(validation_filename)

    def read_file(self, filename: str) -> list:
        data = read_csv(filename,
                        engine="python",
                        sep="\r\n",
                        header=None)
        data = list(data[0])
        return data


class neural_language_model(nn.Module):
    def __init__(self, args, embeddings=None) -> None:
        super(neural_language_model, self).__init__()
        self.window_size = args.N-1
        self.embeding_size = args.d
        self.emb = nn.Embedding(args.vocabulary_size,
                                args.d)
        self.fc1 = nn.Linear(args.d*(args.N-1),
                             args.d_h)
        self.drop1 = nn.Dropout(p=args.dropout)
        self.fc2 = nn.Linear(args.d_h,
                             args.vocabulary_size,
                             bias=False)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(-1, self.window_size*self.embeding_size)
        h = F.relu(self.fc1(x))
        h = self.drop1(h)
        return self.fc2(h)


class model_class:
    def __init__(self, model: neural_language_model, args: Namespace):
        pass

    def get_pred(self, raw_logits):
        probs = F.softmax(raw_logits.detach(), dim=1)
        y_pred = torch.argmax(probs, dim=1).cpu().numpy()
        return y_pred

    def model_eval(self, data, model, gpu=False):
        with torch.no_grad():
            preds = []
            tgts = []
            for window_words, labels in data:
                if gpu:
                    window_words = window_words.cuda()
                outputs = model(window_words)
                # Get prediction
                y_pred = self.get_pred(outputs)
                tgt = labels.numpy()
                tgts.append(tgt)
                preds.append(y_pred)
        tgts = [e for l in tgts for e in l]
        preds = [e for l in preds for e in l]
        return accuracy_score(tgts, preds)
