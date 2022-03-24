from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from .ngrams_class import ngram_model
import torch.nn.functional as F
from argparse import Namespace
from tabulate import tabulate
from numpy import array, mean
from pandas import read_csv
from shutil import copyfile
from os.path import join
import torch.nn as nn
import torch
import time


class Mex_data_class:
    def __init__(self, params: dict, args: Namespace) -> None:
        self.params = params
        self.args = args
        self.read()

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

    def obtain_data_and_labels(self, ngram: ngram_model) -> None:
        self.train_data, self.train_labels = ngram.transform(self.train_data)
        self.validation_data, self.validation_labels = ngram.transform(
            self.validation_data)

    def obtain_loaders(self) -> None:
        self.train_loader = obtain_loader(self.train_data,
                                          self.train_labels,
                                          self.args)
        self.validation_loader = obtain_loader(self.validation_data,
                                               self.validation_labels,
                                               self.args)


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

    def read_model(self, path: str, name: str) -> None:
        filename = join(path, name)
        self.load_state_dict(torch.load(filename)["state_dict"])
        self.train(False)


class model_class:
    def __init__(self, model: neural_language_model, args: Namespace, train_loader, validation_loader):
        self.validation_loader = validation_loader
        self.train_loader = train_loader
        self.model = model
        self.args = args

    def get_pred(self, raw_logits):
        probs = F.softmax(raw_logits.detach(), dim=1)
        y_pred = torch.argmax(probs, dim=1).cpu().numpy()
        return y_pred

    def model_eval(self, data):
        with torch.no_grad():
            preds = []
            tgts = []
            for window_words, labels in data:
                if self.args.use_gpu:
                    window_words = window_words.cuda()
                outputs = self.model(window_words)
                # Get prediction
                y_pred = self.get_pred(outputs)
                tgt = labels.numpy()
                tgts.append(tgt)
                preds.append(y_pred)
        tgts = [e for l in tgts for e in l]
        preds = [e for l in preds for e in l]
        return accuracy_score(tgts, preds)

    def save_checkpoint(state, is_best: bool, checkpoint_path: str, filename: str = 'checkpoint.pt', best_model_name: str = 'model_best.pt') -> None:
        filename = join(checkpoint_path,
                        filename)
        torch.save(state,
                   filename)
        if is_best:
            filename = join(checkpoint_path,
                            best_model_name)
            copyfile(filename,
                     filename)

    def run(self):
        start_time = time.time()
        best_metric = 0
        metric_history = []
        train_metric_history = []
        criterion, optimizer, scheduler = init_models_parameters(self.model,
                                                                 self.args)
        for epoch in range(self.args.num_epochs):
            epoch_start_time = time.time()
            loss_epoch = []
            training_metric = []
            self.model.train()
            for window_words, labels in self.train_loader:
                # If GPU available
                if self.args.use_gpu:
                    window_words = window_words.cuda()
                    labels = labels.cuda()
                # Forward pass
                outputs = self.model(window_words)
                loss = criterion(outputs, labels)
                loss_epoch.append(loss.item())
                # Get Trainning Metrics
                y_pred = self.get_pred(outputs)
                tgt = labels.cpu().numpy()
                training_metric.append(accuracy_score(tgt, y_pred))
                # Backward and Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Get Metric in Trainning Dataset
            mean_epoch_metric = mean(training_metric)
            train_metric_history.append(mean_epoch_metric)
            # Get Metric in Validation Dataset
            self.model.eval()
            tuning_metric = self.model_eval(self.validation_loader)
            metric_history.append(mean_epoch_metric)
            # Update Scheduler
            scheduler.step(tuning_metric)
            # Check for Metric Improvement
            is_improvement = tuning_metric > best_metric
            if is_improvement:
                best_metric = tuning_metric
                n_no_improve = 0
            else:
                n_no_improve += 1
            # Save best model if metric improved
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_metric': best_metric, },
                is_improvement,
                self.args.savedir,
                filename='checkpoint.pt',
                best_model_name='best_model'
            )
            # Early stopping
            if n_no_improve >= self.args.patience:
                print('No improvement. Breaking out of loop')
                break
            print('Train acc: {}'.format(mean_epoch_metric))
            print('Epoch[{}/{}], Loss : {:4f} - Val accuracy: {:4f} - Epoch time: {:2f}'.format(
                epoch + 1,
                self.args.num_epochs,
                mean(loss_epoch),
                tuning_metric,
                time.time() - epoch_start_time))
            print('--- %s seconds ---' % (time.time() - start_time))


def init_models_parameters(model: neural_language_model, args: Namespace) -> tuple:
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           "min",
                                                           patience=args.lr_patience,
                                                           verbose=True,
                                                           factor=args.lr_factor)
    return criterion, optimizer, scheduler


def print_closet_words(embeddings, ngram_data, word, n):
    word_id = torch.LongTensor([ngram_data.word_index[word]])
    word_embed = embeddings(word_id)
    # Compute distances to all words
    dist = torch.norm(embeddings.weight-word_embed, dim=1).detach()
    lst = sorted(enumerate(dist.numpy()),
                 key=lambda x: x[1])
    table = []
    for idx, difference in lst[1:n+1]:
        table += [[ngram_data.index_word[idx],
                   difference]]
    print(tabulate(table,
                   headers=["Word", "Difference"]))


def obtain_loader(data: array, labels: array, args: Namespace) -> DataLoader:
    dataset = TensorDataset(torch.tensor(data,
                                         dtype=torch.int64),
                            torch.tensor(labels,
                                         dtype=torch.int64))
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True)
    return loader
