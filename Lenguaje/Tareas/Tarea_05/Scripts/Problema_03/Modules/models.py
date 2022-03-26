from numpy import array, mean, asanyarray, sum, exp, argmax, log
from torch.utils.data import DataLoader, TensorDataset
from nltk.tokenize import TweetTokenizer as tokenizer
from Modules.ngrams_class import ngram_model
from sklearn.metrics import accuracy_score
from numpy.random import multinomial
from itertools import permutations
import torch.nn.functional as F
from argparse import Namespace
from tabulate import tabulate
from pandas import DataFrame, read_csv
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
        self.train_text = self.read_file(train_filename)
        self.validation_text = self.read_file(validation_filename)

    def read_file(self, filename: str) -> list:
        data = read_csv(filename,
                        engine="python",
                        sep="\r\n",
                        header=None)
        data = list(data[0])
        return data

    def obtain_data_and_labels(self, ngram: ngram_model) -> None:
        self.train_data, self.train_labels = ngram.transform(self.train_text)
        self.validation_data, self.validation_labels = ngram.transform(
            self.validation_text)

    def obtain_loaders(self) -> None:
        self.train_loader = obtain_loader(self.train_data,
                                          self.train_labels,
                                          self.args)
        self.validation_loader = obtain_loader(self.validation_data,
                                               self.validation_labels,
                                               self.args)


class neural_language_model(nn.Module):
    def __init__(self, args, embeddings: array = None) -> None:
        super(neural_language_model, self).__init__()
        self.window_size = args.N-1
        self.embeding_size = args.d
        self.emb = nn.Embedding(args.vocabulary_size,
                                args.d)
        if embeddings is not None:
            for i in range(embeddings.shape[0]):
                for j in range(embeddings.shape[1]):
                    self.emb.weight.data[i][j] = embeddings[i][j]
        self.fc1 = nn.Linear(args.d*(args.N-1),
                             args.d_h)
        self.drop1 = nn.Dropout(p=args.dropout)
        self.fc2 = nn.Linear(args.d_h,
                             args.vocabulary_size,
                             bias=False)
        self.args = args

    def forward(self, x):
        x = self.emb(x)
        x = x.view(-1, self.window_size*self.embeding_size)
        h = torch.tanh(self.fc1(x))
        h = self.fc2(h)
        h = F.log_softmax(h, dim=1)
        h = self.drop1(h)
        return h

    def read_model(self, path: str, name: str) -> None:
        filename = join(path, name)
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(filename)["state_dict"])
        else:
            self.load_state_dict(torch.load(filename,
                                            map_location=torch.device('cpu'))["state_dict"])
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

    def save_checkpoint(self, state,
                        is_best: bool,
                        checkpoint_path: str,
                        filename: str = 'checkpoint.pt',
                        best_model_name: str = 'model_best.pt') -> None:
        print(checkpoint_path, filename)
        name = join(checkpoint_path,
                    filename)
        torch.save(state,
                   name)
        if is_best:
            filename_best = join(checkpoint_path,
                                 best_model_name)
            copyfile(name,
                     filename_best)

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
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_metric': best_metric, }
            self.save_checkpoint(
                state,
                is_improvement,
                self.args.savedir,
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


class generate_text_class:
    def __init__(self, ngram_data: ngram_model, model: neural_language_model, tokenize: tokenizer) -> None:
        self.ngram_data = ngram_data
        self.tokenize = tokenize
        self.model = model

    def parse_text(self, text: str) -> tuple:
        all_tokens = [word.lower()
                      if word in self.ngram_data.word_index else self.ngram_data.eos
                      for word in self.tokenize(text)]
        tokens_id = [self.ngram_data.word_index[word]
                     for word in all_tokens]
        return all_tokens, tokens_id

    def sample_next_word(self, logits: array, temperature: float) -> int:
        logits = asanyarray(logits).astype("float64")
        preds = logits/temperature
        exp_preds = exp(preds)
        preds = exp_preds/sum(exp_preds)
        probability = multinomial(1, preds)
        return argmax(probability)

    def predict_next_token(self, tokens_id: list) -> int:
        word_index_tensor = torch.LongTensor(tokens_id).unsqueeze(0)
        y_raw_predict = self.model(
            word_index_tensor).squeeze(0).detach().numpy()
        y_pred = self.sample_next_word(y_raw_predict, 1.0)
        return y_pred

    def run(self, initial_text: str):
        tokens, window_word_index = self.parse_text(initial_text)
        for i in range(300):
            y_pred = self.predict_next_token(window_word_index)
            next_word = self.ngram_data.index_word[y_pred]
            tokens.append(next_word)
            if next_word == self.ngram_data.eos:
                break
            else:
                window_word_index.pop(0)
                window_word_index.append(y_pred)
        return " ".join(tokens)

    def obtain_closet_words(self, word: str, n: int) -> None:
        print("Palabras cercanas a {}".format(word))
        word_id = torch.LongTensor([self.ngram_data.word_index[word]])
        word_embed = self.model.emb(word_id)
        # Compute distances to all words
        dist = torch.norm(self.model.emb.weight-word_embed, dim=1).detach()
        lst = sorted(enumerate(dist.numpy()),
                     key=lambda x: x[1])
        table = []
        for idx, difference in lst[1:n+1]:
            table += [[self.ngram_data.index_word[idx],
                       difference]]
        print(tabulate(table,
                       headers=["Word", "Difference"]))
        print()


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


def log_likelihood(model: neural_language_model, text: str, ngram_data: ngram_model) -> float:
    x, y = ngram_data.transform(text)
    x, y = x[2:], y[2:]
    x = torch.LongTensor(x).unsqueeze(0)
    logits = model(x).detach()
    probability = F.softmax(logits, dim=1).numpy()
    return sum(log([probability[i][w]
                    for i, w in enumerate(y)]))


def perplexity(model: neural_language_model, text: str, ngram_data: ngram_model) -> float:
    perplexity_value = log_likelihood(model, text, ngram_data)
    perplexity_value = - perplexity_value / len(text)
    return perplexity_value


def syntax_structure(model: neural_language_model, ngram_data: ngram_model, word: str, tokenize: tokenizer) -> None:
    words = tokenize(word)
    perms = [" ".join(perm) for perm in permutations(words)]
    best_log_likelihood = [(log_likelihood(model,
                                           pharse,
                                           ngram_data),
                            pharse)
                           for pharse in perms]
    best_log_likelihood = sorted(best_log_likelihood, reverse=True)
    headers = ["Palabra", "Perplejidad"]
    print("-"*40)
    results = []
    for p, i in best_log_likelihood[:5]:
        results += [[i, p]]
    print(tabulate(results,
                   headers=headers))
    print("-"*40)
    results = []
    for p, i in best_log_likelihood[-5:]:
        results += [[i, p]]
    print(tabulate(results,
                   headers=headers))
