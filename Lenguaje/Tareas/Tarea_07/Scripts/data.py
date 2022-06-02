from transformers import AutoTokenizer, AutoModelForSequenceClassification
from numpy import array, concatenate, unique
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from sklearn.utils import class_weight
from pandas import read_csv, DataFrame
import torch.nn.functional as F
from transformers import AdamW
from argparse import Namespace
from os.path import join
from torch import nn
import numpy as np
import random
import torch
from torch.utils.data import (TensorDataset,
                              DataLoader,
                              RandomSampler,)


def get_params() -> dict:
    params = {
        "root": "/content/drive/MyDrive/Lenguaje/Tarea_07",
        "path data": "Data",
        "train file": "train.csv",
        "val file": "val.csv",
        "test file": "test.csv",
    }
    params["path data"] = join(params["root"],
                               params["path data"])
    params["train file"] = join(params["path data"],
                                params["train file"])
    params["val file"] = join(params["path data"],
                              params["val file"])
    params["test file"] = join(params["path data"],
                               params["test file"])
    return params


def get_args() -> Namespace:
    args = Namespace()
    args.max_tokens = 100
    args.batch_size = 8
    args.epoch = 3
    args.lr = 1e-5
    args.device = torch.device('cuda')
    return args


def set_seeds() -> None:
    seed = 54
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def tokenize_data(data: list, tokenizer, args: Namespace) -> dict:
    token_list = tokenizer.batch_encode_plus(
        data,
        max_length=args.max_tokens,
        pad_to_max_length=True,
        add_special_tokens=True,
        truncation='longest_first')
    return token_list


def get_seq_and_mask(tokens_data: dict, data: DataFrame = None) -> tuple:
    seq = torch.tensor(tokens_data['input_ids'])
    mask = torch.tensor(tokens_data['attention_mask'])
    if data:
        y = torch.tensor(list(map(int,
                                  data["target"].to_numpy())))
        return seq, mask, y
    return seq, mask


def get_weights(data: DataFrame, target: list, args: Namespace) -> array:
    # class weights
    cw = class_weight.compute_class_weight('balanced',
                                           classes=unique(target),
                                           y=data["target"].to_list())
    # to tensor
    weights = torch.tensor(cw,
                           dtype=torch.float)
    # upload to GPU
    weights = weights.to(args.device)  # to GPU
    return weights


def create_dataloader(seq: array, mask: array, target: array = None) -> tuple:
    if target:
        # wrap tensor
        data = TensorDataset(seq,
                             mask,
                             target)
        # sampler for sampling the data during training
        sampler = RandomSampler(data)
        # dataLoader for train set
        dataloader = DataLoader(data,
                                sampler=sampler,
                                batch_size=args.batch_size)
        return dataloader

    data = TensorDataset(seq,
                         mask)
    dataloader = DataLoader(data,
                            batch_size=args.batch_size)
    return dataloader


class RobertuitoClasificator(nn.Module):
    def __init__(self, transformer):
        super(RobertuitoClasificator, self).__init__()
        # pretrained
        self.transformer = transformer

    def forward(self, sent_id, mask):
        # Get cls token
        x = self.transformer(sent_id,
                             attention_mask=mask,
                             return_dict=False)[0]
        return x


class robertito_model:
    def __init__(self, params: dict, args: Namespace, weights: array) -> None:
        self.params = params
        self.args = args
        self._create_model(weights)

    def _create_model(self, weights: array) -> None:
        robertuito = AutoModelForSequenceClassification.from_pretrained(
            "pysentimiento/robertuito-base-uncased")
        robertuito = robertuito.to(self.args.device)
        # Use pre-trained model and upload to current device
        self.model = RobertuitoClasificator(robertuito)
        self.model = self.model.to(self.args.device)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.args.lr,
                               correct_bias=False)
        self.cross_entropy = nn.NLLLoss(weight=weights)

    def _train(self, train_dataloader):
        self.model.train()
        total_loss = 0
        # empty list to save model predictions
        total_preds = []
        # iterate over batches
        for step, batch in enumerate(train_dataloader):
            # progress update after every 50 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step,
                                                           len(train_dataloader)))
            # push the batch to gpu
            batch = [r.to(self.args.device)
                     for r in batch]
            sent_id, mask, labels = batch
            # clear previously calculated gradients
            self.model.zero_grad()
            # get model predictions for the current batch
            preds = self.model(sent_id,
                               mask)
            preds = F.log_softmax(preds,
                                  dim=1)
            # compute the loss between actual and predicted values
            loss = self.cross_entropy(preds,
                                      labels)
            # add on to the total loss
            total_loss = total_loss + loss.item()
            # backward pass to calculate the gradients
            loss.backward()
            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           1.0)
            # update parameters
            self.optimizer.step()
            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds)
        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)
        # predictions are in the form of (no. of batches, size of batch, no. of classes).
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = concatenate(total_preds,
                                  axis=0)
        # returns the loss and predictions
        return avg_loss, total_preds

    # function for evaluating the model
    def _evaluate(self, val_dataloader):
        print("\nEvaluating...")
        # deactivate dropout layers
        self.model.eval()
        total_loss = 0
        # empty list to save the model predictions
        total_preds = []
        targets = []
        predictions = []
        # iterate over batches
        for step, batch in enumerate(val_dataloader):
            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step,
                                                           len(val_dataloader)))
            # push the batch to gpu
            batch = [t.to(self.args.device)
                     for t in batch]
            sent_id, mask, labels = batch
            # deactivate autograd
            with torch.no_grad():
                # model predictions
                preds = self.model(sent_id,
                                   mask)
                lab = F.log_softmax(preds,
                                    dim=1).argmax(1)
                # compute the validation loss between actual and predicted values
                loss = self.cross_entropy(preds,
                                          labels)
                total_loss = total_loss + loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)
            predictions += lab.cpu().tolist()
            targets += labels.cpu().tolist()
        metric = accuracy_score(targets,
                                predictions)
        print(f"Validation accuracy: {metric:.4f}")
        # compute the validation loss of the epoch
        avg_loss = total_loss / len(val_dataloader)
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = concatenate(total_preds,
                                  axis=0)
        return avg_loss, total_preds

    def run(self, train_dataloader, val_dataloader) -> None:
        # empty lists to store training and validation loss of each epoch
        self.train_losses = []
        self.valid_losses = []
        # for each epoch
        for epoch in range(self.args.epoch):
            print('\n Epoch {} / {}'.format(epoch + 1,
                                            self.args.epochs))
            # train model
            train_loss, _ = self._train(train_dataloader)
            # evaluate model
            valid_loss, _ = self._evaluate(val_dataloader)
            # save the best model
            torch.save(self.model.state_dict(),
                       f'saved_weights{epoch}.pt')
            # Results
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, test_dataloader):
        predictions = []
        for batch in test_dataloader:
            sent_id, masks = batch
            sent_id = sent_id.to(self.args.device)
            masks = masks.to(self.args.device)
            output = self.model(sent_id,
                                masks)
            preds = F.softmax(output,
                              dim=1).argmax(1)
            predictions += preds.tolist()
        predictions = DataFrame(predictions,
                                columns=["Expected"])
        predictions.index.name = "Id"
        return predictions


params = get_params()
args = get_args()
tokenizer = AutoTokenizer.from_pretrained(
    "pysentimiento/robertuito-base-uncased")

train_data = read_csv(params["train file"])
val_data = read_csv(params["val file"])
test_data = read_csv(params["test file"])
tokens_train = tokenize_data(train_data["text"].to_list(),
                             tokenizer,
                             args)
tokens_val = tokenize_data(val_data["text"].to_list(),
                           tokenizer,
                           args)
tokens_test = tokenize_data(test_data["text"].to_list(),
                            tokenizer,
                            args)
train_seq, train_mask, y_train = get_seq_and_mask(tokens_train,
                                                  train_data)
val_seq, val_mask, y_val = get_seq_and_mask(tokens_val,
                                            val_data)
test_seq, test_mask = get_seq_and_mask(tokens_test)

weights = get_weights(train_data,
                      y_train,
                      args)
# dataLoader for train set
train_dataloader = create_dataloader(train_seq,
                                     train_mask,
                                     y_train)
val_dataloader = create_dataloader(val_seq,
                                   val_mask,
                                   y_val)
test_dataloader = create_dataloader(test_seq,
                                    test_mask)

robertuito = robertito_model(params,
                             args,
                             weights)
robertuito.run(train_dataloader,
               val_dataloader)
results = robertuito.predict(test_dataloader)
results.to_csv("results.csv")
