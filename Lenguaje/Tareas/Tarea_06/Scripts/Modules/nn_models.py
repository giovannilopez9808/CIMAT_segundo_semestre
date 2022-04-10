from sklearn.metrics import accuracy_score
from argparse import Namespace
from pandas import DataFrame
from shutil import copyfile
from os.path import join
from numpy import mean
import torch.nn as nn
import torch
import time


class CNNTextCls(nn.Module):
    def __init__(self, args, embeddings=None, freeze=False):
        super(CNNTextCls, self).__init__()
        if embeddings is not None:
            self.emb = nn.Embedding.from_pretrained(
                torch.FloatTensor(embeddings))
            if freeze:
                self.emb.weight.requires_grad = False
        else:
            self.emb = nn.Embedding(args.max_vocabulary, args.d)
        conv_block_list = []
        for k in args.filter_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=args.d,
                          out_channels=args.num_filters, kernel_size=k, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=(args.max_seq_len-k+1))
            )
            conv_block_list.append(conv_block)
        self.conv_block_list = nn.ModuleList(conv_block_list)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.num_filters*len(args.filter_sizes), 1)

    def forward(self, x):
        '''
        B: batch size
        L: sequence length
        D: embedding dim
        C: Convolution output channels (number of filters)
        k: Convolution kernel size
        N: Number of convolution blocks

        x shape: (B, L)
        '''
        # (B, L, D)
        x = self.emb(x)
        # (B, D, L) conv1d expects L in last dimension
        x = x.transpose(1, 2)
        x_filter = []
        # Conv1d -> ReLU -> MaxPool1d
        for conv_block in self.conv_block_list:
            # (B, C, L-k+1) -> ReLU -> (B, C, 1) -> (B, C) after squeeze
            x_filter.append(conv_block(x).squeeze(2))
        x_cat = torch.cat(x_filter, dim=1)  # (B, C*N)
        x = self.dropout(x_cat)
        return self.fc(x)


class model_class:
    """
    Modelo que realiza la ejeccución del entrenamiento de la red neuronal dada su estructura y datos
    """

    def __init__(self, model: CNNTextCls, args: Namespace, train_loader, validation_loader):
        self.validation_loader = validation_loader
        self.train_loader = train_loader
        self.model = model
        self.args = args

    def get_pred(self, outputs) -> torch.Tensor:
        result = torch.round(torch.sigmoid(outputs.detach())).cpu().numpy()
        return result

    def model_eval(self, data, gpu=False):
        with torch.no_grad():
            preds, tgts = [], []
            for input, labels in data:
                if gpu:
                    input = input.cuda()
                outputs = self.model(input)
                # Get prediction for Accuracy
                y_pred = self.get_pred(outputs)
                tgt = labels.numpy()
                tgts.append(tgt)
                preds.append(y_pred)
        tgts = [e for l in tgts for e in l]
        preds = [e for l in preds for e in l]
        metrics = {
            "accuracy": accuracy_score(tgts, preds),
        }
        return metrics

    def save_checkpoint(self, state,
                        is_best: bool,
                        checkpoint_path: str,
                        filename: str = 'checkpoint.pt',
                        best_model_name: str = 'model_best.pt') -> None:
        name = join(checkpoint_path,
                    filename)
        torch.save(state,
                   name)
        if is_best:
            filename_best = join(checkpoint_path,
                                 best_model_name)
            copyfile(name,
                     filename_best)

    def run(self) -> DataFrame:
        """
        Ejecuta el entrenamiento de la red neuronal, regresa un dataframe con las estadisticas del entrenamiento
        """
        stadistics = DataFrame(columns=["Train acc",
                                        "Loss",
                                        "Val acc",
                                        "Time"])
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
            for input, labels in self.train_loader:
                # If GPU available
                if self.args.use_gpu:
                    input = input.cuda()
                    labels = labels.cuda()
                # Forward pass
                outputs = self.model(input)
                loss = criterion(outputs, labels)
                loss_epoch.append(loss.item())
                # Get Trainning Metrics
                preds = self.get_pred(outputs)
                tgt = labels.cpu().numpy()
                training_metric.append(accuracy_score(tgt, preds))
                # Backward and Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Get Metric in Trainning Dataset
            mean_epoch_metric = mean(training_metric)
            train_metric_history.append(mean_epoch_metric)
            # Get Metric in Validation Dataset
            self.model.eval()
            tuning_metric = self.model_eval(self.validation_loader,
                                            self.args.use_gpu)
            metric_history.append(tuning_metric["accuracy"])
            # Update Scheduler
            scheduler.step(tuning_metric["accuracy"])
            # Check for Metric Improvement
            is_improvement = tuning_metric["accuracy"] > best_metric
            if is_improvement:
                best_metric = tuning_metric["accuracy"]
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
            finish_time = time.time()-epoch_start_time
            stadistics.loc[epoch+1] = [mean_epoch_metric,
                                       mean(loss_epoch),
                                       tuning_metric["accuracy"],
                                       finish_time]
            print('Epoch[{}/{}], Loss : {:4f} - Train Accuracy: {:.4f} - Val accuracy: {:4f} - Epoch time: {:2f}'.format(
                epoch + 1,
                self.args.num_epochs,
                mean(loss_epoch),
                mean_epoch_metric,
                tuning_metric["accuracy"],
                finish_time))
        return stadistics


def init_models_parameters(model: CNNTextCls, args: Namespace) -> tuple:
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           "max",
                                                           patience=args.lr_patience,
                                                           verbose=True,
                                                           factor=args.lr_factor)
    return criterion, optimizer, scheduler


def save_stadistics(params: dict, stadistics: DataFrame) -> None:
    filename = join(params["path results"],
                    params["stadistics  file"])
    stadistics.index.name = "Epoch"
    stadistics.to_csv(filename)
