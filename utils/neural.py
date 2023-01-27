import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ScoreData(Dataset):
    def __init__(self, scores, labels):
        self.scores = scores
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        score = self.scores[index]
        label = [self.labels[index]]
        return {
            'score': torch.FloatTensor(score),
            'label': torch.FloatTensor(label)
        }


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0,
                 path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_epoch = -1

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func('EarlyStopping counter: '
                            f'{self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.best_epoch

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func('Validation loss decreased '
                            f'({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
                            'Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def split_data(df):
    train_valid, test = train_test_split(
        df, test_size=0.1, shuffle=True, random_state=1, stratify=df['label']
    )
    train, valid = train_test_split(
        train_valid, test_size=0.111, shuffle=True, random_state=1, stratify=train_valid['label']
    )
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    return train, valid, test


def create_dataset(scores):
    score_label_list = []
    for score_type, score_list in scores.items():
        if score_type == 'pop':
            label = 0
        if score_type == 'classic':
            label = 1
        for score in score_list:
            score_label_list.append([score, label])
    df = pd.DataFrame(score_label_list)
    df.columns = ['score', 'label']
    train, valid, test = split_data(df)
    score_split_data = {}
    score_split_data['train'] = ScoreData(train['score'], train['label'])
    score_split_data['valid'] = ScoreData(valid['score'], valid['label'])
    score_split_data['test'] = ScoreData(test['score'], test['label'])
    return score_split_data


def create_model(input_size, drop_rate):
    model = nn.Sequential(
        nn.Linear(input_size * 3, 128),
        nn.ReLU(),
        nn.Dropout(drop_rate),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    return model


def discretize(probs, threshold=0.5):
    discretized = []
    for prob in probs:
        x = 1 if prob[0] >= threshold else 0
        discretized.append(x)
    return discretized


def get_correct_num(preds, labels, threshold=0.5):
    count = 0
    preds = discretize(preds)
    labels = discretize(labels)
    for pred, label in zip(preds, labels):
        if pred == label:
            count += 1
    return count


def calculate_loss_and_accuracy(model, loader, device, criterion=None):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            scores = data['score'].to(device)
            labels = data['label'].to(device)

            outputs = model(scores)
            loss += criterion(outputs, labels).item()

            total += len(labels)
            correct += get_correct_num(outputs, labels)
    return loss / len(loader), correct / total


def print_log(e, epoch, loss_train, acc_train, loss_valid, acc_valid,
               file=None):
    if file is not None:
        with open(file, 'a')as f:
            print(f'finish[{e}/{epoch}] '
                  f'loss_train: {loss_train:.3f} acc_train: {acc_train:.3f} '
                  f'loss_valid: {loss_valid:.3f} acc_valid: {acc_valid:.3f}',
                  file=f)
    else:
        print(f'finish[{e}/{epoch}] '
              f'loss_train: {loss_train:.3f} acc_train: {acc_train:.3f} '
              f'loss_valid: {loss_valid:.3f} acc_valid: {acc_valid:.3f}')
    return

def train(train_data, valid_data, model, optimizer=None, criterion=nn.BCELoss(),
          lr=1e-5, batch_size=8, device='cpu', patience=3, model_save_path='./', log_file=None):
    if optimizer is None:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    dataloader_train = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False)
    early_stopping = EarlyStopping(patience=patience, path=model_save_path)

    for epoch in range(100):
        model.train()
        for i, data in enumerate(dataloader_train):
            scores = data['score'].to(device)
            labels = data['label'].to(device)
            optimizer.zero_grad()

            outputs = model(scores)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = calculate_loss_and_accuracy(
            model, dataloader_train, device, criterion=criterion)
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, dataloader_valid, device, criterion=criterion)

        if log_file is not None:
            print_log(epoch, 100, loss_train, acc_train, loss_valid, acc_valid, file=log_file)

        best_epoch = early_stopping(loss_valid, model, epoch)
        if early_stopping.early_stop:
            print(f"Early Stop!! epoch={best_epoch}")
            if log_file is not None:
                with open(log_file, 'a')as f:
                    print(f"Early Stop!! epoch={best_epoch}", file=f)
            break
    return


def print_estimate(pred_list, label_list):
    print('[confusion matrix]')
    types = ['pop', 'classic']
    cm = confusion_matrix(label_list, pred_list)
    column = pd.MultiIndex.from_arrays([['Pred']*len(types), types])
    index = pd.MultiIndex.from_arrays([['Actual']*len(types), types])
    cm = pd.DataFrame(data=cm, index=index, columns=column)
    print(cm)

    print()
    print('[classification report]')
    report = classification_report(label_list, pred_list, target_names=types)
    print(report)
    return


def estimate(test_data, model, batch_size=1, device='cpu'):
    model.eval()
    pred_list = []
    label_list = []
    loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for data in loader:
            scores = data['score'].to(device)
            labels = data['label'].to(device)

            outputs = model(scores)
            preds = discretize(outputs)
            labels = discretize(labels)
            pred_list.extend(preds)
            label_list.extend(labels)
    print_estimate(pred_list, label_list)
    return