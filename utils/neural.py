import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class ScoreData(Dataset):
    def __init__(self, scores, labels):
        self.scores = scores
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        score = self.scores[index]
        label = self.labels[index]
        return {
            'score': torch.LongTensor(score),
            'labels': torch.Tensor(label)
        }


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