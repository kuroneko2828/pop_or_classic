import os
import joblib
import torch
import torch.nn as nn
import utils.neural as neural
import settings


def train():
    os.remove(settings.LOG_FILE)
    split_data = joblib.load(settings.SPLIT_DATA_PATH)
    model = neural.create_model(settings.INPUT_NOTE_NUM, settings.DROP_RATE)
    if settings.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=settings.LEARNING_RATE)
    if settings.CRITERION == 'BCE':
        criterion = nn.BCELoss()
    neural.train(split_data['train'], split_data['valid'], model, optimizer, criterion,
                 settings.LEARNING_RATE, settings.BATCH_SIZE, settings.DEVICE,
                 settings.PATIENCE, settings.MODEL_SAVE_PATH, settings.LOG_FILE)
    return


if __name__ == '__main__':
    train()