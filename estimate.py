import joblib
import torch
import utils.neural as neural
import settings


def estimate():
    model = neural.create_model(settings.INPUT_NOTE_NUM, settings.DROP_RATE)
    model.load_state_dict(torch.load(settings.MODEL_SAVE_PATH))
    split_data = joblib.load(settings.SPLIT_DATA_PATH)
    neural.estimate(split_data['test'], model)
    return


if __name__ == '__main__':
    estimate()