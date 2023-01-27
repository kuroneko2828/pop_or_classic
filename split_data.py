import joblib
import utils.neural as neural
import settings

def split_data():
    scores = joblib.load(settings.SCORES_PATH)
    score_split_data = neural.create_dataset(scores)
    joblib.dump(score_split_data, settings.SPLIT_DATA_PATH)

if __name__ == '__main__':
    split_data()