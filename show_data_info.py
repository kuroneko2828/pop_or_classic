import joblib
import utils.details as details
import settings


def main():
    scores = joblib.load(settings.SCORES_PATH)
    lengthes = details.get_score_lengthes(scores)
    for type in lengthes.keys():
        print(f'{type}: {lengthes[type]}')
    return


if __name__ == '__main__':
    main()