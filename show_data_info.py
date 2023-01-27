import joblib
import utils.details as details
import settings


def main():
    print('[score_lengthes]')
    scores = joblib.load(settings.SCORES_PATH)
    lengthes = details.get_score_lengthes(scores)
    for type in lengthes.keys():
        print(f'{type}: {lengthes[type]}')
    print('[breakdowns]')
    print('       pop classic')
    split_data = joblib.load(settings.SPLIT_DATA_PATH)
    breakdowns = details.get_breakdown(split_data)
    for type_, breakdown in breakdowns.items():
        print(f'{type_}: {breakdown}')
    return


if __name__ == '__main__':
    main()