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
    print('[beeping_time_avg]')
    beeping_time_avg = details.get_beeping_time_avg(scores)
    for key, value in beeping_time_avg.items():
        print(f'{key}: {value}')
    print('[score time]')
    score_time = details.get_score_time(scores)
    for key, value in score_time.items():
        print(f'{key}: {value}')
    return


if __name__ == '__main__':
    main()