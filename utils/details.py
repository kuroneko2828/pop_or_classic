import utils.neural


def get_score_length(score):
    l = len(score)
    for x in score[::-1]:
        if x == 0:
            l -= 1
        else:
            break
    return int(l / 3)


def get_score_lengthes(scores):
    lengthes = {}
    for key, value in scores.items():
        lengthes[key] = {}
        for score in scores[key]:
            l = get_score_length(score)
            if l in lengthes[key].keys():
                lengthes[key][l] += 1
            else:
                lengthes[key][l] = 1
        lengthes[key] = sorted(lengthes[key].items(), reverse=True)
    return lengthes


def get_breakdown_for_1data(data):
    breakdown = [0, 0]
    for i in range(len(data)):
        breakdown[int(data[i]['label'][0])] += 1
    return breakdown


def get_breakdown(split_data):
    breakdowns = {}
    for type_, data in split_data.items():
        breakdowns[type_] = get_breakdown_for_1data(data)
    return breakdowns


def get_beeping_time_avg(scores):
    beeping_time_avg = {}
    for key, value in scores.items():
        beeping_times = []
        beeping_num = 0
        for score in value:
            beeping_times.extend(score[2::3])
            beeping_num += get_score_length(score)
        beeping_time_avg[key] = sum(beeping_times) / beeping_num
    return beeping_time_avg


def get_score_time(scores):
    score_time = {}
    for key, value in scores.items():
        score_time[key] = 0
        for score in value:
            score_time[key] += sum(score[1::3])
        score_time[key] /= len(value)
    return score_time