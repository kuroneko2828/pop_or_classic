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