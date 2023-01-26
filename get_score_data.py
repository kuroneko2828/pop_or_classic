import utils.get_midi_file as get_midi
import utils.create_data as create_data
import settings
import joblib


def get_score_data():
    pop_files = get_midi.get_pop_files(settings.POP_PATH)
    pop_scores = create_data.create_score_lists(pop_files, 4, settings.INPUT_NOTE_NUM)
    print('OK')
    classic_files = get_midi.get_classic_files(settings.CLASSIC_PATH)
    classic_scores = create_data.create_score_lists(classic_files, 2, settings.INPUT_NOTE_NUM)
    joblib.dump({'pop': pop_scores, 'classic': classic_scores}, settings.SCORES_PATH)
    return


if __name__ == '__main__':
    get_score_data()