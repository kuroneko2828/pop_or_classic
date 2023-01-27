import mido


def get_tempo(config_track):
    # get tempo (micro-seconds/beat)
    for msg in config_track:
        if type(msg) == mido.midifiles.meta.MetaMessage:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
    # type(tempo) == int
    tempo /= 1_000_000 # seconds/beat
    return tempo


def is_start(velocity):
    return velocity != 0


def is_stop(velocity):
    return velocity == 0


def add_beeping_time(beeping_notes, time):
    for note in beeping_notes.keys():
        beeping_notes[note]['beeping_time'] += time
    return


def time_to_second(time, tempo):
    return time / 480 * tempo


def zero_padding(score, input_note_num):
    if len(score) != input_note_num * 3:
        score.extend([0] * (input_note_num * 3 - len(score)))
    return


def get_score(main_track, tempo, input_note_num):
    score = []
    beeping_notes = {}
    for msg in main_track:
        if type(msg) == mido.messages.messages.Message:
            time = msg.time
            add_beeping_time(beeping_notes, time)
            if msg.type == 'note_on':
                note = msg.note
                velocity = msg.velocity
                if is_start(velocity) and len(score) < input_note_num:
                    index = len(score)
                    score.append({'note': note, 'start_time': time_to_second(time, tempo), 'beeping_time': 0})
                    beeping_notes[note] = {'index': index, 'beeping_time': 0}
                if is_stop(velocity):
                    if note in beeping_notes.keys():
                        beeping_note = beeping_notes.pop(note)
                        index = beeping_note['index']
                        beeping_time = beeping_note['beeping_time']
                        score[index]['beeping_time'] = time_to_second(beeping_time, tempo)
                if len(score) >= input_note_num and beeping_notes == {}:
                    break
    score[0]['start_time'] = 0
    return score


def midi_to_score(file, main_track_no, input_note_num):
    midi = mido.MidiFile(file)
    config_track = midi.tracks[0]
    main_track = midi.tracks[main_track_no-1]
    tempo = get_tempo(config_track)
    score = get_score(main_track, tempo, input_note_num)
    return score


def score_to_1d_list(score, input_note_num):
    lst = []
    for note in score:
        lst.extend([note['note'], note['start_time'], note['beeping_time']])
    zero_padding(lst, input_note_num)
    return lst


def create_score_lists(files, main_track_no, input_note_num):
    score_lists = []
    for f in files:
        score = midi_to_score(f, main_track_no, input_note_num)
        score_list = score_to_1d_list(score, input_note_num)
        score_lists.append(score_list)
    return score_lists