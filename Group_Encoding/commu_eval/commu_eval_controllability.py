import re
import os
import pandas as pd
import numpy as np
import functools
import miditoolkit

from commu.preprocessor.encoder.event_tokens import base_event, TOKEN_OFFSET

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
PITCH_RANGE_CUT = {
    "very_low": 36,
    "low": 48,
    "mid_low": 60,
    "mid": 72,
    "mid_high": 84,
    "high": 96,
    "very_high": 128,
}

CHORD_WORDS = base_event[TOKEN_OFFSET.CHORD_START.value - 2:]
scale_map = {"": (0, 4, 7),
             "7": (0, 4, 7, 10),
             "maj7": (0, 4, 7, 11),
             "sus4": (0, 5, 7),
             "7sus4": (0, 5, 7, 10),
             "+": (0, 4, 8),
             "dim": (0, 3, 6),
             "dim7": (0, 3, 6, 9),
             "m": (0, 3, 7),
             "m7": (0, 3, 7, 10),
             "m7b5": (0, 3, 6, 10),
             "sus2": (0, 2, 7),
             "add2": (0, 2, 4, 7),
             "madd2": (0, 2, 3, 7),
             "m6": (0, 3, 7, 9),
             "6": (0, 4, 7, 9)}

chord_lst = [
    "a",
    "a#",
    "b",
    "c",
    "c#",
    "d",
    "d#",
    "e",
    "f",
    "f#",
    "g",
    "g#",
]

def mk_chord_scale_map(chord_words = CHORD_WORDS):
    chord2symbol = {k: v for k, v in zip(chord_lst, range(12))}
    chord2scale = dict()
    for event in chord_words:
        if "#" in event:
            root = event.split('_')[1][0] + re.findall(r"([#])", event)[0]
            quality = event.split('_')[1].replace(root, "")
        else:
            root = event.split('_')[1][0]
            quality = event.split('_')[1][1:]
        if root != "N":
            diff = chord2symbol[root] - chord2symbol['c']
            scale = [(k + diff) % 12 for k in scale_map[quality]]
        else:
            scale = []
        chord2scale["Chord_" + root + quality] = scale
    return chord2scale



def get_pitch_range_v2(midi_obj, keyswitch_velocity = None):
    def _get_avg_note(_tracks):
        total = 0
        count = 0
        for track in _tracks:
            # pretty_midi 에서 메타 트랙은 Instrument 로 파싱되지 않음
            if track.name == "chord":
                continue
            for event in track.notes:
                if event.pitch == 0:
                    continue

                if keyswitch_velocity is not None:
                    if event.velocity != keyswitch_velocity:
                        total += event.pitchls
                        count += 1
                else:
                    total += event.pitch
                    count += 1
        if not count:
            return None
        return total / count

    def _get_pitch_range(avg_pitch_range):
        indexer = {i: k for i, k in enumerate(PITCH_RANGE_CUT.keys())}
        bins = list(PITCH_RANGE_CUT.values())
        digitizer = functools.partial(np.digitize, bins=bins)
        return indexer[digitizer(avg_pitch_range)]


    avg_note = _get_avg_note(midi_obj.instruments)
    if avg_note is None:
        return "n"
    return _get_pitch_range(avg_note)


def in_pitch_range_rate(midifile, pitch_range):
    gen_pr = get_pitch_range_v2(midifile)
    p = (gen_pr == list(PITCH_RANGE_CUT.keys())[pitch_range])
    return p


def in_velocity_range_rate(midifile, min_vel, max_vel):
    p, n = 0, 0
    if not midifile.instruments or not midifile.instruments[0].notes:
        return np.NaN, np.NaN, np.NaN
    velocity = [i.velocity for i in midifile.instruments[0].notes]
    for i in velocity:
        if min_vel <= i <= max_vel:
            p += 1
        else:
            n += 1
    pn = not n
    return p, n, pn

def in_harmony_rate(midifile, chord_progression, key_num):
    scale = key_num // 12
    diff = key_num % 12
    key_scale = []
    if scale == 0:
        for i in MAJOR_SCALE:
            key_scale.append((i + diff) % 12)
    else:
        for i in NATURAL_MINOR_SCALE:
            key_scale.append((i + diff) % 12)

    p, n = 0, 0
    if not midifile.instruments or not midifile.instruments[0].notes:
        return np.NaN, np.NaN, np.NaN
    notes = midifile.instruments[0].notes
    pn = [0] * len(notes)
    chord_times = [240 * i for i in range(len(chord_progression) + 1)] # 240 ticks per 8th note
    chord_scale_map = mk_chord_scale_map()
    for note_idx, note in enumerate(notes):
        for idx, time in enumerate(chord_times[:-1]):
            next_time = chord_times[idx + 1]
            if (time < note.start < next_time) or ( # | ----note----
                    time < note.end < next_time) or ( # ----note---- |
                    note.start < time and note.end > next_time): # --|--note--|--
                if note.pitch % 12 in key_scale:
                    pn[note_idx] += 0
                elif note.pitch % 12 in chord_scale_map['Chord_' + chord_progression[idx].lower()]:
                    pn[note_idx] += 0
                else:
                    pn[note_idx] += 1
            else:
                pass
    for i in pn:
        if i:
            n += 1
        else:
            p += 1
    pn = p / (p + n)
    return p, n, pn


meta_data_path = "/workspace/ComMU-Transformer-conditional-VAE/dataset/val_meta.csv"
metas = pd.read_csv(meta_data_path)
min_vel_arr = metas["min_velocity"]
max_vel_arr = metas["max_velocity"]
chord_arr = metas["chord_progressions"]

n_gen = np.load("/workspace/ComMU-Transformer-conditional-VAE/out_val/val.npy", allow_pickle=True)
np_gen = np.zeros((763, 10, 1012))
i = 0
for ii in n_gen:
    j = 0
    for jj in ii:
        k = 0
        for kk in jj:
            if k >= 1012:
                break
            np_gen[i, j, k] = kk
            k += 1
        j += 1
    i += 1
n_gen = np_gen
n_gen = n_gen[:, 0, :12]

dir = "/workspace/ComMU-Transformer-conditional-VAE/out_val/"

CP = 0
CV = 0
CH = 0
phsum = 0
nhsum = 0
pvsum = 0
nvsum = 0

for i in range(763):
    cur_dir = dir + str(i)
    for (root, directories, files) in os.walk(cur_dir):
        if(len(files) == 10):
            cur_file = root + '/' + files[0]
            midifile = miditoolkit.midi.parser.MidiFile(cur_file)

            pitch_range = int(n_gen[i, 4]) - 631
            min_vel = min_vel_arr[i]
            max_vel = max_vel_arr[i]
            chord_progression = eval(chord_arr[i])[0]
            key_num = int(n_gen[i, 2]) - 602

            P = in_pitch_range_rate(midifile, pitch_range)
            CP += P
            pv, nv, V = in_velocity_range_rate(midifile, min_vel, max_vel)
            pvsum += pv
            nvsum += nv

            CV += V
            ph, nh, pnh = in_harmony_rate(midifile, chord_progression, key_num)
            phsum += ph
            nhsum += nh

            CH += pnh

print("CP :", CP / 763, "CV :", pvsum / (pvsum + nvsum), "CH :", phsum / (phsum + nhsum))
print("CP :", CP / 763, "CV :", CV / 763, "CH :", CH / 763)