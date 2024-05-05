import os
import random
import sys
import mido
import numpy as np
from tqdm import tqdm

# message format:
# note, velocity, time
# fits into 32-bit integer


def reduce_prec(arr, n_prec, v_prec, t_prec):
    """
    reduces the precision of an array of encoded messages.
    n_prec, v_prec, and t_prec are for note, velocity, and time, respectively.
    """
    filter_val = 0
    filter_val += (65535 >> (16 - t_prec)) << (16 - t_prec)
    filter_val += (127 >> (7 - v_prec)) << (23 - v_prec)
    filter_val += (127 >> (7 - n_prec)) << (31 - n_prec)
    return np.bitwise_and(arr, np.int32(filter_val), dtype=np.int32, out=arr)


def encode(m):
    """
    encodes a MIDI message as a 32-bit signed integer.
    """
    if m.type == "note_off":
        return int(m.time)
    return (int(m.note) << 24) + (int(m.velocity) << 16) + int(m.time)


def decode(m):
    """
    decodes a 32-bit signed integer as a MIDI message.
    """
    m = np.int32(m)
    dtime = m & 65535
    vel = (m >> 16) & 127
    note = (m >> 24) & 127
    return mido.Message("note_on", note=note, velocity=vel, time=dtime)


# finds longest track
def find_main_track(file):
    # lengths = [len(track) for track in file.tracks]
    # return file.tracks[lengths.index(max(lengths))]
    return max(file.tracks, key=len)


# filters out any messages but note_on and note_off
def filter_track(track):
    track = list(track)
    result = []
    for i in range(len(track) - 1):
        if track[i].type in ["note_on", "note_off"]:
            result.append(track[i])
        else:
            track[i + 1].time += track[i].time
    if track[-1].type in ["note_on", "note_off"]:
        result.append(track[-1])
    return result


# loads all tracks in a file
def load(path):
    file = mido.MidiFile(path)

    if file.type == 2:
        return [
            np.array([encode(msg) for msg in filter_track(track)], dtype=np.int32)
            for track in file.tracks
        ]

    return [np.array([encode(msg) for msg in filter_track(file)], dtype=np.int32)]
    # track = max(file.tracks, key=len)
    # return [np.array([encode(msg) for msg in filter_track(track)], dtype=np.int32)]


# saves a single track to a file
def save(path, arr):
    file = mido.MidiFile()
    file.tracks.append(mido.MidiTrack([decode(msg) for msg in arr]))
    file.save(path)


# finds all the files with a given extension
def get_all_files(p=".", ext=None, out=None):
    if out is None:
        out = []
    if ext is None or p.endswith(ext):
        out.append(p)
    for path, _, files in os.walk(p):
        for file in files:
            if ext is None or file.endswith(ext):
                out.append(os.path.join(path, file))

    return out


def get_seed(data, k):
    i = random.randint(0, len(data) - k)
    return data[i : i + k]


def generate_iter(func, seed):
    while True:
        m = func(seed)
        seed[:] = np.roll(seed, -1)
        seed[-1] = m
        yield m


def generate(func, seed, n, tq=False):
    arr = np.empty(n, dtype=np.int32)
    for i in tqdm(range(n)) if tq else range(n):
        arr[i] = func(seed)
        seed[:] = np.roll(seed, -1)
        seed[-1] = arr[i]
    return arr, seed


def read_pipe(pipe=sys.stdin):
    while True:
        # yield int(pipe.readline().strip())
        yield input()


# rand_rate is on avg, how many times/sec it will pick randomly
def get_randomwriter(data, k):
    data_dict = {}
    for i in tqdm(range(k, len(data))):
        key = tuple(data[i - k : i])
        if key in data_dict:
            data_dict[key].append(data[i])
        else:
            data_dict[key] = [data[i]]

    def call(seed):
        choices = data_dict.get(tuple(seed), [])
        if len(choices) < 2:
            choices = data
        return random.choice(choices)

    return call
