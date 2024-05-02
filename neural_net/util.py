import os

from typing import Iterable

import numpy as np
import mido
from tqdm import tqdm

import torch
from torch.utils import data as tdata


def cond_tq(cond, arr):
    return tqdm(arr) if cond else arr

V_PREC = 4

N_NOTES = (128 // V_PREC) * 88

DIM = N_NOTES + 3840


"""
Ideas:
Format
 - absolute or relative time?
 - rel-time better for initial prediction
 - "blacking" - inserting notes, or playing alongside?
 - if treated as alongside, rel-time better for both
 - insertion might work better w/ abs-time
 - how to encode abs-time?
 - - won't need positional encoding for abs-time
 - - can't auto-stop though
 - probably better to go with rel-time, playing alongside
"""

# finds all the files with a given extension
def get_all_files(p: str = ".", ext: str = None, out: list = None) -> list[str]:
    if out is None:
        out = []
    if ext is None or p.endswith(ext):
        out.append(p)
    for path, _, files in os.walk(p):
        for file in files:
            if ext is None or file.endswith(ext):
                out.append(os.path.join(path, file))

    return out

def encode(track: mido.MidiTrack) -> Iterable[int]:
    cur_time = 0

    for msg in track:
        if msg.type == "note_off":
            msg = msg.copy(type="note_on", velocity=0)
        
        cur_time += msg.time
        if msg.type == "note_on":
            if cur_time > 0:
                yield max(N_NOTES + cur_time, DIM)
                cur_time = 0

            yield (msg.velocity // V_PREC) * 88 + (msg.note - 21)

def decode(track: Iterable[int]) -> mido.MidiTrack:
    result = mido.MidiTrack()

    cur_time = 0

    for token in track:
        if token > N_NOTES:
            cur_time += token - N_NOTES
        else:
            note, vel = 21 + (token % 88), V_PREC * (token // 88)
            result.append(mido.Message(type="note_on", note=note, velocity=vel, time=cur_time))
            cur_time = 0
    
    return result

def play_track(track: mido.MidiTrack, port: str = "CAISO USB-MIDI") -> None:
    with mido.open_output(port) as out_port:
        file = mido.MidiFile(type=0)
        file.tracks.append(track)
        print(file)
        print(sum(msg.time for msg in track))
        for msg in file.play():
            out_port.send(msg)

def preprocess_data(filenames: list[str], outfile: str, tq: bool = True) -> None:
    with open(outfile, "w") as outfile:
        for filename in cond_tq(tq, filenames):
            file = mido.MidiFile(filename)
            tracks = [file.merged_track] if file.type == 1 else file.tracks
            for track in tracks:
                outfile.write(" ".join(map(str, encode(track))))
                outfile.write("\n")



    

        
    
        
