import colorsys
import random
import os
import sys

import pygame
import numpy as np
import mido
from tqdm import tqdm

import util

# config
k = 10
n_prec, v_prec, t_prec = 7, 5, 16

show_desc = True

scroll_dist = 40

ticks_per_sec = 960
lines_per_sec = 5

port = sys.argv[1] if len(sys.argv) > 1 else None


DESCRIPTION = """
####################################################################################################################################
##                                                                                                                                ##
##  The Sounds of Your Unemployment                                                                                               ##
##                                                                                                                                ##
##  LemonKat                                                                                                                      ##
##                                                                                                                                ##
##  In this modern era, technology is becoming a bigger and bigger part of our daily lives.                                       ##
##  The unstoppable drive of human progress boldly marches ever forward, leaving your job security behind.                        ##
##  The skills the world needs are no longer yours. The sounds played here are the sounds of your unemployment.                   ##
##  I decided to do my part by computer-generating music so we don't need human composers anymore.                                ##
##  As you can see/hear, the Singularity is inevitable and the robot overlords should arrive in a couple weeks.                   ##
##                                                                                                                                ##
####################################################################################################################################
"""
# ########################################################################################
# NOTE_RANGE = 12, 110
NOTE_RANGE = 0, 128

pygame.init()
mido.set_backend("mido.backends.pygame")

colors = [colorsys.hsv_to_rgb(i / 128, 0.99, 0.99) for i in range(*NOTE_RANGE)]
color_codes = [16 + 36 * int(6 * r) + 6 * int(6 * g) + int(6 * b) for r, g, b in colors]
color_strings = [f"\x1b[38;5;{code};m" for code in color_codes]

def play_line(out_channel, line):
    for note in np.nonzero(line & 128)[0]:
        out_channel.send(mido.Message("note_on", note=note + NOTE_RANGE[0], velocity=line[note] & 127, time=0))

def line_iter(iterator):
    cur = np.zeros(NOTE_RANGE[1] - NOTE_RANGE[0], dtype=np.uint8)
    for msg in iterator:
        msg = util.decode(msg)
        cur[msg.note - NOTE_RANGE[0]] = max(cur[msg.note - NOTE_RANGE[0]], 128 + msg.velocity)
        for _ in range(msg.time):
            yield cur
            cur &= 127

def draw_grid(grid):
        result = ["\x1b8"]
        for row in grid[::-1]:
            line = ["##"]
            for note, val in enumerate(row):
                if val & 127:
                    line.append(color_strings[note])
                    line.append("O" if val > 128 else "#")
                else:
                    line.append(" ")
            line.append("\x1b[0m##")
            result.append("".join(line))
        result.append("#" * (NOTE_RANGE[1] - NOTE_RANGE[0] + 4))
        print("\n".join(result), flush=True)

ticks_per_line = int(ticks_per_sec / lines_per_sec)

# arrays = []
# print("Loading data...")
# # note: grabs all files with a ".midi" extension in the data folder
# for file in tqdm(util.get_all_files("data/maestro-v3.0.0", ".midi")[:10]):
#     arrays.extend(util.load(file))

# random.shuffle(arrays)
# data = np.concatenate(arrays, axis=0)
# print("Data loaded.")

# print("Preprocessing...")
# util.reduce_prec(data, n_prec, v_prec, t_prec)
# func = util.get_randomwriter(data, k)
# seed = util.get_seed(data, k)

os.system("clear")
if show_desc:
    for line in DESCRIPTION.strip().split("\n"):
        print(line)
else:
    print("#" * (NOTE_RANGE[1] - NOTE_RANGE[0] + 4))
print("\n" * scroll_dist, end=f"\x1b[{scroll_dist + 1}A\x1b7")
draw_grid(np.zeros((scroll_dist, NOTE_RANGE[1] - NOTE_RANGE[0]), dtype=np.int32))

# using circular array for speed
cur_tick = 0
note_grid = np.zeros((scroll_dist * ticks_per_line, NOTE_RANGE[1] - NOTE_RANGE[0]), dtype=np.uint8)

clock = pygame.time.Clock()
if port is None:
        # for new_line in line_iter(util.generate_iter(func, seed)):
        for new_line in line_iter(util.read_pipe()):
            # print(line)
            note_grid[cur_tick] = new_line
            cur_tick = (cur_tick + 1) % note_grid.shape[0]

            if cur_tick % ticks_per_line == 0:
                draw_grid(np.roll(np.max(np.reshape(note_grid, (scroll_dist, ticks_per_line, NOTE_RANGE[1] - NOTE_RANGE[0])), axis=1), -(cur_tick // ticks_per_line), axis=0))

            clock.tick(ticks_per_sec)
else:
    with mido.open_output(port) as out_port:
        # for new_line in line_iter(util.generate_iter(func, seed)):
        for new_line in line_iter(util.read_pipe()):
            # print(line)
            play_line(out_port, note_grid[cur_tick])
            note_grid[cur_tick] = new_line
            cur_tick = (cur_tick + 1) % note_grid.shape[0]

            if cur_tick % ticks_per_line == 0:
                draw_grid(np.roll(np.max(np.reshape(note_grid, (scroll_dist, ticks_per_line, NOTE_RANGE[1] - NOTE_RANGE[0])), axis=1), -(cur_tick // ticks_per_line), axis=0))

            clock.tick(ticks_per_sec)