import sys

import pygame
import mido

import util as util

ticks_per_sec = 960
lines_per_sec = 5


port = sys.argv[1] if len(sys.argv) > 1 else None

pygame.init()
mido.set_backend("mido.backends.pygame")

ticks_per_line = int(ticks_per_sec / lines_per_sec)
clock = pygame.time.Clock()

if port is None:
    for m in util.read_pipe():
        msg = util.decode(m)
        print(msg)

        for _ in range(msg.time):
            clock.tick(ticks_per_sec)

else:

    with mido.open_output(port) as out_port:
        for m in util.read_pipe():
            msg = util.decode(m)
            out_port.send(msg)

            for _ in range(msg.time):
                clock.tick(ticks_per_sec)

