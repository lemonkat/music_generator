import sys

from tqdm import tqdm
import mido

import util


if __name__ == "__main__":
    tracks = []
    for name in tqdm(util.get_all_files(sys.argv[1], ".midi")):
        file = mido.MidiFile(name)
        file.__iter__
        if file.type == 2:
            tracks.extend(util.filter_track(file.tracks) for file in tracks)

        else:
            tracks.append(util.filter_track(file.merged_track))

    # print({msg.time for track in tracks for msg in track})

    with open(sys.argv[2], "w") as file:
        file.write(str(len(tracks)) + "\n")
        for track in tqdm(tracks):
            file.write(str(len(track)) + "\n")
            line = []
            for msg in track:
                line.append(str(util.encode(msg)))
            file.write(" ".join(line) + "\n")
