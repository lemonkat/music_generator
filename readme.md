A set of computer programs for computer-generating music.
The programs work by picking a random "seed" which is a sequence of MIDI notes from somewhere in the dataset, then repeatedly:
1. scanning through the dataset for all occurences of the seed
2. looking at which MIDI notes come after the seed
3. randomly picking a note from these notes and playing it
4. appending the picked note to the end of the seed, while chopping off the first note in the seed

To use:
0. install dependencies
1. compile `Generator.java`
2. add the Maestro 3 dataset to the data folder (if data does not exist, create a folder called data) (dataset found at https://magenta.tensorflow.org/datasets/maestro)
3. run `python3 load.py data/maestro-v3.0.0 data/maestro.dat` to process the dataset
5. plug a MIDI device into your computer (and turn it on)
6. run `mido-ports` and find the name of the MIDI port corresponding to the device
7. run `java Generator path=data/maestro.dat | python3 inf_roll.py [port name]` where [port name] is the name of the MIDI device

If your terminal isn't big enough for the graphics or doesn't support 24-bit TrueColor RGB, instead run `java Generator path=data/maestro.dat | python3 inf_play.py [port name]`

There's some experiments on using neural networks for the generation, but those are still WIP.

Required libraries:
tqdm
numpy
pygame
mido
PyTorch for neural_net/pt_nn_util.py
TensorFlow for neural_net/tf_nn_util.py