import random

from typing import Iterable

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import util

def one_hot(ind):
    return tf.one_hot(ind, depth=util.DIM, dtype=tf.float32)

# note: maestro contains around 27.6 million notes
# un-windowed and with 16-bit integers, that's about 0.4 GB

def load_data(path: str, k: int = 256, splits: tuple[float] = (1.0,), tq: bool = False) -> list[tf.data.Dataset]:
    x = []
    y = []
    with open(path, mode="r") as file:
        tracks = [np.array([min(int(num), util.DIM - 1) for num in line.split()], dtype=np.int16) for line in (tqdm(file) if tq else file)]
        windowed = np.concatenate([np.lib.stride_tricks.sliding_window_view(track, k + 1, axis=0) for track in tracks], axis=0)

        while len(splits) > 1:
            windowed, selection = train_test_split(windowed, train_size=splits[0])
            x.append(tf.convert_to_tensor(selection[:, :-1], dtype=tf.int16))
            y.append(tf.convert_to_tensor(selection[:, -1], dtype=tf.int16))
            splits = splits[1:]
        return [tf.data.Dataset(d).apply(one_hot) for d in (x + y)]
    
if __name__ == "__main__":
    k = 256

    epochs = 50
    lr = 0.03
    batch_size = 32

    x_train, x_test, y_train, y_test = load_data("data/maestro_full.dat", k, (0.8, 0.2,))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(k, util.DIM)),
            tf.keras.layers.Dense(65536, activation="relu"),
            tf.keras.layers.Dense(65536, activation="relu"),
            tf.keras.layers.Dense(65536, activation="relu"),
            tf.keras.layers.Dense(65536, activation="relu"),
            tf.keras.layers.Dense(65536, activation="relu"),
            tf.keras.layers.Dense(65536, activation="relu"),
            tf.keras.layers.Dense(util.DIM, activation="relu"),
        ]
    )
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.categorical_crossentropy(),
    )
    hist = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    

