import time

from typing import Iterable

import torch
import numpy as np
from torch.utils import data as tdata

import util


class MidiDataset(tdata.IterableDataset):
    def __init__(self, path: str, k: int = 256):
        self._path = path
        self._k = k
        self._len = None

    def __iter__(self) -> Iterable[torch.Tensor]:
        with open(self._path, "r") as file:
            for line in file:
                track = torch.tensor(
                    [min(int(token), util.DIM - 1) for token in line.split()],
                    dtype=torch.int64,
                )

                for idx in range(self._k, len(track) - 1):
                    seq = torch.nn.functional.one_hot(
                        track[idx - self._k : idx],
                        num_classes=util.DIM,
                    ).to(torch.float32)

                    yield seq, track[idx + 1]

    def __len__(self) -> int:
        if self._len is None:
            self._len = 0
            with open(self._path, "r") as file:
                for line in file:
                    self._len += line.count(" ") + 1 - self._k + 1
        return self._len


def floatOR(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1 - (1 - a) * (1 - b)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.dropout(x + self.pe[: x.size(0)])


class MidiRNN(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.rnn = torch.nn.RNN(util.DIM, 5000, 2)
        self.dense = torch.nn.Sequential(
            # torch.nn.Linear(5000, 5000),
            # torch.nn.ReLU(),
            # torch.nn.Linear(5000, 5000),
            # torch.nn.ReLU(),
            torch.nn.Linear(5000, 5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000, util.DIM),
            # torch.nn.Softmax(),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.dense(out)


class MidiTransformer(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self._pos_enc = PositionalEncoding(util.DIM, 0.1, k)
        self._transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                util.DIM,
                16,
            ),
            10,
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(util.DIM, util.DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(util.DIM, util.DIM),
            torch.nn.Softmax(),
        )

    def forward(self, x):
        return self._transformer(self._pos_enc(x))


def fit(
    model: torch.nn.Module,
    ds: tdata.Dataset | tdata.IterableDataset,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.003,
    device: str = None,
) -> None:
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    model = model.to(device)

    model.train()

    dl = tdata.DataLoader(ds, batch_size=batch_size)
    loss_fn = torch.nn.functional.cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    size = len(dl.dataset)

    for epoch in range(epochs):
        for batch, (x, y) in enumerate(dl):
            t0 = time.time_ns()

            x, y = x.to(device), y.to(device)

            t1 = time.time_ns()

            pred = model(x)

            t2 = time.time_ns()

            loss = loss_fn(pred, y)

            t3 = time.time_ns()

            loss.backward()

            t4 = time.time_ns()

            opt.step()

            t5 = time.time_ns()

            opt.zero_grad()

            t6 = time.time_ns()

            diffs = t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5

            if batch % 1 == 0:
                loss, current = loss.item(), batch * batch_size + len(x)
                print(
                    f"epoch: [{epoch} / {epochs}] loss: {loss} [{current}/{size}]",
                    end="\n",
                )
                print(f"times: {[d / 1e9 for d in diffs]}")


if __name__ == "__main__":
    k = 100
    model = MidiRNN(k)
    ds = MidiDataset("data/maestro_full.dat", k)
    fit(model, ds, 100, 64, 0.003)
