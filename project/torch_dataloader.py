import bisect
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data/data/fineweb_2.5B"))


class FineWebDataset(Dataset):
    def __init__(self, split, seq_len=1024, stride=None, data_dir=DATA_DIR):
        self.split = split
        self.seq_len = seq_len
        self.stride = stride or seq_len

        prefix = "val_" if split == "val" else "train_"
        self.shards = sorted(Path(data_dir).glob(f"{prefix}*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"no {prefix}*.bin in {data_dir}")

        self.shard_tokens = [p.stat().st_size // 2 for p in self.shards]

        self.shard_windows = []
        for n in self.shard_tokens:
            usable = n - self.seq_len - 1
            self.shard_windows.append(usable // self.stride + 1 if usable >= 0 else 0)

        self.cumulative = []
        running = 0
        for w in self.shard_windows:
            running += w
            self.cumulative.append(running)
        self.n_windows = running

        self._mmaps = {}

    def _shard(self, i):
        mm = self._mmaps.get(i)
        if mm is None:
            mm = np.memmap(self.shards[i], dtype=np.uint16, mode="r")
            self._mmaps[i] = mm
        return mm

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.n_windows
        if not 0 <= idx < self.n_windows:
            raise IndexError(idx)

        shard_i = bisect.bisect_right(self.cumulative, idx)
        prior = self.cumulative[shard_i - 1] if shard_i else 0
        start = (idx - prior) * self.stride

        tokens = self._shard(shard_i)
        buf = np.asarray(tokens[start : start + self.seq_len + 1]).astype(np.int64)
        return torch.from_numpy(buf[:-1]), torch.from_numpy(buf[1:])


def make_loader(
    split, batch_size, seq_len=1024, stride=None, num_workers=4, data_dir=DATA_DIR
):
    ds = FineWebDataset(split, seq_len=seq_len, stride=stride, data_dir=data_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
