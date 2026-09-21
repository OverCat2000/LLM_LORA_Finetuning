import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATASET = "HuggingFaceFW/fineweb"
CONFIG = "sample-10BT"
SPLIT = "train"

OUT_DIR = Path("data/fineweb_2.5B")

TARGET_TOKENS = 2_500_000_000
SHARD_TOKENS = 100_000_000
VAL_TOKENS = 10_000_000

MIN_CHARS = 200
MIN_LANGUAGE_SCORE = 0.9

NUM_WORKERS = 8

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]
DTYPE = np.uint16


def keep(doc):
    if len(doc["text"]) < MIN_CHARS:
        return False
    if doc.get("language_score", 1.0) < MIN_LANGUAGE_SCORE:
        return False
    return True


def tokenize(doc):
    ids = enc.encode_ordinary(doc["text"])
    ids.append(EOT)
    return np.array(ids, dtype=DTYPE)


def shard_capacity(i):
    return VAL_TOKENS if i == 0 else SHARD_TOKENS


def shard_path(i):
    name = "val" if i == 0 else "train"
    return OUT_DIR / f"{name}_{i:06d}.bin"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")

    with ctx.Pool(NUM_WORKERS) as pool:
        print("resolving dataset (first documents take ~1 min to arrive)...")
        ds = load_dataset(DATASET, name=CONFIG, split=SPLIT, streaming=True)
        docs = (d for d in ds if keep(d))

        shard_idx = 0
        buf = np.empty(shard_capacity(0), dtype=DTYPE)
        used = 0
        total = 0
        n_docs = 0
        written = []
        t0 = time.time()
        pbar = tqdm(total=TARGET_TOKENS, unit="tok", unit_scale=True, mininterval=1.0)

        for tokens in pool.imap(tokenize, docs, chunksize=16):
            n_docs += 1
            pos = 0
            while pos < len(tokens):
                take = min(len(buf) - used, len(tokens) - pos)
                buf[used : used + take] = tokens[pos : pos + take]
                used += take
                pos += take
                total += take
                pbar.update(take)

                if used == len(buf):
                    buf.tofile(shard_path(shard_idx))
                    written.append((shard_path(shard_idx).name, used))
                    shard_idx += 1
                    if len(buf) != shard_capacity(shard_idx):
                        buf = np.empty(shard_capacity(shard_idx), dtype=DTYPE)
                    used = 0

            pbar.set_postfix(docs=n_docs, refresh=False)
            if total >= TARGET_TOKENS:
                break

        pbar.close()
        pool.terminate()
        pool.join()

    if used > 0:
        buf[:used].tofile(shard_path(shard_idx))
        written.append((shard_path(shard_idx).name, used))

    elapsed = time.time() - t0
    print(f"\n{total:,} tokens from {n_docs:,} documents -> {OUT_DIR}")
    for name, count in written:
        print(f"  {name}  {count:,} tokens")
    print(f"  {elapsed:.0f}s  ({total / max(elapsed, 1) / 1e6:.2f}M tok/s)")
    print(f"  vocab_size = {enc.n_vocab}")


if __name__ == "__main__":
    main()
