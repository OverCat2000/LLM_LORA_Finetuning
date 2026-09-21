"""
Pull the tokenized FineWeb shards from S3 onto the GPU box.

Run on the Vast.ai instance:
    pip install boto3 python-dotenv tqdm
    python download_data.py

Expects a .env next to this script (gitignored):
    AWS_ACCESS_KEY_ID=...        # read-only IAM user scoped to this bucket
    AWS_SECRET_ACCESS_KEY=...
    AWS_REGION=us-east-1
    DATA_BUCKET=my-fineweb-2-5b-data-f212ca8d
    DATA_PREFIX=fineweb_2.5B
    DATA_DIR=/workspace/data/fineweb_2.5B

Resumable: files already on disk at the right size are skipped.
"""

import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
from tqdm import tqdm

# Reads .env into os.environ. Existing env vars win, so an exported value
# still overrides the file. Must run before the constants below.
load_dotenv()

BUCKET = os.environ.get("DATA_BUCKET", "my-fineweb-2-5b-data-f212ca8d")
PREFIX = os.environ.get("DATA_PREFIX", "fineweb_2.5B")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Vast.ai containers usually have a big volume at /workspace and a small root fs.
LOCAL_DIR = Path(os.environ.get("DATA_DIR", "/workspace/data/fineweb_2.5B"))

# Files downloaded in parallel; each file also uses multipart threads internally.
MAX_PARALLEL_FILES = 4

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,
    multipart_chunksize=50 * 1024 * 1024,
    max_concurrency=10,
    use_threads=True,
)


def gb(n_bytes):
    return n_bytes / 1024**3


def list_remote():
    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".bin"):
                objects.append((obj["Key"], obj["Size"]))
    return sorted(objects)


class ProgressCallback:
    """Feeds transferred byte counts into a shared tqdm bar.

    boto3 calls this from several transfer threads at once, so updates take a
    lock — tqdm's counter is not thread-safe on its own.
    """

    def __init__(self, bar, lock):
        self.bar = bar
        self.lock = lock

    def __call__(self, bytes_transferred):
        with self.lock:
            self.bar.update(bytes_transferred)


def download_one(key, size, bar, lock):
    # Each thread needs its own client; boto3 clients are not thread-safe.
    s3 = boto3.client("s3", region_name=REGION)
    dest = LOCAL_DIR / Path(key).name

    if dest.exists() and dest.stat().st_size == size:
        return f"skip   {dest.name}  (already complete)"

    tmp = dest.with_suffix(".bin.part")
    s3.download_file(
        BUCKET,
        key,
        str(tmp),
        Config=TRANSFER_CONFIG,
        Callback=ProgressCallback(bar, lock),
    )
    tmp.rename(dest)
    return f"done   {dest.name}  ({size / 1024**2:.0f} MB)"


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        remote = list_remote()
    except NoCredentialsError:
        sys.exit(
            "No AWS credentials found. Export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY first."
        )

    if not remote:
        sys.exit(f"No .bin objects under s3://{BUCKET}/{PREFIX}")

    total_bytes = sum(size for _, size in remote)
    already = sum(
        (LOCAL_DIR / Path(k).name).stat().st_size
        for k, s in remote
        if (LOCAL_DIR / Path(k).name).exists()
        and (LOCAL_DIR / Path(k).name).stat().st_size == s
    )
    free = shutil.disk_usage(LOCAL_DIR).free

    print(f"{len(remote)} shards, {gb(total_bytes):.2f} GB total")
    print(
        f"{gb(total_bytes - already):.2f} GB to fetch, {gb(free):.2f} GB free on {LOCAL_DIR}"
    )

    if total_bytes - already > free:
        sys.exit("Not enough free disk. Resize the instance volume or trim shards.")

    to_fetch = total_bytes - already
    lock = threading.Lock()
    bar = tqdm(
        total=to_fetch,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="downloading",
        smoothing=0.1,
    )

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FILES) as pool:
        futures = [
            pool.submit(download_one, key, size, bar, lock) for key, size in remote
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            # tqdm.write keeps the bar from being chopped up by these lines.
            tqdm.write(f"[{i}/{len(futures)}] {fut.result()}")

    bar.close()

    # Token counts, so you can eyeball that nothing is truncated (uint16 = 2 bytes/token).
    print("\nlocal shards:")
    grand = 0
    for path in sorted(LOCAL_DIR.glob("*.bin")):
        n = path.stat().st_size // 2
        grand += n
        print(f"  {path.name}  {n:,} tokens")
    print(f"  total {grand:,} tokens")


if __name__ == "__main__":
    main()
