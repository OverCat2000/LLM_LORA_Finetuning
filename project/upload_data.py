import os
import boto3
from boto3.s3.transfer import TransferConfig

BUCKET_NAME = "my-fineweb-2-5b-data-f212ca8d"
LOCAL_DIR = "./data/data/fineweb_2.5B"
S3_PREFIX = "fineweb_2.5B"
REGION = "us-east-1"

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,
    multipart_chunksize=50 * 1024 * 1024,
    max_concurrency=10,
    use_threads=True,
)

s3 = boto3.client("s3", region_name=REGION)


def upload_files():
    files = sorted(f for f in os.listdir(LOCAL_DIR) if f.endswith(".bin"))
    print(f"Found {len(files)} files to upload.")

    for fname in files:
        local_path = os.path.join(LOCAL_DIR, fname)
        s3_key = f"{S3_PREFIX}/{fname}" if S3_PREFIX else fname
        size_mb = os.path.getsize(local_path) / 1024 / 1024

        print(f"Uploading {fname} ({size_mb:.1f} MB) → s3://{BUCKET_NAME}/{s3_key}")
        s3.upload_file(local_path, BUCKET_NAME, s3_key, Config=TRANSFER_CONFIG)

    print("✅ Done. All files uploaded.")


if __name__ == "__main__":
    upload_files()
