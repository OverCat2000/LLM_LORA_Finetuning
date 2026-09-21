import os
import re
from pathlib import Path

import torch

from model import unwrap


def _s3():
    import boto3

    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def save(cfg, model, optimizer, step, val_loss, tag=None, run_id=""):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"ckpt_{tag}.pt" if tag else f"ckpt_{step:07d}.pt"
    path = out_dir / name

    torch.save(
        {
            "step": step,
            "val_loss": val_loss,
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "run_id": run_id,
        },
        path,
    )
    print(f"  saved {path} ({path.stat().st_size / 1024**2:.0f} MB)")

    if cfg.s3_bucket:
        key = f"{cfg.s3_prefix}/{name}" if cfg.s3_prefix else name
        try:
            _s3().upload_file(str(path), cfg.s3_bucket, key)
            print(f"  uploaded s3://{cfg.s3_bucket}/{key}")
        except Exception as e:
            print(f"  WARNING: S3 upload failed ({e})")

    if cfg.keep_last > 0:
        steps = sorted(
            (
                p
                for p in out_dir.glob("ckpt_*.pt")
                if re.fullmatch(r"ckpt_\d+\.pt", p.name)
            ),
            key=lambda p: p.stat().st_mtime,
        )
        for old in steps[: -cfg.keep_last]:
            old.unlink()

    return path


def resolve(cfg):
    if not cfg.resume:
        return None

    if cfg.resume.startswith("s3://"):
        bucket, key = cfg.resume[5:].split("/", 1)
        dest = Path(cfg.out_dir) / Path(key).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        _s3().download_file(bucket, key, str(dest))
        return dest

    if cfg.resume == "latest":
        local = sorted(
            Path(cfg.out_dir).glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime
        )
        if local:
            return local[-1]
        if cfg.s3_bucket:
            objs = []
            for page in (
                _s3()
                .get_paginator("list_objects_v2")
                .paginate(Bucket=cfg.s3_bucket, Prefix=cfg.s3_prefix)
            ):
                objs += [
                    o for o in page.get("Contents", []) if o["Key"].endswith(".pt")
                ]
            if objs:
                key = max(objs, key=lambda o: o["LastModified"])["Key"]
                dest = Path(cfg.out_dir) / Path(key).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                _s3().download_file(cfg.s3_bucket, key, str(dest))
                return dest
        return None

    path = Path(cfg.resume)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load(path, model, optimizer=None, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"resumed {path} at step {ckpt['step']:,}")
    return ckpt["step"], ckpt.get("run_id", "")
