import argparse
import dataclasses

import torch
from dotenv import load_dotenv

import checkpoint
from config import Config

# from fineweb_dataset_clean import make_loader
from torch_dataloader import make_loader
from model import build_model
from optim import configure_optimizer
from training_loop import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        if isinstance(f.default, bool):
            p.add_argument(
                f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes")
            )
        else:
            p.add_argument(f"--{f.name}", type=type(f.default))
    return Config(**{k: v for k, v in vars(p.parse_args()).items() if v is not None})


def main():
    load_dotenv()
    cfg = parse_args()

    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device, cfg.compile = "cpu", False

    train_loader = make_loader(
        "train",
        cfg.batch_size,
        cfg.seq_len,
        num_workers=cfg.num_workers,
        data_dir=cfg.data_dir,
    )
    val_loader = make_loader(
        "val", cfg.batch_size, cfg.seq_len, num_workers=2, data_dir=cfg.data_dir
    )

    model = build_model(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.device}  {cfg.model_type}  {n_params / 1e6:.1f}M params")
    print(
        f"train {len(train_loader.dataset):,} samples  val {len(val_loader.dataset):,} samples"
    )

    optimizer = configure_optimizer(model, cfg)

    start_step = 0
    path = checkpoint.resolve(cfg)
    if path:
        start_step = checkpoint.load(path, model, optimizer, cfg.device)

    if cfg.compile:
        model = torch.compile(model)

    Trainer(cfg, model, optimizer, train_loader, val_loader, start_step).fit()


if __name__ == "__main__":
    main()
