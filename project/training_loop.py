import math
import time

import torch

import checkpoint
from optim import lr_at, set_lr


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, cfg, model, optimizer, train_loader, val_loader, start_step=0):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.batches = infinite(train_loader)
        self.step = start_step
        self.best_val = float("inf")
        self.autocast = torch.autocast(
            device_type="cuda" if cfg.device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=cfg.device.startswith("cuda"),
        )

    def train_step(self):
        cfg = self.cfg
        self.model.train()
        lr = set_lr(self.optimizer, lr_at(self.step, cfg))
        self.optimizer.zero_grad(set_to_none=True)

        total = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = next(self.batches)
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            with self.autocast:
                _, loss = self.model(x, y)
                loss = loss / cfg.grad_accum_steps
            loss.backward()
            total += loss.item()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.optimizer.step()
        return total, lr

    @torch.no_grad()
    def evaluate(self):
        cfg = self.cfg
        self.model.eval()
        losses = []
        for i, (x, y) in enumerate(self.val_loader):
            if i >= cfg.eval_batches:
                break
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            with self.autocast:
                _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    def fit(self):
        cfg = self.cfg
        print(
            f"training step {self.step:,} -> {cfg.max_steps:,}, {cfg.tokens_per_step:,} tokens/step"
        )
        t0 = time.time()
        running = 0.0

        while self.step < cfg.max_steps:
            loss, lr = self.train_step()
            running += loss
            self.step += 1

            if self.step % cfg.log_every == 0:
                dt = time.time() - t0
                avg = running / cfg.log_every
                tok_s = cfg.log_every * cfg.tokens_per_step / dt
                eta = (cfg.max_steps - self.step) * dt / cfg.log_every / 3600
                print(
                    f"step {self.step:>7,}  loss {avg:.4f}  ppl {math.exp(min(avg, 20)):>8.1f}  "
                    f"lr {lr:.2e}  {tok_s / 1e3:.1f}k tok/s  eta {eta:.1f}h",
                    flush=True,
                )
                running = 0.0
                t0 = time.time()

            if self.step % cfg.eval_every == 0:
                val = self.evaluate()
                if val < self.best_val:
                    self.best_val = val
                    checkpoint.save(
                        cfg, self.model, self.optimizer, self.step, val, tag="best"
                    )
                print(f"  eval {self.step:,} val {val:.4f}", flush=True)
                t0 = time.time()

            if self.step % cfg.ckpt_every == 0:
                checkpoint.save(
                    cfg, self.model, self.optimizer, self.step, self.best_val
                )
                t0 = time.time()

        val = self.evaluate()
        checkpoint.save(cfg, self.model, self.optimizer, self.step, val, tag="final")
        print(f"done: final val {val:.4f}, best {self.best_val:.4f}")
