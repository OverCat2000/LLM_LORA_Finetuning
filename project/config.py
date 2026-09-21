from dataclasses import asdict, dataclass


@dataclass
class Config:
    # data_dir: str = "/workspace/data/fineweb_2.5B"
    data_dir: str = "./data/data/fineweb_2.5B"
    seq_len: int = 1024
    batch_size: int = 8
    grad_accum_steps: int = 4
    num_workers: int = 4

    model_type: str = "hf-gpt2"
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 700
    max_steps: int = 20_000

    eval_every: int = 500
    eval_batches: int = 50
    ckpt_every: int = 1000
    keep_last: int = 3
    out_dir: str = "/workspace/checkpoints"
    s3_bucket: str = ""
    s3_prefix: str = "checkpoints"
    resume: str = ""

    device: str = "cuda"
    compile: bool = True
    seed: int = 1337
    log_every: int = 10

    def to_dict(self):
        return asdict(self)

    @property
    def tokens_per_step(self):
        return self.batch_size * self.grad_accum_steps * self.seq_len
