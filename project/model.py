import torch.nn as nn
import torch.nn.functional as F


class HFGPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import GPT2Config, GPT2LMHeadModel

        self.net = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=cfg.vocab_size,
                n_positions=cfg.seq_len,
                n_embd=cfg.n_embd,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                resid_pdrop=cfg.dropout,
                embd_pdrop=cfg.dropout,
                attn_pdrop=cfg.dropout,
            )
        )

    def forward(self, x, targets=None):
        logits = self.net(input_ids=x).logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss


def build_model(cfg):
    if cfg.model_type == "hf-gpt2":
        return HFGPT2(cfg)
    if cfg.model_type == "custom":
        from my_gpt import MyGPT

        return MyGPT(cfg)
    raise ValueError(cfg.model_type)


def unwrap(model):
    return getattr(model, "_orig_mod", model)
