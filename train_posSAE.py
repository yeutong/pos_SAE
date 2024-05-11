# %%
from torch import nn
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

from transformer_lens.utils import get_act_name


# %%
@dataclass
class Config:
    hook_name = get_act_name("resid_pre", 0)
    max_epochs = 4

    batch_size = 32
    lr = 1e-4


class SAESelector(nn.Module):
    def __init__(self, cfg, sae):
        super().__init__()
        self.cfg = cfg
        self.d_sae = sae.cfg.d_sae

    def train(self, sampler, model):
        for epoch in range(self.cfg.max_epochs):
            loader = DataLoader(
                sampler, batch_size=self.config.out_batch, shuffle=True, drop_last=True
            )
