from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import TokenPosEmbedding
from .block import DecoderBlock

class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenPosEmbedding(cfg.vocab_size, cfg.n_embd, cfg.block_size, learned=cfg.pos_learned)
        self.blocks = nn.Sequential(*[
            DecoderBlock(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.block_size)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    @torch.no_grad()
    def _slice_ctx(self, idx: torch.Tensor) -> torch.Tensor:
        return idx[:, -self.cfg.block_size:]

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Dönüş: (logits, loss|None)."""
        x = self.embed(idx) 
        x = self.blocks(x)      
        x = self.ln_f(x)
        logits = self.lm_head(x)   

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
