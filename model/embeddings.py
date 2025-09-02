import torch
import torch.nn as nn

class TokenPosEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, learned: bool = True):
        super().__init__()
        self.token = nn.Embedding(vocab_size, n_embd)
        if learned:
            self.pos = nn.Embedding(block_size, n_embd)
            self.register_buffer("pos_ids", torch.arange(block_size), persistent=False)
            self.is_learned = True
        else:
            self.pos = None
            self.is_learned = False

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        tok = self.token(idx) 
        if self.is_learned:
            T = idx.size(1)
            pos = self.pos(self.pos_ids[:T])
            return tok + pos
        return tok
