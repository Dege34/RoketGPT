# modelroketsan.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout, block_size):
        super().__init__()
        # Key, Query, Value; input dim = n_embd, output dim = head_size
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal mask için alt üçgensel matris
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # scaled dot-product attention (scale by head dimension, not model dim)
        wei = q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5)  # (B, T, T)
        mask = self.tril[:T, :T]
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, head_size)
        return wei @ v     # (B, T, head_size)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        # Her head, input dim = n_embd, output = head_size
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, dropout, block_size)
            for _ in range(n_head)
        ])
        # proj sonrası = tekrar n_embd boyutuna getir
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # her head’ten geç, concat et, proje et
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHeadAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd= FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int    = 512,
        n_layer: int   = 8,
        n_head: int    = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.block_size = block_size
        # token + positional embedding
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # transformer blokları
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, dropout, block_size)
            for _ in range(n_layer)
        ])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)       # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device)        # (T,)
        pos_emb = self.position_embedding_table(pos)    # (T, n_embd)
        x = tok_emb + pos_emb                           # (B, T, n_embd)
        x = self.blocks(x)                              # (B, T, n_embd)
        x = self.ln_f(x)                                # (B, T, n_embd)
        logits = self.lm_head(x)                        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T, C),
                targets.view(B*T)
            )
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
