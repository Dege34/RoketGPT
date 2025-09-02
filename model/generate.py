import torch
import torch.nn.functional as F

@torch.no_grad()
def sample(model, idx, max_new_tokens=128, temperature=1.0, top_k=None, top_p=None, repetition_penalty=None):
    for _ in range(max_new_tokens):
        idx_cond = model._slice_ctx(idx) if hasattr(model, "_slice_ctx") else idx
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if repetition_penalty is not None:
            for b in range(idx.size(0)):
                logits[b, idx[b]] /= repetition_penalty

        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)

        if top_k is not None:
            v, _ = torch.topk(probs, top_k)
            thr = v[:, -1].unsqueeze(-1)
            probs = torch.where(probs < thr, torch.zeros_like(probs), probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
