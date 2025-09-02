import torch
from .config import ModelConfig
from .gpt import GPTLanguageModel

def build_model(cfg: ModelConfig) -> GPTLanguageModel:
    return GPTLanguageModel(cfg)

def save_checkpoint(path: str, model: GPTLanguageModel, cfg: ModelConfig, optimizer=None, extra: dict | None = None):
    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)

def load_checkpoint(path: str, device: str = "cpu") -> tuple[GPTLanguageModel, ModelConfig, dict]:
    blob = torch.load(path, map_location=device)
    cfg = ModelConfig(**blob["config"])
    model = GPTLanguageModel(cfg)
    model.load_state_dict(blob["model_state_dict"], strict=False)
    model.to(device)
    return model, cfg, blob.get("extra", {})
