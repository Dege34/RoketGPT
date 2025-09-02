from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 1024
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.1
    pos_learned: bool = True 
