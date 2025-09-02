"""
roketgpt package initializer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

try:
    from importlib.metadata import PackageNotFoundError, version 
except Exception: 
    PackageNotFoundError = Exception  

try:
    __version__: str = version("roketgpt")  
except Exception:  
  
    try:
        from ._version import __version__  
    except Exception:
        __version__ = "0.0.0.dev0"


# public api
__all__ = [
    "__version__",
    "load_model",
    "Retriever",
    "Memory",
    "set_seed",
    "get_logger",
]

if TYPE_CHECKING: 
    from .model import load_model
    from .retrieval import Retriever
    from .memory import Memory


def __getattr__(name: str) -> Any:
    """Lazily import heavy modules only when needed."""
    mapping: Dict[str, Tuple[str, str]] = {
        "load_model": ("roketgpt.model", "load_model"),
        "Retriever": ("roketgpt.retrieval", "Retriever"),
        "Memory": ("roketgpt.memory", "Memory"),
    }
    if name in mapping:
        module_name, attr = mapping[name]
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value  
        return value
    raise AttributeError(f"module 'roketgpt' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


def get_logger(name: str = "roketgpt"):
    """Get a configured logger with a NullHandler by default."""
    import logging
    import os

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    level = os.getenv("ROKETGPT_LOGLEVEL")
    if level:
        try:
            logger.setLevel(getattr(logging, level.upper()))
        except Exception:
            logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = 42) -> None:
    """Best-effort global seeding (Python, NumPy, PyTorch if available)."""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np 

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch 

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    except Exception:
        pass
