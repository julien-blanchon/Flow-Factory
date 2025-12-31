# src/flow_factory/hparams/log_args.py
import os
import yaml
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional
from .abc import ArgABC


@dataclass
class LogArguments(ArgABC):
    r"""Arguments pertaining to data input for training and evaluation."""

    save_dir: str = field(
        default='save',
        metadata={"help": "Directory to save logs and checkpoints. None for no saving."},
    )

    save_freq: int = field(
        default=10,
        metadata={"help": "Model saving frequency (in epochs). 0 for no saving."},
    )
    
    save_model_only : bool = field(
        default=True,
        metadata={"help": "Whether to save the model only, or the complete training state (model and optimizer)."}
    )


    def __post_init__(self):

        # Expand path to user's path
        self.save_dir = os.path.expanduser(self.save_dir)
        self.resume_path = os.path.expanduser(self.resume_path) if self.resume_path is not None else None
        # If save_dir does not exist, create it
        os.makedirs(self.save_dir, exist_ok=True)


    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()