# src/flow_factory/logger/wandb.py
from abc import ABC, abstractmethod
from typing import List, Union, Any, Optional, Dict

from ..hparams import *
from ..models.adapter import BaseSample
from .abc import Logger

class WandbLogger(Logger):
    def __init__(self, config : Arguments):
        """Initialize the logger with an optional run name."""
        self.config = config

    def _init_platform(self):
        import wandb
        self.platform = wandb

    def resume_from(self):
        pass

    def log_data(
        self,
        samples: Union[Dict, BaseSample],
        step: int,
        keys: Optional[str] = None,
    ):
        pass