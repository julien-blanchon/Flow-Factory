# src/flow_factory/logger/abc.py
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any

from ..hparams import *
from ..models.adapter import BaseSample
from .formatting import LogFormatter, LogImage, LogVideo

class Logger(ABC):
    platform: Any

    def __init__(self, config: Arguments):
        self.config = config
        self._init_platform()

    @abstractmethod
    def _init_platform(self):
        pass

    def log_data(
        self,
        samples: Union[Dict, BaseSample],
        step: int,
        keys: Optional[str] = None,
    ):
        """
        Unified logging entry point.
        1. Formats data (paths -> Media, lists -> mean).
        2. Converts generic Media objects to platform specific objects (wandb.Image, etc).
        3. Pushes to platform.
        """
        # 1. Process rules (Mean, Paths, wrappers)
        formatted_dict = LogFormatter.format_dict(samples)
        
        # 2. Filter keys if requested
        if keys:
            valid_keys = keys.split(',')
            formatted_dict = {k: v for k, v in formatted_dict.items() if k in valid_keys}

        # 3. Convert Intermediate Representations (LogImage/LogVideo) to Platform Objects (wandb.Image)
        final_dict = {}
        for k, v in formatted_dict.items():
            final_dict[k] = self._convert_to_platform(v)

        # 4. Actual Logging
        self._log_impl(final_dict, step)

    @abstractmethod
    def _convert_to_platform(self, value: Any) -> Any:
        """Convert LogImage/LogVideo to wandb.Image/swanlab.Image etc."""
        pass

    @abstractmethod
    def _log_impl(self, data: Dict, step: int):
        """Call the specific platform log function."""
        pass