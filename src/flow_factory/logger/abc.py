# src/flow_factory/logger/abc.py
from abc import ABC, abstractmethod
from typing import List, Union, Any, Optional, Dict

from ..hparams import *
from ..models.adapter import BaseSample

class Logger(ABC):
    platform : Any
    def __init__(self, config : Arguments):
        """Initialize the logger with an optional run name."""
        self.config = config
        
        self._init_platform()

    @abstractmethod
    def _init_platform(self):
        pass

    def resume_from(self):
        pass

    @abstractmethod
    def log_data(
        self,
        samples: Union[Dict, BaseSample],
        step: int,
        keys: Optional[str] = None,
    ):
        pass