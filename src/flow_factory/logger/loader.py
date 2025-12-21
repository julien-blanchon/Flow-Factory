from typing import Optional, Union
from .abc import Logger, LogImage, LogVideo
from .swanlab import SwanlabLogger
from .wandb import WandbLogger


def load_logger(config) -> Union[Logger, None]:
    """Load and initialize the appropriate logger based on configuration."""
    if config.logging_backend == 'wandb':
        return WandbLogger(config=config)
    elif config.logging_backend == 'swanlab':
        return SwanlabLogger(config=config)
    else:
        return None