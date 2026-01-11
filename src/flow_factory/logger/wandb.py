# src/flow_factory/logger/wandb.py
from typing import Any, Dict
import wandb
from .abc import Logger
from .formatting import LogImage, LogVideo, LogTable


class WandbLogger(Logger):
    def _init_platform(self):
        wandb.init(
            project=self.config.project,
            name=self.config.run_name,
            config=self.config.to_dict()
        )
        self.platform = wandb

    def _convert_to_platform(self, value: Any) -> Any:
        if isinstance(value, LogImage):
            return wandb.Image(value.value, caption=value.caption)
        elif isinstance(value, LogVideo):
            return wandb.Video(value.value, caption=value.caption, format=value.format)
        elif isinstance(value, LogTable):
            data = [
                [self._convert_to_platform(item) for item in row]
                for row in value.rows
            ]
            return wandb.Table(columns=value.columns, data=data)
        return value        

    def _log_impl(self, data: Dict, step: int):
        self.platform.log(data, step=step)