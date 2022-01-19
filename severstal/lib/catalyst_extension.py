from catalyst.callbacks import CheckpointCallback, DiceCallback
from typing import Dict, Union
import os
from pathlib import Path
from catalyst.core.runner import IRunner
import torch
from datetime import datetime

class CustomCheckpointCallback(CheckpointCallback):
    """
    Eliminates checkpoint saving on each epoch, because, when removed on next epoch,
    they go to google drive recycle bin and quickly overflow it
    """
    @staticmethod
    def perform_saving(
        checkpoint: Dict,
        runner: "IRunner",
        logdir: Union[Path, str],
        suffix: str,
        is_best: bool = False,
        is_last: bool = False,
        extra_suffix: str = "",
    ) -> Union[Path, str]:
        os.makedirs(logdir, exist_ok=True)
        if is_best:
            torch.save(checkpoint, f"{logdir}/best{extra_suffix}.pth")
            if extra_suffix == "":
                print('new optimum is achieved', datetime.now())
        if is_last:
            torch.save(checkpoint, f"{logdir}/last{extra_suffix}.pth")

        temp_fname = f"{logdir}/temp{extra_suffix}.pth"
        open(temp_fname, 'w').close()
        return temp_fname

    def _save_checkpoint(
        self, runner: IRunner, checkpoint: Dict, is_best: bool, is_last: bool
    ) -> str:
        logdir = Path(f"{self.logdir}/")
        suffix = f"{runner.stage_key}.{runner.stage_epoch_step}"
        checkpoint_path = None
        if self.mode in ("all", "full"):
            checkpoint_path = CustomCheckpointCallback.perform_saving(
                runner=runner,
                logdir=logdir,
                checkpoint=checkpoint,
                suffix=f"{suffix}_full",
                is_best=is_best,
                is_last=is_last,
                extra_suffix="_full",
            )
        if self.mode in ("all", "model"):
            exclude = ["criterion", "optimizer", "scheduler"]
            checkpoint_path = CustomCheckpointCallback.perform_saving(
                runner=runner,
                checkpoint={
                    key: value
                    for key, value in checkpoint.items()
                    if all(z not in key for z in exclude)
                },
                logdir=logdir,
                suffix=suffix,
                is_best=is_best,
                is_last=is_last,
            )
        return checkpoint_path
