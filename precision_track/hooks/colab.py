from typing import Dict, Optional

from mmengine.hooks import Hook
from mmengine.logging import print_log

from precision_track.registry import HOOKS
from precision_track.utils import find_checkpoint_hook


@HOOKS.register_module()
class ColabCheckpointHook(Hook):

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        try:
            from google.colab import files

            on_colab = True
        except ImportError:
            on_colab = False

        checkpoint_hook = find_checkpoint_hook(runner)
        if on_colab and checkpoint_hook is not None:
            ckpt_path = checkpoint_hook.last_ckpt
            try:
                files.download(ckpt_path)
                print_log(logger="current", msg=f"Downloaded the '{ckpt_path}' checkpoint file.")
            except AttributeError:
                print_log(
                    logger="current",
                    msg=f"Could not download your current best training checkpoint. " f"You can still download it manually. It is located at '{ckpt_path}'.",
                )

    def after_val_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        self.after_val_epoch(runner)
