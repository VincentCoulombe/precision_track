from typing import Dict, Optional
import os
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.hooks import CheckpointHook

from precision_track.registry import HOOKS


@HOOKS.register_module()
class ColabCheckpointHook(Hook):

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        try:
            from google.colab import files

            on_colab = True
        except ImportError:
            on_colab = False

        checkpoint_hook = self._find_checkpoint_hook(runner)
        if on_colab and checkpoint_hook is not None:
            for key_indicator in checkpoint_hook.key_indicators:
                if len(checkpoint_hook.key_indicators) == 1:
                    best_ckpt_path = checkpoint_hook.best_ckpt_path
                else:
                    best_ckpt_path = checkpoint_hook.best_ckpt_path_dict[key_indicator]

                best_ckpt_path = os.path.abspath(best_ckpt_path)

                try:
                    files.download(best_ckpt_path)
                    print_log(logger="current", msg=f"Downloaded the '{best_ckpt_path}' checkpoint file.")
                except AttributeError:
                    print_log(
                        logger="current",
                        msg=f"Could not download your current best training checkpoint. "
                        f"You can still download it manually. It is located at '{best_ckpt_path}'.",
                    )

    def after_val_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        self.after_val_epoch(runner)

    @staticmethod
    def _find_checkpoint_hook(runner):
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                return hook
        return None
