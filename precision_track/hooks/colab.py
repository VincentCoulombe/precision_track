from typing import Dict, Optional

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.runner import find_latest_checkpoint

from precision_track.registry import HOOKS


@HOOKS.register_module()
class ColabCheckpointHook(Hook):

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        try:
            from google.colab import files  # noqa

            on_colab = True
        except ImportError:
            on_colab = False

        if on_colab:
            latest_checkpoint = find_latest_checkpoint(runner.work_dir)
            if latest_checkpoint:
                files.download(latest_checkpoint)
                print_log(logger="current", msg=f"Downloaded the '{latest_checkpoint}' checkpoint file.")
