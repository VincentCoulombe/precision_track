import os
from typing import Dict, Optional

from mmengine.hooks import Hook
from mmengine.logging import print_log

from precision_track.registry import HOOKS
from precision_track.utils import find_checkpoint_hook


def on_colab() -> bool:
    """Whether the current process runs inside a Google Colab runtime.

    The ``google.colab`` module is importable in some environments that are not Colab
    (Kaggle ships it), so the runtime's environment variables are what actually tell the
    two apart.
    """
    if not ("COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ):
        return False
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


@HOOKS.register_module()
class ColabCheckpointHook(Hook):
    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        if not on_colab():
            return
        from google.colab import files

        checkpoint_hook = find_checkpoint_hook(runner)
        if checkpoint_hook is None:
            return

        # `interval=-1` disables periodic saves and `save_best` only writes once a metric
        # has improved, so either attribute may still be unset at the first validation.
        ckpt_path = getattr(checkpoint_hook, "last_ckpt", None) or getattr(checkpoint_hook, "best_ckpt_path", None)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            return

        try:
            files.download(ckpt_path)
            print_log(logger="current", msg=f"Downloaded the '{ckpt_path}' checkpoint file.")
        except Exception:
            print_log(
                logger="current",
                msg=f"Could not download your current best training checkpoint. " f"You can still download it manually. It is located at '{ckpt_path}'.",
            )
