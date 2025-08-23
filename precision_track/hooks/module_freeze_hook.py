from mmengine.hooks import Hook

from precision_track.registry import HOOKS
from precision_track.utils import freeze_model_part


@HOOKS.register_module()
class ModuleFreezingHook(Hook):
    def __init__(
        self,
        modules_to_freeze: list,
    ):
        assert isinstance(modules_to_freeze, list)
        self.modules_to_freeze = modules_to_freeze

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        for module_to_freeze in self.modules_to_freeze:
            if hasattr(runner.model, module_to_freeze):
                freeze_model_part(getattr(runner.model, module_to_freeze))
