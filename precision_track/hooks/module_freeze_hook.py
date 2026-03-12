from mmengine.hooks import Hook
from mmengine.logging import print_log
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

    def before_optim_wrapper(self, runner, *args, **kwargs) -> None:
        for module_to_freeze in self.modules_to_freeze:
            if hasattr(runner.model, module_to_freeze):
                print_log(f"Freezing -- {module_to_freeze}", logger="current")
                freeze_model_part(getattr(runner.model, module_to_freeze))
