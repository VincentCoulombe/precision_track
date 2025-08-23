from mmengine.hooks import Hook

from precision_track.registry import HOOKS


@HOOKS.register_module()
class SequencePreprocessingHook(Hook):
    def __init__(self):
        pass

    def before_val(self, runner) -> None:
        runner.model.data_preprocessor.mode = "sequence"

    def after_val(self, runner) -> None:
        runner.model.data_preprocessor.mode = "loss"
