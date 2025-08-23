from mmengine.hooks import Hook

from precision_track.registry import HOOKS


@HOOKS.register_module()
class ValidateBeforeTrainingHook(Hook):
    def __init__(self):
        pass

    def before_train(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before train.

        Args:
            runner (Runner): The runner of the training process.
        """
        runner.val_loop.run()
