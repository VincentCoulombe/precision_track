from mmengine.hooks import CheckpointHook


def find_checkpoint_hook(runner):
    for hook in runner.hooks:
        if isinstance(hook, CheckpointHook):
            return hook
    return None
