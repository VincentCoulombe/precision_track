resume = False
load_from = None
backend_args = None
default_scope = "mmengine"
work_dir = "../work_dir/"
auto_scale_lr = dict(base_batch_size=256, enable=True)
compile = False
data_mode = "bottomup"


# hooks
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(interval=-1, type="CheckpointHook", save_best="val/loss", rule="less"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=False),
    gc=dict(type="EmptyCacheHook"),
)

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type="SyncBuffersHook")
]

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

# logger
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True, num_digits=6)
log_level = "INFO"
visualizer = dict()
