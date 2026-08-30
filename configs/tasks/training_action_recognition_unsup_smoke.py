_base_ = "./training_action_recognition_unsup.py"

# A short end-to-end exercise of the unsupervised (MAE) pretraining path: sequence
# generation -> masked reconstruction -> MSEMetric validation -> checkpointing.
# It validates the wiring, not the model. Use ./training_action_recognition_unsup.py
# for real pretraining.

# Settings
num_iter = 20
warmup_iter = 5
val_interval = 10
base_lr = _base_.base_lr

# Sequence generation runs the detector frame by frame (~3s/frame on CPU), so the bundled
# 508-frame clip costs ~26 min. Trim a shorter one to iterate faster:
#   ffmpeg -i ../assets/20mice_sanity_check.avi -frames:v 60 -c:v copy ../assets/smoke_60f.avi
# then point this at it. Real pretraining uses training_action_recognition_unsup.py.
smoke_sequences = ["../assets/20mice_sanity_check.avi"]
# /Settings

# Loops
train_cfg = dict(
    type="IterBasedTrainLoop",
    _scope_=_base_.default_scope,
    val_interval=val_interval,
    max_iters=num_iter,
)
# /Loops

# Optimization
# Recomputed here: the base evaluates these bounds against its own num_iter at import
# time, so overriding num_iter alone would leave the schedules pointing past the run.
param_scheduler = [
    dict(
        type="QuadraticWarmupLR",
        by_epoch=False,
        begin=0,
        end=warmup_iter,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr / 100,
        begin=warmup_iter,
        T_max=9 * (num_iter // 10),
        end=9 * (num_iter // 10),
        by_epoch=False,
    ),
    dict(type="ConstantLR", by_epoch=False, factor=1, begin=9 * (num_iter // 10), end=num_iter),
]
# /Optimization

# CPU-only: the base uses AmpOptimWrapper (fp16), which asserts a CUDA/NPU/MLU device.
# Plain OptimWrapper is the CPU equivalent. Drop this override when running on a GPU.
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=_base_.weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    _delete_=True,
)

# Dataloaders
# The base validation set walks 50k samples, which would dwarf a 20-iteration run.
# num_workers=0 is not a speed choice: under the spawn start method every worker rebuilds
# the whole MAEDataset, including its own ONNX detector session. Eight of them exhausted a
# 7.8GB box and the kernel OOM-killed the run. batch_size 128 is likewise GPU-sized.
# tracking_batch_size is the number of raw frames held in memory during sequence
# generation. At this footage's 2720x2720 that is ~22MB/frame, so the default 30 costs
# ~660MB per dataset before activations -- and train and val each build their own detector.
train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_prefix=dict(sequences=smoke_sequences),
        tracking_batch_size=4,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_prefix=dict(sequences=smoke_sequences),
        tracking_batch_size=4,
        custom_length=5,
    ),
)
# /Dataloaders
