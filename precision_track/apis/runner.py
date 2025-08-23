import copy
import logging
import os
import os.path as osp
import platform
import warnings
from typing import Callable, Dict, List, Optional, Union

import cv2
import mmengine
import torch.multiprocessing as mp
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import LOOPS, VISUALIZERS, DefaultScope
from mmengine.runner import Runner as MMENGINERunner
from mmengine.utils import digit_version
from mmengine.visualization import Visualizer

from precision_track.registry import MODELS
from precision_track.utils import CheckpointLoader, load_checkpoint_to_model

warnings.simplefilter("ignore", category=FutureWarning)


class Runner(MMENGINERunner):

    def __init__(self, cfg: Union[str, Config], launcher: str, mode: str = "test"):
        self.mode = mode
        setup_cache_size_limit_of_dynamo()
        if isinstance(cfg, str):
            cfg = Config.fromfile(cfg)
        model_config = cfg.get("model")
        work_dir = cfg.get("work_dir")
        train_dataloader = cfg.get("train_dataloader")
        val_dataloader = cfg.get("val_dataloader")
        test_dataloader = cfg.get("test_dataloader")
        train_cfg = cfg.get("train_cfg")
        val_cfg = cfg.get("val_cfg")
        test_cfg = cfg.get("test_cfg")
        calibration_cfg = cfg.get("calibration_cfg")
        auto_scale_lr = cfg.get("auto_scale_lr")
        optim_wrapper = cfg.get("optim_wrapper")
        param_scheduler = cfg.get("param_scheduler")
        val_evaluator = cfg.get("val_evaluator")
        test_evaluator = cfg.get("test_evaluator")
        calibration_evaluator = cfg.get("calibration_evaluator")
        default_hooks = cfg.get("default_hooks")
        custom_hooks = cfg.get("custom_hooks")
        data_preprocessor = cfg.get("data_preprocessor")
        resume = cfg.get("resume", False)
        launcher = launcher
        env_cfg = cfg.get("env_cfg", dict(dist_cfg=dict(backend="nccl")))
        log_processor = cfg.get("log_processor")
        log_level = cfg.get("log_level", "INFO")
        visualizer = cfg.get("visualizer")
        default_scope = cfg.get("default_scope", "mmengine")
        randomness = cfg.get("randomness", dict(seed=None))
        experiment_name = cfg.get("experiment_name")
        load_from = None if mode == "test" else cfg.get("load_from")
        cfg = cfg

        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related) or all(item is not None for item in training_related)):
            raise ValueError(
                "train_dataloader, train_cfg, and optim_wrapper should be "
                "either all None or not None, but got "
                f"train_dataloader={train_dataloader}, "
                f"train_cfg={train_cfg}, "
                f"optim_wrapper={optim_wrapper}."
            )
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg
        self.optim_wrapper = optim_wrapper
        self.auto_scale_lr = auto_scale_lr

        if param_scheduler is not None and self.optim_wrapper is None:
            param_scheduler = None

        self._check_scheduler_cfg(param_scheduler)
        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg]
        if not (all(item is None for item in val_related) or all(item is not None for item in val_related)):
            raise ValueError(
                "val_dataloader and val_cfg should be either " "all None or not None, but got " f"val_dataloader={val_dataloader}, val_cfg={val_cfg}, "
            )
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related) or all(item is not None for item in test_related)):
            raise ValueError(
                "test_dataloader, test_cfg, and test_evaluator should be "
                "either all None or not None, but got "
                f"test_dataloader={test_dataloader}, test_cfg={test_cfg}, "
                f"test_evaluator={test_evaluator}"
            )
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        self._launcher = launcher
        if self._launcher == "none":
            self._distributed = False
        else:
            self._distributed = True

        self.setup_env(env_cfg)
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f"{experiment_name}_{self._timestamp}"
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f"{filename_no_ext}_{self._timestamp}"
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)
        if default_scope is not None:
            default_scope = DefaultScope.get_instance(self._experiment_name, scope_name=default_scope)  # type: ignore
        self.default_scope = default_scope

        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)
        self.logger = self.build_logger(log_level=log_level)

        self.message_hub = self.build_message_hub()
        self.visualizer = self.build_visualizer(visualizer)
        for vis_backend in self.visualizer._vis_backends.values():
            vis_backend._init_env()
        if self.cfg:
            self.visualizer.add_config(self.cfg)

        self._load_from = load_from
        self._resume = resume
        self._has_loaded = False

        self._calibration_loop = calibration_cfg
        self._calibration_evaluator = calibration_evaluator

        # build a model
        if isinstance(model_config, dict) and data_preprocessor is not None:
            model_config.setdefault("data_preprocessor", data_preprocessor)
        self.model = MODELS.build(model_config)
        self.model = self.wrap_model(self.cfg.get("model_wrapper_cfg"), self.model)

        if hasattr(self.model, "module"):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._hooks: List[Hook] = []
        self.register_hooks(default_hooks, custom_hooks)

        self.dump_config()

    def __call__(self):
        if self.mode == "test":
            return self.test()
        elif self.mode == "train":
            return self.train()
        else:
            return self.calibrate()

    def calibrate(self):
        loop = LOOPS.build(self._calibration_loop, default_args=dict(runner=self, dataloader=self._test_dataloader, evaluator=self._calibration_evaluator))
        self.call_hook("before_run")
        metrics = loop.run()
        self.call_hook("after_run")
        return metrics

    def build_visualizer(self, visualizer: Optional[Union[Visualizer, Dict]] = None) -> Visualizer:
        """Build a global asscessable Visualizer.

        Args:
            visualizer (Visualizer or dict, optional): A Visualizer object
                or a dict to build Visualizer object. If ``visualizer`` is a
                Visualizer object, just returns itself. If not specified,
                default config will be used to build Visualizer object.
                Defaults to None.

        Returns:
            Visualizer: A Visualizer object build from ``visualizer``.
        """
        if visualizer is None:
            visualizer = dict(name=self._experiment_name, vis_backends=[dict(type="LocalVisBackend")], save_dir=self._log_dir)
            return Visualizer.get_instance(**visualizer)

        if isinstance(visualizer, Visualizer):
            return visualizer

        if isinstance(visualizer, dict):
            if not visualizer.get("type", False):
                visualizer = dict(name=self._experiment_name, vis_backends=[dict(type="LocalVisBackend")], save_dir=self._log_dir)
                return Visualizer.get_instance(**visualizer)
            visualizer.setdefault("name", self._experiment_name)
            visualizer.setdefault("save_dir", self._log_dir)
            return VISUALIZERS.build(visualizer)
        else:
            raise TypeError("visualizer should be Visualizer object, a dict or None, " f"but got {visualizer}")

    def load_checkpoint(self, filename: str, map_location: Union[str, Callable] = "cpu", strict: bool = False, revise_keys: list = [(r"^module.", "")]):

        checkpoint = CheckpointLoader.load_checkpoint(filename, map_location)

        # Add comments to describe the usage of `after_load_ckpt`
        self.call_hook("after_load_checkpoint", checkpoint=checkpoint)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = load_checkpoint_to_model(model, checkpoint, strict, revise_keys=revise_keys)

        self._has_loaded = True

        self.logger.info(f"Load checkpoint from {filename}")

        return checkpoint


def setup_cache_size_limit_of_dynamo():
    """Setup cache size limit of dynamo.

    Note: Due to the dynamic shape of the loss calculation and
    post-processing parts in the object detection algorithm, these
    functions must be compiled every time they are run.
    Setting a large value for torch._dynamo.config.cache_size_limit
    may result in repeated compilation, which can slow down training
    and testing speed. Therefore, we need to set the default value of
    cache_size_limit smaller. An empirical value is 4.
    """

    import torch

    if digit_version(torch.__version__) >= digit_version("2.0.0"):
        if "DYNAMO_CACHE_SIZE_LIMIT" in os.environ:
            import torch._dynamo

            cache_size_limit = int(os.environ["DYNAMO_CACHE_SIZE_LIMIT"])
            torch._dynamo.config.cache_size_limit = cache_size_limit
            print_log(f"torch._dynamo.config.cache_size_limit is force " f"set to {cache_size_limit}.", logger="current", level=logging.WARNING)


def setup_multi_processes(cfg):
    """Setup multi-processing environment variables."""
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != "Windows":
        mp_start_method = cfg.get("mp_start_method", "fork")
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f"Multi-processing start method `{mp_start_method}` is "
                f"different from the previous setting `{current_method}`."
                f"It will be force set to `{mp_start_method}`. You can change "
                f"this behavior by changing `mp_start_method` in your config."
            )
        mp.set_start_method(mp_start_method, force=True)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get("opencv_num_threads", 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    workers_per_gpu = cfg.data.get("workers_per_gpu", 1)
    if "train_dataloader" in cfg.data:
        workers_per_gpu = max(cfg.data.train_dataloader.get("workers_per_gpu", 1), workers_per_gpu)

    if "OMP_NUM_THREADS" not in os.environ and workers_per_gpu > 1:
        omp_num_threads = 1
        warnings.warn(
            f"Setting OMP_NUM_THREADS environment variable for each process "
            f"to be {omp_num_threads} in default, to avoid your system being "
            f"overloaded, please further tune the variable for optimal "
            f"performance in your application as needed."
        )
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    # setup MKL threads
    if "MKL_NUM_THREADS" not in os.environ and workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f"Setting MKL_NUM_THREADS environment variable for each process "
            f"to be {mkl_num_threads} in default, to avoid your system being "
            f"overloaded, please further tune the variable for optimal "
            f"performance in your application as needed."
        )
        os.environ["MKL_NUM_THREADS"] = str(mkl_num_threads)
