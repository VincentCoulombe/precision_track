import argparse
import os

from mmengine.config import Config
from precision_track.apis import PrecisionTrackRunner, SequenceRunner

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# import torch

# if hasattr(torch, "set_default_device"):  # PyTorch ≥2.0
#     torch.set_default_device("cpu")
# torch.backends.cudnn.enabled = False  # ensure no CuDNN paths
# import torch, os

# torch.use_deterministic_algorithms(True)
# torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))


if "DYNAMO_CACHE_SIZE_LIMIT" in os.environ:
    import torch._dynamo

    cache_size_limit = int(os.environ["DYNAMO_CACHE_SIZE_LIMIT"])
    torch._dynamo.config.cache_size_limit = cache_size_limit


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="Path to the desired training config")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main(args):
    cfg = Config.fromfile(args.config)
    runner = cfg.pop("runner", "PrecisionTrackRunner")
    if runner == "PrecisionTrackRunner":
        runner = PrecisionTrackRunner(cfg=cfg, launcher=args.launcher, mode="train")
    elif runner == "SequenceRunner":
        runner = SequenceRunner(cfg=cfg, launcher=args.launcher, mode="train")
    else:
        raise ValueError(f"{runner} not supported.")
    runner()


if __name__ == "__main__":
    # main(parse_args())
    from addict import Dict

    main(Dict({"config": "../configs/tasks/training_action_recognition.py", "launcher": "none"}))
