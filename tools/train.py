import argparse
import os

from precision_track import Runner

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
    runner = Runner(args.config, args.launcher, mode="train")
    runner()


if __name__ == "__main__":
    main(parse_args())
