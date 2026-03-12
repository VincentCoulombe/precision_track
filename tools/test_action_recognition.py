import argparse
import os

from precision_track import Runner
from precision_track.utils import load_user_configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/tasks/testing_action_recognition.py", help="Path to the training config")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main(args):
    system_configs_path = args.config
    user_system_configs_path = "../configs/user_configs.yaml"
    load_user_configs(user_system_configs_path, system_configs_path, dynamic_ar_flag=True)
    runner = Runner(system_configs_path, args.launcher, mode="test")
    runner()


if __name__ == "__main__":
    # main(parse_args())
    from addict import Dict

    main(
        Dict(
            dict(
                config="../configs/tasks/testing_group_action_recognition.py",
                launcher="none",
            )
        )
    )
