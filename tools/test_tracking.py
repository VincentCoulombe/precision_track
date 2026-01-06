import yaml
import argparse
import os
from precision_track import Runner
from precision_track.utils import load_user_configs, check_if_mot_dataset_is_ok, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main(args):
    system_configs_path = "../configs/tasks/testing_tracking.py"
    with open("../configs/user_configs.yaml", "r") as f:
        user_configs = yaml.safe_load(f)
    load_user_configs(user_configs, system_configs_path)
    config = load_config(system_configs_path)
    check_if_mot_dataset_is_ok(config["testing_tracking_data_root"])
    runner = Runner(system_configs_path, args.launcher, mode="test")
    runner()


if __name__ == "__main__":
    main(parse_args())
