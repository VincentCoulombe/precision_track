import yaml
from test_tracking import parse_args

from precision_track import Runner
from precision_track.utils import load_user_configs


def main(args):
    system_configs_path = "../configs/tasks/testing_detection.py"
    with open("../configs/user_configs.yaml", "r") as f:
        user_configs = yaml.safe_load(f)
    load_user_configs(user_configs, system_configs_path)
    runner = Runner(system_configs_path, args.launcher, mode="test")
    runner()


if __name__ == "__main__":
    main(parse_args())
