from test_tracking import parse_args

from precision_track import Runner
from precision_track.utils import load_user_configs


def main(args):
    system_configs_path = "../configs/tasks/testing_detection.py"
    user_system_configs_path = "../configs/user_configs.yaml"
    load_user_configs(user_system_configs_path, system_configs_path, tool="test_detection")
    runner = Runner(system_configs_path, args.launcher, mode="test")
    runner()


if __name__ == "__main__":
    main(parse_args())
