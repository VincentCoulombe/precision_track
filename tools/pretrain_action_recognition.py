import argparse
import logging
import multiprocessing as mp
import os

from mmengine.logging import MMLogger

from precision_track import Runner
from precision_track.utils import find_checkpoint_hook, load_user_configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-supervised (masked autoencoder) pretraining of a MART model. Learns from raw video only -- no action labels required."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/tasks/training_action_recognition_unsup.py",
        help="Path to the pretraining config. Default to ../configs/tasks/training_action_recognition_unsup.py.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main(args):
    """Pretrain MART by masked reconstruction of its own sequence embeddings.

    Deliberately leaner than ``train_action_recognition.py``: the sequences are generated on
    the fly from raw video by ``MAEDataset``, so there is no action-recognition dataset to
    register, and MART is trained from scratch, so there is no MART checkpoint to load. The
    resulting weights are meant to initialise a subsequent supervised run, not to be deployed
    on their own -- the classification head is still untrained.
    """
    logger = MMLogger.get_instance("mmengine", log_level=logging.INFO, file_mode="w")
    system_configs_path = args.config
    user_system_configs_path = "../configs/user_configs.yaml"

    load_user_configs(
        user_system_configs_path,
        system_configs_path,
        dynamic_ar_flag=True,
        tool="pretrain_action_recognition",
        flags=dict(test=False, deploy=False),
    )

    runner = Runner(system_configs_path, args.launcher, mode="train")
    runner()

    checkpoint_hook = find_checkpoint_hook(runner)
    best_ckpt_path = str(getattr(checkpoint_hook, "best_ckpt_path", "") or "")
    if os.path.isfile(best_ckpt_path):
        logger.info(f"Pretrained MART checkpoint: {best_ckpt_path}")
        logger.info("Point 'training_checkpoint' at it to warm-start a supervised train_action_recognition.py run.")
    else:
        logger.warning(
            "No best checkpoint was recorded. This happens when the run stopped before its first "
            "validation step, or when the checkpoint was removed manually."
        )
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main(parse_args()))
