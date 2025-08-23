import cProfile
import os
import pstats

from track import main as TRACK_MAIN


def profile_function(func):

    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        _ = func(*args, **kwargs)
        pr.disable()
        ps = pstats.Stats(pr)
        ps.sort_stats(pstats.SortKey.TIME)
        sink_path = kwargs.get("args", {}).get("sink_path")
        os.makedirs(os.path.dirname(sink_path), exist_ok=True)
        if sink_path:
            ps.dump_stats(sink_path)
        else:
            ps.print_stats()
        return _

    return wrapper


@profile_function
def main(args, track_args):
    TRACK_MAIN(track_args)


if __name__ == "__main__":
    from addict import Dict

    main(
        args={"sink_path": "../profiles/profile_v7_bt.prof"},
        track_args=Dict(
            config="../configs/mice/tracking.py",
            video="../assets/20mice.avi",
        ),
    )
