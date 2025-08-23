import os
from contextlib import contextmanager


@contextmanager
def temp_csv_file(path):
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)
        raw_path, _ = os.path.splitext(path)
        mapping_path = f"{raw_path}_mapping.npy"
        if os.path.exists(mapping_path):
            os.remove(mapping_path)
