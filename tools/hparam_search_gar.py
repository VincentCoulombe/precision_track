#!/usr/bin/env python3
"""
Hyperparameter search for GMART training.

Patches training_group_action_recognition.py with regex for each combination,
launches train_action_recognition.py, then restores the original config.
"""

import itertools
import re
import subprocess
import sys
from pathlib import Path

# ── Search space ──────────────────────────────────────────────────────────────
LEARNING_RATES    = [1e-4, 3e-4, 1e-3]
BATCH_SIZES       = [4, 8]
HEADS_DROPOUTS    = [0.1, 0.2, 0.3]
ENCODER_DROPOUTS  = [0.1]          # add values to sweep encoder dropout too
# ─────────────────────────────────────────────────────────────────────────────

TOOLS_DIR    = Path(__file__).parent.resolve()
CONFIG_PATH  = (TOOLS_DIR / "../configs/tasks/training_group_action_recognition.py").resolve()
TRAIN_SCRIPT = TOOLS_DIR / "train_action_recognition.py"


def patch_config(content: str, lr: float, batch_size: int,
                 heads_dropout: float, encoder_dropout: float) -> str:
    content = re.sub(r"^gar_base_lr = .*$",
                     f"gar_base_lr = {lr}",                  content, flags=re.MULTILINE)
    content = re.sub(r"^batch_size = .*$",
                     f"batch_size = {batch_size}",           content, flags=re.MULTILINE)
    content = re.sub(r"^heads_dropout = .*$",
                     f"heads_dropout = {heads_dropout}",     content, flags=re.MULTILINE)
    content = re.sub(r"^encoder_dropout = .*$",
                     f"encoder_dropout = {encoder_dropout}", content, flags=re.MULTILINE)
    return content


def run_combo(lr, batch_size, heads_dropout, encoder_dropout, idx, total):
    tag = f"lr={lr}  bs={batch_size}  hd={heads_dropout}  ed={encoder_dropout}"
    print(f"\n[{idx}/{total}] {tag}")
    result = subprocess.run(
        [
            sys.executable, str(TRAIN_SCRIPT),
            "--config", str(CONFIG_PATH),
            "--test",   "false",
            "--deploy", "false",
        ],
        cwd=str(TOOLS_DIR),
    )
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  {status}")
    return result.returncode


def main():
    original = CONFIG_PATH.read_text()
    combos   = list(itertools.product(LEARNING_RATES, BATCH_SIZES,
                                      HEADS_DROPOUTS, ENCODER_DROPOUTS))
    print(f"Hyperparameter search: {len(combos)} combinations")
    print(f"Config : {CONFIG_PATH}")
    print(f"Script : {TRAIN_SCRIPT}\n")

    results = []
    try:
        for i, (lr, bs, hd, ed) in enumerate(combos, 1):
            CONFIG_PATH.write_text(patch_config(original, lr, bs, hd, ed))
            rc = run_combo(lr, bs, hd, ed, i, len(combos))
            results.append((lr, bs, hd, ed, rc))
    finally:
        CONFIG_PATH.write_text(original)
        print("\nOriginal config restored.")

    print("\n── Summary " + "─" * 50)
    for lr, bs, hd, ed, rc in results:
        print(f"  [{'OK  ' if rc == 0 else 'FAIL'}]  lr={lr}  bs={bs}  hd={hd}  ed={ed}")


if __name__ == "__main__":
    main()
