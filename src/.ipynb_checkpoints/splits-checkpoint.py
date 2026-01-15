from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List, Set

def split_train_eval_files(
    all_files: list[str],
    eval_basenames: Set[str],
) -> tuple[list[str], list[str]]:
    """
    all_files: list of absolute file paths
    eval_basenames: {"1hour_sigma0.01.csv", ...}
    returns: (train_files, eval_files)
    """
    eval_files = [f for f in all_files if Path(f).name in eval_basenames]
    train_files = [f for f in all_files if Path(f).name not in eval_basenames]
    return train_files, eval_files
