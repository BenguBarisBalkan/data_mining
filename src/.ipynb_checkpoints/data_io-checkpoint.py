from __future__ import annotations
from pathlib import Path
import glob

def list_csv_files(data_dir: Path) -> list[str]:
    data_dir = Path(data_dir)
    return sorted(str(p) for p in data_dir.glob("*.csv"))


def resolve_paths(paths: list[str], base_dir: str | Path) -> list[str]:
    """If notebook runs in /notebooks, this helps normalize relative paths."""
    base_dir = Path(base_dir)
    return [str((base_dir / p).resolve()) if not Path(p).is_absolute() else p for p in paths]
