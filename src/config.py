from dataclasses import dataclass
from pathlib import Path

CLASSES_12 = [
    "BearBat","BearButterfly","BearCrab","BearCypher","BearGartley","BearShark",
    "BullBat","BullButterfly","BullCrab","BullCypher","BullGartley","BullShark"
]

class Config:
    def __init__(self):
        # Backup_Time_Series_Classification/src/config.py -> parents[1] = project root
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data" / "multi_resolution"

        self.seed = 42
        self.seq_len = 64
        self.min_group_len = 1
        self.max_seqs_per_file = None
        self.feature_cols = ["open", "high", "low", "close", "r"]
        self.required_cols = {"date", "open", "high", "low", "close", "r", "pattern"}
        self.junk_cols = {"Unnamed: 0", "Unnamed:0", "index", "level_0"}

