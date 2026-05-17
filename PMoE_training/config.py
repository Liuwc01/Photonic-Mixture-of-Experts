from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "exp_data"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

FEATURE_MAP_SIZE = 26
FEATURE_MAP_PIXELS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE


@dataclass(frozen=True)
class TaskConfig:
    key: str
    display_name: str
    num_classes: int
    num_samples: int
    hard_channels: int
    batch_size: int


TASKS = (
    TaskConfig("fmnist", "Fashion-MNIST", 10, 2000, 6, 10),
    TaskConfig("mnist", "MNIST", 10, 2000, 4, 10),
    TaskConfig("emnist", "EMNIST", 26, 5200, 8, 26),
)

WEIGHTED_CHANNELS = (4, 6, 8)

DEFAULT_SEED = 31
DEFAULT_EPOCHS = 300
DEFAULT_LR = 8e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_ETA_MIN = 1e-6
DEFAULT_CLIP_NORM = 1.0
DEFAULT_TEST_SIZE = 0.2
SHARED_CONV_CHANNELS = 16
SHARED_FC_DIM = 128
