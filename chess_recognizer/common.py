from datetime import datetime
from pathlib import Path

import torch

BOARD_DIMENSIONS = (64, 13)

ROOT_DIR = Path(__file__).parent.parent.resolve()
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"


def get_curr_time_str() -> str:
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")


def save_model(model: torch.nn.Module, name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model, MODELS_DIR / "{}_{}.pt".format(name, get_curr_time_str()))


def load_model(name: str) -> torch.nn.Module:
    return torch.load(MODELS_DIR / name)


if __name__ == "__main__":
    print(load_model("resnet_finetuned_2020.03.06.15.10.40.pt"))
