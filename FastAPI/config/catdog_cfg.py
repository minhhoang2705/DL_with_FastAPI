import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


class CatDogDataConfig:
    N_CLASSES = 2
    IMG_SIZE = 64
    ID2DLABEL = {0: 'cat', 1: 'dog'}
    LABEL2ID = {'cat': 0, 'dog': 1}
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]


class ModelConfig:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    MODEL_NAME = 'resnet18'
    MODEL_WEIGHT = ROOT_DIR / 'weights' / 'catdog_model.pt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
