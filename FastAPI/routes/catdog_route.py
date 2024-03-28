import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, File, UploadFile
from schemas.catdog_schema import CatDogResponse
from config.catdog_cfg import ModelConfig
from models.catdog_predictor import CatDogPredictor

router = APIRouter()
predictor = CatDogPredictor(
    model_name=ModelConfig.MODEL_NAME,
    model_weight=ModelConfig.MODEL_WEIGHT,
    device=ModelConfig.DEVICE
)


@router.post("/predict")
async def predict_catdog(file: UploadFile = File(...)):
    response = await predictor.predict(file.file)
    return CatDogResponse(**response)
