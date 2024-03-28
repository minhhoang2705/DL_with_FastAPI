import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image

from config.catdog_cfg import CatDogDataConfig
from models.catdog_model import CatDogModel

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import Logger

LOGGER = Logger(__file__, log_file="predictor.log")
LOGGER.log.info("Starting Model Serving")


class CatDogPredictor:
    def __init__(self, model_name: str, model_weight: str, device: str = 'cuda'):
        self.model_name = model_name
        self.model_weight = model_weight
        self.device = device
        self.load_model()
        self.create_transform()

    async def predict(self, image):
        pil_img = Image.open(image)
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGB')
        transformed_img = self.transform(pil_img).unsqueeze(0)
        output = await self.model_inference(transformed_img)
        probs, best_prob, predicted_id, predicted_class = self.output2pred(output)

        LOGGER.log_model(self.model_name)
        LOGGER.log_response(best_prob, predicted_id, predicted_class)

        torch.cuda.empty_cache()

        resp_dict = {
            "probs": probs,
            "best_prob": best_prob,
            "predicted_id": predicted_id,
            "predicted_class": predicted_class,
            "predictor_name": self.model_name
        }
        return resp_dict

    async def model_inference(self, input):
        input = input.to(self.device)
        with torch.no_grad():
            output = self.loaded_model(input.to(self.device))
        return output

    def load_model(self):
        try:
            model = CatDogModel(CatDogDataConfig.N_CLASSES)
            model.load_state_dict(torch.load(self.model_weight, map_location=self.device))
            model.to(self.device)
            model.eval()

            self.loaded_model = model
        except Exception as e:
            LOGGER.log.error(f"Error loading model: {e}")
            return None

    def create_transform(self):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((CatDogDataConfig.IMG_SIZE, CatDogDataConfig.IMG_SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=CatDogDataConfig.NORMALIZE_MEAN, std=CatDogDataConfig.NORMALIZE_STD)
        ])

    def output2pred(self, output):
        probabilities = torch.nn.functional.softmax(output, dim=1)
        best_prob = torch.max(probabilities, 1)[0].item()
        predicted_id = torch.max(probabilities, 1)[1].item()
        predicted_class = CatDogDataConfig.ID2DLABEL[predicted_id]
        return probabilities.squeeze().tolist(), round(best_prob, 6), predicted_id, predicted_class
