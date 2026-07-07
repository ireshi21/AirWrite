import string

import cv2
import numpy as np
import torch

from model import CharacterCNN


class CharacterRecognizer:
    def __init__(self, model_path="models/character_cnn.pth"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = CharacterCNN().to(self.device)

        state_dict = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.labels = string.ascii_uppercase
    def _predict_single(self, image):
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

        tensor = torch.from_numpy(image).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, index = probabilities.max(dim=1)

        return self.labels[index.item()], confidence.item()


    

    def predict(self, character_image):
        if character_image is None:
            return None, 0.0

        # Torchvision EMNIST orientation differs from normal display orientation.
        # Transpose AirWrite input to match the model's training distribution.
        character_image = character_image.T.copy()

        return self._predict_single(character_image)