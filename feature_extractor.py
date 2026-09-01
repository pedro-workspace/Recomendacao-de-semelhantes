import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class FeatureExtractor:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        self.model = model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def extract(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        embedding = embedding.squeeze().cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(self.transform(img))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch)
        embeddings = embeddings.squeeze(-1).squeeze(-1).cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings
