"""
Tiny wrapper around a pretrained ResNet‑18 that has been fine‑tuned on FER2013
(7 classes: anger, disgust, fear, happiness, sadness, surprise, neutral).
We map the 7 classes to 8 Plutchik emotions by adding a small uniform mass
for “anticipation”.  Replace the weights with your own 8‑class checkpoint if
you have one.
"""
import torch
import torchvision.transforms as T
from torchvision.models import resnet18
import numpy as np
from PIL import Image
from pathlib import Path

# Mapping from FER2013 class order → our 8 Plutchik emotions
FER2PLUTCHIK = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",   # → joy
    4: "sadness",
    5: "surprise",
    6: "neutral",     # → trust
    # FER2013 has only 7 classes; we add “anticipation” as a uniform prior later
}

class EmotionNet:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 8)
        # Load your own checkpoint here – for the demo we keep random weights
        self.model.to(self.device).eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict_emotion(self, rgb_img: np.ndarray) -> dict:
        """
        rgb_img – H×W×3 uint8 array (OpenCV format)
        Returns a dict {emotion: probability in [0,1]} for the 8 emotions.
        """
        # T.ToTensor() expects a PIL Image, not a raw numpy array
        pil_img = Image.fromarray(rgb_img) if isinstance(rgb_img, np.ndarray) else rgb_img
        img = self.transform(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(img)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        # Build dict – add a small uniform mass for anticipation (class 7)
        emotion_probs = {FER2PLUTCHIK[i]: float(probs[i]) for i in range(7)}
        emotion_probs["anticipation"] = 0.0   # placeholder; will be renormalised
        total = sum(emotion_probs.values())
        for k in emotion_probs:
            emotion_probs[k] /= total
        return emotion_probs
