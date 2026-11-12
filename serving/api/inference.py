"""Model loading, preprocessing and prediction - the same preprocessing that was used for training and testing."""
import json
import math

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras


class Classifier:
    def __init__(self, model_path: str, class_names_path: str, threshold: float = 0.7):
        self.model = keras.models.load_model(model_path)
        with open(class_names_path) as f:
            self.class_names = json.load(f)               # output index i == class_names[i]
        self.size = int(self.model.input_shape[1])        # 288 for the trained model
        if self.model.output_shape[-1] != len(self.class_names):
            raise ValueError("class_names.json does not match the model's output size")
        self.threshold = threshold

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """RGB image -> (1, size, size, 3) float32 with RAW 0-255 pixels. The model rescales internally (do NOT divide by 255)."""
        image = ImageOps.exif_transpose(image).convert("RGB")          # phone photos: respect the camera orientation
        img = tf.convert_to_tensor(np.asarray(image, dtype=np.float32))
        side = tf.reduce_min(tf.shape(img)[:2])
        img = tf.image.resize_with_crop_or_pad(img, side, side)        # centre square crop of the shorter side
        img = tf.image.resize(img, (self.size, self.size))             # bilinear, as in the test pipeline
        return img.numpy()[None]

    def predict(self, image: Image.Image) -> dict:
        probs = self.model.predict(self.preprocess(image), verbose=0)[0]     # already softmax probabilities
        order = np.argsort(-probs)[:3]
        top3 = [{"disease": self.class_names[i], "probability": round(float(probs[i]), 4)} for i in order]
        confident = bool(probs[order[0]] >= self.threshold)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(probs)))     # 0 = sure, 1 = completely unsure
        return {
            "confident": confident,
            "prediction": top3[0]["disease"] if confident else None,
            "message": None if confident else "Not sure - please take a clearer close-up photo of a single leaf.",
            "top3": top3,
            "threshold": self.threshold,
            "_top1": top3[0]["disease"],
            "_confidence": float(probs[order[0]]),
            "_entropy": round(entropy, 4),
        }


def image_stats(image: Image.Image) -> dict:
    """Simple image statistics that we log for drift monitoring (brightness, contrast, sharpness, colour, edges, size)."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    small = image.copy()
    small.thumbnail((256, 256))
    rgb = np.asarray(small, dtype=np.float32)
    gray = rgb.mean(axis=2)
    lap = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
    gy, gx = np.gradient(gray)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    total = rgb.sum() + 1e-6
    return {
        "brightness": round(float(gray.mean() / 255), 4),
        "contrast": round(float(gray.std() / 255), 4),
        "sharpness": round(float(lap.var()), 2),
        "saturation": round(float(hsv[..., 1].mean() / 255), 4),
        "green_ratio": round(float(rgb[..., 1].sum() / total), 4),
        "edge_density": round(float((np.hypot(gx, gy) > 20).mean()), 4),
        "width": int(width),
        "height": int(height),
    }
