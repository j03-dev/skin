import numpy as np
import cv2
from keras.models import Model
from model import cfg


classes = ["mpox", "others"]


def resize_image(img: np.ndarray):
    img_size: float[int, int] = cfg["image_size"]
    img = cv2.resize(img, img_size)  # type: ignore
    img = np.expand_dims(img, axis=0)
    return img


def predict(model: Model, image_path: str) -> tuple[str, float]:
    img = cv2.imread(image_path)
    resized_img = resize_image(img)
    prediction = model.predict(resized_img)
    class_index = np.argmax(prediction, axis=1)[0]

    confidence = float(prediction[0][class_index])
    return classes[class_index], confidence
