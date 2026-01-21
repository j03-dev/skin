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


def predict(model: Model, image_path: str) -> str:
    img = cv2.imread(image_path)
    resized_img = resize_image(img)
    predict_x = model.predict(resized_img)
    classes_x = np.argmax(predict_x, axis=1)[0]
    return classes[classes_x]
