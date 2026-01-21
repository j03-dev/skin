import numpy as np
import cv2
import keras
from keras import layers

classes = ["mpox", "others"]

cfg = {
    "rescaling": 1.0 / 255,
    "image_size": (128, 128),
    "batch_size": 3,
    "seed": 36,
    "optimizer": "adam",
    "epochs": 3,
    "datasets_dir": "assets/datasets",
    "log_dir": ".",
    "model_name": "assets/models/mpox_model-v3.h5",
}


def load_model(path: str):
    return keras.models.load_model(path)


def load_dataset(path: str):
    return keras.utils.image_dataset_from_directory(
        path,
        seed=cfg["seed"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
    )


def model_layers():
    return keras.Sequential(
        [
            layers.Rescaling(cfg["rescaling"]),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(36, activation="softmax"),
        ]
    )


def train():
    model = model_layers()
    model.compile(
        optimizer=cfg["optimizer"],
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=".",
        histogram_freq=1,
        write_images=".",
    )
    datasets_dir = cfg["datasets_dir"]
    training_data = load_dataset(f"{datasets_dir}/Train")
    validation_data = load_dataset(f"{datasets_dir}/Val")

    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=cfg["epochs"],
        callbacks=[tensorboard_callback],
    )
    model.summary()
    model.save(cfg["model_name"])


def resize_image(img: np.ndarray):
    img_size: float[int, int] = cfg["image_size"]
    img = cv2.resize(img, img_size)  # type: ignore
    img = np.expand_dims(img, axis=0)
    return img


def predict(model: keras.Model, image_path: str) -> tuple[str, float]:
    img = cv2.imread(image_path)
    resized_img = resize_image(img)
    prediction = model.predict(resized_img)
    class_index = np.argmax(prediction, axis=1)[0]

    confidence = float(prediction[0][class_index])
    return classes[class_index], confidence


if __name__ == "__main__":
    train()
