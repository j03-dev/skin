import numpy
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from typing import Any

CONFIG = {
    "dataset_dir": "assets/datasets",
    "img_size": (224, 224),
    "batch_size": 32,
    "seed": 42,
    "epochs_head": 15,
    "epochs_fine": 10,
    "lr_head": 1e-4,
    "lr_fine": 1e-5,
    "unfreez_layers": 30,
    "model_path": "assets/models/mpox_model.h5",
}

CLASSES = []


def load_dataset(
    path: str, img_size: tuple[int, int], batch_size: int, seed: int, shuffle=True
):
    return tf.keras.utils.image_dataset_from_directory(
        path, image_size=img_size, batch_size=batch_size, seed=seed, shuffle=shuffle
    )


def prepare_dataset(ds: list[Any] | Any):
    return ds.prefetch(tf.data.AUTOTUNE)


def build_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_backbone(name: str, input_shape):
    backbones = {
        "efficientnet": tf.keras.applications.EfficientNetB0,
        "mobilenet": tf.keras.applications.MobileNetV3Small,
    }

    model_cls = backbones[name]
    backbone = model_cls(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    return backbone


def build_classifier(x):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    return layers.Dense(1, activation="sigmoid")(x)


def build_model(backbone_name: str, img_size: tuple[int, int]):
    inputs = layers.Input(shape=(*img_size, 3))

    augmentation = build_augmentation()
    backbone = build_backbone(backbone_name, (*img_size, 3))
    backbone.trainable = False

    x = augmentation(inputs)
    x = backbone(x, training=False)
    outputs = build_classifier(x)

    return models.Model(inputs, outputs), backbone


def compile_model(model: models.Model, lr: float):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def build_callbacks(cfg: dict):
    return [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(cfg["model_path"], save_best_only=True),
    ]


def train_head(
    model: models.Model, trains_ds: list[Any] | Any, val_ds: list[Any] | Any, cfg: dict
):
    compile_model(model, cfg["lr_head"])
    return model.fit(
        trains_ds,
        validation_data=val_ds,
        epochs=cfg["epochs_head"],
        callbacks=build_callbacks(cfg),
    )


def fine_tune(model, backbone, train_ds, val_ds, cfg):
    backbone.trainable = True

    for layer in backbone.layers[: -cfg["unfreez_layers"]]:
        layer.trainable = False

    compile_model(model, cfg["lr_fine"])

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs_fine"],
        callbacks=build_callbacks(cfg),
    )


def build_pipeline(cfg: dict):
    dataset_dir = cfg["dataset_dir"]
    data_cfg = {
        "img_size": cfg["img_size"],
        "seed": cfg["seed"],
        "batch_size": cfg["batch_size"],
    }
    train_ds = prepare_dataset(load_dataset(dataset_dir + "/Train", **data_cfg))
    val_ds = prepare_dataset(load_dataset(dataset_dir + "/Val", **data_cfg))
    test_ds = prepare_dataset(
        load_dataset(dataset_dir + "/Test", **data_cfg, shuffle=False)
    )

    model, backbone = build_model("efficientnet", cfg["img_size"])

    model.summary()

    train_head(model, train_ds, val_ds, cfg)
    fine_tune(model, backbone, train_ds, val_ds, cfg)

    model.evaluate(test_ds)


def load_model_from(path: str) -> models.Model:
    models.load_model(path)


def predict(model: models.Model, path_image: str) -> str:
    img = cv2.imread(path_image)
    img = cv2.resize(img, CONFIG["img_size"])
    img = img / 255.0
    img = numpy.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]
    return CLASSES[int(prob > 0.5)]


def main():
    build_pipeline(CONFIG)


if __name__ == "__main__":
    main()
