import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CONFIG
# =========================
CONFIG = {
    "dataset_dir": "assets/datasets",
    "img_size": (224, 224),
    "batch_size": 32,
    "seed": 42,
    "epochs_head": 15,
    "epochs_fine": 10,
    "lr_head": 1e-4,
    "lr_fine": 1e-5,
    "unfreeze_layers": 30,
    "model_path": "assets/models/mpox_model-v2.h5",
    "threshold": 0.3,  # medical-sensitive threshold
}

CLASSES = ["Monkeypox", "Others"]


# =========================
# DATA
# =========================
def load_dataset(
    path: str,
    img_size: tuple[int, int],
    batch_size: int,
    seed: int,
    shuffle: bool = True,
):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        label_mode="binary",
    )


def prepare_dataset(ds):
    return ds.prefetch(tf.data.AUTOTUNE)


# =========================
# MODEL
# =========================
def build_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_backbone(name: str, input_shape):
    backbones = {
        "efficientnet": tf.keras.applications.EfficientNetB0,
        "mobilenet": tf.keras.applications.MobileNetV3Small,
    }

    backbone = backbones[name](
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


def build_model(backbone_name: str, img_size):
    inputs = layers.Input(shape=(*img_size, 3))

    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = build_augmentation()(x)

    backbone = build_backbone(backbone_name, (*img_size, 3))
    backbone.trainable = False

    x = backbone(x, training=False)
    outputs = build_classifier(x)

    model = models.Model(inputs, outputs)
    return model, backbone


# =========================
# TRAINING
# =========================
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


def compute_class_weights(train_ds):
    y = np.concatenate([labels.numpy() for _, labels in train_ds])
    y = y.astype(int).flatten()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y,
    )

    return dict(enumerate(weights))


def train_head(model, train_ds, val_ds, cfg, class_weights):
    compile_model(model, cfg["lr_head"])
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs_head"],
        callbacks=build_callbacks(cfg),
        class_weight=class_weights,
    )


def fine_tune(model, backbone, train_ds, val_ds, cfg, class_weights):
    backbone.trainable = True

    for layer in backbone.layers[: -cfg["unfreeze_layers"]]:
        layer.trainable = False

    compile_model(model, cfg["lr_fine"])
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs_fine"],
        callbacks=build_callbacks(cfg),
        class_weight=class_weights,
    )


# =========================
# PIPELINE
# =========================
def build_pipeline(cfg: dict):
    base = cfg["dataset_dir"]
    data_cfg = {
        "img_size": cfg["img_size"],
        "seed": cfg["seed"],
        "batch_size": cfg["batch_size"],
    }

    train_raw = load_dataset(base + "/Train", **data_cfg)
    train_ds = prepare_dataset(train_raw)
    val_ds = prepare_dataset(load_dataset(base + "/Val", **data_cfg))
    test_ds = prepare_dataset(load_dataset(base + "/Test", **data_cfg, shuffle=False))

    print("Class order:", train_raw.class_names)

    model, backbone = build_model("efficientnet", cfg["img_size"])
    model.summary()

    class_weights = compute_class_weights(train_ds)

    train_head(model, train_ds, val_ds, cfg, class_weights)
    fine_tune(model, backbone, train_ds, val_ds, cfg, class_weights)

    model.evaluate(test_ds)
    model.save(cfg["model_path"])


# =========================
# INFERENCE
# =========================
def load_model_from(path: str) -> models.Model:
    return models.load_model(path)


def predict(model: models.Model, image_path: str) -> str:
    img = cv2.imread(image_path)
    img = cv2.resize(img, CONFIG["img_size"])
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img, verbose=0)[0][0]
    label = int(prob > CONFIG["threshold"])

    print(f"Probability Others: {prob:.3f}")
    return CLASSES[label]


# =========================
# MAIN
# =========================
def main():
    build_pipeline(CONFIG)


if __name__ == "__main__":
    main()
