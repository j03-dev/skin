from oxapy import Router, Request, post
from model import load_model
import skin_ai


MODEL_V1 = load_model(path="assets/models/mpox_model-v3.h5")


@post("/predict")
def predict(r: Request):
    image = r.files.get("image")
    assert image, "the image is none, pls upload image"
    path = f"./media/{image.name}"
    image.save(path)
    prediction = skin_ai.predict(MODEL_V1, path)
    return {"prediction": prediction}


def router():
    return Router("/api/v1").route(predict)
