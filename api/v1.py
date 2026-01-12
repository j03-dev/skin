from oxapy import Router, Request, post
import skin_ai

MODEL_V1 = skin_ai.load_model_from(path="../assets/models/skin-ai-v1.h5")


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
