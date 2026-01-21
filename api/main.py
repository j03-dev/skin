from oxapy import HttpServer, Router, Request, post
from model import load_model
import skin_ai


MODEL = load_model(path="assets/models/mpox_model-v3.h5")


@post("/predict")
def predict(r: Request):
    image = r.files.get("image")
    assert image, "the image is none, pls upload image"
    path = f"./media/{image.name}"
    image.save(path)
    prediction = skin_ai.predict(MODEL, path)
    return {"prediction": prediction}


def main():
    HttpServer(("0.0.0.0", 8000)).attach(Router("/api/v1").route(predict)).run()


if __name__ == "__main__":
    main()
