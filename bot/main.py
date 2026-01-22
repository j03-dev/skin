import os
import io
import requests
import skin_ai
import dotenv
from oxapy import HttpServer, Request, Response, Router, Status, exceptions, get, post


dotenv.load_dotenv()

MODEL = skin_ai.load_model(path="assets/models/mpox_model-v3.h5")


@get("/")
def webhook_verify(r: Request):
    query = r.query
    hub_mode: str | None = query.get("hub.mode")
    hub_challenge: str | None = query.get("hub.challenge")
    hub_verify_token: str | None = query.get("hub.verify_token")

    if hub_mode and hub_challenge and hub_verify_token:
        if hub_mode == "subscribe" and hub_verify_token == os.getenv("VERIFY_TOKEN"):
            return Response(hub_challenge, content_type="text/plain")
    raise exceptions.UnauthorizedError(
        "Insufficient arguments provided or Token mismatch"
    )


def predict(image_url: str):
    response = requests.get(image_url)
    response.raise_for_status()
    file = {"image": ("image.jpg", io.BytesIO(response.content), "image/jpeg")}
    response = requests.post("http://locahost:8000/api/v1/predict", files=file)
    response.raise_for_status()
    return response.json()


def send_text(psid: str, text: str):
    url = "https://graph.facebook.com/v19/me/messages"
    params = {"access_token": os.getenv("ACCESS_TOKEN")}
    payload = {"recipient": {"id": psid}, "message": {"text": text}}
    r = requests.post(url, params=params, json=payload)
    r.raise_for_status()


@post("/")
def webhook_core(r: Request):
    data = r.json()
    messaging = data["entry"][0]["messaging"][0]
    sender = messaging["sender"]["id"]
    image_url = messaging["message"]["attachements"][0]["playload"]["url"]
    prediction = predict(image_url)
    send_text(
        sender,
        f"label={prediction['label']} prediction={prediction['confidence']}",
    )
    return Status.OK


def main():
    (
        HttpServer(("0.0.0.0", 5555))
        .attach(
            Router("/webhook").routes(
                [
                    webhook_verify,
                    webhook_core,
                ]
            )
        )
        .run()
    )


if __name__ == "__main__":
    main()
