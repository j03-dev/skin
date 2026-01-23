import os
import io
import requests
import dotenv
import logging
from oxapy import HttpServer, Request, Response, Router, Status, exceptions, get, post


dotenv.load_dotenv()
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")


@get("/")
def webhook_verify(r: Request):
    query = r.query
    hub_mode: str | None = query.get("hub.mode")
    hub_challenge: str | None = query.get("hub.challenge")
    hub_verify_token: str | None = query.get("hub.verify_token")

    if hub_mode and hub_challenge and hub_verify_token:
        if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
            return Response(hub_challenge, content_type="text/plain")
    raise exceptions.UnauthorizedError(
        "Insufficient arguments provided or Token mismatch"
    )


def service_prediction(image_bytes: bytes) -> dict[str, float]:
    file = {"image": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    response = requests.post("http://localhost:8000/api/v1/predict", files=file)
    response.raise_for_status()
    return response.json()


def get_image_content(message: dict) -> bytes:
    image_url = message["attachments"][0]["payload"]["url"]
    response = requests.get(image_url)
    response.raise_for_status()
    return response.content


def send_text(psid: str, text: str):
    url = "https://graph.facebook.com/v21.0/me/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PAGE_ACCESS_TOKEN}",
    }
    payload = {"recipient": {"id": psid}, "message": {"text": text}}
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()


@post("/")
def webhook_core(r: Request):
    data = r.json()
    messaging = data["entry"][0]["messaging"][0]
    sender = messaging["sender"]["id"]
    if message := messaging.get("message"):
        image_bytes = get_image_content(message)
        prediction = service_prediction(image_bytes)
        text_message = (
            f"label={prediction['label']} confidence={prediction['confidence']}"
        )
        send_text(sender, text_message)
    return Status.OK


def log(r: Request, next, **kwargs):
    logging.log(1000, f"{r.uri} {r.method}")
    return next(r, **kwargs)


def main():
    (
        HttpServer(("0.0.0.0", 5555))
        .attach(
            Router("/webhook")
            .middleware(log)
            .routes(
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
