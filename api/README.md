# 🚀 Skin AI - REST API

This package provides a REST API to serve the `skin-ai` mpox detection model. It's built using the `oxapy` web framework.

## Endpoints

### Predict Mpox

- **Endpoint**: `POST /api/v1/predict`
- **Description**: Receives an image and returns a prediction for `mpox`.
- **Request**: `multipart/form-data` with an `image` field.
- **Response**: A JSON object with `label` and `confidence`.

**Example (cURL)**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "image=@/path/to/your/image.jpg"
```

**Example Response**
```json
{
  "label": "mpox",
  "confidence": 0.92
}
```

## Running the API

To start the server, run the following command from the project root:

```bash
uv run python api/main.py
```

The API will be available at `http://localhost:8000`.
