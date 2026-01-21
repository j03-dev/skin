# Skin AI 🧠🩺

**Skin AI** is a simple machine-learning project for detecting **mpox** from skin images.  
It exposes a REST API that allows clients to send an image and receive a prediction.

> ⚠️ **Disclaimer**: This project is for educational and experimental purposes only.  
> It is **not** a medical device and should not be used for clinical diagnosis.

---

## Features

- 🧪 Mpox detection using a trained ML model
- 🚀 REST API built for easy integration
- 📦 Simple local setup using `uv`
- 🔌 JSON-based prediction response

---

## API Overview

### Predict Mpox

**Endpoint**
```

POST /api/v1/predict

```

**URL (local)**
```

[http://localhost:8000/api/v1/predict](http://localhost:8000/api/v1/predict)

````

**Request**
- `multipart/form-data`
- Field:
  - `image`: image file (jpg, png, etc.)

**Example (curl)**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "image=@example.jpg"
````

**Response (example)**

```json
{
  "label": "mpox",
  "confidence": 0.92
}
```

---

## Getting Started

### Prerequisites

* Python 3.10+
* [`uv`](https://github.com/astral-sh/uv)

---

### Installation

Clone the repository and install dependencies:

```bash
uv sync --all-packages
```

---

### Run the API

```bash
uv run api/main.py
```

The API will be available at:

```
http://localhost:8000
```

---

## License

MIT License

---

## Author

Built with ❤️ by **Joe (j03-dev)**
