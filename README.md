# Skin AI 🧠🩺

**Skin AI** is a simple machine-learning project for detecting **mpox** from skin images.
It is structured as a monorepo containing:
-   `skin-ai`: The core ML model and training code.
-   `api`: A REST API to serve the model.
-   `bot`: A Facebook Messenger bot for interaction.

> ⚠️ **Disclaimer**: This project is for educational and experimental purposes only.  
> It is **not** a medical device and should not be used for clinical diagnosis.

---
## Components

### 🤖 Skin-AI (ML Core)
The core component responsible for the machine learning model. It includes scripts for training the model and making predictions.  
[See `skin-ai/README.md` for more details.](./skin-ai/README.md)

### 🚀 API
A REST API built with `oxapy` that exposes the `skin-ai` model for predictions.  
[See `api/README.md` for more details.](./api/README.md)

### 💬 Messenger Bot
A Facebook Messenger bot that allows users to get predictions by sending an image.  
[See `bot/README.md` for more details.](./bot/README.md)

---

## Getting Started

### Prerequisites

* Python 3.10+
* [`uv`](https://github.com/astral-sh/uv)

---

### Installation

Clone the repository and install dependencies from all packages:

```bash
uv sync --all-packages
```

---

### Running the services

#### 1. Run the API

The API serves the prediction model.

```bash
uv run python api/main.py
```

The API will be available at `http://localhost:8000`.

#### 2. Run the Messenger Bot (Optional)

To use the bot, you'll need to set up a Facebook App and get your tokens.

```bash
uv run python bot/main.py
```

The bot webhook will listen on `http://localhost:5555`.

---

## License

MIT License

---

## Author

Built with ❤️ by **Joe (j03-dev)**
