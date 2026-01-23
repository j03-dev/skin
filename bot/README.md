# 💬 Skin AI - Messenger Bot

This package contains a Facebook Messenger bot that allows users to interact with the Skin AI service. Users can send a skin image and receive an `mpox` prediction directly in the chat.

## Features

- Handles Facebook Messenger webhook verification.
- Receives image attachments from users.
- Communicates with the `skin-ai` API to get predictions.
- Sends the prediction results back to the user as a text message.

## Setup

1.  **Environment Variables**: Create a `.env` file in the root of the project with the following content:

    ```env
    VERIFY_TOKEN="YOUR_FACEBOOK_VERIFY_TOKEN"
    PAGE_ACCESS_TOKEN="YOUR_FACEBOOK_PAGE_ACCESS_TOKEN"
    ```
    Replace the placeholder values with your actual tokens from your Facebook Developer App settings.

2.  **Dependencies**: Ensure all project dependencies are installed by running `uv sync --all-packages` from the project root.

## Running the Bot

First, make sure the [API service](../api/) is running. Then, start the bot server:

```bash
uv run python bot/main.py
```

The bot's webhook listener will be running at `http://localhost:5555/webhook`. You will need to use a tool like `ngrok` to expose this endpoint to Facebook.
