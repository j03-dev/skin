# 🤖 Skin-AI (ML Core)

This package contains the core machine learning logic for **Skin AI**. It includes the model architecture, training scripts, and prediction functions for detecting `mpox` from skin images.

## Functionality

- **Model Training**: The `train()` function in `src/skin_ai.py` compiles and trains the Keras model using the datasets in `assets/datasets`.
- **Prediction**: The `predict()` function takes a trained model and image bytes to return a classification label and confidence score.
- **Model**: The pre-trained model is located at `assets/models/mpox_model-v3.h5`.

## Usage

### Training

To retrain the model, run the training script:

```bash
uv run python skin-ai/src/skin_ai.py
```
*Note: Make sure you have the datasets located in `assets/datasets`.*

### As a Library

This package is used by the `api` service to load the model and perform predictions. See `api/main.py` for an example.
