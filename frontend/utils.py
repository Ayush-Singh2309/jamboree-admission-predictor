import requests
import src.config


def predict(payload: dict):

    response = requests.post(
        f"{src.config.UVICORN_URL}/predict",
        json=payload,
        timeout=10
    )
    response.raise_for_status()
    return response.json()

def get_model_info():
    response = requests.get(
        f"{src.config.UVICORN_URL}/model-info",
        timeout=60
    )
    response.raise_for_status()
    return response.json()