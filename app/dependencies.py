from src.model.inference import load_model
import src.config


model = load_model(src.config.MODEL_PATH)

def get_model():
    return model
    