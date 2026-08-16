from src.model.pipeline import train_model
import src.config

def train():
    model, metrics = train_model()

    print("\nModel :", f"{src.config.MODEL_NAME} model")

    print("\nTrain Metrics")
    print(metrics['train'])

    print("\nTest Metrics")
    print(metrics['test'])

if __name__ == '__main__':
    train()