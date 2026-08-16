from src.model.pipeline import train_model


def test_training_pipeline():

    model, metrics = train_model()

    assert model is not None

    assert "train" in metrics

    assert "test" in metrics