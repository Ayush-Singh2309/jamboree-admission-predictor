import pytest

from sklearn.pipeline import Pipeline

from src.model.factory import get_model


def test_linear_model():

    model = get_model("linear")

    assert isinstance(model, Pipeline)


def test_ridge_model():

    model = get_model("ridge")

    assert isinstance(model, Pipeline)


def test_lasso_model():

    model = get_model("lasso")

    assert isinstance(model, Pipeline)


def test_invalid_model():

    with pytest.raises(ValueError):
        get_model("random")