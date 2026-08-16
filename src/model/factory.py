import src.model.linear as lr
import src.model.lasso as lasso
import src.model.ridge as ridge


def get_model(name):
    models = {
        'linear': lr.build_model(),
        'ridge': ridge.build_model(),
        'lasso': lasso.build_model()
    }
    if name not in models.keys():
        raise ValueError("Invalid model name")
    return models[name]