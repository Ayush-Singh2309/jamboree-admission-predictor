from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from src.config import NUM_FEATS

def adj_r2(r2, y):
    """Calculates the adjusted R2 score."""
    m = len(y)
    return 1 - ((1 - r2) * (m - 1) / (m - NUM_FEATS - 1))

def evaluate(model, X, y):
    """Evaluates the model and returns the predictions and metrics"""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    adjr2 = adj_r2(r2, y)
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    return y_pred, {
        'R2': round(r2, 5),
        'Adjusted_R2': round(adjr2, 5),
        'RMSE': round(rmse, 5),
        'MAE': round(mae, 5)
    }
    