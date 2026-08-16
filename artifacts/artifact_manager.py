import json
import numpy as np
import pandas as pd
import src.config
import joblib
import matplotlib.pyplot as plt

class ArtifactManager:

    def save_model(self, model, filename: str = f"{src.config.MODEL_NAME}_model.pkl"):
        path = src.config.MODELS_DIR / filename
        joblib.dump(model, path)
        return path

    def save_metrics(self, metrics: dict, filename: str):
        path = src.config.METRICS_DIR / filename
        with open(path, 'w') as f:
            json.dump(metrics, f)
        return path

    def save_predictions(self, actual: pd.Series, predicted, filename: str):
        path = src.config.PREDICTIONS_DIR / filename
        df = pd.DataFrame({
            "Actual": actual,
            "Predicted": predicted,
        })
        df["Residual"] = df["Actual"] - df["Predicted"]
        df.to_csv(path, index=False)
        return path

    def save_coefficients(self, features: list[str], coefficients: np.ndarray, filename: str):
        path = src.config.COEFFICIENTS_DIR / filename
        df = pd.DataFrame({
            "Feature": features,
            "Coefficient": coefficients
        })
        df.to_csv(path, index=False)
        return path

    def save_plot(self, fig, filename: str):
        path = src.config.PLOTS_DIR / filename
        fig.savefig(path)
        plt.close(fig)
        return path

    def generate_training_artifacts(
        self, 
        model,
        X_train, 
        y_train, 
        X_test, 
        y_test,
        metrics: dict,
        figures: dict[str, plt.Figure]
    ):
        model_path = self.save_model(model)
        train_pred = model.predict(X_train)
        train_pred_path = self.save_predictions(y_train, train_pred, "train_predictions.csv")
        test_pred = model.predict(X_test)
        test_pred_path = self.save_predictions(y_test, test_pred, "test_predictions.csv")
        metrics_path = self.save_metrics(metrics, "metrics.json")
        coefficients_path = self.save_coefficients(X_train.columns, model.named_steps.model.coef_, "coefficients.csv")
        fig_paths = {}
        for name, fig in figures.items():
            fig_path = self.save_plot(fig, f"{name}.png")
            fig_paths[name] = fig_path

        return {
            "model": model_path,
            "train_pred": train_pred_path,
            "test_pred": test_pred_path,
            "metrics": metrics_path,
            "coefficients": coefficients_path,
            "plots": fig_paths
        }