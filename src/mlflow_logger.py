import mlflow
from pathlib import Path
import src.config

class MlflowLogger():
    def set_experiment(self, experiment_name: str):
        """Sets the experiment name"""
        mlflow.set_tracking_uri(src.config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        """Starts an mlflow run"""
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        """Logs parameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        """Logs metrics"""
        mlflow.log_metrics(metrics)

    def log_model(self, model):
        """Logs the model"""
        mlflow.sklearn.log_model(sk_model=model, name=f"{src.config.MODEL_NAME}_model_")

    def log_artifact(self, path: Path, artifact_path: str = "artifacts"):
        """Logs artifact"""
        mlflow.log_artifact(local_path=str(path), artifact_path=artifact_path)
    
    def log_artifacts(self, artifacts: dict[str, Path | dict[str, Path]]):
        """Logs artifacts"""
        for key, value in artifacts.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    self.log_artifact(v, artifact_path="artifacts/plots")
            else:
                self.log_artifact(value)

    def set_tags(self, tags: dict):
        """Sets tags"""
        mlflow.set_tags(tags)

    def end_run(self):
        """Ends the current run"""
        mlflow.end_run()
        
    def log_runs(self, model, params:dict, metrics:dict, tags:dict, artifacts: dict[str, Path | dict[str, Path]]):
        """Logs run"""
        self.set_experiment(src.config.MLFLOW_EXPERIMENT_NAME)
        self.start_run(run_name=f"{src.config.MODEL_NAME} model")
        self.set_tags(tags)
        self.log_params(params)
        self.log_metrics(metrics)
        self.log_model(model)
        self.log_artifacts(artifacts)
        self.end_run()