import mlflow
from mlflow.tracking import MlflowClient

# 1) Connect to MLflow
client = MlflowClient()

# 2) Identify the experiment and best run
experiment = client.get_experiment_by_name("Vehicle-Classification-Tuning")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_accuracy DESC"],
    max_results=1
)
best_run = runs[0]

# 3) Register the best model
model_uri = f"runs:/{best_run.info.run_id}/model"
result = mlflow.register_model(
    model_uri=model_uri,
    name="VehicleClassifier"
)
print(f"Registered new model version: {result.version}")
