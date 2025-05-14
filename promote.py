# promote.py
from mlflow.tracking import MlflowClient

client = MlflowClient()
name = "VehicleClassifier"
version = 4  # or read from args/env

# Promote version 4 to Production (and archive older versions)
client.transition_model_version_stage(
    name=name,
    version=version,
    stage="Production",
    archive_existing_versions=True
)
print(f"Promoted {name} v{version} to Production")
