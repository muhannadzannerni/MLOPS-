import os
import json
import time
import requests
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:1234/invocations"
# Folder containing subfolders named bus/, car/, truck/, motorcycle/
BATCH_DIR = "data/vehicle_type_recognition"
IMG_SIZE = (128, 128)

# ─── PREPARE ─────────────────────────────────────────────────────────────────────
# Load class labels
with open("models/class_labels.json", "r") as f:
    labels_map = json.load(f)
# Build list of (image_path, true_label_index)
samples = []
for idx, label in labels_map.items():
    class_folder = os.path.join(BATCH_DIR, labels_map[idx])
    for fname in os.listdir(class_folder):
        if fname.lower().endswith((".jpg", ".png")):
            samples.append((os.path.join(class_folder, fname), int(idx)))

# ─── MLflow SETUP ────────────────────────────────────────────────────────────────
mlflow.set_experiment("Vehicle-Classification-BatchInference")

# ─── RUN INFERENCE & LOG ─────────────────────────────────────────────────────────
with mlflow.start_run():
    true = []
    pred = []

    for img_path, true_idx in samples:
        # Preprocess
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img) / 255.0
        inst = arr.tolist()

        # Call model API
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={"instances": [inst]}
        )
        p = response.json()["predictions"][0]
        pred_idx = int(np.argmax(p))

        true.append(true_idx)
        pred.append(pred_idx)

    # Compute and log classification report
    report = classification_report(true, pred, target_names=[labels_map[str(i)] for i in range(len(labels_map))], output_dict=True)
    for cls, metrics in report.items():
        if cls in labels_map.values():
            mlflow.log_metric(f"{cls}_precision", metrics["precision"])
            mlflow.log_metric(f"{cls}_recall", metrics["recall"])
            mlflow.log_metric(f"{cls}_f1-score", metrics["f1-score"])
    overall_accuracy = report["accuracy"]
    mlflow.log_metric("overall_accuracy", overall_accuracy)

    # Confusion matrix plot
    cm = confusion_matrix(true, pred, labels=list(range(len(labels_map))))
    disp = ConfusionMatrixDisplay(cm, display_labels=[labels_map[str(i)] for i in range(len(labels_map))])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Batch Inference Confusion Matrix")

    # Save & log artifact
    os.makedirs("batch_artifacts", exist_ok=True)
    cm_path = os.path.join("batch_artifacts", "confusion_matrix.png")
    fig.savefig(cm_path)
    plt.close(fig)
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

    # Log the raw report as JSON
    report_path = os.path.join("batch_artifacts", "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(report_path, artifact_path="classification_report")

    print(f"✅ Batch inference done: accuracy={overall_accuracy:.2f}")
