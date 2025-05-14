import mlflow
import requests
import os
import time
import json
import requests
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Configuration ===
API_URL = "http://127.0.0.1:1234/invocations"
LOG_INTERVAL = 300            # seconds between confusion‐matrix logs
SINGLE_PRED_INTERVAL = 60     # seconds between single‐image logs
TEST_DIR = "data/vehicle_type_recognition"  # root of class‐subfolders

# === Load class labels ===
with open("models/class_labels.json", "r") as f:
    labels_map = json.load(f)
labels = [labels_map[str(i)] for i in range(len(labels_map))]

mlflow.set_experiment("Vehicle-Classification-Monitor")

def send_single_and_log(img_path):
    img = Image.open(img_path).convert("RGB").resize((128, 128))
    arr = np.array(img)/255.0
    inst = arr.tolist()

    resp = requests.post(API_URL,
                         headers={"Content-Type": "application/json"},
                         json={"instances": [inst]})
    pred = resp.json()["predictions"][0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    pred_label = labels[idx]

    with mlflow.start_run():
        mlflow.log_param("single_image", os.path.basename(img_path))
        mlflow.log_metric("pred_confidence", conf)
        mlflow.set_tag("predicted_class", pred_label)
        print(f"[Single] {img_path} → {pred_label} ({conf:.2f})")

def eval_testset_and_log():
    true = []
    pred = []
    # gather one example per class (or more, adjust as needed)
    for cls in labels:
        cls_dir = os.path.join(TEST_DIR, cls)
        img_files = os.listdir(cls_dir)
        # take up to 5 images per class
        for img_name in img_files[:5]:
            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).convert("RGB").resize((128, 128))
            arr = np.array(img)/255.0
            inst = arr.tolist()
            resp = requests.post(API_URL,
                                 headers={"Content-Type": "application/json"},
                                 json={"instances": [inst]})
            p = resp.json()["predictions"][0]
            idx = int(np.argmax(p))
            true.append(labels.index(cls))
            pred.append(idx)

    # compute confusion matrix
    cm = confusion_matrix(true, pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    # save and log artifact
    os.makedirs("monitor_artifacts", exist_ok=True)
    cm_path = os.path.join("monitor_artifacts", "confusion_matrix.png")
    fig.savefig(cm_path)
    plt.close(fig)

    with mlflow.start_run():
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")
        # also log overall accuracy
        acc = np.trace(cm) / np.sum(cm)
        mlflow.log_metric("testset_accuracy", acc)
        print(f"[Batch] Logged confusion matrix with accuracy {acc:.2f}")

if __name__ == "__main__":
    last_cm_time = 0
    test_images = []
    # prepare a rotating list of single‐image paths
    for cls in labels:
        cls_dir = os.path.join(TEST_DIR, cls)
        for img in os.listdir(cls_dir):
            test_images.append(os.path.join(cls_dir, img))
    idx = 0

    while True:
        # Single‐image log
        send_single_and_log(test_images[idx % len(test_images)])
        idx += 1

        # Periodic confusion‐matrix log
        now = time.time()
        if now - last_cm_time > LOG_INTERVAL:
            eval_testset_and_log()
            last_cm_time = now

        time.sleep(SINGLE_PRED_INTERVAL)
