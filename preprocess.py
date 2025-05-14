import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_prepare_data(
    data_dir="data/vehicle_type_recognition",
    img_size=(128, 128),
    batch_size=32
):
    """
    Loads images from `data_dir`, applies rescaling and split into train/val.
    Saves class label mapping to models/class_labels.json.
    Returns (train_gen, val_gen).
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Save class label mapping
    os.makedirs("models", exist_ok=True)
    class_indices = train_gen.class_indices
    labels = {str(v): k for k, v in class_indices.items()}
    with open("models/class_labels.json", "w") as f:
        json.dump(labels, f)

    return train_gen, val_gen


if __name__ == "__main__":
    # Quick test
    train_gen, val_gen = load_and_prepare_data()
    print("Data loaded. Classes:", train_gen.class_indices)
