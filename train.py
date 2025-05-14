import mlflow
import mlflow.keras

# ← Set this script’s own experiment
mlflow.set_experiment("Vehicle-Classification-Train")
mlflow.keras.autolog()

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import load_and_prepare_data

train_gen, val_gen = load_and_prepare_data()

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(
    train_gen,
    epochs=25,
    validation_data=val_gen,
    callbacks=[early_stop]
)
