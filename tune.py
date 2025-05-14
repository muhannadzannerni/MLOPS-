import mlflow
import mlflow.keras

# ← Use a distinct experiment for hyperparameter tuning
mlflow.set_experiment("Vehicle-Classification-Tune")
mlflow.keras.autolog()

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import load_and_prepare_data

train_gen, val_gen = load_and_prepare_data()

for units in [64, 128, 256]:
    with mlflow.start_run():
        mlflow.log_param("dense_units", units)
        model = models.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(units, activation='relu'),
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
