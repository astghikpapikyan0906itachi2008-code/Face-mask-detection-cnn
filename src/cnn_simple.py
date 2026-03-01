from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from config import *

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu",
               input_shape=(*IMAGE_SIZE, CHANNELS)),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
