import numpy as np
import tensorflow as tf
from dataset import load_data
from cnn_simple import build_model
from config import *

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

x_train, x_val, x_test, y_train, y_val, y_test = load_data()

model = build_model()

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

model.save("models/cnn_simple.h5")