import numpy as np
import tensorflow as tf

from dataset import load_data
from config import CLASSES

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)

MODEL_PATH = "models/cnn_simple.h5"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
FINAL_THRESHOLD = 0.6  


_, _, x_test, _, _, y_test = load_data()
model = tf.keras.models.load_model(MODEL_PATH)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")


y_prob = model.predict(x_test).ravel()



print("\n==============================")
print(" Threshold tuning results")
print("==============================")

for t in THRESHOLDS:
    print(f"\n===== Threshold = {t} =====")

    y_pred = (y_prob >= t).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=CLASSES
    ))

print("\n==============================")
print(f" Final evaluation (threshold = {FINAL_THRESHOLD})")
print("==============================")

y_pred_final = (y_prob >= FINAL_THRESHOLD).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_final,
    target_names=CLASSES
))


roc_auc = roc_auc_score(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

print("\n==============================")
print(" Threshold-independent metrics")
print("==============================")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")