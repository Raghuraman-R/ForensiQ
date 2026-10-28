
import os
import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Configuration
# -----------------------------
DATASET_PATH = "dataset"
IMAGE_SIZE = (64, 64)
RANDOM_STATE = 42

CLASSES = {
    "authentic": 0,
    "forged": 1
}

X = []
y = []
skipped = 0

print("\n========== ForensiQ Training ==========\n")

# -----------------------------
# 2. Load and process images
# -----------------------------
for folder, label in CLASSES.items():
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}"
        )

    files = os.listdir(folder_path)
    print(f"Reading {folder}: {len(files)} files")

    for filename in files:
        file_path = os.path.join(folder_path, filename)

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        ):
            continue

        image = cv2.imread(file_path)

        if image is None:
            skipped += 1
            continue

        # Resize and normalize image
        image = cv2.resize(image, IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert pixels to a numeric feature vector
        features = image.astype(np.float32).flatten() / 255.0

        X.append(features)
        y.append(label)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

if len(X) == 0:
    raise ValueError(
        "No valid images found. Check your dataset folders."
    )

print(f"\nValid images loaded: {len(X)}")
print(f"Skipped unreadable images: {skipped}")
print(f"Feature vector size: {X.shape[1]}")

# -----------------------------
# 3. Split into train and test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining images: {len(X_train)}")
print(f"Testing images: {len(X_test)}")

# -----------------------------
# 4. Train baseline classifier
# -----------------------------
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
)

print("\nTraining model... please wait.")
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate model
# -----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== Evaluation ==========")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Authentic", "Forged"],
        zero_division=0
    )
)

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# -----------------------------
# 6. Save trained model
# -----------------------------
os.makedirs("models", exist_ok=True)

model_path = os.path.join("models", "forensiq_model.joblib")

joblib.dump(
    {
        "model": model,
        "image_size": IMAGE_SIZE,
        "classes": CLASSES,
        "feature_type": "resized_rgb_pixels"
    },
    model_path
)

print(f"\nModel saved to: {model_path}")
print("\nTraining complete! ForensiQ baseline is ready.")
print("Remember: predictions are experimental, not proof of forgery.")
