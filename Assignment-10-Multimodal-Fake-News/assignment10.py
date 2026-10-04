import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# Assignment 10
# Fake News Detection from Multimodal Data
# =========================================================

print("\n==============================================")
print("   MULTIMODAL FAKE NEWS DETECTION SYSTEM")
print("==============================================\n")


# =========================================================
# 1. Create Demo Multimodal Dataset
# =========================================================
# Text + Image information are used together.
# For the practical demonstration, sample data is created.

data = {
    "text": [
        "Government announces new education policy for students",
        "Scientists discover a new method to improve renewable energy",
        "Breaking shocking news celebrity has secretly disappeared",
        "Miracle medicine can cure every disease instantly",
        "University announces new scholarship program",
        "Researchers publish a study on climate change",
        "You will not believe this unbelievable secret government message",
        "Doctors are hiding a magical cure from the public",
        "Government launches new digital education initiative",
        "New research improves solar energy efficiency",
        "Famous actor spreads unbelievable conspiracy theory",
        "Secret medicine promises instant weight loss",
        "College announces new technical workshop",
        "Scientists develop improved battery technology",
        "Social media post claims impossible supernatural event",
        "Fake experts claim one pill can solve all health problems"
    ],

    "label": [
        0, 0, 1, 1,
        0, 0, 1, 1,
        0, 0, 1, 1,
        0, 0, 1, 1
    ],

    # Simulated image features
    # 0 = normal/news image
    # 1 = suspicious/clickbait image
    "image_feature": [
        0.20, 0.25, 0.90, 0.85,
        0.15, 0.30, 0.95, 0.88,
        0.20, 0.28, 0.92, 0.86,
        0.18, 0.24, 0.91, 0.89
    ]
}

df = pd.DataFrame(data)

print("Dataset created successfully.")
print("\nDataset:")
print(df)


# =========================================================
# 2. Separate Text and Labels
# =========================================================

texts = df["text"]
labels = df["label"]


# =========================================================
# 3. Convert Text into TF-IDF Features
# =========================================================

print("\nCreating TF-IDF text features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=100
)

text_features = vectorizer.fit_transform(texts)

print("Text feature shape:", text_features.shape)


# =========================================================
# 4. Extract Image Features
# =========================================================
# In a real dataset, image features would be extracted
# from the actual news images.
#
# Here, image_feature represents an image characteristic
# for demonstrating multimodal feature fusion.

image_features = df[["image_feature"]].values

scaler = StandardScaler()

image_features = scaler.fit_transform(image_features)

print("Image feature shape:", image_features.shape)


# =========================================================
# 5. Convert TF-IDF to Dense Array
# =========================================================

text_features_dense = text_features.toarray()


# =========================================================
# 6. Combine Text + Image Features
# =========================================================

multimodal_features = np.hstack(
    (text_features_dense, image_features)
)

print("\nMultimodal feature shape:")
print(multimodal_features.shape)


# =========================================================
# 7. Split Dataset
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    multimodal_features,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 8. Train Logistic Regression Model
# =========================================================

print("\nTraining multimodal classification model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")


# =========================================================
# 9. Make Predictions
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 10. Calculate Accuracy
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================================")
print("MODEL PERFORMANCE")
print("==============================================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")


# =========================================================
# 11. Classification Report
# =========================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Real News", "Fake News"],
        zero_division=0
    )
)


# =========================================================
# 12. Confusion Matrix
# =========================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# 13. Display Confusion Matrix
# =========================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Multimodal Fake News Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Real News", "Fake News"]
)

plt.yticks(
    [0, 1],
    ["Real News", "Fake News"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()


# =========================================================
# 14. Test a New News Article
# =========================================================

new_text = [
    "Scientists announce a new research study about renewable energy"
]

new_text_features = vectorizer.transform(
    new_text
).toarray()

# Example image feature for the new article
new_image_feature = np.array([[0.25]])

new_image_feature = scaler.transform(
    new_image_feature
)

new_multimodal_features = np.hstack(
    (
        new_text_features,
        new_image_feature
    )
)


prediction = model.predict(
    new_multimodal_features
)


# =========================================================
# 15. Display Prediction
# =========================================================

print("\n==============================================")
print("NEW NEWS PREDICTION")
print("==============================================")

if prediction[0] == 0:
    print("Prediction: REAL NEWS")
else:
    print("Prediction: FAKE NEWS")


print("\n==============================================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("==============================================")