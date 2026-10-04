import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. Load Dataset
# ============================================================

file_path = "Assignment-2-Clustering/Student_Survey_AI_Tools_(Responses).xlsx"

df = pd.read_excel(file_path)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# 2. Select Features and Target
# ============================================================

features = [
    "Approximately how many hours per week do you spend usiAi tools",
    "Productivity ",
    "How would you rate your productivity After using AI tools",
    "How would you rate your academic performance after using AI tools?",
    "How much do you trust the information provided by AI tools"
]

target = "Overall,how has Ai affected your Academic productivity?"


# ============================================================
# 3. Check Required Columns
# ============================================================

required_columns = features + [target]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing Columns:")

    for column in missing_columns:
        print("-", column)

    raise SystemExit


# ============================================================
# 4. Create Feature and Target Data
# ============================================================

X = df[features].copy()
y = df[target].copy()


# ============================================================
# 5. Handle Missing Values
# ============================================================

for column in X.columns:

    X[column] = X[column].fillna(
        X[column].mode()[0]
    )

y = y.fillna(
    y.mode()[0]
)


# ============================================================
# 6. Convert Categorical Features to Numerical Values
# ============================================================

print("\nOriginal Feature Types:")

print(X.dtypes)


# One-hot encoding converts survey answers such as
# "Less than 2 hours" into numerical columns.

X = pd.get_dummies(
    X,
    drop_first=False
)


print("\nFeatures After Encoding:")
print(X.shape)


# ============================================================
# 7. Encode Target Variable
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y.astype(str)
)


print("\nTarget Classes:")

print(
    label_encoder.classes_
)


# ============================================================
# 8. Convert Boolean Values to Integer
# ============================================================

X = X.astype(float)


# ============================================================
# 9. Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


# ============================================================
# 10. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 11. Create KNN Model
# ============================================================

model = KNeighborsClassifier(
    n_neighbors=5
)


# ============================================================
# 12. Train KNN Model
# ============================================================

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 13. Make Predictions
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# 14. Calculate Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n=============================================")
print("KNN CLASSIFICATION RESULTS")
print("=============================================")

print(
    "\nAccuracy:",
    accuracy
)

print(
    "\nAccuracy Percentage:",
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 15. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 16. Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 17. Visualize Confusion Matrix
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title(
    "KNN Classification - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)


# X-axis labels

plt.xticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_,
    rotation=45,
    ha="right"
)


# Y-axis labels

plt.yticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)


# Display values inside cells

for i in range(len(cm)):

    for j in range(len(cm[i])):

        plt.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()

plt.show()