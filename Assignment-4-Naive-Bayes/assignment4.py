import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =====================================================
# 1. Load Dataset
# =====================================================

file_path = "Assignment-2-Clustering/Student_Survey_AI_Tools_(Responses).xlsx"

df = pd.read_excel(file_path)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())


# =====================================================
# 2. Select Features and Target
# =====================================================

features = [
    "Ai Tool Usage",
    "How Frequently do you use Ai tools",
    "Approximately how many hours per week do you spend usiAi tools",
    "Productivity ",
    "How would you rate your productivity After using AI tools",
    "How much time o AI tools save you during Your Academic work",
    "How would you rate your academic performance after using AI tools?",
    "How much do Ai ools help you understand different academic concepts ",
    "How Dependent do you feelon  tools for completing your academic work",
    "How much do you trust the information provided by AI tools"
]

target = "Overall,how has Ai affected your Academic productivity?"


# =====================================================
# 3. Create Feature and Target Data
# =====================================================

X = df[features].copy()
y = df[target].copy()


# Remove rows where target is missing
valid_rows = y.notna()

X = X[valid_rows]
y = y[valid_rows]


# =====================================================
# 4. Separate Numerical and Categorical Features
# =====================================================

numeric_features = [
    "How would you rate your productivity After using AI tools",
    "How would you rate your academic performance after using AI tools?"
]

categorical_features = [
    "Ai Tool Usage",
    "How Frequently do you use Ai tools",
    "Approximately how many hours per week do you spend usiAi tools",
    "Productivity ",
    "How much time o AI tools save you during Your Academic work",
    "How much do Ai ools help you understand different academic concepts ",
    "How Dependent do you feelon  tools for completing your academic work",
    "How much do you trust the information provided by AI tools"
]


# =====================================================
# 5. Preprocessing
# =====================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# =====================================================
# 6. Encode Target Variable
# =====================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget Classes:")
print(label_encoder.classes_)


# =====================================================
# 7. Split Dataset
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# =====================================================
# 8. Create Naïve Bayes Model
# =====================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ]
)


# =====================================================
# 9. Train Model
# =====================================================

model.fit(X_train, y_train)


# =====================================================
# 10. Make Predictions
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# 11. Evaluate Model
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n=============================================")
print("Naïve Bayes Classification Results")
print("=============================================")

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(round(accuracy * 100, 2), "%")


print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# =====================================================
# 12. Visualize Confusion Matrix
# =====================================================

plt.figure(figsize=(9, 7))

plt.imshow(cm)

plt.title("Naïve Bayes Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_,
    rotation=30,
    ha="right"
)

plt.yticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)


# Display values inside the matrix
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