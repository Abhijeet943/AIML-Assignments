import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

file_path = os.path.join(
    os.path.dirname(__file__),
    "Student_Survey_AI_Tools_(Responses).xlsx"
)

df = pd.read_excel(file_path)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# ---------------------------------------------------------
# 2. Define Dataset Columns
# ---------------------------------------------------------

hours_column = (
    "Approximately how many hours per week do you spend usiAi tools"
)

productivity_column = (
    "How would you rate your productivity After using AI tools"
)

target_column = (
    "How would you rate your academic performance after using AI tools?"
)


# ---------------------------------------------------------
# 3. Convert AI Usage Hours to Numeric
# ---------------------------------------------------------

def convert_hours(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if "Less than 2" in value:
        return 1.0

    if "2-5" in value:
        return 3.5

    if "5-10" in value:
        return 7.5

    if "10-15" in value:
        return 12.5

    if "15-20" in value:
        return 17.5

    if "20" in value:
        return 20.0

    try:
        return float(value)
    except:
        return np.nan


df["AI_Usage_Hours"] = df[hours_column].apply(convert_hours)


# ---------------------------------------------------------
# 4. Convert Other Features to Numeric
# ---------------------------------------------------------

df["Productivity"] = pd.to_numeric(
    df[productivity_column],
    errors="coerce"
)

df["Academic_Performance"] = pd.to_numeric(
    df[target_column],
    errors="coerce"
)


# ---------------------------------------------------------
# 5. Select Features and Target
# ---------------------------------------------------------

features = [
    "AI_Usage_Hours",
    "Productivity"
]

target = "Academic_Performance"

X = df[features].copy()

y = df[target].copy()


print("\nSelected Features:")
print(X.head())

print("\nTarget Values:")
print(y.head())


# ---------------------------------------------------------
# 6. Handle Missing Values
# ---------------------------------------------------------

data = pd.concat([X, y], axis=1)

# Remove rows where target is missing
data = data.dropna(subset=[target])

# Fill missing feature values with median
for column in features:
    data[column] = data[column].fillna(
        data[column].median()
    )

X = data[features]

y = data[target]


print("\nDataset after handling missing values:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())


# ---------------------------------------------------------
# 7. Split Dataset
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)

print("Testing Data:", X_test.shape)


# ---------------------------------------------------------
# 8. Feature Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 9. Create SVR Model
# ---------------------------------------------------------

model = SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.1
)


# ---------------------------------------------------------
# 10. Train the Model
# ---------------------------------------------------------

model.fit(X_train_scaled, y_train)


# ---------------------------------------------------------
# 11. Make Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test_scaled)


# ---------------------------------------------------------
# 12. Evaluate the Model
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\nSVM Regression Results:")

print("Mean Absolute Error (MAE):", round(mae, 4))

print("Mean Squared Error (MSE):", round(mse, 4))

print("Root Mean Squared Error (RMSE):", round(rmse, 4))

print("R2 Score:", round(r2, 4))


# ---------------------------------------------------------
# 13. Actual vs Predicted Values
# ---------------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")

print(results.head(10))


# ---------------------------------------------------------
# 14. Visualization
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    s=50
)

min_value = min(y_test.min(), y_pred.min())

max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Academic Performance")

plt.ylabel("Predicted Academic Performance")

plt.title("SVM Regression - Actual vs Predicted")

plt.grid(True)

plt.tight_layout()

plt.show()