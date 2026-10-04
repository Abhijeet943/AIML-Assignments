import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


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
# 4. Convert Features to Numeric
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

data = df[features + [target]].copy()

data = data.dropna()

X = data[features]

y = data[target]


print("\nSelected Features:")
print(X.head())

print("\nTarget Values:")
print(y.head())

print("\nDataset after removing missing values:")
print(data.shape)


# ---------------------------------------------------------
# 6. Split Dataset
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
# 7. Standard Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 8. Create Overfitting Model
# ---------------------------------------------------------

polynomial_features = PolynomialFeatures(
    degree=5
)

X_train_poly = polynomial_features.fit_transform(
    X_train_scaled
)

X_test_poly = polynomial_features.transform(
    X_test_scaled
)


# ---------------------------------------------------------
# 9. Linear Regression Without Regularization
# ---------------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train_poly,
    y_train
)

linear_pred = linear_model.predict(
    X_test_poly
)


# ---------------------------------------------------------
# 10. Ridge Regression
# ---------------------------------------------------------

ridge_model = Ridge(
    alpha=1.0
)

ridge_model.fit(
    X_train_poly,
    y_train
)

ridge_pred = ridge_model.predict(
    X_test_poly
)


# ---------------------------------------------------------
# 11. Lasso Regression
# ---------------------------------------------------------

lasso_model = Lasso(
    alpha=0.01,
    max_iter=10000
)

lasso_model.fit(
    X_train_poly,
    y_train
)

lasso_pred = lasso_model.predict(
    X_test_poly
)


# ---------------------------------------------------------
# 12. Evaluate Models
# ---------------------------------------------------------

linear_mse = mean_squared_error(
    y_test,
    linear_pred
)

ridge_mse = mean_squared_error(
    y_test,
    ridge_pred
)

lasso_mse = mean_squared_error(
    y_test,
    lasso_pred
)


linear_r2 = r2_score(
    y_test,
    linear_pred
)

ridge_r2 = r2_score(
    y_test,
    ridge_pred
)

lasso_r2 = r2_score(
    y_test,
    lasso_pred
)


# ---------------------------------------------------------
# 13. Display Results
# ---------------------------------------------------------

print("\n========================================")
print("REGULARIZATION RESULTS")
print("========================================")

print("\nLinear Regression:")
print("MSE:", round(linear_mse, 4))
print("R2 Score:", round(linear_r2, 4))

print("\nRidge Regression:")
print("MSE:", round(ridge_mse, 4))
print("R2 Score:", round(ridge_r2, 4))

print("\nLasso Regression:")
print("MSE:", round(lasso_mse, 4))
print("R2 Score:", round(lasso_r2, 4))


# ---------------------------------------------------------
# 14. Actual vs Predicted
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    ridge_pred,
    s=50
)

min_value = min(
    y_test.min(),
    ridge_pred.min()
)

max_value = max(
    y_test.max(),
    ridge_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Academic Performance")

plt.ylabel("Predicted Academic Performance")

plt.title("Ridge Regression - Actual vs Predicted")

plt.grid(True)

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 15. Compare Models
# ---------------------------------------------------------

models = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression"
]

r2_scores = [
    linear_r2,
    ridge_r2,
    lasso_r2
]

plt.figure(figsize=(8, 6))

plt.bar(
    models,
    r2_scores
)

plt.xlabel("Models")

plt.ylabel("R2 Score")

plt.title("Comparison of Regression Models")

plt.xticks(rotation=15)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()