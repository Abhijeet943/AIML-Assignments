import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------
# 1. Load Dataset
# ---------------------------------------------

file_path = "Assignment-2-Clustering/Student_Survey_AI_Tools_(Responses).xlsx"

df = pd.read_excel(file_path)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# ---------------------------------------------
# 2. Select Features and Target
# ---------------------------------------------

features = [
    "Approximately how many hours per week do you spend usiAi tools",
    "How much do you trust the information provided by AI tools",
    "How would you rate your productivity After using AI tools"
]

target = "How would you rate your academic performance after using AI tools?"


X = df[features].copy()
y = df[target].copy()


# ---------------------------------------------
# 3. Convert Features to Numeric
# ---------------------------------------------

def extract_number(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Extract first number from text
    match = re.search(r"\d+(\.\d+)?", value)

    if match:
        return float(match.group())

    return None


# Convert Hours and Productivity
X[features[0]] = X[features[0]].apply(extract_number)
X[features[2]] = X[features[2]].apply(extract_number)

# Convert target
y = y.apply(extract_number)


# ---------------------------------------------
# 4. Encode Trust Column
# ---------------------------------------------

trust_column = features[1]

# First try numeric conversion
trust_numeric = X[trust_column].apply(extract_number)

# If trust responses are text, convert categories to numbers
if trust_numeric.notna().sum() == 0:

    trust_categories = X[trust_column].fillna("Unknown").astype(str)

    trust_mapping = {
        "Not at all": 1,
        "Very little": 1,
        "Slightly": 2,
        "Somewhat": 3,
        "Moderately": 3,
        "Quite a lot": 4,
        "A lot": 4,
        "Very much": 5,
        "Completely": 5,
        "Very high": 5,
        "High": 4,
        "Medium": 3,
        "Low": 2,
        "Very low": 1
    }

    X[trust_column] = trust_categories.map(trust_mapping)

else:

    X[trust_column] = trust_numeric


# ---------------------------------------------
# 5. Handle Missing Values
# ---------------------------------------------

print("\nMissing Values Before Cleaning:")
print(X.isnull().sum())

print("\nMissing Target Values:")
print(y.isnull().sum())


# Fill missing feature values with median
for column in features:
    X[column] = X[column].fillna(X[column].median())


# Fill missing target values with median
y = y.fillna(y.median())


# Remove any remaining invalid rows
valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]


print("\nMissing Values After Cleaning:")
print(X.isnull().sum())

print("\nFinal Dataset Shape:")
print(X.shape)


# ---------------------------------------------
# 6. Split Dataset
# ---------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ---------------------------------------------
# 7. Create Linear Regression Model
# ---------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# ---------------------------------------------
# 8. Make Predictions
# ---------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------
# 9. Evaluate Model
# ---------------------------------------------

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n---------------------------------------------")
print("Linear Regression Results")
print("---------------------------------------------")

print("Mean Squared Error (MSE):", round(mse, 4))
print("Mean Absolute Error (MAE):", round(mae, 4))
print("R² Score:", round(r2, 4))


# ---------------------------------------------
# 10. Display Feature Coefficients
# ---------------------------------------------

print("\nFeature Coefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", round(coefficient, 4))

print("\nIntercept:", round(model.intercept_, 4))


# ---------------------------------------------
# 11. Actual vs Predicted Values
# ---------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))


# ---------------------------------------------
# 12. Visualization
# ---------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    s=50
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Academic Performance")
plt.ylabel("Predicted Academic Performance")

plt.title("Linear Regression - Actual vs Predicted")

plt.grid(True)

plt.tight_layout()

plt.show()