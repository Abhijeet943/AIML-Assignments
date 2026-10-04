# Assignment 7 – SVM Regression

## Aim

To implement Support Vector Machine (SVM) Regression using the Support Vector Regressor (SVR) algorithm and predict academic performance based on AI tool usage and productivity-related factors.

## Objective

- Load and preprocess the AI Tools Usage & Student Productivity dataset.
- Select relevant numerical features.
- Convert AI tool usage hours into numerical values.
- Handle missing values.
- Split the dataset into training and testing sets.
- Scale the input features using StandardScaler.
- Implement SVM Regression using SVR.
- Predict academic performance.
- Evaluate the model using MAE, MSE, RMSE and R² Score.
- Visualize actual and predicted academic performance.

## Dataset

The dataset used for this assignment is:

**Student_Survey_AI_Tools_(Responses).xlsx**

The dataset contains responses from students regarding their AI tool usage and academic productivity.

### Selected Features

1. AI Usage Hours
2. Academic Usage
3. Productivity

### Target Variable

**Academic Performance**

The target represents the student's academic performance after using AI tools.

## Algorithm Used

### Support Vector Machine Regression (SVR)

Support Vector Regression is the regression version of Support Vector Machine.

In this implementation, the following SVR configuration is used:

- Kernel: RBF
- C: 100
- Gamma: Scale
- Epsilon: 0.1

The RBF kernel is used to model non-linear relationships between the input features and academic performance.

## Preprocessing

The following preprocessing steps were performed:

1. Loaded the Excel dataset using Pandas.
2. Converted AI usage hours into numerical values.
3. Converted academic usage and productivity into numerical values.
4. Converted academic performance into the target variable.
5. Removed records containing missing values.
6. Split the dataset into training and testing sets.
7. Standardized the input features using `StandardScaler`.

## Train-Test Split

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

The model was trained using the training dataset and evaluated using the testing dataset.

## Evaluation Metrics

The following metrics were used:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and indicates the typical prediction error.

### R² Score

R² indicates how well the model explains the variation in the target variable.

## Visualization

An Actual vs Predicted graph was created to compare the actual academic performance with the performance predicted by the SVM Regression model.

The dashed diagonal line represents the ideal prediction where:

```text
Actual = Predicted