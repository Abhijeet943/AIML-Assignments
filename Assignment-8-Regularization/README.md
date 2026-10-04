# Assignment 8: Regularization to Avoid Overfitting

## Aim

To apply regularization techniques to a regression model in order to reduce overfitting and improve model performance.

## Objective

The objective of this practical is to understand how regularization techniques such as Ridge and Lasso Regression can be used to control model complexity and reduce overfitting.

## Dataset

The dataset used for this assignment is:

**AI Tools Usage & Student Productivity Survey**

The dataset contains responses collected from students regarding their usage of AI tools and its effect on academic productivity and performance.

## Features Used

The following features were used:

- AI Usage Hours
- Productivity

### Target Variable

- Academic Performance

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Excel Dataset

## Algorithms Used

### 1. Linear Regression

Linear Regression is used as the baseline regression model.

### 2. Ridge Regression

Ridge Regression applies L2 regularization by adding a penalty to large model coefficients. This helps reduce model complexity and overfitting.

### 3. Lasso Regression

Lasso Regression applies L1 regularization. It can reduce some coefficients toward zero and can therefore help in feature selection.

## Implementation Steps

1. Load the student survey dataset.
2. Convert AI usage hours into numerical values.
3. Convert productivity and academic performance into numerical values.
4. Remove records containing missing values.
5. Separate input features and target variable.
6. Split the dataset into training and testing data.
7. Standardize the input features.
8. Generate polynomial features to create a more complex regression model.
9. Apply Linear Regression without regularization.
10. Apply Ridge Regression.
11. Apply Lasso Regression.
12. Calculate Mean Squared Error (MSE).
13. Calculate R² score.
14. Compare the performance of the models.
15. Visualize actual and predicted academic performance.

## Results

The obtained results were:

| Model | MSE | R² Score |
|---|---:|---:|
| Linear Regression | 0.6955 | 0.5166 |
| Ridge Regression | 0.6918 | 0.5192 |
| Lasso Regression | 0.6758 | 0.5303 |

Lasso Regression achieved the highest R² score among the three models for the test data.

## Visualization

The program generates:

1. Actual vs Predicted Academic Performance graph using Ridge Regression.
2. R² score comparison graph for Linear, Ridge and Lasso Regression.

## Conclusion

Regularization helps control model complexity and can reduce the risk of overfitting. In this practical, Linear Regression was compared with Ridge and Lasso Regression.

For the given student AI-tools dataset, Lasso Regression achieved the highest R² score of approximately **0.5303**, while Ridge Regression also slightly improved the performance compared with the unregularized Linear Regression model.

Thus, regularization is useful for improving the generalization performance of regression models and controlling overfitting.

## How to Run

Make sure the dataset file is present inside the Assignment-8-Regularization folder:

```text
Student_Survey_AI_Tools_(Responses).xlsx