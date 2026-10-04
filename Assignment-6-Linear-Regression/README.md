# Assignment 6 – Linear Regression

## Aim

To implement the Linear Regression algorithm using Python and predict academic performance based on AI tool usage and productivity-related factors.

## Objective

- To understand the concept of Linear Regression.
- To use real-world student survey data.
- To predict academic performance using selected features.
- To evaluate the performance of the regression model.
- To visualize actual and predicted values.

## Dataset

The dataset used is **AI Tools Usage & Student Productivity**.

The dataset contains **1042 student responses** and includes information about AI tool usage, productivity, academic performance, and other student-related factors.

## Features Used

The following features were selected for the Linear Regression model:

1. AI Tool Usage Hours
2. Academic Usage of AI Tools
3. Productivity After Using AI Tools

### Target Variable

**Academic Performance After Using AI Tools**

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Algorithm Used

### Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical value.

The general equation is:

Y = b0 + b1X1 + b2X2 + ... + bnXn

Where:

- Y = Predicted value
- b0 = Intercept
- b1, b2, ..., bn = Model coefficients
- X1, X2, ..., Xn = Input features

In this assignment, Linear Regression is used to predict students' academic performance based on their AI usage and productivity.

## Implementation Steps

1. Load the AI Tools Usage & Student Productivity dataset.
2. Display the dataset columns and shape.
3. Select the required input features.
4. Convert the selected data into numerical format.
5. Handle missing values.
6. Separate input features and target variable.
7. Split the dataset into training and testing sets.
8. Train the Linear Regression model.
9. Generate predictions using the trained model.
10. Display actual and predicted values.
11. Visualize the results using a scatter plot.

## Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

The training data is used to train the Linear Regression model, while the testing data is used to evaluate its predictions.

## Output

The program displays:

- Model intercept
- Actual academic performance
- Predicted academic performance
- Comparison between actual and predicted values
- Actual vs Predicted visualization

## Visualization

The project generates an **Actual vs Predicted Academic Performance** graph.

The graph helps compare the actual academic performance values with the values predicted by the Linear Regression model.

## Advantages

- Simple and easy to understand.
- Easy to implement using Python.
- Works well for understanding relationships between variables.
- Computationally efficient.
- Provides interpretable model coefficients.

## Disadvantages

- Assumes a linear relationship between variables.
- Sensitive to outliers.
- Performance may decrease when the relationship between variables is non-linear.
- Predictions can be affected by missing or noisy data.

## Applications

Linear Regression can be used for:

- Academic performance prediction.
- Student productivity analysis.
- Sales prediction.
- House price prediction.
- Salary prediction.
- Trend analysis.
- Business forecasting.

## Conclusion

Linear Regression was successfully implemented using the AI Tools Usage & Student Productivity dataset. The model was used to predict academic performance based on AI usage and productivity-related features. The actual and predicted values were compared using a visualization, demonstrating how Linear Regression can be applied to real-world student data.

## How to Run

Make sure the virtual environment is activated and run:

```bash
python Assignment-6-Linear-Regression/assignment6.py