# Assignment 4 - Naïve Bayes Classification

## Aim

To apply the Naïve Bayes classification algorithm on the
**AI Tools Usage & Student Productivity** dataset.

## Dataset

The dataset used for this assignment is:

**Student_Survey_AI_Tools_(Responses).xlsx**

The dataset contains information about students' AI tool usage,
productivity, academic performance, and their perception of
AI's impact on productivity.

## Objective

The objective of this practical is to classify students based on
their response to whether AI improves their productivity.

## Features Used

The following features were selected:

- AI Usage Hours
- Productivity
- Academic Performance
- Accuracy Trust

## Target Variable

The target variable is:

`AI_Improves_Productivity`

The target contains five classes:

- Greatly decreased
- Slightly decreased
- No change
- Slightly improved
- Greatly improved

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Excel Dataset

## Algorithm Used

### Gaussian Naïve Bayes

Gaussian Naïve Bayes is a supervised machine learning
classification algorithm based on Bayes' theorem.

It assumes that the numerical features follow a Gaussian
(normal) distribution within each class.

## Methodology

1. Load the Excel dataset using Pandas.
2. Select the required features and target variable.
3. Handle missing values.
4. Encode the target classes using LabelEncoder.
5. Split the dataset into training and testing sets.
6. Create a Gaussian Naïve Bayes model.
7. Train the model using the training data.
8. Predict the classes for the test data.
9. Calculate the accuracy.
10. Generate the confusion matrix and classification report.
11. Visualize the confusion matrix.

## Train-Test Split

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

A random state of 42 was used to obtain reproducible results.

## Results

The Naïve Bayes model successfully classified the test data.

The model achieved an accuracy of approximately:

**21%**

The classification report provides precision, recall,
F1-score, and support for each class.

## Confusion Matrix

The confusion matrix is used to compare the actual classes
with the classes predicted by the Naïve Bayes model.

It helps identify which classes are being correctly
classified and which classes are being confused with each other.

## Conclusion

The Gaussian Naïve Bayes classification algorithm was successfully
implemented on the AI Tools Usage & Student Productivity dataset.

The model classified students into five categories based on
their response to the effect of AI on productivity.

This practical demonstrates how supervised machine learning
can be used to classify student responses using numerical
features related to AI usage and academic performance.