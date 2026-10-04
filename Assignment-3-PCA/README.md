# Assignment 3 - Dimensionality Reduction using PCA

## Aim

To apply Principal Component Analysis (PCA) for dimensionality reduction on the **AI Tools Usage & Student Productivity** dataset.

## Objective

The objective of this assignment is to reduce the number of dimensions in the dataset while preserving as much useful information as possible.

PCA transforms multiple related features into a smaller number of new features called **Principal Components**.

## Dataset

The dataset used in this assignment is:

**Student_Survey_AI_Tools_(Responses).xlsx**

The dataset contains survey responses related to:

- AI tool usage by students
- Academic usage of AI tools
- Productivity
- Time saved
- Academic performance
- Understanding of academic concepts
- AI dependency
- Trust in AI-generated information
- Verification of AI-generated information
- Overall productivity impact

The same dataset is used throughout the AIML assignments.

## Features Used

The following features were selected for PCA:

1. AI Usage Hours
2. Academic Usage
3. Productivity After Using AI
4. Time Saved
5. Academic Performance
6. Understanding
7. AI Dependence
8. AI Trust
9. Verification
10. Overall Productivity

Categorical survey responses were converted into numerical values before applying PCA.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Principal Component Analysis (PCA)
- StandardScaler

## Methodology

### 1. Load Dataset

The Excel dataset was loaded using Pandas.

```python
df = pd.read_excel(file_path)