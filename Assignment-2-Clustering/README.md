# Assignment 2 - K-Means Clustering

## Aim

To apply clustering techniques using unsupervised learning on an AI Tools Usage and Student Productivity dataset.

## Objective

The objective of this assignment is to group students into different clusters based on their AI tool usage and productivity-related characteristics.

## Dataset

The dataset used is the **AI Tools Usage & Student Productivity** survey dataset collected from students.

The dataset contains survey responses related to:

- AI tool usage
- AI usage hours
- Productivity
- Time saved using AI tools
- Academic performance

## Algorithm Used

### K-Means Clustering

K-Means is an unsupervised machine learning algorithm used to divide data into a predefined number of clusters.

In this assignment:

- Number of clusters = 3
- Random state = 42
- Number of records used = 953

## Features Used

The following features were used for clustering:

1. AI Usage Hours
2. Productivity
3. Time Saved
4. Academic Performance

## Implementation

The implementation was performed using Python.

### Libraries Used

- Pandas
- Scikit-learn
- Matplotlib
- OpenPyXL

### Main Steps

1. Load the Excel dataset using Pandas.
2. Select relevant features.
3. Convert categorical survey responses into numerical values.
4. Remove records containing missing values.
5. Standardize the selected features using StandardScaler.
6. Apply K-Means clustering.
7. Divide students into three clusters.
8. Display cluster counts and cluster centers.
9. Visualize the resulting clusters.

## Result

The K-Means algorithm successfully divided the valid student records into three clusters.

| Cluster | Number of Students |
|---------|--------------------:|
| Cluster 0 | 406 |
| Cluster 1 | 223 |
| Cluster 2 | 324 |
| **Total** | **953** |

## Conclusion

K-Means clustering was successfully applied to the AI Tools Usage & Student Productivity dataset. The algorithm grouped students based on similarities in AI usage, productivity, time saved, and academic performance.

This demonstrates how unsupervised learning can be used to identify different patterns of AI tool usage among students.