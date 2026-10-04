# Assignment 10 - Fake News Detection from Multimodal Data

## Aim

To design and implement a machine learning model for detecting fake news using multimodal data consisting of text and image-related features.

## Objective

The objective of this practical is to combine information from different data modalities and classify news as either Real News or Fake News.

## Concept

Multimodal fake news detection uses more than one type of information to identify misleading content.

In this project:

- Text information is converted into numerical features using TF-IDF.
- Image-related information is represented using image features.
- Text and image features are combined using feature fusion.
- Logistic Regression is used for classification.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Pillow
- Matplotlib
- Machine Learning

## Libraries Used

```python
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix