import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

file_path = "Assignment-2-Clustering/Student_Survey_AI_Tools_(Responses).xlsx"

df = pd.read_excel(file_path)

print("Original Dataset Shape:", df.shape)


# --------------------------------------------------
# 2. Select Features for PCA
# --------------------------------------------------

features = pd.DataFrame()

# AI usage hours per week
hours_map = {
    "Less than 2 hours": 1,
    "2-5 hours": 2,
    "5-10 hours": 3,
    "10-15 hours": 4,
    "More than 15 hours": 5
}

features["AI_Usage_Hours"] = (
    df["Approximately how many hours per week do you spend usiAi tools"]
    .map(hours_map)
)

# Percentage of AI usage related to academics
academic_usage_map = {
    "0-25%": 1,
    "25-50%": 2,
    "50-75%": 3,
    "75-100%": 4
}

features["Academic_Usage"] = (
    df["How much  your AI sage is related to Academics "]
    .map(academic_usage_map)
)

# Productivity rating after using AI
features["Productivity_After"] = pd.to_numeric(
    df["How would you rate your productivity After using AI tools"],
    errors="coerce"
)

# Time saved using AI
time_saved_map = {
    "Less than 1 hour": 1,
    "1-3 hours": 2,
    "3-5 hours": 3,
    "More than 5 hours": 4
}

features["Time_Saved"] = (
    df["How much time o AI tools save you during Your Academic work"]
    .map(time_saved_map)
)

# Academic performance after using AI
features["Academic_Performance"] = pd.to_numeric(
    df["How would you rate your academic performance after using AI tools?"],
    errors="coerce"
)

# Understanding academic concepts
understanding_map = {
    "Not at all": 1,
    "Slightly": 2,
    "Moderately": 3,
    "Very much": 4,
    "Extremely": 5
}

features["Understanding"] = (
    df["How much do Ai ools help you understand different academic concepts "]
    .map(understanding_map)
)

# AI dependency
dependency_map = {
    "Not dependent": 1,
    "Slightly dependent": 2,
    "Moderately dependent": 3,
    "Highly dependent": 4,
    "Completely dependent": 5
}

features["AI_Dependence"] = (
    df["How Dependent do you feelon  tools for completing your academic work"]
    .map(dependency_map)
)

# Trust in AI information
trust_map = {
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Complete trust": 4
}

features["AI_Trust"] = (
    df["How much do you trust the information provided by AI tools"]
    .map(trust_map)
)

# Verification frequency
verification_map = {
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4
}

features["Verification"] = (
    df["How often do you verify AI generated information before using it?"]
    .map(verification_map)
)

# Overall productivity impact
overall_productivity_map = {
    "Greatly decreased": 1,
    "Slightly decreased": 2,
    "No change": 3,
    "Slightly improved": 4,
    "Greatly improved": 5
}

features["Overall_Productivity"] = (
    df["Overall,how has Ai affected your Academic productivity?"]
    .map(overall_productivity_map)
)


# --------------------------------------------------
# 3. Remove Missing Values
# --------------------------------------------------

features = features.dropna()

print("Records Used for PCA:", len(features))

print("\nSelected Features:")
print(features.columns.tolist())


# --------------------------------------------------
# 4. Standardize the Data
# --------------------------------------------------

scaler = StandardScaler()

scaled_data = scaler.fit_transform(features)


# --------------------------------------------------
# 5. Apply PCA
# --------------------------------------------------

pca = PCA(n_components=2)

principal_components = pca.fit_transform(scaled_data)


# --------------------------------------------------
# 6. Create PCA DataFrame
# --------------------------------------------------

pca_df = pd.DataFrame(
    principal_components,
    columns=["Principal Component 1", "Principal Component 2"]
)

print("\nPCA Result:")
print(pca_df.head())


# --------------------------------------------------
# 7. Explained Variance
# --------------------------------------------------

explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:")
print(explained_variance)

print("\nTotal Variance Explained:",
      explained_variance.sum())


# --------------------------------------------------
# 8. Visualize PCA Result
# --------------------------------------------------

plt.figure(figsize=(10, 7))

plt.scatter(
    pca_df["Principal Component 1"],
    pca_df["Principal Component 2"],
    s=50,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "PCA - AI Tools Usage & Student Productivity"
)

plt.grid(True)

plt.show()