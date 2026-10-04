import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

file_path  = "Assignment-2-Clustering/Student_Survey_AI_Tools_(Responses).xlsx"

df = pd.read_excel(file_path)

print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())


# ---------------------------------------
# 2. Select Features for Clustering
# ---------------------------------------

features = [
    "Approximately how many hours per week do you spend usiAi tools",
    "How would you rate your productivity After using AI tools",
    "How much time o AI tools save you during Your Academic work",
    "How would you rate your academic performance after using AI tools?"
]


# ---------------------------------------
# 3. Convert Categorical Values to Numbers
# ---------------------------------------

hours_mapping = {
    "Less than 2 hours": 1,
    "2-5 hours": 2,
    "5-10 hours": 3,
    "10-15 hours": 4,
    "More than 15 hours": 5
}

time_saved_mapping = {
    "Less than 1 hour": 1,
    "1-3 hours": 2,
    "3-5 hours": 3,
    "More than 5 hours": 4
}


df["Hours_Numeric"] = df[
    "Approximately how many hours per week do you spend usiAi tools"
].map(hours_mapping)

df["Time_Saved_Numeric"] = df[
    "How much time o AI tools save you during Your Academic work"
].map(time_saved_mapping)


# ---------------------------------------
# 4. Prepare Clustering Data
# ---------------------------------------

X = pd.DataFrame({
    "AI_Usage_Hours": df["Hours_Numeric"],
    "Productivity": df[
        "How would you rate your productivity After using AI tools"
    ],
    "Time_Saved": df["Time_Saved_Numeric"],
    "Academic_Performance": df[
        "How would you rate your academic performance after using AI tools?"
    ]
})


# Remove missing values

X = X.dropna()

print("\nClustering Data:")
print(X.head())

print("\nNumber of records used:", len(X))


# ---------------------------------------
# 5. Standardize Features
# ---------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ---------------------------------------
# 6. Apply K-Means Clustering
# ---------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)


# ---------------------------------------
# 7. Display Cluster Results
# ---------------------------------------

X["Cluster"] = clusters

print("\nCluster Counts:")
print(X["Cluster"].value_counts().sort_index())

print("\nCluster Centers:")
print(kmeans.cluster_centers_)


# ---------------------------------------
# 8. Visualize Clusters
# ---------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X["AI_Usage_Hours"],
    X["Productivity"],
    c=X["Cluster"],
    s=50
)

plt.xlabel("AI Tool Usage Hours")
plt.ylabel("Productivity Rating")
plt.title("K-Means Clustering of AI Tools Usage and Student Productivity")

plt.grid(True)

plt.show()