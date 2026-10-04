import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Sample dataset
data = {
    "Study_Hours": [1, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 8],
    "Marks": [35, 40, 45, 50, 55, 60, 65, 70, 80, 90]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display dataset
print("Dataset:")
print(df)

# Calculate Pearson correlation
correlation, p_value = pearsonr(
    df["Study_Hours"],
    df["Marks"]
)

print("\nPearson Correlation Coefficient:",
      round(correlation, 3))

print("P-value:",
      round(p_value, 5))

# Interpret correlation
if correlation > 0:
    print("\nThere is a positive association between Study Hours and Marks.")
elif correlation < 0:
    print("\nThere is a negative association between Study Hours and Marks.")
else:
    print("\nThere is no linear association between Study Hours and Marks.")

# Create scatter plot with regression line
plt.figure(figsize=(8, 5))

sns.regplot(
    x="Study_Hours",
    y="Marks",
    data=df
)

plt.title("Association Between Study Hours and Marks")
plt.xlabel("Study Hours (Independent Variable)")
plt.ylabel("Marks (Dependent Variable)")
plt.grid(True)

plt.show()