import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# 2. Basic Data Exploration
print("--- First Five Rows ---")
print(df.head())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Correlation matrix
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# 4. Scatter plot: Sepal Length vs Petal Length
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="target",
    palette="viridis"
)

plt.title("Sepal Length vs Petal Length by Class")
plt.show()

# 5. Histogram: Distribution of Sepal Length
plt.figure(figsize=(7, 5))
sns.histplot(df["sepal length (cm)"], kde=True, color="blue")
plt.title("Distribution of Sepal Length")
plt.show()

# Assignment

from sklearn.datasets import load_wine
import numpy as np

# 1. Load and perform complete EDA on Wine dataset
wine = load_wine()

wine_df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

wine_df["target"] = wine.target

print("\n\n========== WINE DATASET EDA ==========")

print("\n--- First Five Rows ---")
print(wine_df.head())

print("\n--- Dataset Information ---")
print(wine_df.info())

print("\n--- Statistical Summary ---")
print(wine_df.describe())

print("\n--- Missing Values ---")
print(wine_df.isnull().sum())

print("\n--- Correlation Matrix ---")
print(wine_df.corr(numeric_only=True))


# 2. Boxplots for all numerical attributes
plt.figure(figsize=(15, 8))

wine_df.drop("target", axis=1).boxplot()

plt.title("Boxplots of Wine Dataset Features")
plt.xticks(rotation=45)
plt.ylabel("Values")
plt.show()


# 3. Correlation Heatmap
plt.figure(figsize=(12, 9))

correlation = wine_df.drop("target", axis=1).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Wine Dataset")
plt.show()



corr_pairs = correlation.where(
    ~np.eye(correlation.shape[0], dtype=bool)
).stack()

strongest_positive = corr_pairs[
    corr_pairs > 0
].sort_values(ascending=False).head(1)

print("\n--- Strongest Positive Correlation ---")
print(strongest_positive)



print("\n========== CONCLUSION ==========")
print("Wine dataset was successfully loaded and analyzed.")
print("EDA, statistical summary, missing value analysis,")
print("boxplot visualization and correlation heatmap were performed.")
print("The strongest positive correlation was also identified.")

