import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Titanic EDA Project")

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "titanic.csv")

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully")
print(df.head())
print(df.info())
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Survival Count
plt.figure(figsize=(5,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Age Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=15, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap
corr = df.corr(numeric_only=True)

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Outliers in Age
plt.figure(figsize=(6,3))
sns.boxplot(x=df["Age"])
plt.title("Age Boxplot")
plt.show()

print("EDA Completed Successfully!")