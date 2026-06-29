import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns 
import os 

print("Understanding Dataset")

script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(script_dir, 'sales_data.csv')

if not os.path.exists(file_name):
    print(f"Error: {file_name} is not found")
    exit()

df = pd.read_csv(file_name)

print("successfully loaded")
print(f"shape of the dataset: Rows: {df.shape[0]}, columns: {df.shape[1]}")

print(df.head())
print(df.tail())
print(df.describe())

print("Handling Missing Values")

print(df.isnull().sum())

median_spending = df['Spending'].median()
df['Spending'] = df['Spending'].fillna(median_spending)

print("Median Spending:", median_spending)

plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Spending')
plt.xlabel('Spending amount')
plt.ylabel('Number of records')
plt.show()

correlation=df.corr(numeric_only=True)
print(correlation)

print("plotting correlation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
sns.boxplot(x=df['Age'],color='lightgreen')
plt.title("Boxplot of customer Age")
plt.xlabel("Age")
plt.show()

print("Find the outliers in age")
outliers=df[df['Age']>100]
print("Found outliers(s):")
print(outliers)
