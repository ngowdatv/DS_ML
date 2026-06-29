import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os


# category_encoders is needed for target encoding
try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders is not installed. Target Encoding will be skipped.")


def main():

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic.csv")

    if not os.path.exists(file_path):
        print("Error: titanic.csv not found!")
        return

    df = pd.read_csv(file_path)

    print(f"Loading dataset successfully. Rows:{df.shape[0]}, Features:{df.shape[1]}\n")

    # Handling missing values

    print("Handling missing data")
    print("Filling missing values in Age, Fare and Sex")

    imputer = SimpleImputer(strategy='median')

    df[['Age']] = imputer.fit_transform(df[['Age']])
    df[['Fare']] = imputer.fit_transform(df[['Fare']])

    print(f"Imputation complete. 'Age' now has {df['Age'].isnull().sum()} missing values")

    # Log Transformation

    print("Evaluating the skewness of the Fare distribution...")

    df['LogFare'] = np.log1p(df['Fare'])

    print(f"Log Transformation applied. New skewness: {df['LogFare'].skew():.2f}\n")

    # High Cardinality

    if TargetEncoder is not None:
        print("Applying Target Encoder")
        encoder = TargetEncoder()
        df['Ticket_Encoded'] = encoder.fit_transform(df['Ticket'], df['Survived'])
    else:
        print("Category Encoders not installed.")

    # Encoding

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df[['Sex']] = imputer.fit_transform(df[['Sex']])

    # ---------------- Feature Selection ----------------

    features_to_test = ['Pclass', 'Sex', 'Age', 'Fare']

    X_features = df[features_to_test].fillna(0)
    y_target = df['Survived']

    selector = SelectKBest(score_func=mutual_info_classif, k=2)

    selector.fit(X_features, y_target)

    winning_features = selector.get_support()

    best_features = X_features.columns[winning_features].tolist()

    print("Best Features:", best_features)

    # splitting data

    x = df[best_features].fillna(0)
    y = df['Survived']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print(f"Training Data Size: {x_train.shape}")

    print(f"Testing Data Size: {x_test.shape}")

    # Training Model

    model = LogisticRegression(max_iter=1000)

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    # Comparing model predictions to the actual real-world values

    actual_survival = y_test.head(3).values
    predicted_survival = predictions[:3]

    for i in range(3):

        predicted = predicted_survival[i]
        actual = actual_survival[i]
        difference = abs(actual - predicted)

        print(f"Model Guessed: {predicted}")
        print(f"Real Answer: {actual}")
        print(f"Difference: {difference}")


if __name__ == "__main__":
    main()