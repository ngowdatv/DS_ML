import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# category_encoders is needed for target encoding
try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders is not installed. Target Encoding will be skipped.")


def main():

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")

    if not os.path.exists(file_path):
        print("Error: train.csv not found!")
        return

    df = pd.read_csv(file_path)

    print(f"Loading dataset successfully. Rows:{df.shape[0]}, Features:{df.shape[1]}\n")

    # Handling missing values
    
    print("Handling missing data")
    print("Artificially creating some 'Hits' (H) data to demonstrate filling")

    df.loc[0:25, 'H'] = np.nan

    imputer = SimpleImputer(strategy='median')
    df[['H']] = imputer.fit_transform(df[['H']])

    print(f"Imputation complete. 'Hits'(H) now has {df['H'].isnull().sum()} missing values")

    # Log Transformation
    print("Evaluating the skewness of the Runs(R) distribution...")

    df['LogRuns'] = np.log1p(df['R'])

    print(f"Log Transformation applied. New skewness: {df['LogRuns'].skew():.2f}\n")

    # High Cardinality
    df['Team_ID'] = ['Team_' + str(np.random.randint(1, 150)) for _ in range(len(df))]

    if TargetEncoder is not None:
        print("Applying Target Encoder")
        encoder = TargetEncoder()
        df['Team_ID_Encoded'] = encoder.fit_transform(df['Team_ID'], df['W'])
    else:
        print("Category Encoders not installed.")

    # ---------------- Feature Selection ----------------

    features_to_test = ['R', 'HR', 'SO', 'SB']

    X_features = df[features_to_test].fillna(0)
    y_target = df['W']

    selector = SelectKBest(score_func=mutual_info_regression, k=2)

    selector.fit(X_features, y_target)

    winning_features = selector.get_support()

    best_features = X_features.columns[winning_features].tolist()

    print("Best Features:", best_features)

    #splitting data

    x=df[best_features]
    y=df['W']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    print(f"Training Data Size: {x_train.shape}")

    print(f"Testing Data Size: {x_test.shape}")

    #Training Model
    model=LinearRegression()
    model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    print(predictions)


if __name__ == "__main__":
    main()

    #Multicollinearity
