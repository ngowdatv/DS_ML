import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer(as_frame=True)
df = data.frame

x = df.drop("target", axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)

gbm = LGBMClassifier()
gbm.fit(x_train, y_train)
lgbm_pred = gbm.predict(x_test)

print("accuracy score:", accuracy_score(y_test, lgbm_pred))
print("\nclassification_report\n", classification_report(y_test, lgbm_pred))

comparison = pd.DataFrame({
    "model": [
        "Random Forest",
        "XGboost",
        "LightGBM"
    ],
    "Accuracy": [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred),
        accuracy_score(y_test, lgbm_pred)
    ]
})

print("Comparison of Models")
print(comparison)


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(
    r"C:\Users\Student\Downloads\customer_churn_dataset-testing-master.csv (1)\customer_churn_dataset-testing-master.csv"
)

df = pd.get_dummies(df, drop_first=True)

x = df.drop("Churn", axis=1)
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)


xgb = XGBClassifier(eval_metric="logloss")
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)


gbm = LGBMClassifier()
gbm.fit(x_train, y_train)
lgbm_pred = gbm.predict(x_test)


print("accuracy score:", accuracy_score(y_test, lgbm_pred))
print("\nclassification_report\n", classification_report(y_test, lgbm_pred))


comparison = pd.DataFrame({
    "model": [
        "Random Forest",
        "XGboost",
        "LightGBM"
    ],
    "Accuracy": [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred),
        accuracy_score(y_test, lgbm_pred)
    ]
})


print("Comparison of Models")
print(comparison)