import pandas as pd
import shap
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score
)


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


print("Accuracy Score:", accuracy_score(y_test, lgbm_pred))
print("\nClassification Report\n", classification_report(y_test, lgbm_pred))


precision = precision_score(y_test, lgbm_pred)
recall = recall_score(y_test, lgbm_pred)
f1 = f1_score(y_test, lgbm_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


print("\nConfusion Matrix")
print(confusion_matrix(y_test, lgbm_pred))


comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost",
        "LightGBM"
    ],
    "Accuracy": [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred),
        accuracy_score(y_test, lgbm_pred)
    ]
})

print("\nComparison of Models")
print(comparison)


cm = confusion_matrix(
    y_test,
    lgbm_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)


lgbm_prob = gbm.predict_proba(x_test)[:, 1]

roc_score = roc_auc_score(
    y_test,
    lgbm_prob
)

print("ROC AUC:", roc_score)


fpr, tpr, thresholds = roc_curve(y_test, lgbm_prob)


plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_score:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
#1.Baseline Model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier(random_state=42)
model.fit(x_train, y_train)
base_pred = model.predict(x_test)
print("Baseline Accuracy:", accuracy_score(y_test, base_pred))

#2.Grid GridSearchCv
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

grid = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=3
)

grid.fit(x_train, y_train)

print("Best Parameters")
print(grid.best_params_)

best_model = grid.best_estimator_

prediction = best_model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, prediction))

#random Search (simple)

random_params={
    'n_estimators' : [50,100,200,300],
    'max_depth' : [3,5,7],
    'learning_rate' : [0.01, 0.1, 0.2]
}

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_distributions=random_params,
    n_iter=10,
    cv=3,
    random_state=42,

)

random_search.fit(x_train, y_train)
print(random_search.best_params_)
best_random_model = random_search.best_estimator_
random_prediction = best_random_model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, random_prediction))

def objective(trial):
    model = XGBClassifier(
             n_estimators=trial.suggest_int("n_estimators",50,300),
             max_depth=trial.suggest_int("max_depth",3,7),
             learning_rate=trial.suggest_float("learning_rate",0.01,0.2)
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return accuracy_score(y_test, predictions)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print(study.best_params)

best_model = XGBClassifier(
    **study.best_params,
    random_state=42
)

best_model.fit(x_train, y_train)

optuna_prediction = best_model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, optuna_prediction))

ros = RandomOverSampler(random_state=42)
x_resampled, y_resampled = ros.fit_resample(x_train, y_train)

rus = RandomUnderSampler(random_state=42)
x_undersampled, y_undersampled = rus.fit_resample(x_train, y_train)

smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x_train, y_train)

print("\nAfter Random Oversampling")
print(y_resampled.value_counts())

print("\nAfter Random Undersampling")
print(y_undersampled.value_counts())

print("\nAfter SMOTE")
print(y_smote.value_counts())

prob = model.predict_proba(x_test)[:, 1]
threshold=0.40
predictions=(prob>=threshold).astype(int)

for threshold in [0.5,0.4,0.2]:
    predictions=(prob>=threshold).astype(int)
    print(precision_score(y_test,predictions))

explainer=shap.TreeExplainer(best_model)
shap_values=explainer.shap_values(x_test)

shap.summary_plot(
    shap_values,
    x_test,
    feature_names=x.columns
)

# ==========================
# SHAP Feature Importance Plot (Horizontal Bar Plot)
# ==========================

import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(x_test)

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    x_test,
    feature_names=x_test.columns,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()