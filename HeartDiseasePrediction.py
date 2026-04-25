import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

heart_data = pd.read_csv("heart.csv")

print("\n INITIAL DATA ")
print(heart_data.head())
print("\nShape:", heart_data.shape)
print("\nInfo:")
print(heart_data.info())
print("\nMissing Values:\n", heart_data.isnull().sum())
print("\nDescribe:\n", heart_data.describe())

heart_data = heart_data.dropna()

print("\nAFTER CLEANING")
print("Shape:", heart_data.shape)

print("\nBEFORE FEATURE ENGINEERING")
print(heart_data.head())

heart_data['age_chol'] = heart_data['age'] * heart_data['chol']
heart_data['bp_chol'] = heart_data['trestbps'] * heart_data['chol']

heart_data['chol_norm'] = heart_data['chol'] / heart_data['chol'].max()
heart_data['bp_norm'] = heart_data['trestbps'] / heart_data['trestbps'].max()

heart_data['risk_score'] = (
    heart_data['age']*0.2 +
    heart_data['chol']*0.2 +
    heart_data['trestbps']*0.2 +
    heart_data['thalach']*0.2 +
    heart_data['oldpeak']*0.2
)

print("\nAFTER FEATURE ENGINEERING")
print(heart_data.head())
# Pie Chart
plt.figure()
heart_data['target'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Target Distribution")
plt.show()

sns.countplot(x='target', data=heart_data)
plt.show()

heart_data.hist(figsize=(15,10))
plt.suptitle("Feature Distributions")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(heart_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(heart_data[['age','chol','trestbps','thalach','target']], hue='target')
plt.show()

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print("\nFEATURES")
print(X.head())

print("\nTARGET")
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

#Handle IMbalaCe
print("\nBefore SMOTE:", np.bincount(Y_train))

smote = SMOTE()
X_train, Y_train = smote.fit_resample(X_train, Y_train)

print("After SMOTE:", np.bincount(Y_train))

#Model Trainnig
models = {
    "Logistic": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ]),
    "RF": Pipeline([
        ('model', RandomForestClassifier())
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(probability=True))
    ]),
    "XGB": Pipeline([
        ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ]),
    "LGBM": Pipeline([
        ('model', LGBMClassifier())
    ])
}

params = {
    "Logistic": {'model__C':[0.1,1]},
    "RF": {'model__n_estimators':[100], 'model__max_depth':[5]},
    "SVM": {'model__C':[1], 'model__kernel':['rbf']},
    "XGB": {'model__n_estimators':[100]},
    "LGBM": {'model__n_estimators':[100]}
}

grids = {}

for name in models:
    print(f"\nTraining {name}...")
    grid = GridSearchCV(models[name], params[name], cv=3)
    grid.fit(X_train, Y_train)
    grids[name] = grid

#Cross Validation
print("\nCROSS VALIDATION")
for name, grid in grids.items():
    scores = cross_val_score(grid.best_estimator_, X, Y, cv=5)
    print(name, ":", scores, "Mean:", scores.mean())

#Evaluation Result + ROC
results = {}

plt.figure()

for name, grid in grids.items():
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:,1]

    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    auc = roc_auc_score(Y_test, y_prob)

    results[name] = acc

    fpr, tpr, _ = roc_curve(Y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} AUC={auc:.2f}")

    print(f"\n{name} Accuracy:", acc)

plt.legend()
plt.title("ROC Curve")
plt.show()

#Confusuion  MAtrix
best_name = max(results, key=results.get)
best_model = grids[best_name]

print("\nBest Model:", best_name)

cm = confusion_matrix(Y_test, best_model.predict(X_test))

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

#Feature Engineering
if best_name in ["RF","XGB","LGBM"]:
    model = best_model.best_estimator_.named_steps['model']
    imp = model.feature_importances_

    feat_imp = pd.Series(imp, index=X.columns).sort_values(ascending=False)

    print("\nTop Features:\n", feat_imp.head())

    sns.barplot(x=feat_imp, y=feat_imp.index)
    plt.title("Feature Importance")
    plt.show()

#Save Model
joblib.dump(best_model, "heart_model.pkl")
print("Model Saved!")