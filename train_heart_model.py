import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, confusion_matrix

from imblearn.over_sampling import SMOTE


def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.dropna()
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['age_chol'] = df['age'] * df['chol']
    df['bp_chol'] = df['trestbps'] * df['chol']
    df['chol_norm'] = df['chol'] / df['chol'].max()
    df['bp_norm'] = df['trestbps'] / df['trestbps'].max()
    df['risk_score'] = (
        df['age']*0.2 +
        df['chol']*0.2 +
        df['trestbps']*0.2 +
        df['thalach']*0.2 +
        df['oldpeak']*0.2
    )
    return df


def build_and_train(data_path: Path, out_model: Path):
    print(f"Loading data from {data_path}")
    df = load_data(data_path)
    df = feature_engineer(df)

    X = df.drop(columns='target')
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Before SMOTE class distribution:", np.bincount(y_train))
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("After SMOTE class distribution:", np.bincount(y_train))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'model__n_estimators': [100],
        'model__max_depth': [5, 8, None]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
    print("Training model with GridSearchCV...")
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)

    print("Cross-validating on full dataset...")
    scores = cross_val_score(grid.best_estimator_, X, y, cv=5)
    print("CV scores:", scores)

    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Test precision:", precision_score(y_test, y_pred))
    print("Test ROC AUC:", roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    print(f"Saving model to {out_model}")
    joblib.dump(grid, out_model)


def main():
    base = Path(__file__).parent
    data_path = base / 'heart.csv'
    out_model = base / 'heart_model.pkl'

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    build_and_train(data_path, out_model)


if __name__ == '__main__':
    main()
