
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.svm import SVC
# from sklearn.metrics import (
#     accuracy_score, classification_report, confusion_matrix,
#     precision_score, recall_score, f1_score,
#     mean_squared_error, mean_absolute_error, r2_score,
# )
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from xgboost import XGBClassifier, XGBRegressor


# # ---------------------------------------------------------------------------
# # Target creation & data preparation
# # ---------------------------------------------------------------------------

# def prepare_targets(df):
#     """
#     Create classification (up/down) and regression (next Close price) targets.
#     Returns X, y_clf, y_reg.
#     """
#     # Assuming 'Pct Change' already exists from eda.py
#     df['Target'] = np.where(df['Pct Change'] > 0, 1, 0)
    
#     y_classification = df['Target'].copy()
#     y_regression     = df['Close'].copy()          # predicting next day's Close
    
#     # Drop targets + any leakage-risk columns
#     drop_cols = ['Target', 'Pct Change', 'Shock Event', 'Close']
#     X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
#     return X, y_classification, y_regression


# def split_scale_pca_shared(X, y_clf, y_reg, test_size=0.2, random_state=42, pca_variance=0.95):
#     """
#     Split → scale → fit ONE PCA on train → transform both train & test.
#     Returns dictionary with consistent transformed splits for both tasks.
#     """
#     # Stratified split for classification (helps balance)
#     X_train, X_test, y_train_clf, y_test_clf = train_test_split(
#         X, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
#     )
    
#     # Regression uses same split indices
#     y_train_reg = y_reg.loc[y_train_clf.index]
#     y_test_reg  = y_reg.loc[y_test_clf.index]

#     # Scale (fit on train only)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled  = scaler.transform(X_test)

#     # One shared PCA (fit on training data only)
#     pca = PCA(n_components=pca_variance)
#     X_train_pca = pca.fit_transform(X_train_scaled)
#     X_test_pca  = pca.transform(X_test_scaled)

#     # Save artifacts
#     joblib.dump(scaler, "scaler.pkl")
#     joblib.dump(pca,    "pca.pkl")

#     return {
#         "X_train": X_train, "X_test": X_test,
#         "y_train_clf": y_train_clf, "y_test_clf": y_test_clf,
#         "y_train_reg": y_train_reg, "y_test_reg": y_test_reg,
#         "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
#         "X_train_pca": X_train_pca,       "X_test_pca": X_test_pca,
#         "pca": pca, "scaler": scaler,     # for later use / inspection
#     }


# # ---------------------------------------------------------------------------
# # Classification helpers
# # ---------------------------------------------------------------------------

# def plot_confusion_matrix(cm, model_name):
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
#     plt.title(f"{model_name} Confusion Matrix")
#     plt.ylabel('True')
#     plt.xlabel('Predicted')
#     plt.show()


# def train_classifiers(X_train, y_train):
#     """Train three classifiers."""
#     classifiers = {
#         "Random Forest": RandomForestClassifier(
#             n_estimators=300, max_depth=20,
#             min_samples_leaf=4, min_samples_split=2, random_state=42),
#         "XGBoost": XGBClassifier(
#             learning_rate=0.01, max_depth=3,
#             n_estimators=200, random_state=42, eval_metric='logloss'),
#         "SVM": SVC(C=10, kernel='linear', probability=True, random_state=42),
#     }
    
#     trained = {}
#     for name, model in classifiers.items():
#         model.fit(X_train, y_train)
#         trained[name] = model
#     return trained


# def evaluate_classifiers(models, X_test, y_test):
#     """Evaluate classification models (accuracy = directional accuracy)."""
#     for name, model in models.items():
#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)
        
#         print(f"\n{name} Classification Evaluation:")
#         print(f"  Directional Accuracy: {acc:.4f} ({acc*100:.2f}%)")
#         print(f"  Precision (Up):       {precision_score(y_test, y_pred):.4f}")
#         print(f"  Recall (Up):          {recall_score(y_test, y_pred):.4f}")
#         print(f"  F1 Score:             {f1_score(y_test, y_pred):.4f}")
#         print(classification_report(y_test, y_pred, digits=4))
        
#         plot_confusion_matrix(confusion_matrix(y_test, y_pred), name)


# # ---------------------------------------------------------------------------
# # Regression helpers
# # ---------------------------------------------------------------------------

# def train_rf_regressor(X_train, y_train):
#     return RandomForestRegressor(
#         n_estimators=200, max_depth=10, max_features=None,
#         min_samples_leaf=2, min_samples_split=2, random_state=42
#     ).fit(X_train, y_train)


# def train_xgb_regressor(X_train, y_train):
#     return XGBRegressor(
#         colsample_bytree=1.0, learning_rate=0.05, max_depth=5,
#         n_estimators=200, subsample=0.8, random_state=42
#     ).fit(X_train, y_train)


# def evaluate_regression(model, X_test, y_test, model_name):
#     y_pred = model.predict(X_test)
#     mse  = mean_squared_error(y_test, y_pred)
#     mae  = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2   = r2_score(y_test, y_pred)
    
#     # Directional accuracy (very useful for market prediction)
#     dir_acc = (np.sign(y_pred) == np.sign(y_test)).mean()
    
#     print(f"\n{model_name} Regression Evaluation:")
#     print(f"  MSE          : {mse:.6f}")
#     print(f"  MAE          : {mae:.6f}")
#     print(f"  RMSE         : {rmse:.6f}")
#     print(f"  R²           : {r2:.4f}")
#     print(f"  Directional Acc: {dir_acc:.4f} ({dir_acc*100:.2f}%)")
    
#     return y_pred


# # ---------------------------------------------------------------------------
# # Optional tuning (you can call these separately)
# # ---------------------------------------------------------------------------

# def tune_regressors(X_train, y_train):
#     # same as before - you can keep or remove
#     pass  # ← implement if needed


# def tune_classifiers(X_train, y_train):
#     # same as before
#     pass  # ← implement if needed


# # ---------------------------------------------------------------------------
# # Main pipeline entry point
# # ---------------------------------------------------------------------------

# def run_model_pipeline(df):
#     """
#     End-to-end pipeline:
#       - Prepare targets
#       - Split + scale + shared PCA
#       - Train & evaluate classifiers (directional focus)
#       - Train & evaluate regressors (with directional accuracy)
#     """
#     print("Preparing targets...")
#     X, y_clf, y_reg = prepare_targets(df)
    
#     print("Splitting, scaling, and applying shared PCA...")
#     splits = split_scale_pca_shared(X, y_clf, y_reg)
    
#     print(f"PCA components retained: {splits['pca'].n_components_} "
#           f"({splits['pca'].explained_variance_ratio_.sum():.3%} variance)")
    
#     # ─── Classification ────────────────────────────────────────
#     print("\n=== Training & Evaluating Classifiers ===")
#     classifiers = train_classifiers(splits["X_train_pca"], splits["y_train_clf"])
#     evaluate_classifiers(classifiers, splits["X_test_pca"], splits["y_test_clf"])
    
#     # ─── Regression ────────────────────────────────────────────
#     print("\n=== Training & Evaluating Regressors ===")
#     rf_reg  = train_rf_regressor(splits["X_train_pca"], splits["y_train_reg"])
#     evaluate_regression(rf_reg, splits["X_test_pca"], splits["y_test_reg"], "Random Forest Regressor")
    
#     xgb_reg = train_xgb_regressor(splits["X_train_pca"], splits["y_train_reg"])
#     evaluate_regression(xgb_reg, splits["X_test_pca"], splits["y_test_reg"], "XGBoost Regressor")
    
#     return classifiers, rf_reg, xgb_reg, splits


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Target creation & data preparation
# ---------------------------------------------------------------------------

def prepare_targets(df):
    """
    Create classification (up/down) and regression targets.
    Returns X, y_clf, y_reg.
    """
    # Assuming 'Pct Change' already exists from eda.py
    df['Target'] = np.where(df['Pct Change'] > 0, 1, 0)
    
    y_classification = df['Target'].copy()
    y_regression     = df['Close'].copy()          # predicting next day's Close
    
    # Drop targets + any leakage-risk columns
    drop_cols = ['Target', 'Pct Change', 'Shock Event', 'Close']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    return X, y_classification, y_regression


def split_scale_pca_shared(X, y_clf, y_reg, test_size=0.2, random_state=42, pca_variance=0.95):
    """
    Split → scale → fit ONE PCA on train → transform both train & test.
    """
    # Stratified split for classification
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )
    
    y_train_reg = y_reg.loc[y_train_clf.index]
    y_test_reg  = y_reg.loc[y_test_clf.index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=pca_variance)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca,    "pca.pkl")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train_clf": y_train_clf, "y_test_clf": y_test_clf,
        "y_train_reg": y_train_reg, "y_test_reg": y_test_reg,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "X_train_pca": X_train_pca,       "X_test_pca": X_test_pca,
        "pca": pca, "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


def train_classifiers(X_train, y_train):
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20,
            min_samples_leaf=4, min_samples_split=2, random_state=42),
        "XGBoost": XGBClassifier(
            learning_rate=0.01, max_depth=3,
            n_estimators=200, random_state=42, eval_metric='logloss'),
        "SVM": SVC(C=10, kernel='linear', probability=True, random_state=42),
    }
    
    trained = {}
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_classifiers(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n{name} Classification Evaluation:")
        print(f"  Directional Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision (Up):       {precision_score(y_test, y_pred):.4f}")
        print(f"  Recall (Up):          {recall_score(y_test, y_pred):.4f}")
        print(f"  F1 Score:             {f1_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, digits=4))
        
        plot_confusion_matrix(confusion_matrix(y_test, y_pred), name)


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def train_rf_regressor(X_train, y_train):
    return RandomForestRegressor(
        n_estimators=200, max_depth=10, max_features=None,
        min_samples_leaf=2, min_samples_split=2, random_state=42
    ).fit(X_train, y_train)


def train_xgb_regressor(X_train, y_train):
    return XGBRegressor(
        colsample_bytree=1.0, learning_rate=0.05, max_depth=5,
        n_estimators=200, subsample=0.8, random_state=42
    ).fit(X_train, y_train)


def evaluate_regression(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    
    dir_acc = (np.sign(y_pred) == np.sign(y_test)).mean()
    
    print(f"\n{model_name} Regression Evaluation:")
    print(f"  MSE            : {mse:.6f}")
    print(f"  MAE            : {mae:.6f}")
    print(f"  RMSE           : {rmse:.6f}")
    print(f"  R²             : {r2:.4f}")
    print(f"  Directional Acc: {dir_acc:.4f} ({dir_acc*100:.2f}%)")
    
    return y_pred

# ---------------------------------------------------------------------------
# Hyperparameter tuning – REAL implementations
# ---------------------------------------------------------------------------

def tune_regressors(X_train, y_train):
    """
    Tune RandomForestRegressor and XGBRegressor using GridSearchCV.
    Uses TimeSeriesSplit to respect temporal order.
    """
    from sklearn.model_selection import TimeSeriesSplit

    print("Tuning regressors...")
    tscv = TimeSeriesSplit(n_splits=5)

    # ─── Random Forest ────────────────────────────────────────
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf_gs = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=rf_param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    rf_gs.fit(X_train, y_train)
    
    print("\nBest RandomForestRegressor params:", rf_gs.best_params_)
    print("Best CV neg MSE:", rf_gs.best_score_)

    # ─── XGBoost ──────────────────────────────────────────────
    # Manual search avoids sklearn/xgboost tag compatibility issues in GridSearchCV.
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    def _slice(data, idx):
        return data.iloc[idx] if hasattr(data, 'iloc') else data[idx]

    best_xgb_params = None
    best_xgb_score = -np.inf

    for params in ParameterGrid(xgb_param_grid):
        fold_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr = _slice(X_train, train_idx)
            X_val = _slice(X_train, val_idx)
            y_tr = _slice(y_train, train_idx)
            y_val = _slice(y_train, val_idx)

            model = XGBRegressor(random_state=42, **params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_scores.append(-mean_squared_error(y_val, preds))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_xgb_score:
            best_xgb_score = mean_score
            best_xgb_params = params

    best_xgb = XGBRegressor(random_state=42, **best_xgb_params)
    best_xgb.fit(X_train, y_train)

    print("\nBest XGBRegressor params:", best_xgb_params)
    print("Best CV neg MSE:", best_xgb_score)

    return rf_gs.best_estimator_, best_xgb


def tune_classifiers(X_train, y_train):
    """
    Tune classifiers using GridSearchCV + TimeSeriesSplit.
    Scoring = 'f1' (good default for directional trading)
    """
    from sklearn.model_selection import TimeSeriesSplit

    print("Tuning classifiers...")
    tscv = TimeSeriesSplit(n_splits=5)

    # ─── Random Forest Classifier ─────────────────────────────
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_gs = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    rf_gs.fit(X_train, y_train)
    print("\nBest RF Classifier params:", rf_gs.best_params_)

    # ─── XGBoost Classifier ───────────────────────────────────
    # Manual search avoids sklearn/xgboost tag compatibility issues in GridSearchCV.
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1]
    }

    def _slice(data, idx):
        return data.iloc[idx] if hasattr(data, 'iloc') else data[idx]

    best_xgb_params = None
    best_xgb_score = -np.inf

    for params in ParameterGrid(xgb_param_grid):
        fold_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr = _slice(X_train, train_idx)
            X_val = _slice(X_train, val_idx)
            y_tr = _slice(y_train, train_idx)
            y_val = _slice(y_train, val_idx)

            model = XGBClassifier(random_state=42, eval_metric='logloss', **params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_scores.append(f1_score(y_val, preds))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_xgb_score:
            best_xgb_score = mean_score
            best_xgb_params = params

    best_xgb = XGBClassifier(random_state=42, eval_metric='logloss', **best_xgb_params)
    best_xgb.fit(X_train, y_train)
    print("Best XGBoost Classifier params:", best_xgb_params)

    # ─── SVM ──────────────────────────────────────────────────
    svm_param_grid = {
        'C': [1, 10],
        'kernel': ['linear', 'rbf']
    }
    
    svm_gs = GridSearchCV(
        estimator=SVC(random_state=42),
        param_grid=svm_param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    svm_gs.fit(X_train, y_train)
    print("Best SVM params:", svm_gs.best_params_)

    return rf_gs.best_estimator_, best_xgb, svm_gs.best_estimator_

# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_model_pipeline(df):
    print("Preparing targets...")
    X, y_clf, y_reg = prepare_targets(df)
    
    print("Splitting, scaling, and applying shared PCA...")
    splits = split_scale_pca_shared(X, y_clf, y_reg)
    
    print(f"PCA components retained: {splits['pca'].n_components_} "
          f"({splits['pca'].explained_variance_ratio_.sum():.3%} variance)")
    
    print("\n=== Training & Evaluating Classifiers ===")
    classifiers = train_classifiers(splits["X_train_pca"], splits["y_train_clf"])
    evaluate_classifiers(classifiers, splits["X_test_pca"], splits["y_test_clf"])
    
    print("\n=== Training & Evaluating Regressors ===")
    rf_reg  = train_rf_regressor(splits["X_train_pca"], splits["y_train_reg"])
    evaluate_regression(rf_reg, splits["X_test_pca"], splits["y_test_reg"], "Random Forest Regressor")
    
    xgb_reg = train_xgb_regressor(splits["X_train_pca"], splits["y_train_reg"])
    evaluate_regression(xgb_reg, splits["X_test_pca"], splits["y_test_reg"], "XGBoost Regressor")
    
    return classifiers, rf_reg, xgb_reg, splits