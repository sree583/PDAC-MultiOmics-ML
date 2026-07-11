import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Machine Learning Architectures from your diagram
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Ensure reproducibility across runs
np.random.seed(42)

# =====================================================================
# STEP 1: DATA INTEGRATION (Multi-Omics + Metabolomics)
# =====================================================================
def load_and_integrate_data(n_samples=600):
    print("[1/5] Simulating Multi-Omics and Metabolomics integration...")
    
    # Simulating 40 Transcriptomic/Genomic features
    omic_cols = [f"omic_feature_{i}" for i in range(1, 41)]
    df_omics = pd.DataFrame(np.random.normal(0, 1.2, size=(n_samples, 40)), columns=omic_cols)
    
    # Simulating 10 Serum Metabolomics features (Crucial for PDAC biomarker panels)
    metabolite_cols = [f"metabolite_{m}" for m in ["lipid_A", "lipid_B", "amino_acid_X", "amino_acid_Y", "m5", "m6", "m7", "m8", "m9", "m10"]]
    df_metabolites = pd.DataFrame(np.random.normal(0, 1.0, size=(n_samples, 10)), columns=metabolite_cols)
    
    # Horizontal integration across the same sample space
    X = pd.concat([df_omics, df_metabolites], axis=1)
    
    # Creating a synthetic ground truth label (0 = Control, 1 = Early Stage PDAC)
    # Introducing strong biological signals in specific features to simulate real biomarker patterns
    biomarker_signal = (X['metabolite_lipid_A'] * 2.2) + (X['metabolite_amino_acid_X'] * 1.8) - (X['omic_feature_12'] * 1.5)
    probabilities = 1 / (1 + np.exp(-biomarker_signal))
    y = np.where(probabilities > 0.5, 1, 0)
    
    return X, y

# =====================================================================
# STEP 2: CLINICAL PREPROCESSING & SPLITTING
# =====================================================================
def preprocess_pipeline(X, y):
    print("[2/5] Preprocessing datasets and executing stratified train-test splits...")
    
    # Stratified split ensures the ratio of PDAC cases remains uniform across sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    # Z-score normalization for genomic and metabolic intensity matrices
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reconvert to DataFrame to maintain feature name integrity for visualization
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_df, X_test_df, y_train, y_test

# =====================================================================
# STEP 3: TRAINING GRADIENT BOOSTING ARCHITECTURES
# =====================================================================
def train_models(X_train, y_train):
    print("[3/5] Initializing and training XGBoost & CatBoost classifiers...")
    
    # Model A: Optimized XGBoost
    xgb_clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    xgb_clf.fit(X_train, y_train)
    
    # Model B: CatBoost (Excellent handling of high-dimensional feature spaces)
    cat_clf = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.04,
        verbose=0, # Keeps console clear of iteration logs
        random_seed=42
    )
    cat_clf.fit(X_train, y_train)
    
    return xgb_clf, cat_clf

# =====================================================================
# STEP 4: CLINICAL DIAGNOSTIC EVALUATION
# =====================================================================
def evaluate_diagnostic_performance(model, X_test, y_test, label="Model"):
    print(f"\n==========================================")
    print(f" DIAGNOSTIC METRICS FOR: {label}")
    print(f"==========================================")
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix breakdown for Sensitivity/Specificity calculations
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    
    sensitivity = tp / (tp + fn)  # True Positive Rate (Crucial for not missing true cancer instances)
    specificity = tn / (tn + fp)  # True Negative Rate (Crucial to prevent false alarms)
    roc_auc = roc_auc_score(y_test, probabilities)
    
    print(f"ROC-AUC Score                  : {roc_auc:.4f}")
    print(f"Clinical Sensitivity (Recall)  : {sensitivity * 100:.2f}%")
    print(f"Clinical Specificity           : {specificity * 100:.2f}%")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))

# =====================================================================
# STEP 5: PIPELINE EXECUTION & FEATURE DISCOVERY VISUALIZATION
# =====================================================================
if __name__ == "__main__":
    # Execute Pipeline stages
    X, y = load_and_integrate_data()
    X_train, X_test, y_train, y_test = preprocess_pipeline(X, y)
    xgb_model, cat_model = train_models(X_train, y_train)
    
    # Evaluate Architectures
    evaluate_diagnostic_performance(xgb_model, X_test, y_test, "XGBoost (Optimized Framework)")
    evaluate_diagnostic_performance(cat_model, X_test, y_test, "CatBoost Framework")
    
    # Generate Feature Importance Visualization (Biomarker Discovery)
    print("\n[5/5] Extracting top features and saving diagnostic visualization...")
    importances = xgb_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1][:15] # Top 15 biomarkers
    
    plt.figure(figsize=(11, 7))
    sns.barplot(x=importances[sorted_indices], y=X_train.columns[sorted_indices], palette="plasma")
    plt.title("Identified Multi-Omics & Metabolomic Biomarkers Ranked by Prediction Impact (XGBoost)")
    plt.xlabel("Relative Relative Importance Weight")
    plt.ylabel("Biological/Metabolic Feature Marker")
    plt.tight_layout()
    
    # Save chart to disk to display in your repository
    plt.savefig("biomarker_importance_ranking.png", dpi=300)
    print("Pipeline finished successfully! Feature ranking visualization saved as 'biomarker_importance_ranking.png'.")
