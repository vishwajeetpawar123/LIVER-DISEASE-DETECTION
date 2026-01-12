from data_processor import LiverDataProcessor
from model_engine import StackingEnsemble
from xai_engine import XAIEngine
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import joblib

def main():
    print("==================================================")
    print("   LIVER DISEASE PREDICTION: MULTI-CLASS ENGINE   ")
    print("==================================================")
    
    # 1. Setup Data Paths
    dataset_path = os.path.join("DATASET", "indian_liver_patient.csv")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    # 2. Data Processing Pipeline
    print("\n[PHASE 1] Data Processing & Label Engineering")
    processor = LiverDataProcessor(dataset_path)
    X, y = processor.get_processed_data()
    
    # Save processed metadata and preprocessors
    feature_names = X.columns.tolist()
    processor.save_preprocessors() # SAVES preprocessor.pkl
    
    print(f"\nFinal Dataset Shape: {X.shape}")
    print(f"Target Classes: {np.unique(y)}")
    
    # 3. Model Training (Stratified Split 80/20)
    print("\n[PHASE 2] Model Training & Stacking")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Just take one fold split for Train/Test simulation (80/20)
    train_idx, test_idx = next(skf.split(X, y))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = StackingEnsemble()
    model.train(X_train, y_train)
    
    # 4. Evaluation
    print("\n[PHASE 3] Evaluation")
    model.evaluate(X_test, y_test)
    
    # Save Model
    model.save_model("liver_stacking_model.pkl")
    
    # 5. Explainable AI
    print("\n[PHASE 4] Explainability (XAI)")
    
    # We use a small subset of Train for XAI speed
    try:
        xai = XAIEngine(model, X_train, feature_names)
        
        # 5a. LIME for a test instance (e.g., a Cirrhosis case if available, else random)
        # Try to find a class 3 (Cirrhosis) instance in test set
        cirrhosis_indices = y_test[y_test == 3].index
        if len(cirrhosis_indices) > 0:
            idx = cirrhosis_indices[0]
            instance_idx = "cirrhosis_case"
        else:
            idx = X_test.index[0]
            instance_idx = "random_case"
            
        instance = X_test.loc[idx].values
        xai.generate_lime_explanation(instance, instance_idx=instance_idx)
        
        # 5b. SHAP Summary
        # xai.generate_shap_plots() # Uncomment if you want to run SHAP (slow)
        print("Skipping SHAP generation for speed (uncomment in main.py to run).")
        
    except Exception as e:
        print(f"XAI Error: {e}")

if __name__ == "__main__":
    main()
