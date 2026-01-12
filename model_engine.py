import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
from imblearn.combine import SMOTETomek
import joblib

class StackingEnsemble:
    def __init__(self):
        # Base Learners (Level 1) - "Dream Team"
        # We wrap them in CalibratedClassifierCV to ensure probability outputs are reliable.
        
        # 1. Random Forest (Robust Trees)
        self.rf = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=42),
            method='sigmoid', cv=3
        )
        
        # 2. SVM (Boundary / Margin)
        self.svm = CalibratedClassifierCV(
            SVC(probability=True, kernel='rbf', random_state=42),
            method='sigmoid', cv=3
        )
        
        # 3. XGBoost (Boosting)
        self.xgb = CalibratedClassifierCV(
            XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=100),
            method='sigmoid', cv=3
        )
        
        # 4. Neural Network (MLP) - for diversity
        self.mlp = CalibratedClassifierCV(
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, activation='relu', solver='adam', random_state=42),
            method='sigmoid', cv=3
        )
        
        # Meta Learner (Level 2)
        # optimize for Multi-Class Log Loss (multi_logloss)
        self.meta_model = LGBMClassifier(objective='multiclass', metric='multi_logloss', random_state=42)
        
        self.trained_base_models = [] 
        
    def _fit_model_safely(self, base_model_attr, X_train, y_train):
        """
        Helper to fit a model safely. 
        If calibration is enabled (model is CalibratedClassifierCV) and samples are < 3,
        it unwraps the base estimator and fits that instead.
        """
        from sklearn.base import clone
        
        # Get the template model from self (e.g., self.rf)
        model_template = getattr(self, base_model_attr)
        
        # Check min samples for calibration safety
        min_samples = y_train.value_counts().min()
        
        # Check if it's a CalibratedClassifierCV
        is_calibrated = isinstance(model_template, CalibratedClassifierCV)
        
        if is_calibrated and min_samples < 3:
            # Fallback to base estimator
            # Try 'estimator' (sklearn 1.2+) or 'base_estimator' (older)
            if hasattr(model_template, 'estimator'):
                base = model_template.estimator
            elif hasattr(model_template, 'base_estimator'):
                base = model_template.base_estimator
            else:
                base = model_template # Should not happen
            
            model = clone(base)
            # print(f"    [Warning] Class samples={min_samples} < 3. Skipping Calibration for {base_model_attr}.")
        else:
            model = clone(model_template)
            
        model.fit(X_train, y_train)
        return model

    def get_oof_predictions(self, X, y, n_splits=5):
        """
        Generates Out-of-Fold (OOF) predictions for the meta-learner.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        n_samples = len(y)
        n_classes = len(np.unique(y))
        if n_classes < 4: n_classes = 4 
        
        # 4 Models * n_classes
        oof_preds = np.zeros((n_samples, 4 * n_classes))
        
        print(f"Starting Stacking with {n_splits}-Fold CV (Enhanced 'Dream Team' Stack)...")
        
        fold = 1
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            # --- APPLY SMOTE-TOMEK ---
            print(f"  Fold {fold}: Balancing training data...")
            train_counts = y_train.value_counts()
            k = min(5, train_counts.min() - 1)
            if k < 1: k = 1
            
            from imblearn.over_sampling import SMOTE
            smote_algo = SMOTE(k_neighbors=k, random_state=42)
            smt = SMOTETomek(smote=smote_algo, random_state=42)
            
            try:
                X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"  Fold {fold} SMOTE Failed: {e}. using unbalanced data.")
                X_train_res, y_train_res = X_train, y_train
            
            # Fit Models Safely (Clone + Unwraps if needed)
            rf_fold = self._fit_model_safely('rf', X_train_res, y_train_res)
            svm_fold = self._fit_model_safely('svm', X_train_res, y_train_res)
            xgb_fold = self._fit_model_safely('xgb', X_train_res, y_train_res)
            mlp_fold = self._fit_model_safely('mlp', X_train_res, y_train_res)
            
            # Predict
            p1 = rf_fold.predict_proba(X_val)
            p2 = svm_fold.predict_proba(X_val)
            p3 = xgb_fold.predict_proba(X_val)
            p4 = mlp_fold.predict_proba(X_val)
            
            combined_preds = np.hstack([p1, p2, p3, p4])
            oof_preds[val_index] = combined_preds
            
            fold += 1
            
        return oof_preds

    def train(self, X, y):
        """
        Full training pipeline.
        """
        # 1. Generate Meta-Features via CV
        print("Generating Meta-Features (Level 1)...")
        meta_features_X = self.get_oof_predictions(X, y)
        
        # 2. Train Meta-Learner
        print("Training Meta-Learner (Level 2) with Multi-Class Log Loss...")
        self.meta_model.fit(meta_features_X, y)
        
        # 3. Retrain Base Models on FULL dataset
        print("Retraining Base Models on Full Balanced Dataset...")
        
        counts = y.value_counts()
        k = min(5, counts.min() - 1)
        if k < 1: k = 1
        
        from imblearn.over_sampling import SMOTE
        smote_algo = SMOTE(k_neighbors=k, random_state=42)
        smt = SMOTETomek(smote=smote_algo, random_state=42)
        
        try:
            X_res, y_res = smt.fit_resample(X, y)
        except:
             X_res, y_res = X, y
        
        # Fit final models (updating self attributes)
        self.rf = self._fit_model_safely('rf', X_res, y_res)
        self.svm = self._fit_model_safely('svm', X_res, y_res)
        self.xgb = self._fit_model_safely('xgb', X_res, y_res)
        self.mlp = self._fit_model_safely('mlp', X_res, y_res)
        
        print("Training Complete.")
        
    def predict(self, X):
        """
        Predicts using the stacked ensemble.
        """
        p1 = self.rf.predict_proba(X)
        p2 = self.svm.predict_proba(X)
        p3 = self.xgb.predict_proba(X)
        p4 = self.mlp.predict_proba(X)
        
        meta_features = np.hstack([p1, p2, p3, p4])
        return self.meta_model.predict(meta_features)

    def evaluate(self, X_test, y_test):
        """Evaluates model performance."""
        y_pred = self.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        kappa = cohen_kappa_score(y_test, y_pred)
        
        print("\n--- Evaluation Metrics ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print("\nClassification Report:\n")
        
        # Dynamic target names
        unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
        all_labels = {0: 'Normal', 1: 'Fatty Liver', 2: 'Fibrosis', 3: 'Cirrhosis'}
        target_names = [all_labels[c] for c in unique_classes if c in all_labels]
        
        print(classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names))
        
        return acc, f1, kappa

    def save_model(self, path='stacking_model.pkl'):
        model_pack = {
            'rf': self.rf,
            'svm': self.svm,
            'xgb': self.xgb,
            'mlp': self.mlp,
            'meta': self.meta_model
        }
        joblib.dump(model_pack, path)
        print(f"Model saved to {path}")
