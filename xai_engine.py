import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np

class XAIEngine:
    def __init__(self, model, X_train, feature_names):
        """
        model: Instance of StackingEnsemble
        X_train: Training data (Pandas DataFrame) for background distribution
        feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        
    def predict_proba_stacking(self, X):
        """
        Wrapper function for LIME/SHAP that mimics sklearn's predict_proba.
        Input: numpy array or dataframe
        Output: probability matrix (n_samples, n_classes)
        """
        # Ensure X is in legitimate format for base models
        # The base models expect standard scaled features. 
        # If X is numpy array, we might need a dataframe if models preserved column names?
        # Our base models (RF, SVM, XGB) trained on DataFrames/Arrays. StackingEnsemble internally handles it.
        
        # We need to manually reconstruct the meta-features logic here
        # Because 'self.model' is the StackingEnsemble class instance
        
        p1 = self.model.rf.predict_proba(X)
        p2 = self.model.svm.predict_proba(X)
        p3 = self.model.xgb.predict_proba(X)
        p4 = self.model.mlp.predict_proba(X) # MLP
        
        meta_features = np.hstack([p1, p2, p3, p4])
        return self.model.meta_model.predict_proba(meta_features)

    def generate_shap_plots(self):
        """
        Generates SHAP summary plot.
        Note: Calculating SHAP for a stacked model is complex.
        Method: We explain the Meta-Learner first, then ideally we'd map back to inputs.
        SIMPLIFICATION: We will calculate SHAP values for the XGBoost base model 
        as a proxy for "strongest learner" feature importance, or 
        KernelExplainer on the whole stack (Computationally Expensive).
        
        Let's use KernelExplainer on the `predict_proba_stacking` wrapper.
        warning: This might be slow. We'll summarize X_train using kmeans.
        """
        print("Generating SHAP Summary Plot (this may take time)...")
        
        # Summarize background data to 50 samples to speed up
        background = shap.kmeans(self.X_train, 10)
        
        explainer = shap.KernelExplainer(self.predict_proba_stacking, background)
        
        # Calculate shap values for a subset of data (e.g., 50 samples)
        shap_values = explainer.shap_values(self.X_train.iloc[:50, :])
        
        # Plot
        plt.figure()
        shap.summary_plot(shap_values, self.X_train.iloc[:50, :], feature_names=self.feature_names, show=False)
        plt.savefig('shap_summary.png', bbox_inches='tight')
        print("SHAP Summary plot saved to 'shap_summary.png'")

    def generate_lime_explanation(self, instance, instance_idx=0):
        """
        Generates LIME explanation for a single instance.
        """
        print(f"Generating LIME explanation for instance {instance_idx}...")
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.X_train),
            feature_names=self.feature_names,
            class_names=['Normal', 'Fatty Liver', 'Fibrosis', 'Cirrhosis'],
            mode='classification'
        )
        
        exp = explainer.explain_instance(
            data_row=instance, 
            predict_fn=self.predict_proba_stacking
        )
        
        exp.save_to_file(f'lime_explanation_{instance_idx}.html')
        print(f"LIME explanation saved to 'lime_explanation_{instance_idx}.html'")
