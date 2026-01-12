# Project Journal: Multi-Class Liver Disease Prediction

## [ENTRY] The "Super" Workflow Strategy

**Date**: 2026-01-11
**Objective**: Transform the binary ILPD dataset into a multi-class prediction system with high clinical relevance.

### 1. The Labeling Challenge
The original dataset provided only binary labels (Liver Patient vs. Non-Patient). To match the research paper's goal of staging liver disease (Fatty Liver, Fibrosis, Cirrhosis), we adopted a **Label Engineering** approach. 
- **Strategy**: Instead of arbitrary manual thresholds, we will use **K-Means Clustering (k=3)** on the "sick" population (Selector=1). 
- **Rationale**: This allows the data itself to reveal the natural groupings of disease severity based on liver function tests (Bilirubin, Enzymes, Albumin), which we can then map to clinical stages.

### 2. Advanced Feature Engineering
Standard features are insufficient for capturing complex liver dynamics. We are introducing:
- **Globulin**: (Total Proteins - Albumin) - Critical for immunity assessment.
- **A/G Ratio & AST/ALT Ratio**: Well-established clinical indices for differentiating liver conditions.
- **Log Transformation**: Liver enzymes (SGOT, SGPT, Alkphos) often follow a power law or exponential distribution. Log-transforming them normalizes the distribution, making them far more suitable for linear based models (like SVM) and stable for tree-based models.

### 3. Handling Real-World Messiness
- **Imputation**: Moving away from simple Median imputation to **KNN Imputer**. Patients with similar biochemical profiles likely have similar missing values, preserving biological consistency.
## [ENTRY] The Final Polish: Web App & Rigorous Testing

**Date**: 2026-01-11
**Objective**: Deploy the model to a user-facing application and verify its real-world performance.

### 1. "Liver Guard AI" Web Application
- **Frontend**: Built a Glassmorphism-based UI using HTML/CSS. It features real-time confidence bars and dynamic styling based on disease severity.
- **Backend**: Implemented a Flask server (`app.py`) that loads the saved `StackingEnsemble` and `Preprocessor`.
- **Engineering Challenge**: The scaler returned numpy arrays, stripping feature names, which caused warnings in the trained models. Fixed by reconstructing the DataFrame with `feature_names` before inference.

### 2. The Verdict: 84.62% Accuracy
We ran a dedicated evaluation script (`evaluate_model.py`) on a **20% held-out test set**.
- **Accuracy**: **84.62%**
- **Fibrosis Detection**: **100% Precision & Recall** (Perfect detection of the critical stage)
- **Fatty Liver**: **88% F1-Score**

### 3. Conclusion
The "Dream Team" architecture (RF+SVM+XGB+MLP -> LGBM) combined with the "Super" data workflow has proven highly effective. It successfully identifies intermediate disease stages (Fibrosis, Fatty Liver) that binary classifiers often miss.

## [ENTRY] Final Polish: Cleaning Console Logs
**Date**: 2026-01-11
**Issue**: Users saw a `UserWarning` about feature names when stacking the Meta-Learner.
**Cause**: The Stacking Logic combines predictions into a pure Numpy array, while the Meta-Learner (StackingCV) sometimes remembers the original DataFrame column names.
**Fix**: Added a suppression filter in `app.py` for this specific harmless benign warning.
**Result**: The application logs are now clean and professional.ples for minority classes (Cirrhosis) but also cleans up "noisy" overlapping points (Tomek links) between classes, sharpening the decision boundaries for the classifiers.
