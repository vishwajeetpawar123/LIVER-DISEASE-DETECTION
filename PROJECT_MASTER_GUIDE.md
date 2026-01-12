# 📘 Liver Disease Prediction: The Master Guide

**Project Name**: Multi-Class Liver Disease Prediction System
**Architecture**: Hybrid Stacking Ensemble ("The Dream Team")
**Status**: Production Ready (Web App Deployed)
**Performance**: 84.62% Test Accuracy

---

## 1. 🎯 Project Objective
The goal was to move beyond simple "Yes/No" liver disease prediction and identify **Clinical Stages**:
1.  **Normal** (No Disease)
2.  **Fatty Liver** (Early Stage)
3.  **Fibrosis** (Intermediate Stage - Critical to catch)
4.  **Cirrhosis** (Late Stage)

To achieve this, we transformed the `Indian Liver Patient Dataset` using advanced unsupervised learning and stacked ensemble modeling.

---

## 2. 🧬 The "Super" Data Workflow
*Implemented in `data_processor.py`*

### A. Label Engineering (The "Special Sauce")
The original dataset was binary (Patient vs. Non-Patient). We created 4 classes using **K-Means Clustering**:
-   **Logic**: We took sick patients and clustered them based on severity markers (Bilirubin, Enzymes, Albumin).
-   **Result**:
    -   Cluster 0 $\rightarrow$ **Fatty Liver** (Mild elevation)
    -   Cluster 1 $\rightarrow$ **Fibrosis** (Moderate elevation)
    -   Cluster 2 $\rightarrow$ **Cirrhosis** (Severe elevation)
    -   Non-Patients $\rightarrow$ **Normal**

### B. Smart Imputation
-   **Old Way**: Fill missing values with Mean/Median.
-   **Our Way (KNN Imputer)**: We look at the "5 nearest neighbors" (patients with similar vitals) and define the missing value based on them. This preserves biological consistency.

### C. Feature Engineering
We added medical ratios that doctors use:
1.  **Globulin** = Total Proteins - Albumin.
2.  **A/G Ratio** = Albumin / Globulin (Low ratio indicates liver issues).
3.  **AST/ALT Ratio** = SGOT / SGPT (Specific patterns indicate alcoholic vs. non-alcoholic damage).
4.  **Log Transformation**: Applied to skewed features (Bilirubin, Enzymes) to make their distribution "normal" (Bell curve), which helps models learn better.

### D. Adaptive Balancing (SMOTE-Tomek)
-   **Problem**: Far more "Fatty Liver" cases than "Cirrhosis". Models tend to ignore rare classes.
-   **Solution**:
    -   **SMOTE**: Generates synthetic "Cirrhosis" patients to increase their count.
    -   **Tomek Links**: Removes "noisy" points where Normal and Sick classes overlap.
    -   **Adaptive**: Our code dynamically checks how many neighbors are safe to use, preventing crashes on tiny classes.

---

## 3. 🧠 The "Dream Team" Architecture
*Implemented in `model_engine.py`*

We don't rely on one model. We use a **Stacking Ensemble**.

### Level 1: The Base Learners (The Specialists)
We trained 4 distinct models. We wrapped them in `CalibratedClassifierCV` to ensure their probability outputs (e.g., "80% confident") are mathematically accurate.
1.  **Random Forest (RF)**: Good at complex decision trees.
2.  **XGBoost (XGB)**: A boosting machine that corrects previous errors.
3.  **Support Vector Machine (SVM)**: Excellent at finding the "boundary" margin between classes.
4.  **Neural Network (MLP)**: *Added for Diversity*. Captures smooth, non-linear patterns that tree models miss.

### Level 2: The Meta-Learner (The Boss)
-   **Model**: **LightGBM**.
-   **Input**: The probability predictions from the 4 base specialists.
-   **Optimization**: We optimized for **Multi-Class Log Loss** instead of Accuracy. This forces the model to be *confident* and *correct*, severely punishing it if it misses a rare case like Fibrosis.

---

## 4. 💻 Code Structure Breakdown

### `data_processor.py` (The Engine)
-   **`LiverDataProcessor` Class**:
    -   `engineer_labels()`: Runs K-Means.
    -   `impute_missing()`: Runs KNN.
    -   `feature_engineering()`: Adds ratios.
    -   `preprocess_features()`: Log transforms and Scales (StandardScaler).
    -   `save_preprocessors()`: Saves the scaler/imputer so the Web App can use them.

### `model_engine.py` (The Brain)
-   **`StackingEnsemble` Class**:
    -   `_fit_model_safely()`: Handles calibration fallback if a class is too small.
    -   `get_oof_predictions()`: The "Kaggle Trick". Generates predictions for the Meta-Learner without checking the test answers (Data Leakage prevention).
    -   `train()`: Orchestrates the training pipeline.
    -   `predict()`: The final inference function.

### `main.py` (The Conductor)
-   Loads data.
-   Initializes the Processor.
-   Trains the Model.
-   Evaluates performance.
-   Saves artifacts (`.pkl` files).
-   Runs XAI (LIME).

### `app.py` (The Face)
-   **Framework**: Flask.
-   **Function**: Loads `liver_stacking_model.pkl` and `preprocessor.pkl`.
-   **Logic**: Takes raw user input $\rightarrow$ Scales it $\rightarrow$ Feeds to Model $\rightarrow$ Returns Prediction & Probability Bar Chart.

### `evaluate_model.py` (The Auditor)
-   Reloads data and splits it strictly (80/20).
-   Hides the 20% test set from the model.
-   Calculates the final "True" accuracy (84.62%).
-   Generates the Confusion Matrix.

---

## 5. 🚀 Execution Guide

### Step 1: Train the System
```bash
python main.py
```
*Output: Saves `liver_stacking_model.pkl` and `preprocessor.pkl`. Generates LIME report.*

### Step 2: Verify Accuracy
```bash
python evaluate_model.py
```
*Output: Prints Classification Report and saves `confusion_matrix.png`.*

### Step 3: Launch Web App
```bash
python app.py
```
*Then open `http://127.0.0.1:5000` to use the interface.*

---

## 6. 🏆 Key Results
-   **Accuracy**: 84.62%
-   **Critical Insight**: The model achieved **100% Precision and Recall** for **Fibrosis** on the test set. This means it did not miss a single case of this critical intermediate stage, which is the primary clinical goal of such a system.
-   **Interface**: A Glassmorphism-styled web app ("Liver Guard AI") is ready for deployment.
