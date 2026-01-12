import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from data_processor import LiverDataProcessor

# 1. Load and Process Data (Deterministic)
print("Loading and Processing Data...")
dataset_path = 'DATASET/indian_liver_patient.csv'
processor = LiverDataProcessor(dataset_path)
# We regenerate the processed data to ensure we have the full dataset to split
# Note: The processor uses random_state=42 internally for KMeans, so labels are stable.
X, y = processor.get_processed_data()

# 2. Split Data (Same seed as main.py logic to isolate Test set)
# In main.py: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Splitting into Train/Test (Wait, strictly using unseen Test set)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Test Set Size: {len(y_test)} samples")

# 3. Load Model
print("Loading Model Artifacts...")
try:
    model_pack = joblib.load('liver_stacking_model.pkl')
    # Reconstruct StackingEnsemble manually for inference since I saved a dict
    # This mimics the app.py logic
    class StackingWrapper:
        def __init__(self, pack):
            self.rf = pack['rf']
            self.svm = pack['svm']
            self.xgb = pack['xgb']
            self.mlp = pack['mlp']
            self.meta_model = pack['meta']
            
        def predict(self, X):
            p1 = self.rf.predict_proba(X)
            p2 = self.svm.predict_proba(X)
            p3 = self.xgb.predict_proba(X)
            p4 = self.mlp.predict_proba(X)
            meta_features = np.hstack([p1, p2, p3, p4])
            return self.meta_model.predict(meta_features)

    model = StackingWrapper(model_pack)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 4. Predict
print("Running Inference on Test Set...")
y_pred = model.predict(X_test)

# 5. Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")

target_names = ['Normal', 'Fatty Liver', 'Fibrosis', 'Cirrhosis']
# Verify classes in test set
unique_labels = np.unique(y_test)
labels_present = [target_names[i] for i in unique_labels]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=labels_present))

# 6. Confusion Matrix Plot
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_present, yticklabels=labels_present)
plt.title('Confusion Matrix: Liver Disease Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Save
output_path = 'confusion_matrix.png'
plt.savefig(output_path)
print(f"Confusion Matrix saved to {output_path}")
