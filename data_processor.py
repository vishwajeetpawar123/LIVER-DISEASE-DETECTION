import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.model_selection import StratifiedKFold
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

class LiverDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = None
        self.label_encoder = None
        self.imputer = None
        
    def load_data(self):
        """Loads the dataset and handles initial column naming."""
        print(f"Loading data from {self.filepath}...")
        self.data = pd.read_csv(self.filepath)
        
        # Standardize column names based on dataset structure
        # Expected columns: Age, Gender, Total_Bilirubin, Direct_Bilirubin, 
        # Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase,
        # Total_Protiens, Albumin, Albumin_and_Globulin_Ratio, Dataset
        
        # Rename 'Dataset' to 'Selector' for clarity if needed, or stick to provided names.
        # Check actual columns first
        print(f"Initial Columns: {self.data.columns.tolist()}")
        
        # Ensure consistent naming
        column_map = {
            'Dataset': 'Selector',
            'Albumin_and_Globulin_Ratio': 'Albumin_and_Globulin_Ratio'
        }
        self.data.rename(columns=column_map, inplace=True)
        return self.data

    def engineer_labels(self):
        """
        Implements the Label Engineering strategy:
        1. Split Selector=1 (Patients) and Selector=2 (Non-Patients).
        2. Use KMeans to cluster Patients into 3 groups.
        3. Map Clusters to Severity (Fatty Liver, Fibrosis, Cirrhosis).
        4. Combine with Normal (0) to create 4-class target.
        """
        print("Starting Label Engineering (Clustering)...")
        
        # 1. Separate Patients and Non-Patients
        # Selector: 1 = Liver Patient, 2 = Non-Liver Patient
        patients = self.data[self.data['Selector'] == 1].copy()
        normals = self.data[self.data['Selector'] == 2].copy()
        
        # Assign Class 0 to Normals
        normals['Target'] = 0
        
        # 2. Cluster Patients
        # We need to impute missing values for clustering first, as KMeans doesn't handle NaNs
        # Using a temporary simple imputer for the clustering step only
        cluster_features = ['Total_Bilirubin', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Albumin']
        
        # Handle cases where column names might slightly differ (e.g. typos in dataset)
        available_features = [c for c in cluster_features if c in patients.columns]
        
        temp_imputer = KNNImputer(n_neighbors=5)
        patients_imputed = patients.copy()
        patients_imputed[available_features] = temp_imputer.fit_transform(patients[available_features])
        
        # Perform KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(patients_imputed[available_features])
        
        # 3. Sort Clusters by Severity
        # We assume higher Bilirubin/Enzymes indiciate higher severity.
        patients['temp_cluster'] = clusters
        
        # Calculate mean Bilirubin per cluster to rank them
        cluster_means = patients.groupby('temp_cluster')['Total_Bilirubin'].mean().sort_values()
        
        # Map: Lowest Mean -> 1 (Fatty), Medium -> 2 (Fibrosis), Highest -> 3 (Cirrhosis)
        mapping = {}
        sorted_clusters = cluster_means.index.tolist() # [id_low, id_mid, id_high]
        
        mapping[sorted_clusters[0]] = 1 # Fatty Liver
        mapping[sorted_clusters[1]] = 2 # Fibrosis
        mapping[sorted_clusters[2]] = 3 # Cirrhosis
        
        patients['Target'] = patients['temp_cluster'].map(mapping)
        patients.drop(columns=['temp_cluster'], inplace=True)
        
        print(f"Cluster severity analysis (Mean Bilirubin): \n{cluster_means}")
        print(f"Cluster Mapping: {mapping}")
        
        # 4. Combine
        self.data = pd.concat([normals, patients], axis=0).sort_index()
        print(f"Target Distribution:\n{self.data['Target'].value_counts().sort_index()}")
        
        self.data.drop(columns=['Selector'], inplace=True) # Remove original binary selector
        return self.data

    def impute_missing(self):
        """Imputes missing values using KNNImputer."""
        print("Imputing missing values with KNN...")
        
        # Identify numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('Target') # Don't impute the target!
        
        self.imputer = KNNImputer(n_neighbors=5)
        self.data[numeric_cols] = self.imputer.fit_transform(self.data[numeric_cols])
        return self.data

    def feature_engineering(self):
        """Adds derived features: Globulin, A/G Ratio, AST/ALT Ratio."""
        print("Engineering new features...")
        
        # Globulin = Total_Protiens - Albumin
        self.data['Globulin'] = self.data['Total_Protiens'] - self.data['Albumin']
        
        # Albumin/Globulin Ratio
        self.data['Albumin_Globulin_Ratio_Calc'] = self.data['Albumin'] / (self.data['Globulin'] + 1e-6)
        
        # AST/ALT Ratio
        self.data['AST_ALT_Ratio'] = self.data['Aspartate_Aminotransferase'] / (self.data['Alamine_Aminotransferase'] + 1e-6)
        
        print("Features added: Globulin, Albumin_Globulin_Ratio_Calc, AST_ALT_Ratio")
        return self.data

    def preprocess_features(self):
        """Performs Log Scaling, Encoding, and Standardization."""
        print("Preprocessing features (Log Transform + Scaling)...")
        
        # 1. Encode Gender
        self.label_encoder = LabelEncoder()
        if 'Gender' in self.data.columns:
            self.data['Gender'] = self.label_encoder.fit_transform(self.data['Gender'].astype(str))
        
        # 2. Log Transform skew features
        skewed_features = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alamine_Aminotransferase', 
                           'Aspartate_Aminotransferase', 'Alkaline_Phosphotase']
        
        for feat in skewed_features:
            if feat in self.data.columns:
                self.data[feat] = np.log1p(self.data[feat])
        
        # 3. Standardization (Z-Score)
        # Separate X and y
        self.y = self.data['Target']
        self.X = self.data.drop(columns=['Target'])
        self.feature_names = self.X.columns.tolist()
        
        self.scaler = StandardScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.feature_names)
        
        return self.X, self.y

    def balance_data(self):
        """Applies SMOTETomek to balance the classes."""
        print("Balancing classes with SMOTETomek...")
        class_counts = self.y.value_counts().sort_index()
        print(f"Before Resampling: \n{class_counts}")
        
        # Check minimum class size
        min_class_size = class_counts.min()
        print(f"Minimum class size: {min_class_size}")
        
        # SMOTE requires k_neighbors < n_samples. Default is 5.
        # If min_samples <= 6, we need to reduce k_neighbors
        k_neighbors = min(5, min_class_size - 1)
        if k_neighbors < 1:
            k_neighbors = 1
            
        print(f"Using SMOTE k_neighbors={k_neighbors}")
        
        # We need to pass the custom SMOTE to SMOTETomek
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        smt = SMOTETomek(smote=smote, random_state=42)
        
        try:
            X_res, y_res = smt.fit_resample(self.X, self.y)
            print(f"After Resampling: \n{y_res.value_counts().sort_index()}")
            self.X = X_res
            self.y = y_res
        except Exception as e:
            print(f"SMOTETomek Failed: {e}. Returning unbalanced data.")
            
        return self.X, self.y

    def save_preprocessors(self, path='preprocessor.pkl'):
        """Saves the scaler, label encoder, and imputer."""
        payload = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }
        joblib.dump(payload, path)
        print(f"Preprocessors saved to {path}")

    def load_preprocessors(self, path='preprocessor.pkl'):
        """Loads preprocessors for inference."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
            
        payload = joblib.load(path)
        self.scaler = payload['scaler']
        self.label_encoder = payload['label_encoder']
        self.imputer = payload['imputer']
        self.feature_names = payload['feature_names']
        print(f"Preprocessors loaded from {path}")

    def get_processed_data(self):
        """Runs the full pipeline and returns X, y. (Restored)"""
        self.load_data()
        self.engineer_labels()   
        self.impute_missing()    
        self.feature_engineering() 
        self.preprocess_features() 
        return self.X, self.y
