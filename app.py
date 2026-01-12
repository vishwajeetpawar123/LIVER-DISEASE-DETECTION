from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress warnings about feature names (caused by numpy/pandas mismatch in stacking)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load Artifacts
print("Loading Model and Preprocessors...")
try:
    # Load Model Pack
    model_pack = joblib.load('liver_stacking_model.pkl')
    stacking_ensemble = model_pack # It seems I saved a dict.
    # Wait, StackingEnsemble class structure in model_engine.py is complex.
    # I saved a dict {'rf':..., 'svm':..., 'meta':...}
    # But for inference I need the StackingEnsemble methods (predict/predict_proba)
    # The saved dict is NOT the class instance.
    # I need to Re-Instantiate StackingEnsemble and populate it with loaded models.
    
    from model_engine import StackingEnsemble
    model = StackingEnsemble()
    model.rf = model_pack['rf']
    model.svm = model_pack['svm']
    model.xgb = model_pack['xgb']
    model.mlp = model_pack['mlp']
    model.meta_model = model_pack['meta']
    
    # Load Preprocessors
    preprocessors = joblib.load('preprocessor.pkl')
    scaler = preprocessors['scaler']
    label_encoder = preprocessors['label_encoder']
    feature_names = preprocessors['feature_names']
    
    print("Artifacts Loaded Successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded.'})
    
    try:
        # Get Data from Form
        data = request.form.to_dict()
        
        # Helper: Safe float conversion
        def get_val(key):
            return float(data.get(key, 0))
            
        # 1. Feature Engineering
        # Input Maps to: Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, 
        # Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio
        
        # Gender Encode
        gender_input = data.get('Gender', 'Male')
        gender_val = 0 if gender_input == 'Female' else 1 # Assuming LabelEncoder behavior (Female=0, Male=1 usually)
        # Better to check label_encoder classes if possible, but safe assumption for 2 classes.
        
        # Calculate Derived Features
        Total_Protiens = get_val('Total_Protiens')
        Albumin = get_val('Albumin')
        Globulin = Total_Protiens - Albumin
        Albumin_Globulin_Ratio_Calc = Albumin / (Globulin + 1e-6)
        
        Aspartate_Aminotransferase = get_val('Aspartate_Aminotransferase')
        Alamine_Aminotransferase = get_val('Alamine_Aminotransferase')
        AST_ALT_Ratio = Aspartate_Aminotransferase / (Alamine_Aminotransferase + 1e-6)
        
        # Construct DataFrame in EXACT order of training
        # Features: ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
        # 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 
        # 'Albumin_and_Globulin_Ratio', 'Globulin', 'Albumin_Globulin_Ratio_Calc', 'AST_ALT_Ratio']
        
        input_dict = {
            'Age': get_val('Age'),
            'Gender': gender_val,
            'Total_Bilirubin': get_val('Total_Bilirubin'),
            'Direct_Bilirubin': get_val('Direct_Bilirubin'),
            'Alkaline_Phosphotase': get_val('Alkaline_Phosphotase'),
            'Alamine_Aminotransferase': Alamine_Aminotransferase,
            'Aspartate_Aminotransferase': Aspartate_Aminotransferase,
            'Total_Protiens': Total_Protiens,
            'Albumin': Albumin,
            'Albumin_and_Globulin_Ratio': get_val('Albumin_and_Globulin_Ratio'),
            'Globulin': Globulin,
            'Albumin_Globulin_Ratio_Calc': Albumin_Globulin_Ratio_Calc,
            'AST_ALT_Ratio': AST_ALT_Ratio
        }
        
        df = pd.DataFrame([input_dict], columns=feature_names)
        
        # 2. Log Transform Skewed
        skewed_features = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alamine_Aminotransferase', 
                           'Aspartate_Aminotransferase', 'Alkaline_Phosphotase']
        for feat in skewed_features:
            df[feat] = np.log1p(df[feat])
            
        # 3. Scale
        df_scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=feature_names) # Convert back to DF to fix warnings
        
        # 4. Predict
        # Manual Stacking Prediction Logic
        p1 = model.rf.predict_proba(df_scaled)
        p2 = model.svm.predict_proba(df_scaled)
        p3 = model.xgb.predict_proba(df_scaled)
        p4 = model.mlp.predict_proba(df_scaled)
        
        meta_features = np.hstack([p1, p2, p3, p4])
        probs = model.meta_model.predict_proba(meta_features)[0]
        prediction = np.argmax(probs)
        
        class_map = {0: 'Normal', 1: 'Fatty Liver', 2: 'Fibrosis', 3: 'Cirrhosis'}
        result = class_map.get(prediction, "Unknown")
        
        return jsonify({
            'result': result, 
            'confidence': f"{np.max(probs)*100:.2f}%",
            'probabilities': {class_map[i]: f"{probs[i]*100:.2f}%" for i in range(4)}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
