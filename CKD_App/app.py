import gradio as gr
import joblib
import numpy as np

print("--- 1. Loading Model and Scaler 'Brain' ---")
# Load the scaler and model
try:
    scaler = joblib.load('ckd_scaler.pkl')
    model = joblib.load('ckd_model.pkl')
    print("...Assets loaded successfully.")
except FileNotFoundError:
    print("--- ERROR: 'ckd_scaler.pkl' or 'ckd_model.pkl' not found! ---")
    print("--- Make sure they are in the same folder as app.py ---")
    exit()

# This is the list of 51 features your model expects IN ORDER
feature_names = [
    'Age', 'Gender', 'Ethnicity', 'SocioeconomicStatus', 'EducationLevel', 'BMI', 
    'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 
    'FamilyHistoryKidneyDisease', 'FamilyHistoryHypertension', 'FamilyHistoryDiabetes', 
    'PreviousAcuteKidneyInjury', 'UrinaryTractInfections', 'SystolicBP', 'DiastolicBP', 
    'FastingBloodSugar', 'HbA1c', 'SerumCreatinine', 'BUNLevels', 'GFR', 
    'ProteinInUrine', 'ACR', 'SerumElectrolytesSodium', 'SerumElectrolytesPotassium', 
    'SerumElectrolytesCalcium', 'SerumElectrolytesPhosphorus', 'HemoglobinLevels', 
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 
    'ACEInhibitors', 'Diuretics', 'NSAIDsUse', 'Statins', 'AntidiabeticMedications', 
    'Edema', 'FatigueLevels', 'NauseaVomiting', 'MuscleCramps', 'Itching', 
    'QualityOfLifeScore', 'HeavyMetalsExposure', 'OccupationalExposureChemicals', 
    'WaterQuality', 'MedicalCheckupsFrequency', 'MedicationAdherence', 'HealthLiteracy'
]

# --- 2. Define the Prediction Function ---
def predict_ckd(*patient_features):
    try:
        # 1. Convert inputs to a numpy array
        final_features = np.array(patient_features).reshape(1, -1)
        
        # 2. Scale the data
        scaled_features = scaler.transform(final_features)
        
        # 3. Get prediction probabilities
        probability = model.predict_proba(scaled_features)
        
        # 4. Format the output nicely
        conf_no_ckd = probability[0][0]
        conf_ckd = probability[0][1]
        
        output_dict = {
            "No CKD": conf_no_ckd,
            "CKD": conf_ckd
        }
        
        return output_dict

    except Exception as e:
        return {"Error": str(e)}

# --- 3. Create the Gradio UI ---
print("--- 3. Creating Gradio Interface ---")

# Create a list of 51 numeric input boxes
inputs = [gr.Number(label=name) for name in feature_names]
# Create an output component (a label)
output = gr.Label(num_top_classes=2, label="Prediction Results")

# --- 4. Launch the App ---
app = gr.Interface(
    fn=predict_ckd,      # The function to call
    inputs=inputs,       # The list of 51 input boxes
    outputs=output,      # The output label
    title="Chronic Kidney Disease (CKD) Predictor",
    description="Enter the 51 patient features to get a CKD prediction. The model is a Random Forest with ~98.5% accuracy."
)

print("--- 4. Launching App... ---")
print("Open the following URL in your web browser:")
app.launch()  # This will print a local URL (like http://127.0.0.1:7860)