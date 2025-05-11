import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

# === Load input feature names from correct file ===
file_path = '3D-Database-May-2025.xlsx'  # Must match training file exactly
df_C_S = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
df_R = pd.read_excel(file_path, sheet_name=1, engine='openpyxl')

# Explicit input feature names
compressive_features_list = [
    'Cement content (%)', 'Limestone content (%)', 'Silica fume content (%)',
    'SCM content (%)', 'Nc', 'SSA of SCM (m2/g)', 'Water/Binder',
    'Sand/Binder', 'Aggregate/Binder', 'Fiber length (mm)', 'Fiber Volume (%)',
    'Fiber Type', 'Age'
]

rheology_features_list = [
    'Cement content (%)', 'Limestone content (%)', 'Silica fume content (%)',
    'SCM content (%)', 'Nc', 'SSA of SCM (m2/g)', 'Water/Binder',
    'Sand/Binder', 'Aggregate/Binder', 'Fiber length (mm)', 'Fiber Volume (%)',
    'Fiber Type', 'Mini-slump after joint'
]

# Additional inputs for Nc calculation
extra_features = ['CaO in SCM', 'Al2O3 in SCM', 'SiO2 in SCM']

input_features = list(set(compressive_features_list + rheology_features_list) - {'Nc'}) + extra_features

# === Load the correct trained model ===
with open('3DP_May_2025.pkl', 'rb') as f:
    models = cloudpickle.load(f)

# === Streamlit UI ===
st.title("3DP Concrete Property Predictor")
st.write("Enter values for each feature below:")

user_input = {}
for feature in input_features:
    user_input[feature] = st.number_input(f"{feature}:", value=0.0, step=0.01, format="%.6f")

# === Compute Nc from SCM components ===
CAO = user_input.pop('CaO in SCM')
Al2O = user_input.pop('Al2O3 in SCM')
SiO2 = user_input.pop('SiO2 in SCM')

total = CAO + Al2O + SiO2
norm_CAO = CAO / total if total != 0 else 0
norm_Al2O = Al2O / total if total != 0 else 0
reg = norm_Al2O - norm_CAO

if reg < (-2/3):
    Nc = (11 + norm_Al2O - 10 * norm_CAO) / (3 - 2 * norm_CAO + 2 * norm_Al2O)
elif (-2/3) <= reg <= 0:
    Nc = (11 + 10 * norm_Al2O - 10 * norm_CAO) / (3 - 2 * norm_CAO + 2 * norm_Al2O)
else:
    Nc = (11 + 13 * norm_Al2O - 13 * norm_CAO) / (3 - 2 * norm_CAO + 2 * norm_Al2O)

user_input['Nc'] = Nc

# Explicit DataFrame creation
compressive_df = pd.DataFrame([{feature: user_input[feature] for feature in compressive_features_list}])
rheology_df = pd.DataFrame([{feature: user_input[feature] for feature in rheology_features_list}])

# === Prediction Button ===
if st.button("Predict"):
    compressive_strength = models['stacking_model_C'].predict(compressive_df)[0]
    water_retention = models['stacking_model_R1'].predict(rheology_df)[0]
    dynamic_yield_stress = models['stacking_model_R2'].predict(rheology_df)[0]
    plastic_viscosity = models['stacking_model_R3'].predict(rheology_df)[0]
    static_floc_stress = models['stacking_model_R4'].predict(rheology_df)[0]
    athix = models['stacking_model_R5'].predict(rheology_df)[0]

    st.subheader("Predicted Properties")
    st.write(f"**Compressive Strength (MPa):** {compressive_strength:.3f}")
    st.write(f"**Water Retention:** {water_retention:.3f}")
    st.write(f"**Dynamic Yield Stress (Pa):** {dynamic_yield_stress:.3f}")
    st.write(f"**Plastic Viscosity (Pa·s):** {plastic_viscosity:.3f}")
    st.write(f"**Static Flocculation Stress (Pa):** {static_floc_stress:.3f}")
    st.write(f"**Athix (Pa/min):** {athix:.3f}")
