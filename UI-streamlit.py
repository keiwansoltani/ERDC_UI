import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

# === Load input feature names from correct file ===
file_path = '3D_DB_Feb2025.xlsx'  # Same file used for training
excel_data = pd.ExcelFile(file_path)

df_C_S = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
df_R = pd.read_excel(file_path, sheet_name=1, engine='openpyxl')

# Extract input feature names
feature_names_C_S = df_C_S.columns[:-1]  # All except the target (Compressive strength)
feature_names_R = df_R.columns[11]       # 'Mini-slump after joint'

# Build input list, excluding Nc (we'll compute it)
input_features = [f for f in feature_names_C_S if f != 'Nc']
input_features += [feature_names_R, 'CaO in SCM', 'Al2O3 in SCM', 'SiO2 in SCM']

# === Load the trained model ===
with open('3DP_APR2025.pkl', 'rb') as f:
    models = cloudpickle.load(f)

# === Streamlit UI ===
st.title("3DP Concrete Property Predictor")
st.write("Enter values for each feature below:")

user_input = {}
for feature in input_features:
    user_input[feature] = st.number_input(f"{feature}:", value=1e-10, step=0.0, format="%.6f")

# === Compute Nc from SCM components ===
CAO = user_input.pop('CaO in SCM')
Al2O = user_input.pop('Al2O3 in SCM')
SiO2 = user_input.pop('SiO2 in SCM')

total = CAO + Al2O + SiO2
norm_CAO = CAO / total
norm_Al2O = Al2O / total
reg = norm_Al2O - norm_CAO

if reg < (-2/3):
    Nc = (11 + norm_Al2O - 10 * norm_CAO) / (3 - 2 * norm_CAO + 2 * norm_Al2O)
elif (-2/3) <= reg <= 0:
    Nc = (11 + 10 * norm_Al2O - 10 * norm_CAO) / (3 - 2 * norm_CAO + 2 * norm_Al2O)
else:
    Nc = (11 + 13 * norm_Al2O - 13 * norm_CAO) / (3 - 2 * norm_CAO + 2 * norm_Al2O)

# Insert Nc into the right place
items = list(user_input.items())
items.insert(3, ('Nc', Nc))  # Nc is expected as the 4th feature
user_input = dict(items)

# Split inputs for compressive vs rheology models
compressive_features = {k: v for k, v in user_input.items() if k != 'Mini-slump after joint'}
rheology_features = {k: v for k, v in user_input.items() if k != 'Age'}

compressive_df = pd.DataFrame([compressive_features])
rheology_df = pd.DataFrame([rheology_features])

# === Prediction Button ===
if st.button("Predict"):
    compressive_strength = models['stacking_model_C'].predict(compressive_df)[0]
    water_retention      = models['stacking_model_R1'].predict(rheology_df)[0]
    dynamic_yield_stress = models['stacking_model_R2'].predict(rheology_df)[0]
    plastic_viscosity    = models['stacking_model_R3'].predict(rheology_df)[0]
    static_floc_stress   = models['stacking_model_R4'].predict(rheology_df)[0]
    athix                = models['stacking_model_R5'].predict(rheology_df)[0]

    st.subheader("Predicted Properties")
    st.write(f"**Compressive Strength (MPa):** {compressive_strength:.3f}")
    st.write(f"**Water Retention:** {water_retention:.3f}")
    st.write(f"**Dynamic Yield Stress (Pa):** {dynamic_yield_stress:.3f}")
    st.write(f"**Plastic Viscosity (Pa·s):** {plastic_viscosity:.3f}")
    st.write(f"**Static Flocculation Stress (Pa):** {static_floc_stress:.3f}")
    st.write(f"**Athix (Pa/min):** {athix:.3f}")
