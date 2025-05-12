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

# Predefined SCM compositions
scm_defaults = {
    "Calcined clay": [12.067, 0.73, 35.27, 57.05],
    "F fly ash": [2.153, 11.83, 19.48, 43.59],
    "C fly ash": [2.973, 29.51, 18.07, 37.29],
    "Ground granulated blast furnace slag": [1.084, 42.73, 8.58, 35.9],
    "MSWI ash": [12.086, 52.85, 6.85, 12.91],
    "Steel slag": [0.854, 62.75, 9.9, 16.09],
    "Glass powder": [0.233, 12.54, 1.16, 74.8]
}

# === Load the correct trained model ===
with open('3DP_May_2025.pkl', 'rb') as f:
    models = cloudpickle.load(f)

# === Streamlit UI ===
st.title("3DP Concrete Property Predictor")
st.write("Enter values for each feature below:")

user_input = {}

fiber_type_options = {
    "None": 0,
    "Steel": 1,
    "PVA": 2,
    "PP": 3,
    "Glass": 4,
    "Hemp": 5
}

# === SCM Composition Section ===
with st.expander("🧪 SCM Composition"):
    scm_choice = st.selectbox("Choose SCM Type:", list(scm_defaults.keys()))
    use_default_scm = st.checkbox("Default Composition", help="Check to auto-fill values for selected SCM type")

    if use_default_scm:
        default_vals = scm_defaults[scm_choice]
    else:
        default_vals = [0.0, 0.0, 0.0, 0.0]

    ssa_val = st.number_input("SSA of SCM (m2/g):", value=default_vals[0], step=0.01, format="%.6f")
    cao_val = st.number_input("CaO in SCM:", value=default_vals[1], step=0.01, format="%.6f")
    al2o_val = st.number_input("Al2O3 in SCM:", value=default_vals[2], step=0.01, format="%.6f")
    sio2_val = st.number_input("SiO2 in SCM:", value=default_vals[3], step=0.01, format="%.6f")

    user_input['SSA of SCM (m2/g)'] = ssa_val
    user_input['CaO in SCM'] = cao_val
    user_input['Al2O3 in SCM'] = al2o_val
    user_input['SiO2 in SCM'] = sio2_val

# === Mix Design Inputs ===
with st.expander("🔧 Mix Design Inputs"):
    for feature in compressive_features_list:
        if feature in ['Nc', 'SSA of SCM (m2/g)', 'Fiber Type', 'Fiber length (mm)', 'Fiber Volume (%)', 'Age']:
            continue
        user_input[feature] = st.number_input(f"{feature}:", value=0.0, step=0.01, format="%.6f")

# === Fiber Properties ===
with st.expander("🧵 Fiber Properties"):
    fiber_label = st.selectbox("Fiber Type:", list(fiber_type_options.keys()))
    user_input['Fiber Type'] = fiber_type_options[fiber_label]

    user_input['Fiber length (mm)'] = st.number_input("Fiber length (mm):", value=0.0, step=0.01, format="%.6f")
    user_input['Fiber Volume (%)'] = st.number_input("Fiber Volume (%):", value=0.0, step=0.01, format="%.6f")

# === Rheology Input ===
with st.expander("📊 Rheology Input"):
    if 'Mini-slump after joint' not in user_input:
        user_input['Mini-slump after joint'] = st.number_input("Mini-slump after joint:", value=0.0, step=0.01, format="%.6f")

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
user_input['CaO in SCM'] = CAO
user_input['Al2O3 in SCM'] = Al2O
user_input['SiO2 in SCM'] = SiO2

# Predict for both ages 7 and 28
age_predictions = {}
for age in [7, 28]:
    user_input['Age'] = age
    compressive_df = pd.DataFrame([{feature: user_input[feature] for feature in compressive_features_list}])
    # Use same rheology_df for both ages since age is not a rheology input
    rheology_df = pd.DataFrame([{feature: user_input[feature] for feature in rheology_features_list}])

    compressive_strength = models['stacking_model_C'].predict(compressive_df)[0]
    water_retention = models['stacking_model_R1'].predict(rheology_df)[0]
    dynamic_yield_stress = models['stacking_model_R2'].predict(rheology_df)[0]
    plastic_viscosity = models['stacking_model_R3'].predict(rheology_df)[0]
    static_floc_stress = models['stacking_model_R4'].predict(rheology_df)[0]
    athix = models['stacking_model_R5'].predict(rheology_df)[0]

    age_predictions[age] = {
        "Compressive Strength (MPa)": compressive_strength,
        "Water Retention": water_retention,
        "Dynamic Yield Stress (Pa)": dynamic_yield_stress,
        "Plastic Viscosity (Pa·s)": plastic_viscosity,
        "Static Flocculation Stress (Pa)": static_floc_stress,
        "Athix (Pa/min)": athix
    }

# === Prediction Button ===
if st.button("Predict"):
    for age, results in age_predictions.items():
        st.subheader(f"Predicted Properties for {age} days")
        for key, value in results.items():
            st.write(f"**{key}:** {value:.3f}")
