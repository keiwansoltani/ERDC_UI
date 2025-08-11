import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle


# === Load input feature names from correct file ===
file_path = '3D-DB-July-2025.xlsx'  # Must match training file exactly
df_C_S = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
df_R = pd.read_excel(file_path, sheet_name=1, engine='openpyxl')
df_3D_L = pd.read_excel(file_path,sheet_name=2, engine="openpyxl")     # Layer
df_3D_S = pd.read_excel(file_path,sheet_name=3, engine="openpyxl")     # Strength

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
    "Glass powder": [0.233, 12.54, 1.16, 74.8],
    "None": [0.0, 0.0, 0.0, 0.0]
}

# === Load the correct trained model ===
with open('3DP_August_2025.pkl', 'rb') as f:
    models = cloudpickle.load(f)

# Initialize session state for main prediction trigger
if "predicted_main" not in st.session_state:
    st.session_state["predicted_main"] = False

# MST logo and ERDC logo

col1, col2, col3 ,colm4= st.columns([1, 4, 1,1])
with col1:
    st.image("MST-logo.png",  width=220)
with col3:
    st.image("ERDC-logo.jpg",  width=220)


# === Streamlit UI ===
st.title("3DP Concrete Property Predictor")

tab1, tab2 = st.tabs(["Prediction", "Optimization"])
with tab1:
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


    # Prediction of the rheology properties
    rheology_df = pd.DataFrame([{feature: user_input[feature] for feature in rheology_features_list}])

    water_retention = models['stacking_model_R1'].predict(rheology_df)[0]
    dynamic_yield_stress = models['stacking_model_R2'].predict(rheology_df)[0]
    plastic_viscosity = models['stacking_model_R3'].predict(rheology_df)[0]
    static_floc_stress = models['stacking_model_R4'].predict(rheology_df)[0]
    athix = models['stacking_model_R5'].predict(rheology_df)[0]

    total_predictions = {
        "Water Retention": water_retention,
        "Dynamic Yield Stress (Pa)": dynamic_yield_stress,
        "Plastic Viscosity (Pa·s)": plastic_viscosity,
        "Static Flocculation Stress (Pa)": static_floc_stress,
        "Athix (Pa/min)": athix
        }
    # Predict of compressive Strength for both ages 7 and 28
    age_predictions = {}
    for age in [7, 28]:
        user_input['Age'] = age
        compressive_df = pd.DataFrame([{feature: user_input[feature] for feature in compressive_features_list}])
        compressive_strength = models['stacking_model_C'].predict(compressive_df)[0]
        age_predictions[age] = {
            "Compressive Strength (MPa)": compressive_strength
        }
    # === Predict button ===
    predict_clicked = st.button("Predict")
    if predict_clicked or st.session_state["predicted_main"]:
        if predict_clicked:
            st.session_state["predicted_main"] = True  # Set the flag
        def check_pass_fail(name, value):
            status = "Fail"
            color = "red"

            if name == "Water Retention":
                if value <= 12:
                    status, color = "PASS", "green"
            elif name == "Dynamic Yield Stress (Pa)":
                if 400 <= value <= 700:
                    status, color = "PASS", "green"
            elif name == "Plastic Viscosity (Pa·s)":
                if 3 <= value <= 14:
                    status, color = "PASS", "green"
            elif name == "Static Flocculation Stress (Pa)":
                if 200 <= value <= 500:
                    status, color = "PASS", "green"
            elif name == "Athix (Pa/min)":
                if 8 <= value <= 25:
                    status, color = "PASS", "green"
            display =  f"<b>{name}:</b> {value:.3f} <span style='background-color:{color};color:white;padding:0.2em 0.5em;border-radius:0.3em;'>{status}</span>"
            return status, display
        
        # Check rheology results
        st.subheader("Predicted Rheological Properties with Status")
        all_pass = True
        for key, value in total_predictions.items():
            status, display = check_pass_fail(key, value)
            if status != "PASS":
                all_pass = False
            st.markdown(display, unsafe_allow_html=True)

        # Check compressive strength at 7 and 28 days
        st.markdown("##### Compressive Strength (MPa) with Pass/Fail")
        for age, results in age_predictions.items():
            for key, value in results.items():
                status = "Fail"
                color = "red"
                if age == 1 and value >= 10:
                    status, color = "PASS", "green"
                elif age == 7 and value >=20:
                    status, color = "PASS", "green"
                elif age == 28 and value >= 25:
                    status, color = "PASS", "green"
                else:
                    all_pass = False
                st.markdown(f"<b>{age} days:</b> {value:.3f} <span style='background-color:{color};color:white;padding:0.2em 0.5em;border-radius:0.3em;'>{status}</span>", unsafe_allow_html=True)        # # Rheology properties 
        
        # If not all passed, show a warning
        if not all_pass:
            st.warning("⚠️ The predicted results did not pass all quality checks. 3DP Layer and 3DP Strength predictions may not be accurate.")
        
        if st.session_state["predicted_main"]:
            # === 3DP Layer Prediction Toggle and Section ===
            enable_3dp_layer = st.toggle("Enable 3DP Layer Prediction", key="layer_toggle")
            if enable_3dp_layer:
                with st.expander(" 3DP Printing Parameters"):
                    printing_speed = st.number_input("Printing Speed (mm/s):", value=0.0, step=0.01, format="%.3f", key="layer_speed")
                    nozzle_size = st.number_input("Nozzle Size (mm):", value=0.0, step=0.01, format="%.3f", key="layer_nozzle")
                    single_layer_thickness = st.number_input("Single Layer Thickness (mm):", value=0.0, step=0.01, format="%.3f", key="layer_thickness")
                
                if st.button("Predict 3DP Layers & Strength"):
                # Prepare input for layer model
                    # Automatically filter features that match model expectations
                    expected_features = models["stacking_model_L"].feature_names_in_
                    layer_input = {key: user_input[key] for key in expected_features if key in user_input}

                    layer_input["Printing speed (mm/s)"] = printing_speed
                    layer_input["nozzle size (mm)"] = nozzle_size
                    layer_input["single layer height (mm)"] = single_layer_thickness
                    layer_df = pd.DataFrame([layer_input])
                    max_layers = models["stacking_model_L"].predict(layer_df)[0]
                    st.subheader("Predicted 3DP Maximum Printing Layers")
                    st.write(f"**Maximum Printing Layers:** {int(round(max_layers))}")
                    
                    # === 3DP Strength Prediction Toggle and Section ===
                    strength_predictions = {}
                    expected_features_S = models["stacking_model_S"].feature_names_in_
                    for age in [7, 28]:
                        # Build input from user_input and manually entered parameters
                        combined_input = user_input.copy()
                        combined_input["Age"] = age
                        combined_input["Printing speed (mm/s)"] = printing_speed
                        combined_input["nozzle size (mm)"] = nozzle_size
                        combined_input["single layer height (mm)"] = single_layer_thickness

                        # Filter only the features expected by the model
                        strength_input = {key: combined_input[key] for key in expected_features_S if key in combined_input}
                        strength_df = pd.DataFrame([strength_input])
                        strength_predictions[age] = models["stacking_model_S"].predict(strength_df)[0]
                    # Display results
                    st.subheader("Predicted 3DP Compressive Strength")
                    for age, value in strength_predictions.items():
                        st.write(f"**{age} days Strength (MPa):** {value:.2f}")

with tab2:
    st.subheader("Optimize Mix Design for Target Strength at 28 Days")

    opt_user_input = {}

    # === SCM Composition Section ===
    with st.expander("🧪 SCM Composition"):
        scm_choice = st.selectbox("Choose SCM Type:", list(scm_defaults.keys()), key="opt_scm_type")
        use_default_scm = st.checkbox("Default Composition", value=True, key="opt_use_default")

        if use_default_scm:
            default_vals = scm_defaults[scm_choice]
        else:
            default_vals = [0.0, 0.0, 0.0, 0.0]

        ssa_val = st.number_input("SSA of SCM (m2/g):", value=default_vals[0], step=0.01, format="%.6f", key="opt_ssa")
        cao_val = st.number_input("CaO in SCM:", value=default_vals[1], step=0.01, format="%.6f", key="opt_cao")
        al2o_val = st.number_input("Al2O3 in SCM:", value=default_vals[2], step=0.01, format="%.6f", key="opt_al2o")
        sio2_val = st.number_input("SiO2 in SCM:", value=default_vals[3], step=0.01, format="%.6f", key="opt_sio2")

        opt_user_input['SSA of SCM (m2/g)'] = ssa_val
        opt_user_input['CaO in SCM'] = cao_val
        opt_user_input['Al2O3 in SCM'] = al2o_val
        opt_user_input['SiO2 in SCM'] = sio2_val

    # === Mix Design Inputs (excluding Cement, SCM, W/B, Nc, Age) ===
    with st.expander("🔧 Mix Design Inputs"):
        for feature in compressive_features_list:
            if feature in ['Cement content (%)', 'SCM content (%)', 'Water/Binder', 'Nc', 'SSA of SCM (m2/g)', 'Fiber Type', 'Fiber length (mm)', 'Fiber Volume (%)', 'Age']:
                continue
            opt_user_input[feature] = st.number_input(f"{feature}:", value=0.0, step=0.01, format="%.6f", key=f"opt_{feature}")

    # === Fiber Properties ===
    with st.expander("🧵 Fiber Properties"):
        fiber_label = st.selectbox("Fiber Type:", list(fiber_type_options.keys()), key="opt_fiber_type")
        opt_user_input['Fiber Type'] = fiber_type_options[fiber_label]

        opt_user_input['Fiber length (mm)'] = st.number_input("Fiber length (mm):", value=0.0, step=0.01, format="%.6f", key="opt_fiber_len")
        opt_user_input['Fiber Volume (%)'] = st.number_input("Fiber Volume (%):", value=0.0, step=0.01, format="%.6f", key="opt_fiber_vol")

    # === Target Strength Input ===
    target_strength = st.number_input("Target Compressive Strength (MPa):", value=50.0, step=0.1, key="opt_target_strength")

    # === Compute Nc ===
    CAO = opt_user_input.pop('CaO in SCM')
    Al2O = opt_user_input.pop('Al2O3 in SCM')
    SiO2 = opt_user_input.pop('SiO2 in SCM')

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

    opt_user_input['Nc'] = Nc
    opt_user_input['CaO in SCM'] = CAO
    opt_user_input['Al2O3 in SCM'] = Al2O
    opt_user_input['SiO2 in SCM'] = SiO2

    # === Optimization Button & Logic ===
    results_list = []

    if st.button("Start Optimization"):
        #st.write("Running optimization...")

        limestone = opt_user_input['Limestone content (%)']
        silica_fume = opt_user_input['Silica fume content (%)']
        K = 100 - limestone - silica_fume

        for wb in np.arange(0.30, 0.51, 0.01):
            for cement_pct in np.arange(0.5, 1.01, 0.05):  # 50% to 100%
                cement = cement_pct * K
                scm = K - cement

                current_input = opt_user_input.copy()
                current_input['Cement content (%)'] = cement
                current_input['SCM content (%)'] = scm
                current_input['Water/Binder'] = wb
                current_input['Age'] = 28

                df_input = pd.DataFrame([{feature: current_input[feature] for feature in compressive_features_list}])
                predicted_strength = models['stacking_model_C'].predict(df_input)[0]

                if predicted_strength >= target_strength:
                    results_list.append({
                        "Cement content (%)": cement,
                        "SCM content (%)": scm,
                        "Water/Binder": wb,
                        "Predicted Strength (MPa)": predicted_strength
                    })

        if results_list:
            st.success(f"Found {len(results_list)} valid combinations!")
            df_results = pd.DataFrame(results_list)
            st.dataframe(df_results)
        else:
            st.warning("No combination met the target strength.")
