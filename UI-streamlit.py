import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

# === Load input feature names from correct file ===
file_path = '3D-DB-Aug-2025.xlsx'  # Must match training file exactly
df_C_S = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
df_R = pd.read_excel(file_path, sheet_name=1, engine='openpyxl')
df_3D_L = pd.read_excel(file_path, sheet_name=2, engine="openpyxl")     # Layer
df_3D_S = pd.read_excel(file_path, sheet_name=3, engine="openpyxl")     # Strength

# Explicit input feature names (MODEL KEYS — do not change)
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

# Display labels (UI ONLY)
feature_labels = {
    # Core mix design (inputs used by models)
    "Cement content (%)": "Cement content (%, 40-100)",
    "Limestone content (%)": "Limestone content (%, 0-40)",
    "Silica fume content (%)": "Silica fume content (%, 0-30)",
    "SCM content (%)": "SCM content (%, 0-60)",
    "Nc": "Nc (alkalinity index)",
    "SSA of SCM (m2/g)": "SSA of SCM (m²/g, 0–14)",
    "Water/Binder": "Water-to-Binder ratio (0.2-0.5)",
    "Sand/Binder": "Sand-to-Binder ratio (0-2.5)",
    "Aggregate/Binder": "Coarse Aggregate-to-Binder ratio (0-0.5)",
    "Fiber length (mm)": "Fiber length (mm, 0–50)",
    "Fiber Volume (%)": "Fiber volume fraction (%, 0–2)",
    "Fiber Type": "Fiber type",
    "Age": "Age (days)",

    # Rheology-only input
    "Mini-slump after joint": "Mini-slump after joint (mm, 120–240)",

    # SCM composition helpers (you store these in user_input too)
    "CaO in SCM": "CaO in SCM (%, 0–65)",
    "Al2O3 in SCM": "Al₂O₃ in SCM (%, 5–35)",
    "SiO2 in SCM": "SiO₂ in SCM (%, 15-75)",

    # 3DP layer/strength extra params
    "Printing speed (mm/s)": "Printing Speed (mm/s, 20–50)",
    "nozzle size (mm)": "Nozzle Size (mm, 20–40)",
    "single layer height (mm)": "Single Layer Thickness (mm, 10–25)",
}

# Additional inputs for Nc calculation (MODEL KEYS)
extra_features = ['CaO in SCM', 'Al2O3 in SCM', 'SiO2 in SCM']

# Predefined SCM compositions: [SSA, CaO, Al2O3, SiO2]
# (Keep numeric values as you had; labels for SSA/CaO/Al2O3/SiO2 are handled via feature_labels)
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
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("MST-logo.png",  width=200)
with col3:
    st.image("ERDC-logo.jpg",  width=200)

# === Streamlit UI ===
st.title("3DP Concrete Property Predictor")

tab1, tab2 = st.tabs(["Prediction", "Optimization"])

# ----------------------
# Tab 1: Prediction
# ----------------------
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

        default_vals = scm_defaults[scm_choice] if use_default_scm else [0.0, 0.0, 0.0, 0.0]

        ssa_val = st.number_input(f"{feature_labels['SSA of SCM (m2/g)']}:", value=default_vals[0], step=0.01, format="%.2f")
        cao_val = st.number_input(f"{feature_labels['CaO in SCM']}:", value=default_vals[1], step=0.01, format="%.2f")
        al2o_val = st.number_input(f"{feature_labels['Al2O3 in SCM']}:", value=default_vals[2], step=0.01, format="%.2f")
        sio2_val = st.number_input(f"{feature_labels['SiO2 in SCM']}:", value=default_vals[3], step=0.01, format="%.2f")

        # MODEL KEYS stored in user_input
        user_input['SSA of SCM (m2/g)'] = ssa_val
        user_input['CaO in SCM'] = cao_val
        user_input['Al2O3 in SCM'] = al2o_val
        user_input['SiO2 in SCM'] = sio2_val

    # === Mix Design Inputs ===
    with st.expander("🔧 Mix Design Inputs"):
        for feature in compressive_features_list:
            # Skip fields handled elsewhere or auto-set
            if feature in ['Nc', 'SSA of SCM (m2/g)', 'Fiber Type', 'Fiber length (mm)', 'Fiber Volume (%)', 'Age']:
                continue
            label = feature_labels.get(feature, feature)
            user_input[feature] = st.number_input(f"{label}:", value=0.0, step=0.01, format="%.2f")

    # === Fiber Properties ===
    with st.expander("🧵 Fiber Properties"):
        fiber_label = st.selectbox(feature_labels['Fiber Type'] + ":", list(fiber_type_options.keys()))
        user_input['Fiber Type'] = fiber_type_options[fiber_label]

        user_input['Fiber length (mm)'] = st.number_input(f"{feature_labels['Fiber length (mm)']}:", value=0.0, step=0.01, format="%.2f")
        user_input['Fiber Volume (%)'] = st.number_input(f"{feature_labels['Fiber Volume (%)']}:", value=0.0, step=0.01, format="%.2f")

    # === Rheology Input ===
    with st.expander("📊 Rheology Input"):
        if 'Mini-slump after joint' not in user_input:
            user_input['Mini-slump after joint'] = st.number_input(f"{feature_labels['Mini-slump after joint']}:", value=0.0, step=0.01, format="%.2f")

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

    # === Predictions ===
    # Rheology
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

    # Compressive Strength (7 & 28 days)
    age_predictions = {}
    for age in [7, 28]:
        user_input['Age'] = age
        compressive_df = pd.DataFrame([{feature: user_input[feature] for feature in compressive_features_list}])
        compressive_strength = models['stacking_model_C'].predict(compressive_df)[0]
        age_predictions[age] = {"Compressive Strength (MPa)": compressive_strength}

    # --- Ranges for PASS/FAIL logic ---
    # Rheology thresholds (low, high) — choose set by fiber presence
    RANGES_WITH_FIBER = {
        "Water Retention": (None, 12),                # <= 12  (I-CAR)
        "Dynamic Yield Stress (Pa)": (583, 1066),     # 583–1066 (I-CAR)
        "Plastic Viscosity (Pa·s)": (2.5, 4.4),       # 2.5–4.4 (I-CAR)
        "Static Flocculation Stress (Pa)": (260, 744),# 260–744 (I-CAR)
        "Athix (Pa/min)": (8, 25),                    # 8–25  (I-CAR)
    }
    RANGES_NO_FIBER = {
        "Water Retention": (None, 8),                 # <= 8   (I-CAR)
        "Dynamic Yield Stress (Pa)": (583, 905),
        "Plastic Viscosity (Pa·s)": (4.6, 6.3),
        "Static Flocculation Stress (Pa)": (100, 422),
        "Athix (Pa/min)": (10, 20),
    }

    # Compressive strength thresholds (MPa) — choose set by fiber presence
    # (Values taken from your latest message; adjust as needed)
    COMPRESSIVE_WITH_FIBER = {
        7: 10,
        28: 25,
    }
    COMPRESSIVE_NO_FIBER = {
        7: 30,
        28: 40,
    }

    # === Predict button ===
    predict_clicked = st.button("Predict")
    if predict_clicked or st.session_state["predicted_main"]:
        if predict_clicked:
            st.session_state["predicted_main"] = True  # Set the flag

        def check_pass_fail(name, value, fiber_type_num: int):
            """Return ('PASS'|'Fail', html_badge) based on fiber_type-specific ranges."""
            ranges = RANGES_NO_FIBER if fiber_type_num == 0 else RANGES_WITH_FIBER
            lo, hi = ranges.get(name, (None, None))  # default: no bounds → Fail unless within both

            status = "Fail"
            color = "red"

            ok_lo = (lo is None) or (value >= lo)
            ok_hi = (hi is None) or (value <= hi)

            if ok_lo and ok_hi:
                status, color = "PASS", "green"

            display = (
                f"<b>{name}:</b> {value:.3f} "
                f"<span style='background-color:{color};color:white;padding:0.2em 0.5em;border-radius:0.3em;'>{status}</span>"
            )
            return status, display

        def check_strength_pass_fail(age: int, value: float, fiber_type_num: int):
            """Return PASS/Fail for compressive strength given fiber type."""
            thresholds = COMPRESSIVE_NO_FIBER if fiber_type_num == 0 else COMPRESSIVE_WITH_FIBER
            min_strength = thresholds.get(age, None)

            status = "Fail"
            color = "red"
            if min_strength is not None and value >= min_strength:
                status, color = "PASS", "green"

            display = (
                f"<b>{age} days:</b> {value:.3f} "
                f"<span style='background-color:{color};color:white;padding:0.2em 0.5em;border-radius:0.3em;'>{status}</span>"
            )
            return status, display

        # Check rheology results
        st.subheader("Predicted Rheological Properties with Status")
        all_pass = True
        fiber_type_num = user_input.get('Fiber Type', 0)  # 0 if missing
        for key, value in total_predictions.items():
            status, display = check_pass_fail(key, value, fiber_type_num)
            if status != "PASS":
                all_pass = False
            st.markdown(display, unsafe_allow_html=True)

        # Check compressive strength at 7 and 28 days
        st.markdown("##### Compressive Strength (MPa) with Pass/Fail")
        for age, results in age_predictions.items():
            for _, value in results.items():
                status, display = check_strength_pass_fail(age, value, fiber_type_num)
                if status != "PASS":
                    all_pass = False
                st.markdown(display, unsafe_allow_html=True)

        # If not all passed, show a warning
        if not all_pass:
            st.warning("⚠️ The predicted results did not pass all quality checks. 3DP Layer and 3DP Strength predictions may not be accurate.")

        # === 3DP Layer & Strength (optional section) ===
        if st.session_state["predicted_main"]:
            enable_3dp_layer = st.toggle("Enable 3DP Layer Prediction", key="layer_toggle")
            if enable_3dp_layer:
                with st.expander("3DP Printing Parameters"):
                    printing_speed = st.number_input(f"{feature_labels['Printing speed (mm/s)']}:", value=0.0, step=0.01, format="%.2f", key="layer_speed")
                    nozzle_size = st.number_input(f"{feature_labels['nozzle size (mm)']}:", value=0.0, step=0.01, format="%.2f", key="layer_nozzle")
                    single_layer_thickness = st.number_input(f"{feature_labels['single layer height (mm)']}:", value=0.0, step=0.01, format="%.2f", key="layer_thickness")

                if st.button("Predict 3DP Layers & Strength"):
                    # Prepare input for layer model (align strictly to model features)
                    expected_features = models["stacking_model_L"].feature_names_in_
                    layer_input = {key: user_input[key] for key in expected_features if key in user_input}

                    layer_input["Printing speed (mm/s)"] = printing_speed
                    layer_input["nozzle size (mm)"] = nozzle_size
                    layer_input["single layer height (mm)"] = single_layer_thickness

                    layer_df = pd.DataFrame([layer_input])
                    max_layers = models["stacking_model_L"].predict(layer_df)[0]
                    st.subheader("Predicted 3DP Maximum Printing Layers")
                    st.write(f"**Maximum Printing Layers:** {int(round(max_layers))}")

                    # 3DP Strength Prediction
                    strength_predictions = {}
                    expected_features_S = models["stacking_model_S"].feature_names_in_
                    for age in [7, 28]:
                        combined_input = user_input.copy()
                        combined_input["Age"] = age
                        combined_input["Printing speed (mm/s)"] = printing_speed
                        combined_input["nozzle size (mm)"] = nozzle_size
                        combined_input["single layer height (mm)"] = single_layer_thickness

                        strength_input = {key: combined_input[key] for key in expected_features_S if key in combined_input}
                        strength_df = pd.DataFrame([strength_input])
                        strength_predictions[age] = models["stacking_model_S"].predict(strength_df)[0]

                    st.subheader("Predicted 3DP Compressive Strength")
                    for age, value in strength_predictions.items():
                        st.write(f"**{age} days Strength (MPa):** {value:.2f}")

# ----------------------
# Tab 2: Optimization
# ----------------------
with tab2:
    st.subheader("Optimize Mix Design for Target Strength at 28 Days")

    opt_user_input = {}

    # === SCM Composition Section ===
    with st.expander("🧪 SCM Composition"):
        scm_choice = st.selectbox("Choose SCM Type:", list(scm_defaults.keys()), key="opt_scm_type")
        use_default_scm = st.checkbox("Default Composition", value=True, key="opt_use_default")

        default_vals = scm_defaults[scm_choice] if use_default_scm else [0.0, 0.0, 0.0, 0.0]

        ssa_val = st.number_input(f"{feature_labels['SSA of SCM (m2/g)']}:", value=default_vals[0], step=0.01, format="%.2f", key="opt_ssa")
        cao_val = st.number_input(f"{feature_labels['CaO in SCM']}:", value=default_vals[1], step=0.01, format="%.2f", key="opt_cao")
        al2o_val = st.number_input(f"{feature_labels['Al2O3 in SCM']}:", value=default_vals[2], step=0.01, format="%.2f", key="opt_al2o")
        sio2_val = st.number_input(f"{feature_labels['SiO2 in SCM']}:", value=default_vals[3], step=0.01, format="%.2f", key="opt_sio2")

        # MODEL KEYS stored
        opt_user_input['SSA of SCM (m2/g)'] = ssa_val
        opt_user_input['CaO in SCM'] = cao_val
        opt_user_input['Al2O3 in SCM'] = al2o_val
        opt_user_input['SiO2 in SCM'] = sio2_val

    # === Mix Design Inputs (excluding Cement, SCM, W/B, Nc, Age) ===
    with st.expander("🔧 Mix Design Inputs"):
        for feature in compressive_features_list:
            if feature in ['Cement content (%)', 'SCM content (%)', 'Water/Binder', 'Nc',
                           'SSA of SCM (m2/g)', 'Fiber Type', 'Fiber length (mm)', 'Fiber Volume (%)', 'Age']:
                continue
            label = feature_labels.get(feature, feature)
            opt_user_input[feature] = st.number_input(f"{label}:", value=0.0, step=0.01, format="%.2f", key=f"opt_{feature}")

    # === Fiber Properties ===
    with st.expander("🧵 Fiber Properties"):
        fiber_label = st.selectbox(feature_labels['Fiber Type'] + ":", list(fiber_type_options.keys()), key="opt_fiber_type")
        opt_user_input['Fiber Type'] = fiber_type_options[fiber_label]

        opt_user_input['Fiber length (mm)'] = st.number_input(f"{feature_labels['Fiber length (mm)']}:", value=0.0, step=0.01, format="%.2f", key="opt_fiber_len")
        opt_user_input['Fiber Volume (%)'] = st.number_input(f"{feature_labels['Fiber Volume (%)']}:", value=0.0, step=0.01, format="%.2f", key="opt_fiber_vol")

    # === Target Strength Input ===
    target_strength = st.number_input("Target Compressive Strength (MPa, 20–60):", value=50.0, step=0.1, key="opt_target_strength")

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
                        "SCM content (%)": int(scm),
                        "Water/Binder": wb,
                        "Predicted Strength (MPa)": predicted_strength
                    })

        if results_list:
            st.success(f"Found {len(results_list)} valid combinations!")
            df_results = pd.DataFrame(results_list)
            st.dataframe(df_results)
        else:
            st.warning("No combination met the target strength.")
