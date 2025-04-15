import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import cloudpickle

# === Load Excel Data ===
file_path = '3D_DB_Feb2025.xlsx'  # Make sure this file is in the same folder
df_C_S = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
df_R = pd.read_excel(file_path, sheet_name=1, engine='openpyxl')

# === Preprocess Data ===
def preprocess_data(df, target_column):
    return df.dropna(subset=[target_column])

df_C_S_cleaned = preprocess_data(df_C_S, df_C_S.columns[-1])
X_train_C = df_C_S_cleaned.iloc[:, :-1]
y_train_C = df_C_S_cleaned.iloc[:, -1]
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_train_C, y_train_C, test_size=0.1, random_state=42)

df_R_cleaned = df_R.dropna()
X_rheo = df_R_cleaned.iloc[:, :12]
y_R1 = df_R_cleaned.iloc[:, 12]
y_R2 = df_R_cleaned.iloc[:, 13]
y_R3 = df_R_cleaned.iloc[:, 14]
y_R4 = df_R_cleaned.iloc[:, 15]
y_R5 = df_R_cleaned.iloc[:, 16]

# === Base Models ===
def get_base_models(lr):
    return [
        ('rf', RandomForestRegressor(n_estimators=250, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=250, learning_rate=lr, random_state=42)),
        ('lr', LinearRegression())
    ]

# === Loop over Learning Rates ===
learning_rates = [0.01, 0.05, 0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19, 0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50]

# Track best models and results
results = {
    'C': {'model': None, 'mae': float('inf'), 'r2': None, 'lr': None},
    'R1': {'model': None, 'mae': float('inf'), 'r2': None, 'lr': None},
    'R2': {'model': None, 'mae': float('inf'), 'r2': None, 'lr': None},
    'R3': {'model': None, 'mae': float('inf'), 'r2': None, 'lr': None},
    'R4': {'model': None, 'mae': float('inf'), 'r2': None, 'lr': None},
    'R5': {'model': None, 'mae': float('inf'), 'r2': None, 'lr': None},
}

for lr in learning_rates:
    print(f"\n🔁 Training with learning rate: {lr}")

    # Compressive Strength - Stacking
    base_models = get_base_models(lr)
    model_C = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
    model_C.fit(X_train_C, y_train_C)
    pred_C = model_C.predict(X_test_C)
    mae_C = mean_absolute_error(y_test_C, pred_C)
    r2_C = r2_score(y_test_C, pred_C)

    if mae_C < results['C']['mae']:
        results['C'].update({'model': model_C, 'mae': mae_C, 'r2': r2_C, 'lr': lr})

    # Rheology models - Gradient Boosting
    for label, y in zip(['R1', 'R2', 'R3', 'R4', 'R5'], [y_R1, y_R2, y_R3, y_R4, y_R5]):
        model = GradientBoostingRegressor(n_estimators=250, learning_rate=lr, random_state=42)
        model.fit(X_rheo, y)
        pred = model.predict(X_rheo)
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)

        if mae < results[label]['mae']:
            results[label].update({'model': model, 'mae': mae, 'r2': r2, 'lr': lr})

# === Report Results ===
print("\n✅ Best Models Summary:")
for label, res in results.items():
    print(f"\n🔹 Model {label}:")
    print(f"   • Best Learning Rate: {res['lr']}")
    print(f"   • MAE: {res['mae']:.4f}")
    print(f"   • R²: {res['r2']:.4f}")

# === Save Best Models ===
models_to_save = {
    'stacking_model_C': results['C']['model'],
    'stacking_model_R1': results['R1']['model'],
    'stacking_model_R2': results['R2']['model'],
    'stacking_model_R3': results['R3']['model'],
    'stacking_model_R4': results['R4']['model'],
    'stacking_model_R5': results['R5']['model'],
}

with open("3DP_Apr2025.pkl", "wb") as f:
    cloudpickle.dump(models_to_save, f)

print("\n📦 Models saved to 3DP_Apr2025.pkl successfully.")
