import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def entrenar_random_forest():
    print("Cargando Dataset V2...")
    df = pd.read_csv('../pulse/bernabeu_pulse_dataset_v2.csv')
    
    # Feature Engineering (One-Hot Encoding del tipo de zona)
    df_modelo = pd.get_dummies(df, columns=['tipo_zona'], drop_first=False)
    
    # Las features de entrada que el Director indicará en Streamlit
    features = [
        'minutos_al_kickoff', 'aforo_stadium', 'clima_lluvia', 'nivel_rivalidad', 
        'factor_estrella', 'id_puerta', 'tipo_zona_Seguridad_Acceso', 
        'tipo_zona_Retail_Tienda', 'tipo_zona_VIP_Catering', 'tipo_zona_Gradas_Tornos'
    ]
    
    X = df_modelo[features]
    y = df_modelo['tasa_llegada_lambda'] # Target 1
    y_staff = df_modelo['personal_necesario_ideal'] # Target 2 (El que pide el jefe)
    
    print("Entrenando RandomForestRegressor (Staff Ideal)...")
    rf_staff = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_staff.fit(X, y_staff)
    
    y_pred_staff = rf_staff.predict(X)
    r2_staff = r2_score(y_staff, y_pred_staff)
    mae_staff = mean_absolute_error(y_staff, y_pred_staff)
    
    print("Entrenando RandomForestRegressor (Tasa Llegada)...")
    rf_lambda = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_lambda.fit(X, y)
    
    print(f"Métricas Modelo Staff => R2: {r2_staff:.4f} | Error Absoluto MAE: {mae_staff:.2f} empleados")
    
    import os
    os.makedirs('../webapp/modelo', exist_ok=True)
    joblib.dump(rf_staff, '../webapp/modelo/rf_staff_model.pkl')
    joblib.dump(rf_lambda, '../webapp/modelo/rf_lambda_model.pkl')
    
    # Guardo las features esperadas para Streamlit
    joblib.dump(features, '../webapp/modelo/model_features.pkl')
    
    print("✅ Modelos exportados a disk exitosamente (.pkl)")

if __name__ == "__main__":
    entrenar_random_forest()
