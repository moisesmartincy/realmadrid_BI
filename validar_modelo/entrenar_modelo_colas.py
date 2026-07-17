import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def entrenar_modelo_ligero():
    print("Cargando Dataset V2...")
    df = pd.read_csv('../pulse/bernabeu_pulse_dataset_v2.csv')
    
    df_modelo = pd.get_dummies(df, columns=['tipo_zona'], drop_first=False)
    
    features = [
        'minutos_al_kickoff', 'aforo_stadium', 'clima_lluvia', 'nivel_rivalidad', 
        'factor_estrella', 'id_puerta', 'tipo_zona_Seguridad_Acceso', 
        'tipo_zona_Retail_Tienda', 'tipo_zona_VIP_Catering', 'tipo_zona_Gradas_Tornos'
    ]
    
    X = df_modelo[features]
    y_lambda = df_modelo['tasa_llegada_lambda']
    y_staff = df_modelo['personal_necesario_ideal']
    
    # Usamos Decision Tree en vez de Random Forest/XGBoost para EVITAR el crash de memoria en Windows
    print("Entrenando Modelo ML de Árbol de Decisión (Bajo Consumo de Memoria)...")
    dt_staff = DecisionTreeRegressor(max_depth=12, random_state=42)
    dt_staff.fit(X, y_staff)
    
    dt_lambda = DecisionTreeRegressor(max_depth=12, random_state=42)
    dt_lambda.fit(X, y_lambda)
    
    r2_staff = r2_score(y_staff, dt_staff.predict(X))
    mae_staff = mean_absolute_error(y_staff, dt_staff.predict(X))
    
    print(f"Métricas Modelo Operativo => R2: {r2_staff:.4f} | Error Absoluto MAE: {mae_staff:.2f} empleados")
    
    # Exportar sin fallos
    import os
    os.makedirs('../webapp/modelo', exist_ok=True)
    joblib.dump(dt_staff, '../webapp/modelo/modelo_staff_ml.pkl', compress=3)
    joblib.dump(dt_lambda, '../webapp/modelo/modelo_lambda_ml.pkl', compress=3)
    
    print("✅ Modelos ML entrenados y exportados a tu disco (.pkl) exitosamente.")

if __name__ == "__main__":
    entrenar_modelo_ligero()
