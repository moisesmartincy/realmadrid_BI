import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import json
import os

# =============================================================================
# ENTRENAMIENTO DEL MODELO - PREDICCION DE TOP PRODUCTOS
# =============================================================================
# Usa un MultiOutputRegressor para predecir simultáneamente la demanda (0-100)
# de 7 categorías de productos diferentes basándose en el contexto.
# =============================================================================

def cargar_y_preparar_datos(csv_path='ftr_top_merchandising.csv'):
    print("[1/6] Cargando dataset...")
    df = pd.read_csv(csv_path)
    print(f"       Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    columnas_targets = [
        'demanda_bebida_fria', 'demanda_bebida_caliente', 'demanda_comida',
        'demanda_camisetas', 'demanda_bufandas', 'demanda_chubasqueros',
        'demanda_conmemorativo'
    ]
    columnas_features = [c for c in df.columns if c not in columnas_targets]

    X = df[columnas_features].copy()
    y = df[columnas_targets].copy()

    print(f"       Features: {len(columnas_features)} columnas")
    print(f"       Targets Multinivel: {len(columnas_targets)} productos distintos")

    return X, y, columnas_features, columnas_targets


def preprocesar_features(X, encoders=None, fit=True):
    print("[2/6] Preprocesando features...")

    if encoders is None:
        encoders = {}

    X_procesado = X.copy()
    columnas_categoricas = ['competicion', 'mes', 'horario']

    for col in columnas_categoricas:
        if col in X_procesado.columns:
            if fit:
                le = LabelEncoder()
                X_procesado[col] = le.fit_transform(X_procesado[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                X_procesado[col] = X_procesado[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    print(f"       Columnas codificadas: {columnas_categoricas}")
    return X_procesado, encoders


def entrenar_modelo(X_train, y_train):
    print("[3/6] Entrenando modelo MultiOutput (XGBoost)...")

    estimador_base = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0
    )

    modelo = MultiOutputRegressor(estimador_base)
    
    # Entrenar el ensamble (esto entrena un modelo independiente para cada producto, pero unificado)
    modelo.fit(X_train, y_train)

    print(f"       MultiOutputRegressor entrenado con éxito.")
    return modelo


def evaluar_modelo(modelo, X_train, y_train, X_test, y_test, columnas_targets):
    print("\n[4/6] Evaluando modelo...")

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # Convertimos a DataFrame para fácil manejo
    df_pred_test = pd.DataFrame(y_pred_test, columns=columnas_targets)

    print("\n" + "=" * 60)
    print("RESULTADOS DEL MODELO (SCORE GLOBAL)")
    print("=" * 60)

    # Métricas Globales
    mae_global = mean_absolute_error(y_test, y_pred_test)
    r2_global = r2_score(y_test, y_pred_test)
    
    print(f"\n[GLOBAL]")
    print(f"   MAE: {mae_global:.2f} puntos de demanda (0-100)")
    print(f"   R2 Score: {r2_global:.4f}")

    # Métricas Específicas por Producto
    print(f"\n[PRECISION POR PRODUCTO (R2 Score)]")
    for i, col in enumerate(columnas_targets):
        r2_prod = r2_score(y_test.iloc[:, i], y_pred_test[:, i])
        mae_prod = mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i])
        print(f"   - {col:25}: R2={r2_prod:.4f}  |  MAE={mae_prod:.2f}")

    return r2_global, mae_global


def guardar_modelo(modelo, encoders, columnas_features, columnas_targets, ruta='modelo_top_productos_entrenado'):
    print(f"\n[5/6] Guardando modelo en '{ruta}/'...")
    os.makedirs(ruta, exist_ok=True)

    joblib.dump(modelo, os.path.join(ruta, 'multioutput_xgboost.pkl'))
    joblib.dump(encoders, os.path.join(ruta, 'feature_encoders.pkl'))

    with open(os.path.join(ruta, 'columnas_features.json'), 'w') as f:
        json.dump(columnas_features, f, indent=2)
        
    with open(os.path.join(ruta, 'columnas_targets.json'), 'w') as f:
        json.dump(columnas_targets, f, indent=2)

    metadata = {
        'modelo': 'MultiOutputRegressor (XGBoost)',
        'tipo': 'regresion_multivariable',
        'target': 'lista_demanda_productos (0-100)',
        'n_features': len(columnas_features),
        'n_targets': len(columnas_targets),
        'columnas_features': columnas_features,
        'columnas_targets': columnas_targets
    }
    with open(os.path.join(ruta, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"       Archivos guardados correctamente.")


if __name__ == '__main__':
    X, y, columnas_features, columnas_targets = cargar_y_preparar_datos('ftr_top_merchandising.csv')
    X_procesado, encoders = preprocesar_features(X, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_procesado, y, test_size=0.20, random_state=42
    )
    
    modelo = entrenar_modelo(X_train, y_train)
    r2, mae = evaluar_modelo(modelo, X_train, y_train, X_test, y_test, columnas_targets)
    guardar_modelo(modelo, encoders, columnas_features, columnas_targets, ruta='modelo_top_productos_entrenado')

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Modelo Multiple guardado en: modelo_top_productos_entrenado/")
    print(f"Para predecir el top de productos: python predecir_top_productos.py")
