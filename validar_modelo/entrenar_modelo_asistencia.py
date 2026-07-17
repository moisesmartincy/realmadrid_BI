import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import json
import os

# =============================================================================
# ENTRENAMIENTO DEL MODELO - PREDICCION DE ASISTENCIA AL ESTADIO
# =============================================================================
# Modelo: XGBoost Regressor
# Input: Variables de contexto (clima, competicion, dia, rival, etc.)
# Output: Predicción de espectadores (asistencia total)
# =============================================================================


def cargar_y_preparar_datos(csv_path='ftr_asistencia_historica.csv'):
    print("[1/6] Cargando dataset...")
    df = pd.read_csv(csv_path)
    print(f"       Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    columnas_excluir = ['asistencia']
    columnas_features = [c for c in df.columns if c not in columnas_excluir]

    X = df[columnas_features].copy()
    y = df['asistencia'].copy()

    print(f"       Features: {len(columnas_features)} columnas")
    print(f"       Target: 'asistencia' (min={y.min()}, max={y.max()}, media={int(y.mean())})")

    return X, y, columnas_features


def preprocesar_features(X, encoders=None, fit=True):
    print("[2/6] Preprocesando features...")

    if encoders is None:
        encoders = {}

    X_procesado = X.copy()
    columnas_categoricas = ['competicion', 'mes', 'dia_semana', 'hora_partido', 'clima']

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


def entrenar_modelo(X_train, y_train, X_test, y_test):
    print("[3/6] Entrenando modelo XGBoost Regressor...")

    modelo = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=1.2,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0,
    )

    modelo.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    print(f"       Modelo entrenado con {modelo.n_estimators} arboles.")
    return modelo


def evaluar_modelo(modelo, X_train, y_train, X_test, y_test):
    print("\n[4/6] Evaluando modelo...")

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print("\n" + "=" * 60)
    print("RESULTADOS DEL MODELO")
    print("=" * 60)

    print(f"\n[METRICAS]")
    print(f"                  {'Train':>10}  {'Test':>10}")
    print(f"   MAE (personas):{int(mae_train):>10}  {int(mae_test):>10}")
    print(f"   RMSE (personas):{int(rmse_train):>10}  {int(rmse_test):>10}")
    print(f"   R2 Score:      {r2_train:>10.4f}  {r2_test:>10.4f}")

    if r2_train - r2_test > 0.10:
        print("   [!] ATENCION: Posible overfitting")
    else:
        print("   [OK] No hay overfitting significativo")

    # Top 15 features
    print(f"\n[TOP FEATURES MAS IMPORTANTES]")
    importances = modelo.feature_importances_
    feature_names = modelo.feature_names_in_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for fname, fimp in feat_imp[:12]:
        bar = "#" * int(fimp * 100)
        print(f"   {fname:>20}: {fimp:.4f} {bar}")

    # Ejemplos (Test Set)
    print(f"\n[EJEMPLOS PREDICCION vs REAL] (10 aleatorios)")
    indices = np.random.choice(len(y_test), 10, replace=False)
    for idx in indices:
        real = y_test.iloc[idx]
        pred = int(y_pred_test[idx])
        diff = abs(real - pred)
        print(f"   Real: {real:>6} | Predicho: {pred:>6} | Error: {diff:>5}")

    return r2_test, mae_test


def guardar_modelo(modelo, encoders, columnas_features, ruta='modelo_asistencia_entrenado'):
    print(f"\n[5/6] Guardando modelo en '{ruta}/'...")
    os.makedirs(ruta, exist_ok=True)

    joblib.dump(modelo, os.path.join(ruta, 'xgboost_regressor.pkl'))
    joblib.dump(encoders, os.path.join(ruta, 'feature_encoders.pkl'))

    with open(os.path.join(ruta, 'columnas_features.json'), 'w') as f:
        json.dump(columnas_features, f, indent=2)

    metadata = {
        'modelo': 'XGBoost Regressor',
        'tipo': 'regresion',
        'target': 'asistencia',
        'n_features': len(columnas_features),
        'n_estimators': modelo.n_estimators,
        'columnas': columnas_features,
    }
    with open(os.path.join(ruta, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"       Archivos guardados:")
    for f in os.listdir(ruta):
        size = os.path.getsize(os.path.join(ruta, f))
        print(f"         - {f} ({size / 1024:.1f} KB)")


def validacion_cruzada(X, y):
    print("\n[6/6] Validacion cruzada (5-fold)...")

    modelo_cv = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, verbosity=0,
    )

    scores_r2 = cross_val_score(modelo_cv, X, y, cv=5, scoring='r2')
    print(f"       R2 Medio: {scores_r2.mean():.4f} (+/- {scores_r2.std():.4f})")

    scores_mae = cross_val_score(modelo_cv, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"       MAE Medio: {-scores_mae.mean():.0f} personas de error promedio")


if __name__ == '__main__':
    X, y, columnas_features = cargar_y_preparar_datos('ftr_asistencia_historica.csv')
    X_procesado, encoders = preprocesar_features(X, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_procesado, y, test_size=0.20, random_state=42
    )
    
    modelo = entrenar_modelo(X_train, y_train, X_test, y_test)
    r2, mae = evaluar_modelo(modelo, X_train, y_train, X_test, y_test)
    guardar_modelo(modelo, encoders, columnas_features, ruta='modelo_asistencia_entrenado')
    validacion_cruzada(X_procesado, y)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"R2 Score: {r2:.4f}")
    print(f"Error MAE: {int(mae)} espectadores")
    print(f"Modelo guardado en: modelo_asistencia_entrenado/")
    print(f"Para predecir: python predecir_asistencia.py")
