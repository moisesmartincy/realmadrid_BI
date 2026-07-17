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
# ENTRENAMIENTO DEL MODELO - PREDICCION DE VALOR DE MERCADO DE JUGADORES
# =============================================================================
# Modelo: XGBoost Regressor (prediccion de valor continuo en M euros)
# Input: Stats del jugador (goles, asistencias, xG, edad, liga, etc.)
# Output: Valor de mercado estimado en millones de euros
# =============================================================================


def cargar_y_preparar_datos(csv_path='ftr_valoracion_jugadores.csv'):
    """Carga el CSV y prepara features + target."""

    print("[1/6] Cargando dataset...")
    df = pd.read_csv(csv_path)
    print(f"       Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Target = valor_mercado
    columnas_excluir = ['valor_mercado']
    columnas_features = [c for c in df.columns if c not in columnas_excluir]

    X = df[columnas_features].copy()
    y = df['valor_mercado'].copy()

    print(f"       Features: {len(columnas_features)} columnas")
    print(f"       Target: 'valor_mercado' (min={y.min():.1f}M, max={y.max():.1f}M, media={y.mean():.1f}M)")

    return X, y, columnas_features


def preprocesar_features(X, encoders=None, fit=True):
    """Convierte variables categoricas a numericas."""

    print("[2/6] Preprocesando features...")

    if encoders is None:
        encoders = {}

    X_procesado = X.copy()

    columnas_categoricas = ['posicion', 'liga', 'nacionalidad', 'pie_dominante']

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
    """Entrena XGBoost Regressor optimizado para prediccion de valor."""

    print("[3/6] Entrenando modelo XGBoost Regressor...")

    modelo = XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.5,
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
    """Evalua el modelo con metricas de regresion."""

    print("\n[4/6] Evaluando modelo...")

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    print("\n" + "=" * 60)
    print("RESULTADOS DEL MODELO")
    print("=" * 60)

    # --- Metricas Train ---
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    # --- Metricas Test ---
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\n[METRICAS]")
    print(f"                  {'Train':>10}  {'Test':>10}")
    print(f"   MAE (M euros): {mae_train:>10.2f}  {mae_test:>10.2f}")
    print(f"   RMSE (M euros):{rmse_train:>10.2f}  {rmse_test:>10.2f}")
    print(f"   R2 Score:      {r2_train:>10.4f}  {r2_test:>10.4f}")

    # Overfitting check
    if r2_train - r2_test > 0.10:
        print("   [!] ATENCION: Posible overfitting")
    else:
        print("   [OK] No hay overfitting significativo")

    # --- Analisis de errores por rango de valor ---
    print(f"\n[ERROR POR RANGO DE VALOR]")
    rangos = [(0, 5), (5, 20), (20, 50), (50, 100), (100, 300)]
    for low, high in rangos:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() > 0:
            mae_rango = mean_absolute_error(y_test[mask], y_pred_test[mask])
            n = mask.sum()
            print(f"   {low:>3}-{high:>3}M: MAE={mae_rango:.2f}M ({n} jugadores)")

    # --- Feature Importance (top 15) ---
    print(f"\n[TOP 15 FEATURES MAS IMPORTANTES]")
    importances = modelo.feature_importances_
    feature_names = modelo.feature_names_in_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for fname, fimp in feat_imp[:15]:
        bar = "#" * int(fimp * 100)
        print(f"   {fname:>25}: {fimp:.4f} {bar}")

    # --- Ejemplos de prediccion vs real ---
    print(f"\n[EJEMPLOS PREDICCION vs REAL] (10 aleatorios del test)")
    indices = np.random.choice(len(y_test), 10, replace=False)
    for idx in indices:
        real = y_test.iloc[idx]
        pred = y_pred_test[idx]
        diff = abs(real - pred)
        print(f"   Real: {real:>7.1f}M | Predicho: {pred:>7.1f}M | Error: {diff:>5.1f}M")

    return r2_test, mae_test


def guardar_modelo(modelo, encoders, columnas_features, ruta='modelo_jugador_entrenado'):
    """Guarda el modelo y artefactos."""

    print(f"\n[5/6] Guardando modelo en '{ruta}/'...")
    os.makedirs(ruta, exist_ok=True)

    joblib.dump(modelo, os.path.join(ruta, 'xgboost_regressor.pkl'))
    joblib.dump(encoders, os.path.join(ruta, 'feature_encoders.pkl'))

    with open(os.path.join(ruta, 'columnas_features.json'), 'w') as f:
        json.dump(columnas_features, f, indent=2)

    metadata = {
        'modelo': 'XGBoost Regressor',
        'tipo': 'regresion',
        'target': 'valor_mercado (millones de euros)',
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
    """Validacion cruzada 5-fold."""

    print("\n[6/6] Validacion cruzada (5-fold)...")

    modelo_cv = XGBRegressor(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, verbosity=0,
    )

    # R2 scores
    scores_r2 = cross_val_score(modelo_cv, X, y, cv=5, scoring='r2')
    print(f"       R2 por fold: {[f'{s:.4f}' for s in scores_r2]}")
    print(f"       R2 Media: {scores_r2.mean():.4f} (+/- {scores_r2.std():.4f})")

    # MAE scores (negativo por convencion sklearn)
    scores_mae = cross_val_score(modelo_cv, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"       MAE Media: {-scores_mae.mean():.2f}M (+/- {scores_mae.std():.2f})")

    if scores_r2.std() < 0.03:
        print("       [OK] Modelo estable entre folds")
    else:
        print("       [!] Varianza alta entre folds")


# =============================================================================
# EJECUCION PRINCIPAL
# =============================================================================
if __name__ == '__main__':

    # 1. Cargar datos
    X, y, columnas_features = cargar_y_preparar_datos('ftr_valoracion_jugadores.csv')

    # 2. Preprocesar
    X_procesado, encoders = preprocesar_features(X, fit=True)

    # 3. Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_procesado, y,
        test_size=0.20,
        random_state=42,
    )
    print(f"       Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

    # 4. Entrenar
    modelo = entrenar_modelo(X_train, y_train, X_test, y_test)

    # 5. Evaluar
    r2, mae = evaluar_modelo(modelo, X_train, y_train, X_test, y_test)

    # 6. Guardar modelo
    guardar_modelo(modelo, encoders, columnas_features, ruta='modelo_jugador_entrenado')

    # 7. Validacion cruzada
    validacion_cruzada(X_procesado, y)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}M euros")
    print(f"Modelo guardado en: modelo_jugador_entrenado/")
    print(f"Para predecir, ejecuta: python predecir_jugador.py")
