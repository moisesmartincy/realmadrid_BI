import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib
import json
import os

# =============================================================================
# ENTRENAMIENTO DEL MODELO - PREDICCION DE PARTIDOS REAL MADRID
# =============================================================================
# Modelo: XGBoost (Multiclass Classification)
# Input: Variables del partido (rival, formacion, bajas, xG, etc.)
# Output: Probabilidades de win/draw/loss
# =============================================================================


def cargar_y_preparar_datos(csv_path='ftr_rendimiento_partidos.csv'):
    """Carga el CSV y prepara features + target."""

    print("[1/6] Cargando dataset...")
    df = pd.read_csv(csv_path)
    print(f"       Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # -----------------------------------------------------------------
    # Separar features (X) y target (y)
    # -----------------------------------------------------------------
    # Columnas target (las que el modelo debe predecir)
    columnas_target = ['prob_win', 'prob_draw', 'prob_loss', 'resultado']

    # Columnas que NO son features (son derivadas o redundantes)
    # Las columnas 'dif_*' las mantenemos porque son informativas
    columnas_excluir = columnas_target

    columnas_features = [c for c in df.columns if c not in columnas_excluir]

    X = df[columnas_features].copy()
    y = df['resultado'].copy()

    print(f"       Features: {len(columnas_features)} columnas")
    print(f"       Target: 'resultado' ({y.nunique()} clases: {list(y.unique())})")

    return X, y, columnas_features


def preprocesar_features(X, encoders=None, fit=True):
    """
    Convierte variables categoricas a numericas.
    Si fit=True, ajusta los encoders. Si fit=False, usa encoders existentes.
    Retorna X procesado y los encoders (para reutilizar en prediccion).
    """

    print("[2/6] Preprocesando features...")

    if encoders is None:
        encoders = {}

    X_procesado = X.copy()

    # Columnas categoricas que necesitan encoding
    columnas_categoricas = [
        'rival', 'competicion', 'formacion_rm', 'formacion_rival',
        'motivacion_rm', 'motivacion_rival'
    ]

    for col in columnas_categoricas:
        if col in X_procesado.columns:
            if fit:
                le = LabelEncoder()
                X_procesado[col] = le.fit_transform(X_procesado[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                # Manejar valores no vistos durante entrenamiento
                X_procesado[col] = X_procesado[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    print(f"       Columnas codificadas: {columnas_categoricas}")

    return X_procesado, encoders


def entrenar_modelo(X_train, y_train, X_test, y_test):
    """Entrena XGBoost con los mejores hiperparametros para este problema."""

    print("[3/6] Entrenando modelo XGBoost...")

    # Codificar target (win=2, draw=0, loss=1 o similar)
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)

    # -----------------------------------------------------------------
    # Hiperparametros optimizados para prediccion de futbol
    # -----------------------------------------------------------------
    modelo = XGBClassifier(
        n_estimators=500,           # Numero de arboles
        max_depth=6,                # Profundidad maxima (evita overfitting)
        learning_rate=0.05,         # Tasa de aprendizaje (bajo = mejor generalizacion)
        subsample=0.8,              # Usar 80% de datos por arbol
        colsample_bytree=0.8,       # Usar 80% de features por arbol
        min_child_weight=5,         # Minimo de muestras por hoja
        gamma=0.1,                  # Regularizacion
        reg_alpha=0.1,              # L1 regularizacion
        reg_lambda=1.0,             # L2 regularizacion
        objective='multi:softprob', # Clasificacion multiclase con probabilidades
        num_class=3,                # win, draw, loss
        eval_metric='mlogloss',     # Metrica de evaluacion
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )

    # Entrenar sin early stopping como parametro directo
    modelo.fit(
        X_train, y_train_encoded,
        eval_set=[(X_test, y_test_encoded)],
        verbose=50  # Mostrar progreso cada 50 arboles
    )

    print(f"       Modelo entrenado con {modelo.n_estimators} arboles.")

    return modelo, le_target


def evaluar_modelo(modelo, le_target, X_train, y_train, X_test, y_test):
    """Evalua el modelo con metricas detalladas."""

    print("\n[4/6] Evaluando modelo...")

    y_test_encoded = le_target.transform(y_test)

    # Predicciones
    y_pred_encoded = modelo.predict(X_test)
    y_pred = le_target.inverse_transform(y_pred_encoded)

    # Probabilidades
    y_proba = modelo.predict_proba(X_test)

    # -----------------------------------------------------------------
    # Metricas
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTADOS DEL MODELO")
    print("=" * 60)

    # Accuracy
    acc_train = accuracy_score(le_target.transform(y_train), modelo.predict(X_train))
    acc_test = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"\n[ACCURACY]")
    print(f"   Train: {acc_train:.4f} ({acc_train*100:.1f}%)")
    print(f"   Test:  {acc_test:.4f} ({acc_test*100:.1f}%)")

    # Si hay mucha diferencia entre train y test = overfitting
    if acc_train - acc_test > 0.05:
        print("   [!] ATENCION: Posible overfitting (train >> test)")
    else:
        print("   [OK] No hay overfitting significativo")

    # Classification Report
    print(f"\n[CLASSIFICATION REPORT]")
    clases_ordenadas = le_target.classes_
    print(classification_report(y_test, y_pred, target_names=clases_ordenadas))

    # Confusion Matrix
    print(f"[CONFUSION MATRIX]")
    cm = confusion_matrix(y_test, y_pred, labels=clases_ordenadas)
    cm_df = pd.DataFrame(cm, index=clases_ordenadas, columns=clases_ordenadas)
    print(cm_df.to_string())

    # Feature Importance (top 15)
    print(f"\n[TOP 15 FEATURES MAS IMPORTANTES]")
    importances = modelo.feature_importances_
    feature_names = modelo.feature_names_in_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for fname, fimp in feat_imp[:15]:
        bar = "#" * int(fimp * 100)
        print(f"   {fname:>25}: {fimp:.4f} {bar}")

    # Calibracion de probabilidades
    print(f"\n[CALIBRACION DE PROBABILIDADES]")
    print(f"   Promedio prob predicha vs frecuencia real:")
    for i, clase in enumerate(clases_ordenadas):
        prob_media = y_proba[:, i].mean()
        freq_real = (y_test == clase).mean()
        print(f"   {clase:>5}: prob_media={prob_media:.3f}, freq_real={freq_real:.3f}")

    return acc_test


def guardar_modelo(modelo, le_target, encoders, columnas_features, ruta='modelo_rm'):
    """Guarda el modelo y todos los artefactos necesarios para prediccion."""

    print(f"\n[5/6] Guardando modelo en carpeta '{ruta}/'...")

    # Crear carpeta
    os.makedirs(ruta, exist_ok=True)

    # 1. Guardar modelo XGBoost
    joblib.dump(modelo, os.path.join(ruta, 'xgboost_model.pkl'))

    # 2. Guardar encoder del target
    joblib.dump(le_target, os.path.join(ruta, 'label_encoder_target.pkl'))

    # 3. Guardar encoders de features
    joblib.dump(encoders, os.path.join(ruta, 'feature_encoders.pkl'))

    # 4. Guardar nombres de columnas (para validar input)
    with open(os.path.join(ruta, 'columnas_features.json'), 'w') as f:
        json.dump(columnas_features, f, indent=2)

    # 5. Guardar metadata del modelo
    metadata = {
        'modelo': 'XGBoost Multiclass',
        'n_features': len(columnas_features),
        'clases': list(le_target.classes_),
        'n_estimators': modelo.n_estimators,
        'columnas': columnas_features,
    }
    with open(os.path.join(ruta, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"       Archivos guardados:")
    for f in os.listdir(ruta):
        size = os.path.getsize(os.path.join(ruta, f))
        print(f"         - {f} ({size / 1024:.1f} KB)")


def validacion_cruzada(modelo_clase, X, y, le_target):
    """Validacion cruzada 5-fold para verificar estabilidad."""

    print("\n[6/6] Validacion cruzada (5-fold)...")

    y_encoded = le_target.transform(y)

    modelo_cv = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )

    scores = cross_val_score(modelo_cv, X, y_encoded, cv=5, scoring='accuracy')
    print(f"       Scores por fold: {[f'{s:.4f}' for s in scores]}")
    print(f"       Media: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"       [OK] Modelo estable" if scores.std() < 0.02 else "       [!] Alta varianza entre folds")


# =============================================================================
# EJECUCION PRINCIPAL
# =============================================================================
if __name__ == '__main__':

    # 1. Cargar datos
    X, y, columnas_features = cargar_y_preparar_datos('ftr_rendimiento_partidos.csv')

    # 2. Preprocesar
    X_procesado, encoders = preprocesar_features(X, fit=True)

    # 3. Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_procesado, y,
        test_size=0.20,
        random_state=42,
        stratify=y  # Mantener proporcion de clases
    )
    print(f"       Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

    # 4. Entrenar
    modelo, le_target = entrenar_modelo(X_train, y_train, X_test, y_test)

    # 5. Evaluar
    accuracy = evaluar_modelo(modelo, le_target, X_train, y_train, X_test, y_test)

    # 6. Guardar modelo
    guardar_modelo(modelo, le_target, encoders, columnas_features, ruta='modelos_entrenados')

    # 7. Validacion cruzada
    validacion_cruzada(XGBClassifier, X_procesado, y, le_target)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Accuracy final: {accuracy*100:.1f}%")
    print(f"Modelo guardado en: modelos_entrenados/")
    print(f"Para predecir, ejecuta: python predecir.py")