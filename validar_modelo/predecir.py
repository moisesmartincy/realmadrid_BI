import pandas as pd
import numpy as np
import joblib
import json
import os

# =============================================================================
# PREDICCION DE PARTIDOS DEL REAL MADRID
# =============================================================================
# Carga el modelo entrenado y predice probabilidades de win/draw/loss
# para un partido dado.
# =============================================================================

RUTA_MODELO = 'modelos_entrenados'


def cargar_modelo(ruta=RUTA_MODELO):
    """Carga el modelo y todos los artefactos necesarios."""

    print(f"Cargando modelo desde '{ruta}/'...")

    modelo = joblib.load(os.path.join(ruta, 'xgboost_model.pkl'))
    le_target = joblib.load(os.path.join(ruta, 'label_encoder_target.pkl'))
    encoders = joblib.load(os.path.join(ruta, 'feature_encoders.pkl'))

    with open(os.path.join(ruta, 'columnas_features.json'), 'r') as f:
        columnas_features = json.load(f)

    with open(os.path.join(ruta, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    print(f"Modelo cargado: {metadata['modelo']}")
    print(f"Clases: {metadata['clases']}")
    print(f"Features: {metadata['n_features']}")

    return modelo, le_target, encoders, columnas_features


def predecir_partido(modelo, le_target, encoders, columnas_features, datos_partido):
    """
    Predice las probabilidades de un partido.

    Args:
        datos_partido: dict con las variables del partido.

    Returns:
        dict con probabilidades y prediccion.
    """

    # Crear DataFrame con una sola fila
    df_input = pd.DataFrame([datos_partido])

    # Verificar que todas las columnas necesarias esten presentes
    columnas_faltantes = [c for c in columnas_features if c not in df_input.columns]
    if columnas_faltantes:
        print(f"[ERROR] Faltan columnas: {columnas_faltantes}")
        return None

    # Ordenar columnas igual que en entrenamiento
    df_input = df_input[columnas_features]

    # Aplicar encoding (sin fit, usando encoders guardados)
    columnas_categoricas = [
        'rival', 'competicion', 'formacion_rm', 'formacion_rival',
        'motivacion_rm', 'motivacion_rival'
    ]

    for col in columnas_categoricas:
        if col in df_input.columns and col in encoders:
            le = encoders[col]
            valor = str(df_input[col].iloc[0])
            if valor in le.classes_:
                df_input[col] = le.transform([valor])[0]
            else:
                print(f"[AVISO] Valor '{valor}' no visto en entrenamiento para '{col}'. Usando -1.")
                df_input[col] = -1

    # Predecir probabilidades
    probabilidades = modelo.predict_proba(df_input)[0]
    prediccion_encoded = modelo.predict(df_input)[0]
    prediccion = le_target.inverse_transform([prediccion_encoded])[0]

    # Mapear probabilidades a clases
    clases = le_target.classes_
    resultado = {
        'prediccion': prediccion,
        'probabilidades': {
            clase: round(float(prob), 4)
            for clase, prob in zip(clases, probabilidades)
        },
        'confianza': round(float(max(probabilidades)), 4),
    }

    return resultado


def mostrar_prediccion(rival, resultado):
    """Muestra la prediccion de forma visual."""

    print("\n" + "=" * 60)
    print(f"  REAL MADRID vs {rival.upper()}")
    print("=" * 60)

    probs = resultado['probabilidades']

    # Ordenar: win, draw, loss
    orden = ['win', 'draw', 'loss']
    etiquetas = {
        'win': 'Victoria RM',
        'draw': 'Empate',
        'loss': 'Derrota RM',
    }

    for key in orden:
        if key in probs:
            pct = probs[key] * 100
            barra = "#" * int(pct / 2)
            espacios = " " * (50 - len(barra))
            print(f"  {etiquetas[key]:>14}: {pct:5.1f}% |{barra}{espacios}|")

    print(f"\n  >>> Prediccion: {etiquetas.get(resultado['prediccion'], resultado['prediccion']).upper()}")
    print(f"  >>> Confianza:  {resultado['confianza']*100:.1f}%")
    print("=" * 60)


# =============================================================================
# EJEMPLOS DE PREDICCION
# =============================================================================
if __name__ == '__main__':

    # Cargar modelo
    modelo, le_target, encoders, columnas_features = cargar_modelo()

    # -----------------------------------------------------------------
    # EJEMPLO 1: Real Madrid vs FC Barcelona (Clasico en el Bernabeu)
    # -----------------------------------------------------------------
    partido_clasico = {
        'rival': 'FC Barcelona',
        'competicion': 'Liga',
        'es_local': 1,
        'valor_mercado_rm': 1350,
        'valor_mercado_rival': 1100,
        'dif_valor_mercado': 250,
        'ranking_rm': 1,
        'ranking_rival': 2,
        'racha_rm': 12,           # 4W 0D 1L en ultimos 5
        'racha_rival': 11,        # 3W 2D 0L
        'dif_racha': 1,
        'goles_favor_rm': 2.4,
        'goles_contra_rm': 0.6,
        'goles_favor_rival': 2.2,
        'goles_contra_rival': 0.8,
        'prom_xg_5_rm': 2.1,
        'prom_xg_5_rival': 1.8,
        'dif_xg': 0.3,
        'prom_xt_5_rm': 1.7,
        'prom_xt_5_rival': 1.5,
        'dif_xt': 0.2,
        'formacion_rm': '4-3-3',
        'formacion_rival': '4-3-3',
        'bajas_rm': 1,            # 1 baja (ej: Carvajal)
        'bajas_rival': 0,
        'dias_descanso_rm': 5,
        'dias_descanso_rival': 6,
        'fatiga_rm': 2,
        'fatiga_rival': 2,
        'motivacion_rm': 'alta',
        'motivacion_rival': 'alta',
    }

    resultado1 = predecir_partido(modelo, le_target, encoders, columnas_features, partido_clasico)
    if resultado1:
        mostrar_prediccion('FC Barcelona', resultado1)

    # -----------------------------------------------------------------
    # EJEMPLO 2: Real Madrid vs Getafe (Liga, fuera de casa)
    # -----------------------------------------------------------------
    partido_getafe = {
        'rival': 'Getafe CF',
        'competicion': 'Liga',
        'es_local': 0,            # Fuera del Bernabeu
        'valor_mercado_rm': 1350,
        'valor_mercado_rival': 95,
        'dif_valor_mercado': 1255,
        'ranking_rm': 1,
        'ranking_rival': 15,
        'racha_rm': 13,
        'racha_rival': 5,
        'dif_racha': 8,
        'goles_favor_rm': 2.6,
        'goles_contra_rm': 0.4,
        'goles_favor_rival': 0.8,
        'goles_contra_rival': 1.6,
        'prom_xg_5_rm': 2.3,
        'prom_xg_5_rival': 0.9,
        'dif_xg': 1.4,
        'prom_xt_5_rm': 1.9,
        'prom_xt_5_rival': 0.7,
        'dif_xt': 1.2,
        'formacion_rm': '4-3-3',
        'formacion_rival': '4-4-2',
        'bajas_rm': 0,
        'bajas_rival': 2,
        'dias_descanso_rm': 6,
        'dias_descanso_rival': 7,
        'fatiga_rm': 1,
        'fatiga_rival': 3,
        'motivacion_rm': 'media',
        'motivacion_rival': 'alta',  # Getafe siempre motivado vs Madrid
    }

    resultado2 = predecir_partido(modelo, le_target, encoders, columnas_features, partido_getafe)
    if resultado2:
        mostrar_prediccion('Getafe CF', resultado2)

    # -----------------------------------------------------------------
    # EJEMPLO 3: Real Madrid vs Manchester City (Champions, Bernabeu)
    # -----------------------------------------------------------------
    partido_champions = {
        'rival': 'Manchester City',
        'competicion': 'Champions',
        'es_local': 1,
        'valor_mercado_rm': 1350,
        'valor_mercado_rival': 1250,
        'dif_valor_mercado': 100,
        'ranking_rm': 1,
        'ranking_rival': 0,       # 0 = no aplica ranking liga
        'racha_rm': 10,
        'racha_rival': 12,
        'dif_racha': -2,
        'goles_favor_rm': 1.8,
        'goles_contra_rm': 0.8,
        'goles_favor_rival': 2.4,
        'goles_contra_rival': 0.6,
        'prom_xg_5_rm': 1.9,
        'prom_xg_5_rival': 2.1,
        'dif_xg': -0.2,
        'prom_xt_5_rm': 1.5,
        'prom_xt_5_rival': 1.8,
        'dif_xt': -0.3,
        'formacion_rm': '4-3-3',
        'formacion_rival': '4-2-3-1',
        'bajas_rm': 2,            # 2 bajas clave
        'bajas_rival': 1,
        'dias_descanso_rm': 4,
        'dias_descanso_rival': 4,
        'fatiga_rm': 3,
        'fatiga_rival': 3,
        'motivacion_rm': 'alta',
        'motivacion_rival': 'alta',
    }

    resultado3 = predecir_partido(modelo, le_target, encoders, columnas_features, partido_champions)
    if resultado3:
        mostrar_prediccion('Manchester City', resultado3)

    # -----------------------------------------------------------------
    # EJEMPLO 4: Real Madrid vs Rayo Vallecano (Liga, local con equipo fresco)
    # -----------------------------------------------------------------
    partido_rayo = {
        'rival': 'Rayo Vallecano',
        'competicion': 'Liga',
        'es_local': 1,
        'valor_mercado_rm': 1350,
        'valor_mercado_rival': 85,
        'dif_valor_mercado': 1265,
        'ranking_rm': 1,
        'ranking_rival': 12,
        'racha_rm': 15,           # 5 victorias seguidas
        'racha_rival': 4,
        'dif_racha': 11,
        'goles_favor_rm': 3.0,
        'goles_contra_rm': 0.2,
        'goles_favor_rival': 0.6,
        'goles_contra_rival': 2.0,
        'prom_xg_5_rm': 2.8,
        'prom_xg_5_rival': 0.7,
        'dif_xg': 2.1,
        'prom_xt_5_rm': 2.2,
        'prom_xt_5_rival': 0.5,
        'dif_xt': 1.7,
        'formacion_rm': '4-3-3',
        'formacion_rival': '4-4-2',
        'bajas_rm': 0,
        'bajas_rival': 1,
        'dias_descanso_rm': 7,
        'dias_descanso_rival': 5,
        'fatiga_rm': 1,
        'fatiga_rival': 3,
        'motivacion_rm': 'alta',
        'motivacion_rival': 'baja',
    }

    resultado4 = predecir_partido(modelo, le_target, encoders, columnas_features, partido_rayo)
    if resultado4:
        mostrar_prediccion('Rayo Vallecano', resultado4)
