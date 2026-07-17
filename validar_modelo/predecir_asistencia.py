import pandas as pd
import numpy as np
import joblib
import json
import os

# =============================================================================
# PREDICCION DE ASISTENCIA AL ESTADIO
# =============================================================================

RUTA_MODELO = 'modelo_asistencia_entrenado'

def cargar_modelo(ruta=RUTA_MODELO):
    print(f"Cargando modelo desde '{ruta}/'...")

    modelo = joblib.load(os.path.join(ruta, 'xgboost_regressor.pkl'))
    encoders = joblib.load(os.path.join(ruta, 'feature_encoders.pkl'))

    with open(os.path.join(ruta, 'columnas_features.json'), 'r') as f:
        columnas_features = json.load(f)

    return modelo, encoders, columnas_features

def predecir_asistencia(modelo, encoders, columnas_features, datos_partido):
    df_input = pd.DataFrame([datos_partido])

    columnas_faltantes = [c for c in columnas_features if c not in df_input.columns]
    if columnas_faltantes:
        print(f"[ERROR] Faltan columnas: {columnas_faltantes}")
        return None

    df_input = df_input[columnas_features]

    # Aplicar encoders
    columnas_categoricas = ['competicion', 'mes', 'dia_semana', 'hora_partido', 'clima']
    for col in columnas_categoricas:
        if col in df_input.columns and col in encoders:
            le = encoders[col]
            valor = str(df_input[col].iloc[0])
            if valor in le.classes_:
                df_input[col] = le.transform([valor])[0]
            else:
                print(f"[AVISO] Valor '{valor}' desconocido para '{col}'. Usando -1.")
                df_input[col] = -1

    asistencia = modelo.predict(df_input)[0]
    return int(np.clip(asistencia, 0, 81000))

def mostrar_prediccion(titulo, datos, asistencia):
    print("\n" + "=" * 60)
    print(f"  PARTIDO: {titulo.upper()}")
    print("=" * 60)
    
    print(f"  --- Contexto del Partido ---")
    print(f"  Competicin:   {datos['competicion']} (Rival Nivel {datos['rival_nivel']}/5)")
    print(f"  Es Clsico:    {'SI' if datos['es_clasico'] else 'NO'}")
    print(f"  Fecha:         {datos['dia_semana']} a las {datos['hora_partido']} ({datos['mes']})")
    print(f"  Es Feriado:    {'SI' if datos['es_feriado'] else 'NO'}")
    
    print(f"\n  --- Contexto Externo ---")
    print(f"  Clima:         {datos['clima']} ({datos['temperatura_c']}C)")
    print(f"  Precio Medio:  {datos['precio_promedio']} EUR")
    print(f"  Promocin:     {'Activa' if datos['promocion_activa'] else 'Ninguna'}")
    
    print(f"\n  --- Contexto Deportivo ---")
    print(f"  Racha:         {int(datos['racha_victorias_pct']*100)}% victorias")
    print(f"  Faltan estrellas: {datos['bajas_estrellas']} jugadores clave ausentes")

    # Calculos extra para renderizar
    pct_llenado = (asistencia / 81000) * 100
    barra_len = int(pct_llenado / 2)
    barra = "#" * barra_len
    
    print(f"\n  {'=' * 40}")
    print(f"  ASISTENCIA ESTIMADA: {asistencia:,} PERSONAS")
    print(f"  LLENADO DEL ESTADIO: {pct_llenado:.1f}%")
    print(f"  [{barra.ljust(50)}]")
    print(f"  {'=' * 40}")
    print("=" * 60)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    modelo, encoders, columnas_features = cargar_modelo()

    # EJEMPLO 1: El Clsico / Semifinal Champions (Partido Top)
    partido_top = {
        'competicion': 'Champions', 'rival_nivel': 5, 'es_clasico': 1,
        'mes': 'Abril', 'dia_semana': 'Mircoles', 'hora_partido': '21:00',
        'es_feriado': 0, 'clima': 'Despejado', 'temperatura_c': 18,
        'racha_victorias_pct': 0.85, 'posicion_liga': 1, 'bajas_estrellas': 0,
        'promocion_activa': 0, 'precio_promedio': 180.0
    }
    
    # EJEMPLO 2: Partido de Liga menor con lluvia fuerte (Caso Pesimista)
    partido_pesimista = {
        'competicion': 'Liga', 'rival_nivel': 1, 'es_clasico': 0,
        'mes': 'Noviembre', 'dia_semana': 'Lunes', 'hora_partido': '21:00',
        'es_feriado': 0, 'clima': 'Lluvia Fuerte', 'temperatura_c': 5,
        'racha_victorias_pct': 0.30, 'posicion_liga': 4, 'bajas_estrellas': 2,
        'promocion_activa': 0, 'precio_promedio': 70.0
    }

    # EJEMPLO 3: Partido dominguero familiar en primavera (Caso Buen Ambiente)
    partido_familiar = {
        'competicion': 'Liga', 'rival_nivel': 2, 'es_clasico': 0,
        'mes': 'Mayo', 'dia_semana': 'Domingo', 'hora_partido': '16:15',
        'es_feriado': 0, 'clima': 'Despejado', 'temperatura_c': 24,
        'racha_victorias_pct': 0.90, 'posicion_liga': 1, 'bajas_estrellas': 0,
        'promocion_activa': 1, 'precio_promedio': 55.0
    }

    p1 = predecir_asistencia(modelo, encoders, columnas_features, partido_top)
    mostrar_prediccion('El Clasico / Noche de Champions Mgica', partido_top, p1)

    p2 = predecir_asistencia(modelo, encoders, columnas_features, partido_pesimista)
    mostrar_prediccion('Lunes lluvioso vs Rival modesto', partido_pesimista, p2)

    p3 = predecir_asistencia(modelo, encoders, columnas_features, partido_familiar)
    mostrar_prediccion('Domingo por la tarde en Primavera', partido_familiar, p3)
