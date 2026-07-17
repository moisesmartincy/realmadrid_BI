import pandas as pd
import numpy as np
import joblib
import json
import os

# =============================================================================
# PREDICCION DE FATIGA DE JUGADORES
# =============================================================================
# Carga el modelo de regresion entrenado y predice el nivel de fatiga 
# de un jugador (0-100) dadas sus estadisticas de carga y recuperacion.
# =============================================================================

RUTA_MODELO = 'modelo_fatiga_entrenado'

def cargar_modelo(ruta=RUTA_MODELO):
    print(f"Cargando modelo desde '{ruta}/'...")

    modelo = joblib.load(os.path.join(ruta, 'xgboost_regressor.pkl'))
    encoders = joblib.load(os.path.join(ruta, 'feature_encoders.pkl'))

    with open(os.path.join(ruta, 'columnas_features.json'), 'r') as f:
        columnas_features = json.load(f)

    with open(os.path.join(ruta, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    print(f"Modelo cargado: {metadata['modelo']}")
    print(f"Target: {metadata['target']}")
    print(f"Features: {metadata['n_features']}")

    return modelo, encoders, columnas_features

def predecir_fatiga(modelo, encoders, columnas_features, datos_jugador):
    """
    Predice el nivel de fatiga.
    Retorna float 0-100.
    """
    df_input = pd.DataFrame([datos_jugador])

    # Revisar features faltantes
    columnas_faltantes = [c for c in columnas_features if c not in df_input.columns]
    if columnas_faltantes:
        print(f"[ERROR] Faltan columnas: {columnas_faltantes}")
        return None

    df_input = df_input[columnas_features]

    # Aplicar encoders
    columnas_categoricas = ['posicion', 'tipo_partido', 'superficie']
    for col in columnas_categoricas:
        if col in df_input.columns and col in encoders:
            le = encoders[col]
            valor = str(df_input[col].iloc[0])
            if valor in le.classes_:
                df_input[col] = le.transform([valor])[0]
            else:
                print(f"[AVISO] Valor '{valor}' desconocido para '{col}'. Usando -1.")
                df_input[col] = -1

    fatiga = modelo.predict(df_input)[0]
    fatiga = np.clip(float(fatiga), 0, 100)
    return round(fatiga, 1)

def mostrar_prediccion(nombre, datos, fatiga):
    print("\n" + "=" * 60)
    print(f"  JUGADOR: {nombre.upper()}")
    print("=" * 60)
    
    print(f"  --- Perfil y Estado ---")
    print(f"  Posicion:  {datos['posicion']} | Edad: {datos['edad']}")
    print(f"  Peso: {datos['peso_kg']} kg | Sem. Temp: {datos['semana_temporada']}")
    print(f"  Lesion reciente: {'SI' if datos['lesion_reciente'] else 'NO'}")

    print(f"\n  --- Cargas (Ultimo partido y acumuladas) ---")
    print(f"  Min. Ultimo Partido: {datos['minutos_ultimo_partido']} | Titular: {'SI' if datos['titular'] else 'NO'}")
    print(f"  Min. (7d / 30d):     {datos['minutos_7d']} / {datos['minutos_30d']}")
    print(f"  Part. (7d / 14d):    {datos['partidos_7d']} / {datos['partidos_14d']}")
    
    print(f"\n  --- Fisiologia (Ultimo partido) ---")
    print(f"  Dist. total (km):  {datos['distancia_km']} | Dist. alta vel: {datos['dist_alta_intensidad_km']}")
    print(f"  Sprints:           {datos['sprints']}")
    print(f"  FC media:          {datos['fc_media_bpm']} bpm | Zona roja: {datos['pct_zona_roja']*100:.1f}%")
    print(f"  RPE (Percepcion):  {datos['rpe']}/10")

    print(f"\n  --- Recuperacion ---")
    print(f"  Dias de Descanso:  {datos['dias_descanso']}")
    print(f"  Sueno (horas):     {datos['horas_sueno']} | Calidad (1-10): {datos['calidad_sueno']}")
    print(f"  Entrenamientos:    {datos['entrenamientos_entre_partidos']} | Carga de Entr: {datos['carga_entrenamiento']}")
    
    print(f"\n  --- Contexto Proximo ---")
    print(f"  Partido vs Nivel {datos['nivel_rival']} | Viaje: {datos['viaje_km']} km | Temp: {datos['temperatura']}C")

    # Barra visual de fatiga
    barra_len = int(fatiga / 2)
    barra = "#" * barra_len
    
    # Categorizar
    estado = "FRESCO" if fatiga < 30 else "LIGERA FATIGA" if fatiga < 60 else "ALTA FATIGA" if fatiga < 85 else "RIESGO CRITICO"
    
    print(f"\n  {'=' * 40}")
    print(f"  NIVEL DE FATIGA ESTIMADO: {fatiga}% - {estado}")
    print(f"  |{barra.ljust(50)}|")
    print(f"  {'=' * 40}")
    print("=" * 60)

if __name__ == '__main__':
    modelo, encoders, columnas_features = cargar_modelo()

    # EJEMPLO 1: Jugador Titular Exhausto (ej: Fede Valverde tras semana dura)
    exhausto = {
        'edad': 26, 'posicion': 'Mediocentro', 'peso_kg': 78, 'imc': 22.5,
        'lesiones_temporada': 0, 'lesion_reciente': 0, 'semana_temporada': 40,
        'titular': 1, 'minutos_ultimo_partido': 95, 'minutos_7d': 270, 'minutos_30d': 850,
        'partidos_7d': 3, 'partidos_14d': 5,
        'distancia_km': 12.5, 'dist_alta_intensidad_km': 2.8, 'sprints': 25,
        'aceleraciones': 60, 'deceleraciones': 55,
        'fc_media_bpm': 165, 'pct_zona_roja': 0.45, 'rpe': 8.5,
        'dias_descanso': 2, 'calidad_sueno': 4.5, 'horas_sueno': 5.5,
        'entrenamientos_entre_partidos': 1, 'carga_entrenamiento': 250,
        'tipo_partido': 'Champions', 'nivel_rival': 5, 'es_local': 0,
        'superficie': 'cesped_natural', 'viaje_km': 2500, 'temperatura': 25
    }
    
    # EJEMPLO 2: Jugador Fresco que vuelve de lesion (ej: Camavinga)
    fresco_lesion = {
        'edad': 22, 'posicion': 'Mediocentro', 'peso_kg': 75, 'imc': 23.0,
        'lesiones_temporada': 2, 'lesion_reciente': 1, 'semana_temporada': 15,
        'titular': 0, 'minutos_ultimo_partido': 20, 'minutos_7d': 20, 'minutos_30d': 50,
        'partidos_7d': 1, 'partidos_14d': 1,
        'distancia_km': 2.5, 'dist_alta_intensidad_km': 0.5, 'sprints': 3,
        'aceleraciones': 8, 'deceleraciones': 7,
        'fc_media_bpm': 135, 'pct_zona_roja': 0.10, 'rpe': 3.0,
        'dias_descanso': 6, 'calidad_sueno': 8.5, 'horas_sueno': 8.5,
        'entrenamientos_entre_partidos': 4, 'carga_entrenamiento': 600,
        'tipo_partido': 'Liga', 'nivel_rival': 2, 'es_local': 1,
        'superficie': 'cesped_natural', 'viaje_km': 0, 'temperatura': 18
    }

    # EJEMPLO 3: Veterano dosificado (ej: Modric)
    veterano = {
        'edad': 38, 'posicion': 'Mediocentro', 'peso_kg': 72, 'imc': 22.2,
        'lesiones_temporada': 1, 'lesion_reciente': 0, 'semana_temporada': 30,
        'titular': 1, 'minutos_ultimo_partido': 65, 'minutos_7d': 90, 'minutos_30d': 320,
        'partidos_7d': 2, 'partidos_14d': 3,
        'distancia_km': 8.1, 'dist_alta_intensidad_km': 0.8, 'sprints': 10,
        'aceleraciones': 25, 'deceleraciones': 22,
        'fc_media_bpm': 148, 'pct_zona_roja': 0.20, 'rpe': 6.5,
        'dias_descanso': 4, 'calidad_sueno': 7.5, 'horas_sueno': 8.0,
        'entrenamientos_entre_partidos': 2, 'carga_entrenamiento': 300,
        'tipo_partido': 'Liga', 'nivel_rival': 3, 'es_local': 0,
        'superficie': 'cesped_natural', 'viaje_km': 500, 'temperatura': 12
    }
    
    # EJEMPLO 4: Defensa Central con alta carga (ej: Rudiger)
    defensa = {
        'edad': 31, 'posicion': 'Defensa Central', 'peso_kg': 85, 'imc': 24.5,
        'lesiones_temporada': 0, 'lesion_reciente': 0, 'semana_temporada': 35,
        'titular': 1, 'minutos_ultimo_partido': 90, 'minutos_7d': 180, 'minutos_30d': 720,
        'partidos_7d': 2, 'partidos_14d': 4,
        'distancia_km': 9.8, 'dist_alta_intensidad_km': 1.2, 'sprints': 15,
        'aceleraciones': 35, 'deceleraciones': 30,
        'fc_media_bpm': 150, 'pct_zona_roja': 0.25, 'rpe': 7.0,
        'dias_descanso': 3, 'calidad_sueno': 6.5, 'horas_sueno': 7.0,
        'entrenamientos_entre_partidos': 2, 'carga_entrenamiento': 350,
        'tipo_partido': 'Liga', 'nivel_rival': 4, 'es_local': 1,
        'superficie': 'cesped_natural', 'viaje_km': 0, 'temperatura': 20
    }

    import warnings
    warnings.filterwarnings('ignore') # Ignorar warnings de pandas en el script

    f1 = predecir_fatiga(modelo, encoders, columnas_features, exhausto)
    mostrar_prediccion('Fede Valverde (Caso Exhausto)', exhausto, f1)

    f2 = predecir_fatiga(modelo, encoders, columnas_features, fresco_lesion)
    mostrar_prediccion('Eduardo Camavinga (Vuelve Lesion)', fresco_lesion, f2)

    f3 = predecir_fatiga(modelo, encoders, columnas_features, veterano)
    mostrar_prediccion('Luka Modric (Veterano Dosificado)', veterano, f3)

    f4 = predecir_fatiga(modelo, encoders, columnas_features, defensa)
    mostrar_prediccion('Antonio Rudiger (Defensa Iron Man)', defensa, f4)
