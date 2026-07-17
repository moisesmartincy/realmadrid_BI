import pandas as pd
import numpy as np
import joblib
import json
import os

# =============================================================================
# PREDICCION DE VALOR DE MERCADO DE JUGADORES
# =============================================================================
# Carga el modelo entrenado y predice el valor de mercado de un jugador
# a partir de sus estadisticas.
# =============================================================================

RUTA_MODELO = 'modelo_jugador_entrenado'


def cargar_modelo(ruta=RUTA_MODELO):
    """Carga el modelo y artefactos guardados."""

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


def predecir_valor(modelo, encoders, columnas_features, datos_jugador):
    """
    Predice el valor de mercado de un jugador.

    Args:
        datos_jugador: dict con las estadisticas del jugador.

    Returns:
        float: valor de mercado estimado en millones de euros.
    """

    df_input = pd.DataFrame([datos_jugador])

    # Verificar columnas
    columnas_faltantes = [c for c in columnas_features if c not in df_input.columns]
    if columnas_faltantes:
        print(f"[ERROR] Faltan columnas: {columnas_faltantes}")
        return None

    df_input = df_input[columnas_features]

    # Encoding categoricas
    columnas_categoricas = ['posicion', 'liga', 'nacionalidad', 'pie_dominante']
    for col in columnas_categoricas:
        if col in df_input.columns and col in encoders:
            le = encoders[col]
            valor = str(df_input[col].iloc[0])
            if valor in le.classes_:
                df_input[col] = le.transform([valor])[0]
            else:
                print(f"[AVISO] Valor '{valor}' no visto para '{col}'. Usando -1.")
                df_input[col] = -1

    # Predecir
    valor_predicho = modelo.predict(df_input)[0]
    valor_predicho = max(0.3, round(float(valor_predicho), 2))

    return valor_predicho


def mostrar_prediccion(nombre, datos, valor_predicho):
    """Muestra la prediccion de forma visual."""

    print("\n" + "=" * 60)
    print(f"  JUGADOR: {nombre.upper()}")
    print("=" * 60)
    print(f"  Posicion:    {datos['posicion']}")
    print(f"  Edad:        {datos['edad']} anios")
    print(f"  Liga:        {datos['liga']}")
    print(f"  Nacion:      {datos['nacionalidad']}")

    print(f"\n  --- Stats Ofensivas ---")
    print(f"  Goles:       {datos['goles']}  |  xG: {datos['xG']}")
    print(f"  Asistencias: {datos['asistencias']}  |  xA: {datos['xA']}")
    print(f"  Tiros:       {datos['tiros']}  |  Regates: {datos['regates']}")

    print(f"\n  --- Stats Defensivas ---")
    print(f"  Tackles:     {datos['tackles']}  |  Intercepciones: {datos['intercepciones']}")
    print(f"  Duelos aereos: {datos['duelos_aereos_pct']*100:.0f}%")

    print(f"\n  --- Contexto ---")
    print(f"  Partidos:    {datos['partidos_jugados']}  |  Minutos: {datos['minutos']}")
    print(f"  Contrato:    {datos['contrato_anios']} anios  |  Lesiones: {datos['lesiones_anio']}")
    print(f"  Club valor:  {datos['valor_club']}M  |  Rendimiento: {datos['rendimiento_ultimos_10']}")
    print(f"  Internacional: {'Si' if datos['es_internacional'] else 'No'} ({datos['partidos_seleccion']} caps)")

    # Resultado
    barra_len = min(50, int(valor_predicho / 4))
    barra = "#" * barra_len
    print(f"\n  {'=' * 40}")
    print(f"  VALOR ESTIMADO: {valor_predicho:.1f} MILLONES EUR")
    print(f"  |{barra}|")
    print(f"  {'=' * 40}")
    print("=" * 60)


# =============================================================================
# EJEMPLOS DE PREDICCION
# =============================================================================
if __name__ == '__main__':

    modelo, encoders, columnas_features = cargar_modelo()

    # -----------------------------------------------------------------
    # EJEMPLO 1: Vinicius Jr (Extremo elite, Real Madrid)
    # -----------------------------------------------------------------
    vinicius = {
        'edad': 25,
        'posicion': 'Extremo',
        'altura_cm': 176,
        'pie_dominante': 'Derecho',
        'liga': 'LaLiga',
        'nacionalidad': 'Brasilena',
        'contrato_anios': 4,
        'lesiones_anio': 1,
        'es_internacional': 1,
        'partidos_seleccion': 35,
        'goles_seleccion': 8,
        'valor_club': 1350,
        'partidos_jugados': 35,
        'minutos': 2900,
        'goles': 15,
        'asistencias': 10,
        'xG': 14.2,
        'xA': 8.5,
        'tiros': 95,
        'regates': 120,
        'pases_clave': 65,
        'pases_progresivos': 85,
        'precision_pases': 0.82,
        'tackles': 25,
        'intercepciones': 15,
        'duelos_aereos_pct': 0.35,
        'paradas_p90': 0.0,
        'porcentaje_paradas': 0.0,
        'porteria_cero': 0,
        'rendimiento_ultimos_10': 8.2,
    }

    valor1 = predecir_valor(modelo, encoders, columnas_features, vinicius)
    if valor1:
        mostrar_prediccion('Vinicius Jr', vinicius, valor1)

    # -----------------------------------------------------------------
    # EJEMPLO 2: Pedri (Mediocentro joven, Barcelona)
    # -----------------------------------------------------------------
    pedri = {
        'edad': 22,
        'posicion': 'Mediocentro',
        'altura_cm': 174,
        'pie_dominante': 'Derecho',
        'liga': 'LaLiga',
        'nacionalidad': 'Espanola',
        'contrato_anios': 5,
        'lesiones_anio': 2,
        'es_internacional': 1,
        'partidos_seleccion': 25,
        'goles_seleccion': 3,
        'valor_club': 1100,
        'partidos_jugados': 30,
        'minutos': 2500,
        'goles': 5,
        'asistencias': 8,
        'xG': 4.0,
        'xA': 7.2,
        'tiros': 40,
        'regates': 55,
        'pases_clave': 70,
        'pases_progresivos': 140,
        'precision_pases': 0.91,
        'tackles': 45,
        'intercepciones': 35,
        'duelos_aereos_pct': 0.42,
        'paradas_p90': 0.0,
        'porcentaje_paradas': 0.0,
        'porteria_cero': 0,
        'rendimiento_ultimos_10': 7.8,
    }

    valor2 = predecir_valor(modelo, encoders, columnas_features, pedri)
    if valor2:
        mostrar_prediccion('Pedri', pedri, valor2)

    # -----------------------------------------------------------------
    # EJEMPLO 3: Courtois (Portero elite, Real Madrid)
    # -----------------------------------------------------------------
    courtois = {
        'edad': 33,
        'posicion': 'Portero',
        'altura_cm': 199,
        'pie_dominante': 'Izquierdo',
        'liga': 'LaLiga',
        'nacionalidad': 'Otra',  # Belga
        'contrato_anios': 3,
        'lesiones_anio': 1,
        'es_internacional': 1,
        'partidos_seleccion': 102,
        'goles_seleccion': 0,
        'valor_club': 1350,
        'partidos_jugados': 30,
        'minutos': 2700,
        'goles': 0,
        'asistencias': 0,
        'xG': 0.0,
        'xA': 0.0,
        'tiros': 0,
        'regates': 0,
        'pases_clave': 2,
        'pases_progresivos': 45,
        'precision_pases': 0.78,
        'tackles': 0,
        'intercepciones': 0,
        'duelos_aereos_pct': 0.60,
        'paradas_p90': 3.5,
        'porcentaje_paradas': 0.78,
        'porteria_cero': 12,
        'rendimiento_ultimos_10': 7.5,
    }

    valor3 = predecir_valor(modelo, encoders, columnas_features, courtois)
    if valor3:
        mostrar_prediccion('Thibaut Courtois', courtois, valor3)

    # -----------------------------------------------------------------
    # EJEMPLO 4: Haaland (Delantero goleador, Man City)
    # -----------------------------------------------------------------
    haaland = {
        'edad': 25,
        'posicion': 'Delantero Centro',
        'altura_cm': 194,
        'pie_dominante': 'Izquierdo',
        'liga': 'Premier League',
        'nacionalidad': 'Otra',  # Noruego
        'contrato_anios': 5,
        'lesiones_anio': 0,
        'es_internacional': 1,
        'partidos_seleccion': 35,
        'goles_seleccion': 30,
        'valor_club': 1250,
        'partidos_jugados': 34,
        'minutos': 2850,
        'goles': 28,
        'asistencias': 5,
        'xG': 25.0,
        'xA': 4.2,
        'tiros': 130,
        'regates': 30,
        'pases_clave': 25,
        'pases_progresivos': 20,
        'precision_pases': 0.75,
        'tackles': 10,
        'intercepciones': 5,
        'duelos_aereos_pct': 0.62,
        'paradas_p90': 0.0,
        'porcentaje_paradas': 0.0,
        'porteria_cero': 0,
        'rendimiento_ultimos_10': 8.5,
    }

    valor4 = predecir_valor(modelo, encoders, columnas_features, haaland)
    if valor4:
        mostrar_prediccion('Erling Haaland', haaland, valor4)

    # -----------------------------------------------------------------
    # EJEMPLO 5: Jugador joven promesa (18 anios, pocos partidos)
    # -----------------------------------------------------------------
    promesa = {
        'edad': 18,
        'posicion': 'Mediapunta',
        'altura_cm': 178,
        'pie_dominante': 'Derecho',
        'liga': 'LaLiga',
        'nacionalidad': 'Espanola',
        'contrato_anios': 5,
        'lesiones_anio': 0,
        'es_internacional': 1,
        'partidos_seleccion': 3,
        'goles_seleccion': 1,
        'valor_club': 800,
        'partidos_jugados': 15,
        'minutos': 900,
        'goles': 4,
        'asistencias': 3,
        'xG': 3.2,
        'xA': 2.5,
        'tiros': 25,
        'regates': 30,
        'pases_clave': 20,
        'pases_progresivos': 35,
        'precision_pases': 0.83,
        'tackles': 12,
        'intercepciones': 8,
        'duelos_aereos_pct': 0.38,
        'paradas_p90': 0.0,
        'porcentaje_paradas': 0.0,
        'porteria_cero': 0,
        'rendimiento_ultimos_10': 7.2,
    }

    valor5 = predecir_valor(modelo, encoders, columnas_features, promesa)
    if valor5:
        mostrar_prediccion('Joven Promesa (18)', promesa, valor5)

    # -----------------------------------------------------------------
    # EJEMPLO 6: Veterano bajo nivel (35 anios, liga menor)
    # -----------------------------------------------------------------
    veterano = {
        'edad': 35,
        'posicion': 'Defensa Central',
        'altura_cm': 186,
        'pie_dominante': 'Derecho',
        'liga': 'Ligue 1',
        'nacionalidad': 'Francesa',
        'contrato_anios': 1,
        'lesiones_anio': 3,
        'es_internacional': 0,
        'partidos_seleccion': 0,
        'goles_seleccion': 0,
        'valor_club': 80,
        'partidos_jugados': 20,
        'minutos': 1600,
        'goles': 1,
        'asistencias': 0,
        'xG': 0.8,
        'xA': 0.2,
        'tiros': 8,
        'regates': 3,
        'pases_clave': 5,
        'pases_progresivos': 30,
        'precision_pases': 0.84,
        'tackles': 40,
        'intercepciones': 30,
        'duelos_aereos_pct': 0.58,
        'paradas_p90': 0.0,
        'porcentaje_paradas': 0.0,
        'porteria_cero': 0,
        'rendimiento_ultimos_10': 6.3,
    }

    valor6 = predecir_valor(modelo, encoders, columnas_features, veterano)
    if valor6:
        mostrar_prediccion('Veterano 35 (Ligue 1)', veterano, valor6)
