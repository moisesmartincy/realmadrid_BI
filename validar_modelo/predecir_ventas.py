import pandas as pd
import numpy as np
import joblib
import json
import os

# =============================================================================
# PREDICCION DE VENTAS TOTALES EN EL ESTADIO
# =============================================================================

RUTA_MODELO = 'modelo_total_productos_ganados_entrenado'

def cargar_modelo(ruta=RUTA_MODELO):
    print(f"Cargando modelo desde '{ruta}/'...")

    modelo = joblib.load(os.path.join(ruta, 'xgboost_regressor.pkl'))
    encoders = joblib.load(os.path.join(ruta, 'feature_encoders.pkl'))

    with open(os.path.join(ruta, 'columnas_features.json'), 'r') as f:
        columnas_features = json.load(f)

    return modelo, encoders, columnas_features

def predecir_ventas(modelo, encoders, columnas_features, datos_partido):
    df_input = pd.DataFrame([datos_partido])

    columnas_faltantes = [c for c in columnas_features if c not in df_input.columns]
    if columnas_faltantes:
        print(f"[ERROR] Faltan columnas: {columnas_faltantes}")
        return None

    df_input = df_input[columnas_features]

    # Aplicar encoders
    columnas_categoricas = ['horario', 'resultado_descanso']
    for col in columnas_categoricas:
        if col in df_input.columns and col in encoders:
            le = encoders[col]
            valor = str(df_input[col].iloc[0])
            if valor in le.classes_:
                df_input[col] = le.transform([valor])[0]
            else:
                print(f"[AVISO] Valor '{valor}' desconocido para '{col}'. Usando -1.")
                df_input[col] = -1

    ventas = modelo.predict(df_input)[0]
    return float(max(0.0, ventas))

def mostrar_prediccion(titulo, datos, ventas):
    print("\n" + "=" * 70)
    print(f"  EVENTO: {titulo.upper()}")
    print("=" * 70)
    
    print(f"  --- Contexto Asistencia y Pblico ---")
    print(f"  Asistencia:      {datos['asistencia']:,} personas")
    print(f"  Turismo:         {int(datos['pct_turismo'] * 100)}% de los asistentes")
    print(f"  Zonas Premium:   {int(datos['ocupacion_premium'] * 100)}% ocupacin VIP")
    
    print(f"\n  --- Contexto de Nivel y Deporte ---")
    print(f"  Nivel del Rival: {datos['nivel_rival']}/5  |  Es Clasico: {'SI' if datos['es_clasico'] else 'NO'}")
    print(f"  Resultado al descanso: {datos['resultado_descanso']}")
    
    print(f"\n  --- Clima y Logstica ---")
    print(f"  Horario Partido: {datos['horario']}")
    print(f"  Apertura Puertas {datos['minutos_apertura_pre']} min antes")
    print(f"  Clima y Lluvia:  {datos['temp_clima_c']}C  |  Lluvia: {'SI' if datos['lluvia'] else 'NO'}")
    
    print(f"\n  {'=' * 45}")
    moneda_str = f"EUR {ventas:,.2f}"
    print(f"  INGRESOS EST. F&B + MERCH: {moneda_str}")
    print(f"  Gasto Medio por Persona: EUR {(ventas / datos['asistencia']):.2f}")
    print(f"  {'=' * 45}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    modelo, encoders, columnas_features = cargar_modelo()

    # EJEMPLO 1: El Clsico, Noche Ideal, Atope de Turistas
    caso_clasico = {
        'asistencia': 81000, 'pct_turismo': 0.60, 'nivel_rival': 5, 
        'es_clasico': 1, 'horario': '21:00', 'temp_clima_c': 20, 
        'lluvia': 0, 'minutos_apertura_pre': 180, 
        'resultado_descanso': 'Ganando', 'ocupacion_premium': 1.00
    }
    
    # EJEMPLO 2: Partido de liga local con lluvia, bajo nivel
    caso_modesto = {
        'asistencia': 50000, 'pct_turismo': 0.10, 'nivel_rival': 1, 
        'es_clasico': 0, 'horario': '16:15', 'temp_clima_c': 8, 
        'lluvia': 1, 'minutos_apertura_pre': 60, 
        'resultado_descanso': 'Empatando', 'ocupacion_premium': 0.45
    }

    # EJEMPLO 3: Partido dominguero caluroso con familia, rival bueno
    caso_familiar = {
        'asistencia': 70000, 'pct_turismo': 0.30, 'nivel_rival': 3, 
        'es_clasico': 0, 'horario': '18:30', 'temp_clima_c': 35, 
        'lluvia': 0, 'minutos_apertura_pre': 90, 
        'resultado_descanso': 'Perdiendo', 'ocupacion_premium': 0.70
    }

    v1 = predecir_ventas(modelo, encoders, columnas_features, caso_clasico)
    mostrar_prediccion('El Clasico Historico - Record Ventas Turistas', caso_clasico, v1)

    v2 = predecir_ventas(modelo, encoders, columnas_features, caso_modesto)
    mostrar_prediccion('Lluvia y Rival de Tabla Baja', caso_modesto, v2)

    v3 = predecir_ventas(modelo, encoders, columnas_features, caso_familiar)
    mostrar_prediccion('Tarde Calurosa - Consumo Bebidas Frias', caso_familiar, v3)
