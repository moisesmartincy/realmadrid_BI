import pandas as pd
import numpy as np
import joblib
import json
import os

# =============================================================================
# PREDICCION DE TOP PRODUCTOS A VENDER
# =============================================================================

RUTA_MODELO = 'modelo_top_productos_entrenado'

def cargar_modelo(ruta=RUTA_MODELO):
    print(f"Cargando modelo desde '{ruta}/'...")

    modelo = joblib.load(os.path.join(ruta, 'multioutput_xgboost.pkl'))
    encoders = joblib.load(os.path.join(ruta, 'feature_encoders.pkl'))

    with open(os.path.join(ruta, 'columnas_features.json'), 'r') as f:
        columnas_features = json.load(f)
        
    with open(os.path.join(ruta, 'columnas_targets.json'), 'r') as f:
        columnas_targets = json.load(f)

    return modelo, encoders, columnas_features, columnas_targets

def predecir_top(modelo, encoders, columnas_features, columnas_targets, datos_contexto):
    df_input = pd.DataFrame([datos_contexto])

    columnas_faltantes = [c for c in columnas_features if c not in df_input.columns]
    if columnas_faltantes:
        print(f"[ERROR] Faltan columnas: {columnas_faltantes}")
        return None

    df_input = df_input[columnas_features]

    # Aplicar encoders
    columnas_categoricas = ['competicion', 'mes', 'horario']
    for col in columnas_categoricas:
        if col in df_input.columns and col in encoders:
            le = encoders[col]
            valor = str(df_input[col].iloc[0])
            if valor in le.classes_:
                df_input[col] = le.transform([valor])[0]
            else:
                print(f"[AVISO] Valor '{valor}' desconocido para '{col}'. Usando -1.")
                df_input[col] = -1

    # Predicción (array 2D con las 7 demandas)
    prediccion_cruda = modelo.predict(df_input)[0]
    
    # Procesar predicciones en un diccionario amigable
    resultados = {}
    for i, target in enumerate(columnas_targets):
        nombre_limpio = target.replace('demanda_', '').replace('_', ' ').title()
        demanda = max(0.0, float(prediccion_cruda[i]))
        resultados[nombre_limpio] = demanda

    # Ordenar productos de mayor a menor demanda
    top_productos = sorted(resultados.items(), key=lambda x: x[1], reverse=True)
    return top_productos

def mostrar_top(titulo, datos, top_productos):
    print("\n" + "=" * 65)
    print(f"  EVENTO: {titulo.upper()}")
    print("=" * 65)
    
    print(f"  --- Contexto del Partido ---")
    print(f"  Mes y Hora:    {datos['mes']} a las {datos['horario']}")
    print(f"  Clima:         {datos['temp_clima_c']}C  |  Lluvia: {'SI' if datos['lluvia'] else 'NO'}")
    print(f"  Asistencia:    {datos['asistencia']:,} espectadores ({int(datos['pct_turismo']*100)}% Turismo)")
    print(f"  Importancia:   Nivel Rival {datos['nivel_rival']}/5  |  Clasico: {'SI' if datos['es_clasico'] else 'NO'}")
    
    print(f"\n  {'=' * 20} TOP PRODUCTOS ESPERADOS {'=' * 20}")
    
    for i, (producto, puntaje) in enumerate(top_productos):
        # Crear barrita visual
        barra_len = min(40, int(puntaje / 2.5))
        barra = "#" * barra_len
        
        # Destacar el TOP 3
        rank = f"{i+1}."
        print(f"  {rank:<3} {producto:<17} | {puntaje:>5.1f} | {barra}")
        
    print(f"  {'=' * 65}")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    modelo, encoders, columnas_features, columnas_targets = cargar_modelo()

    # EJEMPLO 1: Noche de Champions Fría y Lluviosa con público local
    noche_fria_lluvia = {
        'competicion': 'Champions', 'mes': 'Diciembre', 'nivel_rival': 3, 
        'es_clasico': 0, 'horario': '21:00', 'temp_clima_c': 4, 
        'lluvia': 1, 'pct_turismo': 0.10, 'asistencia': 65000
    }
    
    # EJEMPLO 2: Tarde de Agosto, calor extremo, lleno de turistas (Primer partido Liga)
    tarde_calor_turistas = {
        'competicion': 'Liga', 'mes': 'Agosto', 'nivel_rival': 2, 
        'es_clasico': 0, 'horario': '16:15', 'temp_clima_c': 35, 
        'lluvia': 0, 'pct_turismo': 0.60, 'asistencia': 78000
    }

    # EJEMPLO 3: El Gran Clásico en primavera
    gran_clasico = {
        'competicion': 'Liga', 'mes': 'Abril', 'nivel_rival': 5, 
        'es_clasico': 1, 'horario': '21:00', 'temp_clima_c': 18, 
        'lluvia': 0, 'pct_turismo': 0.40, 'asistencia': 81000
    }

    p1 = predecir_top(modelo, encoders, columnas_features, columnas_targets, noche_fria_lluvia)
    mostrar_top('Noche Invernal de Champions (Lluvia)', noche_fria_lluvia, p1)

    p2 = predecir_top(modelo, encoders, columnas_features, columnas_targets, tarde_calor_turistas)
    mostrar_top('Tarde de Agosto (Ola de Calor y Turistas)', tarde_calor_turistas, p2)

    p3 = predecir_top(modelo, encoders, columnas_features, columnas_targets, gran_clasico)
    mostrar_top('El Clásico en Primavera', gran_clasico, p3)
