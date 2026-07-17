import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def probar_sarimax():
    print("--------------------------------------------------")
    print("Iniciando Prueba Estocástica: Bernabéu Pulse")
    print("--------------------------------------------------")
    
    # 1. Cargar Datos
    print("1. Cargando dataset generado (bernabeu_pulse_dataset.csv)...")
    try:
        df = pd.read_csv('bernabeu_pulse_dataset.csv')
    except FileNotFoundError:
        print("❌ Error: No se encontró 'bernabeu_pulse_dataset.csv'. Por favor ejecuta primero el script generador.")
        return
        
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])

    # 2. Transformaciones previas
    # Filtraremos solo la Puerta 1 (la más concurrida) para un análisis univariado
    print("2. Filtrando flujos para la 'Puerta 1'...")
    df_p1 = df[df['id_puerta'] == 1].copy()

    # Como entre partidos hay saltos de días y horas vacías (donde el estadio está cerrado),
    # SARIMA falla. Para evitarlo en esta prueba rápida, usaremos el índice como el bloque
    # secuencial del Tiempo de Partido. Trataremos "11 intervalos" como nuestra estacionalidad pura (s=11).
    df_p1 = df_p1.sort_values(by=['id_partido', 'minutos_al_kickoff']).reset_index(drop=True)

    # Separar Endógenas (Y = Flujo) y Exógenas (X = Variables Ambientales)
    y = df_p1['flujo_personas']
    X = df_p1[['clima_lluvia', 'nivel_rivalidad', 'factor_estrella']]

    # 3. Train / Test Split (El Target)
    # Dejamos exactamente los últimos 11 intervalos (El último partido entero) para ver si
    # el modelo logra predecir *ese* partido observando solo sus factores de entorno.
    intervalos_por_partido = 11
    train_size = len(df_p1) - intervalos_por_partido
    
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]

    print(f"\n3. Entrenando SARIMAX en {len(y_train)} registros históricos.")
    print("   Parámetros (p,d,q)= (1,0,1) | Estacionalidad (P,D,Q,s) = (1,0,1,11)")
    print("   [⏳ Esto tardará un poco dependiendo de tu CPU (Aprox 5 - 15 segs)]...")
    
    # Modelo SARIMAX
    modelo = SARIMAX(y_train, 
                     exog=X_train,
                     order=(1, 0, 1),
                     seasonal_order=(1, 0, 1, intervalos_por_partido),
                     enforce_stationarity=False,
                     enforce_invertibility=False)

    resultado = modelo.fit(disp=False)
    print("✅ Entrenamiento completado. AIC del modelo:", round(resultado.aic, 2))

    # 4. Predicciones con Fan Chart
    print("\n4. Pronosticando flujos para el partido final usando las exógenas del Test...")
    
    # Pronóstico con exógenas (Le pasamos si llovió y qué rivalidad hubo en el partido Test)
    pred_obj = resultado.get_forecast(steps=intervalos_por_partido, exog=X_test)
    pred_mean = pred_obj.predicted_mean
    pred_ci = pred_obj.conf_int(alpha=0.1) # 90% de confianza estadística

    # Preparando Eje X para la gráfica ("Minutos al Inicio")
    eje_x = df_p1.loc[train_size:, 'minutos_al_kickoff'].values
    valores_reales = y_test.values

    # 5. Visualización Avanzada (Dashboard Style)
    plt.figure(figsize=(11, 6))
    
    # Líneas principales
    plt.plot(eje_x, valores_reales, label='Flujo Real medido (Puerta 1)', color='#112F8B', linewidth=2.5, marker='o') # Azul RM
    plt.plot(eje_x, pred_mean, label='Predicción Estocástica (SARIMAX)', color='#E1A522', linewidth=2.5, linestyle='--', marker='x') # Dorado RM
    
    # El famoso Fan Chart !!!
    plt.fill_between(eje_x, 
                     pred_ci.iloc[:, 0], 
                     pred_ci.iloc[:, 1], 
                     color='#E1A522', alpha=0.25, label='Cono de Confianza (90%)')

    # Detectar el ambiente del partido de prueba para mostrar en el título
    clima_str = "🌧️ Lluvia" if X_test['clima_lluvia'].iloc[0] == 1 else "☀️ Despejado"
    riv_str = ["🟢 Baja", "🟡 Media", "🔴 Alta"][int(X_test['nivel_rivalidad'].iloc[0] - 1)]

    plt.title(f"Bernabéu Pulse: Predicción Real vs Modelo SARIMAX\n(Contexto partido a predecir - Clima: {clima_str} | Rivalidad: {riv_str})", fontweight='bold', fontsize=14)
    plt.xlabel('Minutos respecto al Pitazo Inicial (0 = Kickoff)', fontsize=11)
    plt.ylabel('Cantidad de Personas en ese corte de 15 min', fontsize=11)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axvline(0, color='red', linestyle='-', alpha=0.6, label='🚨 KICKOFF', linewidth=2)
    
    # Limitar el eje Y para que no empiece de negativo debido a las bandas anchas
    plt.ylim(bottom=0)
    
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    # Guardar en local y mostrar
    nombre_img = "grafico_sarimax_prueba.png"
    plt.savefig(nombre_img, dpi=300)
    print(f"\n✅ ¡ÉXITO! Visualización exportada como '{nombre_img}'.")
    print("Abre esa imagen para ver si el cono de confianza envuelve la curva real eficientemente.")

if __name__ == "__main__":
    probar_sarimax()
