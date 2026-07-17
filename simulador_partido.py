import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generar_datos_historicos_partidos(num_partidos=80, output_file='historico_partidos_bernabeu.csv'):
    # Semillas para reproducibilidad
    np.random.seed(42)
    random.seed(42)

    CAPACIDAD_MAXIMA = 81044 # Capacidad actual aprox del Santiago Bernabéu

    # Listas de variables
    rivales_top = ["Barcelona", "Atlético", "Dortmund", "Bayern", "Man City", "PSG", "Liverpool", "Juventus", "Milan"]
    rivales_med = ["Sevilla", "Real Sociedad", "Athletic", "Betis", "Villarreal", "Valencia", "Chelsea", "Leipzig"]
    rivales_bajos = ["Getafe", "Alavés", "Cádiz", "Mallorca", "Osasuna", "Rayo Vallecano", "Elche", "Granada", "Celta", "Espanyol"]

    # Distancia en kilómetros aproximada desde Madrid
    distancias = {
        "Barcelona": 505, "Atlético": 8, "Dortmund": 1430, "Bayern": 1480, 
        "Man City": 1450, "PSG": 1050, "Liverpool": 1450, "Juventus": 1060, "Milan": 1180,
        "Sevilla": 390, "Real Sociedad": 350, "Athletic": 320, "Betis": 390, 
        "Villarreal": 310, "Valencia": 300, "Chelsea": 1260, "Leipzig": 1750,
        "Getafe": 14, "Alavés": 280, "Cádiz": 490, "Mallorca": 550, 
        "Osasuna": 310, "Rayo Vallecano": 6, "Elche": 350, "Granada": 360, 
        "Celta": 460, "Espanyol": 500
    }

    competiciones = ["Liga", "Champions", "Copa del Rey"]
    
    # Iniciar fecha alrededor de agosto 2021
    fecha_actual = datetime(2021, 8, 15)
    
    datos = []
    
    for i in range(num_partidos):
        # Avanzar la fecha
        dias_salto = random.randint(5, 14)
        while fecha_actual.month in [6, 7] and dias_salto < 30:
            fecha_actual += timedelta(days=30) 
            
        fecha_actual += timedelta(days=dias_salto)
        
        # Determinar competicion
        competicion = np.random.choice(competiciones, p=[0.70, 0.20, 0.10])
        
        # Determinar rival y la importancia base
        if competicion == "Champions":
            rival = random.choice(rivales_top + rivales_med)
            importancia = 1 if rival in rivales_top else np.random.choice([0, 1], p=[0.4, 0.6])
        elif competicion == "Copa del Rey":
            rival = random.choice(rivales_bajos + rivales_med)
            importancia = 0
        else: # Liga
            rival = random.choice(rivales_top + rivales_med + rivales_bajos)
            importancia = 1 if rival in rivales_top else (1 if np.random.rand() < 0.15 else 0)
            
        # Generar variables contextuales independientes
        posicion_tabla = np.random.choice([1, 2, 3], p=[0.60, 0.30, 0.10])
        hora = np.random.choice(["tarde", "noche"], p=[0.35, 0.65])
        
        mes = fecha_actual.month
        if mes in [6, 7, 8, 9]:
            temperatura = int(np.random.normal(28, 4))
        elif mes in [12, 1, 2]:
            temperatura = int(np.random.normal(8, 4))
        else:
            temperatura = int(np.random.normal(16, 5))
            
        racha_equipo = int(np.random.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
                                            p=[0.01, 0.01, 0.02, 0.02, 0.04, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15]))
        
        cracks_disponibles = int(np.random.choice([1, 0], p=[0.85, 0.15]))
        distancia_rival_km = distancias.get(rival, 500)

        # ==========================================
        # MODELO ESTRUCTURAL DE ASISTENCIA
        # ==========================================
        # 1. Intercepto base y Precio según competición e importancia
        if importancia == 1 and competicion in ["Liga", "Champions"]:
            asistencia_base = 78500
            precio_base = 135
        elif competicion == "Champions":
            asistencia_base = 73000
            precio_base = 95
        elif competicion == "Copa del Rey":
            asistencia_base = 55000
            precio_base = 45
        else: # Liga regular
            asistencia_base = 67000
            precio_base = 65
            
        # Generar el precio cobrado con algo de varianza
        precio_promedio = int(np.random.normal(precio_base, 15))
        
        # Efecto Cuadrático de Precio (Punto Óptimo)
        # La licenciada quiere un punto medio: precios muy bajos indican baja expectativa,
        # precios muy altos reducen la demanda drásticamente.
        # Asumimos que el precio percibido como "justo/hype" máximo es de 75 euros.
        diferencia_optimo = precio_promedio - 75
        efecto_precio = -4.5 * (diferencia_optimo ** 2) 
        
        asistencia_esperada = asistencia_base + efecto_precio
            
        # 2. Efecto de Cracks Disponibles
        if cracks_disponibles == 0:
            asistencia_esperada -= 2500  # Fuerte impacto negativo si faltan estrellas
            
        # 3. Efecto de Racha del Equipo
        if racha_equipo >= 12:
            asistencia_esperada += 2000  # Euforia
        elif racha_equipo <= 6:
            asistencia_esperada -= 3500  # Desencanto
            
        # 4. Efecto de Posición en Tabla
        if posicion_tabla == 1:
            asistencia_esperada += 1000
        elif posicion_tabla == 3:
            asistencia_esperada -= 2500
            
        # 5. Efecto de Distancia del Rival y Clima (Interactúan con la Importancia)
        if importancia == 0:
            # Si el partido no es top, la distancia importa mucho más (menos fans visitantes y menos turismo)
            if distancia_rival_km < 100:
                asistencia_esperada += 2000 # Derbis regionales llenan más
            elif distancia_rival_km > 600:
                asistencia_esperada -= 1500 # Equipos lejanos traen poca afición
                
            # Clima extremo y horarios afectan más a partidos de baja importancia
            if temperatura < 8 or temperatura > 32:
                asistencia_esperada -= 1800
            if hora == "noche" and temperatura < 10:
                asistencia_esperada -= 1200 # Noches frías desmotivan
                
        # Añadir ruido blanco (varianza natural)
        asistencia = int(np.random.normal(asistencia_esperada, 1200))
            
        # Clipping estricto a las capacidades del estadio
        asistencia = max(min(asistencia, CAPACIDAD_MAXIMA), 35000)
        precio_promedio = max(min(precio_promedio, 250), 30)
        capacidad_utilizada = round((asistencia / CAPACIDAD_MAXIMA) * 100, 1)
        
        # ==========================================
        # RESULTADOS DEL PARTIDO
        # ==========================================
        # La probabilidad de ganar baja un poco si faltan cracks o si hay mala racha
        prob_victoria = 0.75
        if cracks_disponibles == 0: prob_victoria -= 0.10
        if racha_equipo <= 6: prob_victoria -= 0.10
        
        prob_empate = (1 - prob_victoria) * 0.7
        prob_derrota = (1 - prob_victoria) * 0.3
        
        resultado = np.random.choice(["victoria", "empate", "derrota"], p=[prob_victoria, prob_empate, prob_derrota])
        
        if resultado == "victoria":
            goles_favor = int(np.random.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.35, 0.3, 0.15, 0.07, 0.03]))
        elif resultado == "empate":
            goles_favor = int(np.random.choice([0, 1, 2, 3], p=[0.25, 0.45, 0.25, 0.05]))
        else: # derrota
            goles_favor = int(np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1]))
            
        # Insertar valores base del usuario explícitamente en el inicio si se requiere
        if i == 0:
            datos.append(["2024-09-15", "Barcelona", "Liga", 79500, 95, 1, 1, "noche", 22, "victoria", 3, 98.1, 15, 1, 505])
        elif i == 1:
            datos.append(["2024-09-22", "Atlético", "Liga", 76200, 85, 1, 1, "tarde", 25, "empate", 1, 94.0, 13, 0, 8])
        elif i == 2:
            datos.append(["2024-09-29", "Getafe", "Liga", 64800, 45, 1, 0, "tarde", 18, "victoria", 2, 79.9, 10, 1, 14])
        elif i == 3:
            datos.append(["2024-10-06", "Villarreal", "Liga", 67300, 55, 2, 0, "noche", 16, "victoria", 1, 83.0, 12, 1, 310])
        elif i == 4:
            datos.append(["2024-10-20", "Dortmund", "Champions", 80100, 120, 2, 1, "noche", 14, "victoria", 4, 98.8, 14, 1, 1430])
        else:
            datos.append([
                fecha_actual.strftime("%Y-%m-%d"), rival, competicion, asistencia, precio_promedio,
                posicion_tabla, importancia, hora, temperatura, resultado, goles_favor, capacidad_utilizada,
                racha_equipo, cracks_disponibles, distancia_rival_km
            ])
            
    df = pd.DataFrame(datos, columns=[
        "fecha", "rival", "competicion", "asistencia", "precio_promedio", 
        "posicion_tabla", "importancia", "hora", "temperatura", 
        "resultado", "goles_favor", "capacidad_utilizada",
        "racha_equipo", "cracks_disponibles", "distancia_rival_km"
    ])
    
    df['fecha_dt'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(by='fecha_dt').drop('fecha_dt', axis=1).reset_index(drop=True)
    
    df.to_csv(output_file, index=False)
    print(f"Archivo generado exitosamente: {output_file}")
    print(f"Total registros: {len(df)}")
    print(df.head(10))

if __name__ == "__main__":
    archivo_csv = "historico_partidos_bernabeu.csv"
    if os.path.exists(archivo_csv):
        os.remove(archivo_csv)
        print(f"Archivo anterior eliminado: {archivo_csv}")
    generar_datos_historicos_partidos(80, archivo_csv)
