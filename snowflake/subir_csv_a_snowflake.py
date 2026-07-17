import os
import pandas as pd
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

# =========================================================================
# SCRIPT DE INGESTA MASIVA MASIVA: LOCAL -> SNOWFLAKE CLOUD
# Sube todos los CSVs de la Arquitectura Medallón y los de Machine Learning
# =========================================================================

# ---------------------------------------------------------
# 1. TUS CREDENCIALES (Reemplaza con las tuyas)
# ---------------------------------------------------------
SNOWFLAKE_USER = 'DW_USER'
SNOWFLAKE_PASSWORD = 'PASSWORD_SEGURO'
SNOWFLAKE_ACCOUNT = 'TVTFDWU-HY98136' # Ej: xy12345.us-east-1
SNOWFLAKE_DATABASE = 'REALMADRID_DB'
SNOWFLAKE_WAREHOUSE = 'COMPUTE_WH'
SNOWFLAKE_ROLE = 'DW_ROLE' # O DATA_ENGINEER_ROLE

BASE_DIR = 'REALMADRID_DB'

# Archivos sueltos en el escritorio que corresponden a Machine Learning
ML_FILES = [
    'ftr_rendimiento_partidos.csv',   # Partidos (Simulacion 1)
    'ftr_valoracion_jugadores.csv',     # Valor mercado (Simulacion 2)
    'ftr_fatiga_medica.csv',       # Fatiga muscular (Simulacion 3)
    'ftr_asistencia_historica.csv',      # Asistencia (Simulacion 4)
    'ftr_ventas_matchday.csv',          # Ventas (Simulacion 5)
    'ftr_top_merchandising.csv'           # Top Merch (Simulacion 6)
]

def obtener_conexion(schema):
    """Crea la conexión segura usando SQLAlchemy orientada a un esquema específico."""
    engine = create_engine(URL(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        database=SNOWFLAKE_DATABASE,
        schema=schema,
        warehouse=SNOWFLAKE_WAREHOUSE,
        role=SNOWFLAKE_ROLE,
    ))
    return engine

def subir_carpeta_a_snowflake(ruta_carpeta, esquema_destino):
    print(f"\n[+] Escaneando carpeta local '{ruta_carpeta}' para subir a esquema '{esquema_destino}'...")
    if not os.path.exists(ruta_carpeta):
        print(f"    -> La ruta {ruta_carpeta} no existe. Omitiendo.")
        return

    # Usamos motor de BD
    engine = obtener_conexion(esquema_destino)

    archivos_subidos = 0
    # Walk recorre incluso subcarpetas (ej. postgresql_erp dentro de BRONZE)
    for root, dirs, files in os.walk(ruta_carpeta):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                # Nombre de la tabla sin extensión ".csv"
                table_name = os.path.splitext(file)[0].upper()
                
                try:
                    print(f"    -> Subiendo tabla: {table_name} ...", end=" ")
                    df = pd.read_csv(file_path)
                    
                    # Mandamos el DataFrame directo a Snowflake
                    df.to_sql(
                        name=table_name,
                        con=engine,
                        schema=esquema_destino,
                        if_exists='replace', # Si existe, la borra y la crea de nuevo
                        index=False,
                        chunksize=16000     # Sube en paquetes para optimizar red
                    )
                    archivos_subidos += 1
                    print(f"¡Exito! ({len(df)} filas)")
                except Exception as e:
                    print(f"Error: {e}")

    print(f"[*] Terminado. Se subieron {archivos_subidos} tablas al esquema {esquema_destino}.")

def subir_archivos_ml_a_feature_store():
    print(f"\n[+] Buscando archivos de Machine Learning para el FEATURE_STORE...")
    esquema = 'FEATURE_STORE'
    engine = obtener_conexion(esquema)
    
    # Apunta a la nueva carpeta donde el usuario organizó manualmente los archivos
    current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'REALMADRID_DB', 'FEATURE_STORE')
    archivos_subidos = 0
    
    for file in ML_FILES:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            table_name = os.path.splitext(file)[0].upper()
            try:
                print(f"    -> Subiendo DB de Machine Learning: {table_name} ...", end=" ")
                df = pd.read_csv(file_path)
                df.to_sql(name=table_name, con=engine, schema=esquema, if_exists='replace', index=False)
                archivos_subidos += 1
                print("¡Exito!")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"    -> (No se encontró el archivo {file})")
            
    print(f"[*] Terminado. Se subieron {archivos_subidos} tablas al FEATURE_STORE.")

if __name__ == '__main__':
    print("===================================================================")
    print(" INICIANDO MIGRACION A SNOWFLAKE CLOUD")
    print("===================================================================")
    
    # 1. Subir Bronce
    subir_carpeta_a_snowflake(os.path.join(BASE_DIR, 'BRONZE'), 'BRONZE')
    
    # 2. Subir Plata
    subir_carpeta_a_snowflake(os.path.join(BASE_DIR, 'SILVER'), 'SILVER')
    
    # 3. Subir Oro (Star Schema)
    subir_carpeta_a_snowflake(os.path.join(BASE_DIR, 'GOLD_BI'), 'GOLD_BI')
    
    # 4. Subir capas de Inteligencia Artificial
    subir_archivos_ml_a_feature_store()
    
    print("\n[OK] !TODOS LOS DATOS HAN SIDO INGESTEADOS A SNOWFLAKE CORRECTAMENTE!")
