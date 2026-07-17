import os
import pandas as pd
import snowflake.connector

# =========================================================================
# REVERSE ETL PIPELINE: SNOWFLAKE -> LOCAL MACHINE
# Extraccion "oficial" de features de Machine Learning desde la Nube
# =========================================================================

# ---------------------------------------------------------
# CREDENCIALES (El mismo Data Warehouse User)
# ---------------------------------------------------------
SNOWFLAKE_USER = 'DW_USER'
SNOWFLAKE_PASSWORD = 'PASSWORD_SEGURO'    
SNOWFLAKE_ACCOUNT = 'TVTFDWU-HY98136'  
SNOWFLAKE_DATABASE = 'REALMADRID_DB'
SNOWFLAKE_SCHEMA = 'FEATURE_STORE'
SNOWFLAKE_WAREHOUSE = 'COMPUTE_WH'
SNOWFLAKE_ROLE = 'DW_ROLE'

# Carpeta Oficial de Extracción que construirá la confianza para el jurado
TARGET_DIR = 'tablas_ml_snowflake'

def obtener_conexion():
    """Conecta con la API Nativa de Snowflake (Más rápido que SQLAlchemy)"""
    conn = snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        warehouse=SNOWFLAKE_WAREHOUSE,
        role=SNOWFLAKE_ROLE
    )
    return conn

def extraer_features_masivamente():
    """Busca todas las tablas usando API nativa y descarga a CSV localmente."""
    print(f"\n[+] Iniciando Reverse ETL Data Pipeline hacia Snowflake '{SNOWFLAKE_SCHEMA}'...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    destino_path = os.path.join(current_dir, TARGET_DIR)
    os.makedirs(destino_path, exist_ok=True)
    
    try:
        conn = obtener_conexion()
        cursor = conn.cursor()
        
        # Consulta Metadata de base de datos
        print("    -> Interrogando catalogo de tablas en Cloud...")
        cursor.execute(f"SHOW TABLES IN SCHEMA {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA};")
        
        # En fetchall(), la columna 1 correpsonde a "name" del SHOW TABLES
        tablas = [row[1] for row in cursor.fetchall()]
        print(f"[*] Entendido. Se han detectado {len(tablas)} tablas de IA (Features) en la Nube.")
        
        for index, table_name in enumerate(tablas, 1):
            print(f"\n    [Extraccion {index}/{len(tablas)}] Descargando: {table_name}")
            
            # Usando comillas dobles para respetar mayusculas/minusculas
            cursor.execute(f'SELECT * FROM "{SNOWFLAKE_DATABASE}"."{SNOWFLAKE_SCHEMA}"."{table_name}"')
            
            # API Nativa super rápida de Snowflake -> Pandas
            df_data = cursor.fetch_pandas_all()
            
            # Renombramos y guardamos
            table_name_lower = table_name.lower()
            nuevo_nombre_archivo = f"{table_name_lower}_cloud_extract.csv"
            ruta_csv = os.path.join(destino_path, nuevo_nombre_archivo)
            
            df_data.to_csv(ruta_csv, index=False)
            print(f"       ✅ Guardado en disco local '{nuevo_nombre_archivo}' con {len(df_data)} filas.")

    except Exception as e:
        print(f"\n❌ Error fatal de conexion o extraccion: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == '__main__':
    print("===================================================================")
    print(" INGENIERIA INVERSA: DESCARGA DE DATOS ENTRENAMIENTO (SNOWFLAKE)")
    print("===================================================================")
    extraer_features_masivamente()
    print(f"\n[OK] !TODOS LOS DATASETS DESCARGADOS CON EXITO EN '{TARGET_DIR}'!")
