import os
import snowflake.connector

# =========================================================================
# SCRIPT DE MLOPS: SUBIDA DE MODELOS ENTRENADOS A SNOWFLAKE (STAGE)
# Toma los .pkl de Machine Learning y los aloja seguros en la nube
# =========================================================================

# ---------------------------------------------------------
# CREDENCIALES (Reemplaza con el rol Data Warehouse)
# ---------------------------------------------------------
SNOWFLAKE_USER = 'DW_USER'
SNOWFLAKE_PASSWORD = 'PASSWORD_SEGURO'    
SNOWFLAKE_ACCOUNT = 'TVTFDWU-HY98136'  # Ej: xyz123.us-east-1
SNOWFLAKE_DATABASE = 'REALMADRID_DB'
SNOWFLAKE_SCHEMA = 'FEATURE_STORE'
SNOWFLAKE_WAREHOUSE = 'COMPUTE_WH'
SNOWFLAKE_ROLE = 'DW_ROLE'

# Nombre del "Stage" virtual que albergará los archivos .pkl en la nube
STAGE_NAME = "ML_MODELS_STAGE"

def obtener_conexion():
    """Establece conexión directa con el cliente de Snowflake."""
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        role=SNOWFLAKE_ROLE
    )
    return conn

def crear_stage_si_no_existe(cursor):
    """Crea la 'carpeta' virtual de alta tecnología dentro de Snowflake."""
    print(f"\n[+] Configurando el entorno de Machine Learning en Snowflake...")
    sql_crear_stage = f"CREATE STAGE IF NOT EXISTS {STAGE_NAME} DIRECTORY = (ENABLE = TRUE) COMMENT = 'Almacen de Modelos XGBoost y Metadata';"
    cursor.execute(sql_crear_stage)
    print(f"    -> Comando enviado. Stage '{STAGE_NAME}' validado en el esquema '{SNOWFLAKE_SCHEMA}'.")

def subir_modelos_ml(cursor):
    """Busca todas las carpetas locales de modelos y usa el comando PUT para subirlas."""
    print(f"\n[+] Buscando modelos entrenados en el escritorio local...")
    
    current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'REALMADRID_DB', 'FEATURE_STORE')
    archivos_subidos = 0

    # Explorar todas las carpetas que hemos creado que digan "modelo_"
    for root, dirs, files in os.walk(current_dir):
        # Filtramos para no subir basura, solo entraremos en entornos creados por XGBoost
        folder_name = os.path.basename(root)
        if folder_name.startswith("modelo") and "entrenado" in folder_name:
            print(f"\n   Encontrada carpeta MLOps: '{folder_name}'")
            
            for file in files:
                # Solo queremos subir los binarios generados (.pkl o .json metadata)
                if file.endswith('.pkl') or file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    
                    # Convertir rutas de Windows a formato aceptado por Snowflake URI (file://...)
                    file_path_snow = file_path.replace('\\', '/')
                    
                    # Target path en el Stage remoto (creamos subcarpetas en la nube para mantener orden)
                    remote_path = f"@{STAGE_NAME}/{folder_name}"
                    
                    try:
                        print(f"       🔼 Subiendo algoritmo: {file} ...", end=" ")
                        
                        # EL COMANDO PUT: Encripta, transfiere y registra el archivo en Snowpark
                        sql_put = f"PUT 'file://{file_path_snow}' {remote_path} AUTO_COMPRESS=TRUE OVERWRITE=TRUE;"
                        cursor.execute(sql_put)
                        
                        archivos_subidos += 1
                        print("¡Completado!")
                    except Exception as e:
                        print(f"\n       ❌ Error al subir {file}: {e}")

    print(f"\n[*] MLOps Deployment Terminado. Se han transportado {archivos_subidos} archivos binarios de Inteligencia Artificial a la nube segura de Snowflake.")

if __name__ == '__main__':
    print("===================================================================")
    print(" CARGA DE ARCHIVOS BINARIOS DE MODELOS A LA NUBE (XGBOOST)")
    print("===================================================================")
    try:
        conn = obtener_conexion()
        cursor = conn.cursor()
        
        # Flujo
        crear_stage_si_no_existe(cursor)
        subir_modelos_ml(cursor)
        
    except Exception as general_err:
        print(f"\nERROR DE CONEXION: {general_err}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
