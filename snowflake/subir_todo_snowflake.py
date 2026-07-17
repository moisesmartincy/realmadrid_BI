import os
import snowflake.connector
import pandas as pd
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

# =========================================================================
# SCRIPT MAESTRO DE MIGRACIÓN TOTAL A SNOWFLAKE
# Sube: CSVs de catálogos, CSVs de series temporales, modelos ML (.pkl.gz),
#        modelos Pulse (.pkl), modelo Deep Learning (.h5)
# Después de ejecutar esto, el sistema NO depende de nada local.
# =========================================================================

# ---------------------------------------------------------
# CREDENCIALES CENTRALES (igual que los otros scripts)
# ---------------------------------------------------------
SNOWFLAKE_USER     = 'DW_USER'
SNOWFLAKE_PASSWORD = 'PASSWORD_SEGURO'
SNOWFLAKE_ACCOUNT  = 'TVTFDWU-HY98136'
SNOWFLAKE_DATABASE = 'REALMADRID_DB'
SNOWFLAKE_WAREHOUSE = 'COMPUTE_WH'
SNOWFLAKE_ROLE     = 'DW_ROLE'

# Esquemas donde van los datos
SCHEMA_FEATURES    = 'FEATURE_STORE'
SCHEMA_CATALOGOS   = 'CATALOGOS'
SCHEMA_FORECASTING = 'FORECASTING'

# Nombre del Stage para TODOS los archivos binarios (modelos)
STAGE_NAME = 'ML_MODELS_STAGE'

# ---------------------------------------------------------
# RUTAS LOCALES BASE (relativas al script en /snowflake/)
# ---------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Raíz: csvModelosMachine/
WEBAPP_DIR          = os.path.join(BASE, 'webapp')
FILES_DIR           = os.path.join(WEBAPP_DIR, 'files')          # CSVs catálogos
FILES2_DIR          = os.path.join(WEBAPP_DIR, 'files2')         # CSVs forecasting + pkl series
MODELO_DIR          = os.path.join(WEBAPP_DIR, 'modelo')         # Pulse pkl
CLOUD_CACHE_DIR     = os.path.join(WEBAPP_DIR, 'backend', 'cloud_models_cache')  # Modelos ML
DEEP_LEARNING_DIR   = os.path.join(WEBAPP_DIR, 'deep_learning')  # .h5 y CSV sentimientos

# =========================================================================
# UTILIDADES DE CONEXIÓN
# =========================================================================

def obtener_conector():
    """Retorna snowflake.connector para comandos PUT/GET/DDL."""
    return snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SCHEMA_FEATURES,
        role=SNOWFLAKE_ROLE
    )

def obtener_engine_sqlalchemy(schema):
    """Retorna SQLAlchemy engine para subir DataFrames CSV."""
    return create_engine(URL(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        database=SNOWFLAKE_DATABASE,
        schema=schema,
        warehouse=SNOWFLAKE_WAREHOUSE,
        role=SNOWFLAKE_ROLE,
    ))

# =========================================================================
# PASO 1 — CREAR ESQUEMAS Y STAGE SI NO EXISTEN
# =========================================================================

def setup_snowflake(cursor):
    print("\n" + "="*60)
    print(" [1] CONFIGURANDO ENTORNO EN SNOWFLAKE")
    print("="*60)

    schemas = [SCHEMA_FEATURES, SCHEMA_CATALOGOS, SCHEMA_FORECASTING]
    for sch in schemas:
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_DATABASE}.{sch};")
        print(f"    -> Esquema '{sch}' validado.")

    cursor.execute(f"""
        CREATE STAGE IF NOT EXISTS {SNOWFLAKE_DATABASE}.{SCHEMA_FEATURES}.{STAGE_NAME}
        DIRECTORY = (ENABLE = TRUE)
        COMMENT = 'Stage único para todos los modelos ML, DL y Pulse del sistema Real Madrid AI Hub';
    """)
    print(f"    -> Stage '{STAGE_NAME}' validado en '{SCHEMA_FEATURES}'.")


# =========================================================================
# PASO 2 — SUBIR CSVs DE CATÁLOGOS (webapp/files/) → SCHEMA: CATALOGOS
# =========================================================================

def subir_csvs_catalogos():
    print("\n" + "="*60)
    print(f" [2] SUBIENDO CSVs DE CATÁLOGOS (webapp/files/) → {SCHEMA_CATALOGOS}")
    print("="*60)

    engine = obtener_engine_sqlalchemy(SCHEMA_CATALOGOS)
    subidos = 0

    for f in os.listdir(FILES_DIR):
        if f.endswith('.csv'):
            path = os.path.join(FILES_DIR, f)
            tabla = os.path.splitext(f)[0].upper()
            try:
                df = pd.read_csv(path)
                df.to_sql(tabla, con=engine, schema=SCHEMA_CATALOGOS,
                          if_exists='replace', index=False, chunksize=5000)
                print(f"    -> ✅ {tabla}  ({len(df)} filas)")
                subidos += 1
            except Exception as e:
                print(f"    -> ❌ {tabla}: {e}")

    print(f"    [*] Total subidos: {subidos} tablas.")


# =========================================================================
# PASO 3 — SUBIR CSVs DE FORECASTING (webapp/files2/) → SCHEMA: FORECASTING
# =========================================================================

def subir_csvs_forecasting():
    print("\n" + "="*60)
    print(f" [3] SUBIENDO CSVs DE FORECASTING (webapp/files2/) → {SCHEMA_FORECASTING}")
    print("="*60)

    engine = obtener_engine_sqlalchemy(SCHEMA_FORECASTING)
    subidos = 0

    for f in os.listdir(FILES2_DIR):
        if f.endswith('.csv'):
            path = os.path.join(FILES2_DIR, f)
            tabla = os.path.splitext(f)[0].upper()
            try:
                df = pd.read_csv(path)
                df.to_sql(tabla, con=engine, schema=SCHEMA_FORECASTING,
                          if_exists='replace', index=False, chunksize=5000)
                print(f"    -> ✅ {tabla}  ({len(df)} filas)")
                subidos += 1
            except Exception as e:
                print(f"    -> ❌ {tabla}: {e}")

    print(f"    [*] Total subidos: {subidos} tablas.")


# =========================================================================
# PASO 4 — SUBIR CSV SENTIMIENTOS DE DEEP LEARNING → SCHEMA: FEATURE_STORE
# =========================================================================

def subir_csvs_deep_learning():
    print("\n" + "="*60)
    print(f" [4] SUBIENDO CSV SENTIMIENTOS (webapp/deep_learning/) → {SCHEMA_FEATURES}")
    print("="*60)

    engine = obtener_engine_sqlalchemy(SCHEMA_FEATURES)
    subidos = 0

    for f in os.listdir(DEEP_LEARNING_DIR):
        if f.endswith('.csv'):
            path = os.path.join(DEEP_LEARNING_DIR, f)
            tabla = 'DATASET_SENTIMIENTOS_DL'
            try:
                df = pd.read_csv(path)
                df.to_sql(tabla, con=engine, schema=SCHEMA_FEATURES,
                          if_exists='replace', index=False, chunksize=8000)
                print(f"    -> ✅ {tabla}  ({len(df)} filas)")
                subidos += 1
            except Exception as e:
                print(f"    -> ❌ {tabla}: {e}")

    print(f"    [*] Total subidos: {subidos} CSV de DL.")


# =========================================================================
# PASO 5 — SUBIR TODOS LOS MODELOS BINARIOS (PUT) AL STAGE
# Incluye: cloud_models_cache/ (XGBoost .pkl.gz),
#          modelo/ (Pulse .pkl),
#          deep_learning/ (.h5),
#          files2/ (modelos series temporales .pkl)
# =========================================================================

def _put_file(cursor, local_path, remote_subpath):
    """Ejecuta el comando PUT de Snowflake para un archivo binario."""
    snow_path = local_path.replace('\\', '/')
    remote = f"@{SNOWFLAKE_DATABASE}.{SCHEMA_FEATURES}.{STAGE_NAME}/{remote_subpath}"
    sql = f"PUT 'file://{snow_path}' {remote} AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
    cursor.execute(sql)


def subir_modelos_binarios(cursor):
    print("\n" + "="*60)
    print(f" [5] SUBIENDO MODELOS BINARIOS AL STAGE '{STAGE_NAME}'")
    print("="*60)
    total = 0

    # ── 5a. Modelos ML desde cloud_models_cache/ (recursivo)
    print("\n  [5a] Modelos XGBoost (cloud_models_cache/):")
    EXTENSIONES_ML = ('.pkl', '.pkl.gz', '.json', '.json.gz')
    for root, dirs, files in os.walk(CLOUD_CACHE_DIR):
        # Calcular la ruta relativa respecto a cloud_models_cache/
        rel = os.path.relpath(root, CLOUD_CACHE_DIR).replace('\\', '/')
        if rel == '.':
            rel = ''
        for f in files:
            if f.endswith(EXTENSIONES_ML):
                local_path = os.path.join(root, f)
                remote_sub = f"cloud_models_cache/{rel}/{f}".replace('//', '/')
                try:
                    _put_file(cursor, local_path, remote_sub)
                    print(f"      -> ✅ {remote_sub}")
                    total += 1
                except Exception as e:
                    print(f"      -> ❌ {remote_sub}: {e}")

    # ── 5b. Modelos Pulse (modelo_lambda_ml.pkl, modelo_staff_ml.pkl)
    print("\n  [5b] Modelos Pulse (webapp/modelo/):")
    for f in os.listdir(MODELO_DIR):
        if f.endswith('.pkl'):
            local_path = os.path.join(MODELO_DIR, f)
            remote_sub = f"modelo_pulse/{f}"
            try:
                _put_file(cursor, local_path, remote_sub)
                print(f"      -> ✅ {remote_sub}")
                total += 1
            except Exception as e:
                print(f"      -> ❌ {remote_sub}: {e}")

    # ── 5c. Modelos de Series Temporales (.pkl en files2/)
    print("\n  [5c] Modelos Series Temporales (webapp/files2/):")
    for f in os.listdir(FILES2_DIR):
        if f.endswith('.pkl'):
            local_path = os.path.join(FILES2_DIR, f)
            remote_sub = f"modelo_forecasting/{f}"
            try:
                _put_file(cursor, local_path, remote_sub)
                print(f"      -> ✅ {remote_sub}")
                total += 1
            except Exception as e:
                print(f"      -> ❌ {remote_sub}: {e}")

    # ── 5d. Modelo Deep Learning (.h5)
    print("\n  [5d] Modelo Deep Learning (.h5):")
    for f in os.listdir(DEEP_LEARNING_DIR):
        if f.endswith('.h5'):
            local_path = os.path.join(DEEP_LEARNING_DIR, f)
            remote_sub = f"modelo_deep_learning/{f}"
            try:
                _put_file(cursor, local_path, remote_sub)
                print(f"      -> ✅ {remote_sub}")
                total += 1
            except Exception as e:
                print(f"      -> ❌ {remote_sub}: {e}")

    print(f"\n  [*] Total archivos binarios subidos al Stage: {total}")


# =========================================================================
# RESUMEN FINAL — Imprime qué hay en el Stage tras la subida
# =========================================================================

def listar_stage(cursor):
    print("\n" + "="*60)
    print(f" [6] CONTENIDO FINAL DEL STAGE '{STAGE_NAME}'")
    print("="*60)
    try:
        cursor.execute(f"LIST @{SNOWFLAKE_DATABASE}.{SCHEMA_FEATURES}.{STAGE_NAME};")
        rows = cursor.fetchall()
        for row in rows:
            name = row[0]
            size = row[1]
            print(f"    {name}  ({size} bytes)")
        print(f"\n  Total en Stage: {len(rows)} archivos.")
    except Exception as e:
        print(f"  Error al listar Stage: {e}")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  MIGRACION TOTAL -> SNOWFLAKE")
    print("  Real Madrid AI Hub - Sistema sin dependencias locales")
    print("=" * 60)

    # ── Paso 1-4: CSVs vía SQLAlchemy (no requieren cursor connector)
    subir_csvs_catalogos()
    subir_csvs_forecasting()
    subir_csvs_deep_learning()

    # ── Pasos 5-6: modelos binarios vía snowflake.connector (PUT)
    try:
        conn   = obtener_conector()
        cursor = conn.cursor()

        setup_snowflake(cursor)
        subir_modelos_binarios(cursor)
        listar_stage(cursor)

        print("\n" + "=" * 60)
        print("  ✅ MIGRACIÓN COMPLETA. El sistema puede operar 100% desde Snowflake.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR DE CONEXIÓN: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
