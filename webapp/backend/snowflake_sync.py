import os
import json
import joblib
import pandas as pd
import snowflake.connector

# =========================================================================
# SNOWFLAKE SYNC — Backend unificado del Real Madrid AI Hub
# =========================================================================

# ── CREDENCIALES ──────────────────────────────────────────────────────────
SNOWFLAKE_USER      = 'DW_USER'
SNOWFLAKE_PASSWORD  = 'PASSWORD_SEGURO'
SNOWFLAKE_ACCOUNT   = 'TVTFDWU-HY98136'
SNOWFLAKE_DATABASE  = 'REALMADRID_DB'
SNOWFLAKE_SCHEMA    = 'FEATURE_STORE'
SNOWFLAKE_WAREHOUSE = 'COMPUTE_WH'
SNOWFLAKE_ROLE      = 'DW_ROLE'
STAGE_NAME          = 'ML_MODELS_STAGE'

SCHEMA_CATALOGOS   = 'CATALOGOS'
SCHEMA_FORECASTING = 'FORECASTING'

# Rutas locales
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))   # webapp/backend/
_WEBAPP_DIR  = os.path.dirname(_BACKEND_DIR)                # webapp/
_CACHE_DIR   = os.path.join(_BACKEND_DIR, 'cloud_models_cache')

# ── SINGLETON DE CONEXIÓN ────────────────────────────────────────────────
_conexion_cache = None

def obtener_conexion():
    """Mantiene una conexión Singleton reutilizable a Snowflake."""
    global _conexion_cache
    try:
        if _conexion_cache is not None and not _conexion_cache.is_closed():
            return _conexion_cache
    except Exception:
        pass
    _conexion_cache = snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        warehouse=SNOWFLAKE_WAREHOUSE,
        role=SNOWFLAKE_ROLE
    )
    return _conexion_cache


# =========================================================================
# SYNC DE MODELOS BINARIOS (Stage → disco local)
# =========================================================================

def sincronizar_modelos_cloud():
    """
    Descarga todos los modelos del Stage ML_MODELS_STAGE al disco local.

    Reglas de mapeo Stage → Local:
      - Archivos bajo cloud_models_cache/* en el stage  →  IGNORADOS (duplicados)
      - modelo_pulse/*               →  webapp/modelo_pulse/
      - modelo_forecasting/*         →  webapp/modelo_forecasting/
      - modelo_deep_learning/*       →  webapp/modelo_deep_learning/
      - modelos_entrenados/*         →  webapp/backend/cloud_models_cache/modelos_entrenados/
      - modelo_*_entrenado/*         →  webapp/backend/cloud_models_cache/modelo_*_entrenado/

    Snowflake PUT provoca doble-anidación para archivos .gz/.pkl:
      stage: modelo_pulse/file.pkl/file.pkl
      → descargamos a:  webapp/modelo_pulse/file.pkl
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)

    print(f"[SYNC] Intentando conectar a Snowflake para descargar artefactos IA desde @{STAGE_NAME} ...")

    try:
        conn   = obtener_conexion()
        cursor = conn.cursor()
        cursor.execute(
            f"LIST @{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{STAGE_NAME}"
        )
        archivos = cursor.fetchall()

        ok = 0
        skip = 0
        for arch in archivos:
            nombre_stage = arch[0]   # ej: ml_models_stage/modelos_entrenados/xg.pkl.gz
            partes = nombre_stage.split('/')

            # ── Ignorar duplicados que tienen "cloud_models_cache" en la ruta ──
            if 'cloud_models_cache' in partes:
                skip += 1
                continue

            # Partes relativas (sin el nombre del stage al inicio)
            rel_parts = partes[1:]
            if not rel_parts:
                skip += 1
                continue

            top_folder = rel_parts[0]   # Primera carpeta (ej: "modelo_pulse")
            filename   = rel_parts[-1]  # Nombre real del archivo

            # ── Determinar directorio local destino ──────────────────────
            CARPETAS_EXTERNAS = ('modelo_pulse', 'modelo_forecasting', 'modelo_deep_learning')

            if top_folder in CARPETAS_EXTERNAS:
                # Estos tienen doble-anidación: modelo_pulse/file.pkl/file.pkl
                # → descargamos el archivo directamente a webapp/modelo_pulse/
                local_dir = os.path.join(_WEBAPP_DIR, top_folder)
            else:
                # Modelos XGBoost → cloud_models_cache/<subfolder>/
                subfolder_parts = rel_parts[:-1]
                if subfolder_parts:
                    local_dir = os.path.join(_CACHE_DIR, *subfolder_parts)
                else:
                    local_dir = _CACHE_DIR

            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)

            # ── Saltar si ya existe con contenido ───────────────────────
            if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                skip += 1
                continue

            # ── GET del archivo ──────────────────────────────────────────
            stage_path   = '/'.join(rel_parts)          # Ruta relativa dentro del stage
            local_snow   = local_dir.replace('\\', '/')

            query_get = (
                f"GET @{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{STAGE_NAME}/{stage_path}"
                f" 'file://{local_snow}'"
            )
            try:
                cursor.execute(query_get)
                print(f"  ✓ {filename}")
                ok += 1
            except Exception as e:
                print(f"  ✗ {filename}: {e}")

        print(f"[OK] Sync completo. Descargados={ok} | Salteados={skip}")
        cursor.close()
    except Exception as e:
        print(f"[WARNING] No se pudo sincronizar con Snowflake (se usarán los modelos locales de caché): {e}")
    
    return _CACHE_DIR


# =========================================================================
# LECTURA DE TABLAS DESDE SNOWFLAKE
# Resuelve automáticamente mayúsculas/minúsculas (SQLAlchemy crea en lower)
# =========================================================================

def traer_tabla(nombre_tabla: str, schema: str = SNOWFLAKE_SCHEMA,
                parse_dates: list = None) -> pd.DataFrame:
    """
    Descarga una tabla de Snowflake como DataFrame.
    Intenta primero sin comillas (case-insensitive, Snowflake pliega a UPPER),
    y si falla intenta con comillas en minúsculas (tablas creadas por SQLAlchemy).
    """
    try:
        conn   = obtener_conexion()
        cursor = conn.cursor()
        # Intento 1: sin comillas → Snowflake interpreta como UPPERCASE (robusto)
        try:
            sql = f'SELECT * FROM {SNOWFLAKE_DATABASE}.{schema}.{nombre_tabla.upper()}'
            cursor.execute(sql)
        except Exception:
            # Intento 2: con comillas en minúsculas → tablas creadas vía SQLAlchemy
            sql = (f'SELECT * FROM "{SNOWFLAKE_DATABASE}"."{schema}"'
                   f'."{nombre_tabla.lower()}"')
            cursor.execute(sql)

        df = cursor.fetch_pandas_all()

        # Normalizar columnas a minúsculas para compatibilidad con código existente
        df.columns = [c.lower() for c in df.columns]

        if parse_dates:
            for col in parse_dates:
                match = [c for c in df.columns if c.lower() == col.lower()]
                if match:
                    df[match[0]] = pd.to_datetime(df[match[0]])

        cursor.close()
        return df

    except Exception as e:
        print(f"[WARNING] traer_tabla({nombre_tabla}, schema={schema}): {e}")
        return pd.DataFrame()


def traer_json_tabla(nombre_tabla: str, schema: str = SNOWFLAKE_SCHEMA) -> dict:
    """Descarga una tabla con columna JSON_DATA y la parsea."""
    try:
        conn   = obtener_conexion()
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT JSON_DATA FROM {SNOWFLAKE_DATABASE}.{schema}.{nombre_tabla.upper()}'
        )
        row = cursor.fetchone()
        cursor.close()
        return json.loads(row[0]) if row else {}
    except Exception as e:
        print(f"[WARNING] traer_json_tabla({nombre_tabla}): {e}")
        return {}


# =========================================================================
# HELPERS DE RUTAS DE MODELOS
# =========================================================================

def ruta_modelo(subcarpeta: str) -> str:
    """Ruta local de un modelo XGBoost dentro de cloud_models_cache/."""
    return os.path.join(_CACHE_DIR, subcarpeta)


def ruta_modelo_pulse() -> str:
    """Ruta local de los modelos Pulse descargados del Stage."""
    return os.path.join(_WEBAPP_DIR, 'modelo_pulse')


def ruta_modelo_forecasting() -> str:
    """Ruta local de los modelos de Forecasting descargados del Stage."""
    return os.path.join(_WEBAPP_DIR, 'modelo_forecasting')


def ruta_modelo_dl() -> str:
    """Ruta local del modelo Deep Learning descargado del Stage."""
    return os.path.join(_WEBAPP_DIR, 'modelo_deep_learning')


# =========================================================================
# ALIAS LEGADO
# =========================================================================

def traer_datos_tabla(nombre_tabla: str) -> pd.DataFrame:
    """Alias de traer_tabla para compatibilidad con código legado."""
    return traer_tabla(nombre_tabla, schema=SNOWFLAKE_SCHEMA)
