"""
=============================================================================
SERVICIO DE APIS EXTERNAS - FOOTBALL DATA + WEATHER
=============================================================================
Módulo backend para consumir datos en tiempo real de:
- football-data.org (fixtures, standings)
- weatherapi.com (pronóstico del clima)
=============================================================================
"""
import requests
from datetime import datetime, timedelta
import streamlit as st

FOOTBALL_API_KEY = "4bff0cdc3664422794febf9fa279ed9c"
WEATHER_API_KEY = "30654c544eba479998311726261404"
REAL_MADRID_ID = 86
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": FOOTBALL_API_KEY}

@st.cache_data(ttl=3600)
def obtener_proximos_partidos():
    """Obtiene los próximos partidos programados del Real Madrid."""
    try:
        r = requests.get(
            f"{BASE_URL}/teams/{REAL_MADRID_ID}/matches?status=SCHEDULED",
            headers=HEADERS, timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            partidos = []
            for m in data.get("matches", []):
                es_local = m["homeTeam"]["id"] == REAL_MADRID_ID
                rival_name = m["awayTeam"]["name"] if es_local else m["homeTeam"]["name"]
                rival_crest = m["awayTeam"].get("crest", "") if es_local else m["homeTeam"].get("crest", "")
                
                partidos.append({
                    "fecha": m["utcDate"][:10],
                    "hora": m["utcDate"][11:16],
                    "rival": rival_name,
                    "rival_crest": rival_crest,
                    "competicion": m["competition"]["name"],
                    "competicion_code": m["competition"].get("code", ""),
                    "competicion_emblem": m["competition"].get("emblem", ""),
                    "es_local": es_local,
                    "localidad": "Local" if es_local else "Visitante",
                    "jornada": m.get("matchday", ""),
                    "stage": m.get("stage", ""),
                })
            return partidos
        return []
    except Exception as e:
        print(f"[API ERROR] football-data fixtures: {e}")
        return []

@st.cache_data(ttl=3600)
def obtener_ultimos_resultados(limit=5):
    """Obtiene los últimos partidos jugados del Real Madrid."""
    try:
        r = requests.get(
            f"{BASE_URL}/teams/{REAL_MADRID_ID}/matches?status=FINISHED&limit={limit}",
            headers=HEADERS, timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            resultados = []
            for m in data.get("matches", [])[-limit:]:
                es_local = m["homeTeam"]["id"] == REAL_MADRID_ID
                score = m.get("score", {}).get("fullTime", {})
                goles_rm = score.get("home", 0) if es_local else score.get("away", 0)
                goles_rival = score.get("away", 0) if es_local else score.get("home", 0)
                
                if goles_rm > goles_rival:
                    resultado = "Victoria"
                elif goles_rm == goles_rival:
                    resultado = "Empate"
                else:
                    resultado = "Derrota"
                
                resultados.append({
                    "fecha": m["utcDate"][:10],
                    "rival": m["awayTeam"]["name"] if es_local else m["homeTeam"]["name"],
                    "rival_crest": (m["awayTeam"] if es_local else m["homeTeam"]).get("crest", ""),
                    "competicion": m["competition"]["name"],
                    "es_local": es_local,
                    "goles_rm": goles_rm,
                    "goles_rival": goles_rival,
                    "resultado": resultado,
                })
            return resultados
        return []
    except Exception as e:
        print(f"[API ERROR] football-data results: {e}")
        return []

@st.cache_data(ttl=3600)
def obtener_tabla_posiciones():
    """Obtiene la tabla de posiciones de La Liga."""
    try:
        r = requests.get(
            f"{BASE_URL}/competitions/PD/standings",
            headers=HEADERS, timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            tabla = []
            for t in data.get("standings", [{}])[0].get("table", []):
                tabla.append({
                    "posicion": t["position"],
                    "equipo": t["team"]["name"],
                    "escudo": t["team"].get("crest", ""),
                    "pj": t["playedGames"],
                    "g": t["won"],
                    "e": t["draw"],
                    "p": t["lost"],
                    "gf": t["goalsFor"],
                    "gc": t["goalsAgainst"],
                    "dg": t["goalDifference"],
                    "pts": t["points"],
                })
            return tabla
        return []
    except Exception as e:
        print(f"[API ERROR] football-data standings: {e}")
        return []

@st.cache_data(ttl=3600)
def obtener_datos_rm_standings():
    """Extrae los datos del RM de la tabla para alimentar al modelo."""
    tabla = obtener_tabla_posiciones()
    rm_data = None
    for t in tabla:
        if "Real Madrid" in t["equipo"]:
            rm_data = t
            break
    return rm_data

@st.cache_data(ttl=3600)
def obtener_clima(ciudad="Madrid", fecha=None):
    """Obtiene pronóstico del clima para una ciudad y fecha."""
    try:
        if fecha:
            # Verificar si la fecha está dentro de los próximos 14 días
            fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
            hoy = datetime.now()
            dias_diff = (fecha_dt - hoy).days
            
            if dias_diff <= 14 and dias_diff >= 0:
                r = requests.get(
                    f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={ciudad}&days={dias_diff + 1}&lang=es",
                    timeout=10
                )
                if r.status_code == 200:
                    data = r.json()
                    for day in data.get("forecast", {}).get("forecastday", []):
                        if day["date"] == fecha:
                            d = day["day"]
                            return {
                                "temp_c": d["avgtemp_c"],
                                "temp_max": d["maxtemp_c"],
                                "temp_min": d["mintemp_c"],
                                "condicion": d["condition"]["text"],
                                "icono": "https:" + d["condition"]["icon"],
                                "lluvia_pct": d["daily_chance_of_rain"],
                                "viento_kph": d["maxwind_kph"],
                                "humedad": d["avghumidity"],
                            }
            
            # Si está fuera de rango, devolver estimación por mes
            mes = fecha_dt.month
            temp_estimada = {1:8, 2:10, 3:13, 4:16, 5:21, 6:27, 7:32, 8:31, 9:25, 10:18, 11:12, 12:9}
            return {
                "temp_c": temp_estimada.get(mes, 18),
                "temp_max": temp_estimada.get(mes, 18) + 5,
                "temp_min": temp_estimada.get(mes, 18) - 5,
                "condicion": "Estimación Histórica",
                "icono": "",
                "lluvia_pct": 20,
                "viento_kph": 15,
                "humedad": 50,
            }
        
        # Clima actual
        r = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={ciudad}&lang=es",
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            c = data["current"]
            return {
                "temp_c": c["temp_c"],
                "condicion": c["condition"]["text"],
                "icono": "https:" + c["condition"]["icon"],
                "lluvia_pct": c.get("precip_mm", 0),
                "viento_kph": c["wind_kph"],
                "humedad": c["humidity"],
            }
        return None
    except Exception as e:
        print(f"[API ERROR] weatherapi: {e}")
        return None


def calcular_racha(resultados):
    """Calcula la racha de victorias reciente (últimos 5 partidos) como porcentaje."""
    if not resultados:
        return 0.6  # Default
    ultimos = resultados[-5:]
    victorias = sum(1 for r in ultimos if r["resultado"] == "Victoria")
    return round(victorias / len(ultimos), 2)
