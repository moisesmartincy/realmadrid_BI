"""
Explorador de APIs para el módulo de predicción de partidos.
Investiga qué datos disponibles hay en football-data.org y weatherapi.
"""
import requests
import json

FOOTBALL_API_KEY = "4bff0cdc3664422794febf9fa279ed9c"
WEATHER_API_KEY = "30654c544eba479998311726261404"
REAL_MADRID_ID = 86  # ID en football-data.org
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": FOOTBALL_API_KEY}

# =============================================
# 1. PARTIDOS PROGRAMADOS DEL REAL MADRID
# =============================================
print("=" * 70)
print("1. PARTIDOS PROGRAMADOS (SCHEDULED) DEL REAL MADRID")
print("=" * 70)
r = requests.get(f"{BASE_URL}/teams/{REAL_MADRID_ID}/matches?status=SCHEDULED", headers=HEADERS)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Total partidos: {data.get('resultSet', {}).get('count', 0)}")
    for m in data.get("matches", [])[:8]:
        print(f"  {m['utcDate'][:10]} | {m['homeTeam']['name']} vs {m['awayTeam']['name']} | {m['competition']['name']} | Jornada {m.get('matchday', '?')}")
else:
    print(f"Error: {r.text[:300]}")

# =============================================
# 2. TABLA DE POSICIONES - LA LIGA
# =============================================
print("\n" + "=" * 70)
print("2. TABLA DE POSICIONES - LA LIGA (PD)")
print("=" * 70)
r2 = requests.get(f"{BASE_URL}/competitions/PD/standings", headers=HEADERS)
print(f"Status: {r2.status_code}")
if r2.status_code == 200:
    data2 = r2.json()
    standings = data2.get("standings", [{}])[0].get("table", [])
    print(f"{'Pos':>3} {'Equipo':30s} {'PJ':>4} {'G':>3} {'E':>3} {'P':>3} {'GF':>4} {'GC':>4} {'DG':>4} {'Pts':>4}")
    print("-" * 70)
    for t in standings[:20]:
        print(f"{t['position']:>3} {t['team']['name']:30s} {t['playedGames']:>4} {t['won']:>3} {t['draw']:>3} {t['lost']:>3} {t['goalsFor']:>4} {t['goalsAgainst']:>4} {t['goalDifference']:>4} {t['points']:>4}")
    
    # Mostrar estructura completa de un equipo
    print("\n  ESTRUCTURA COMPLETA DE UN REGISTRO:")
    print(json.dumps(standings[0], indent=2, default=str))
else:
    print(f"Error: {r2.text[:300]}")

# =============================================
# 3. PARTIDOS TERMINADOS RECIENTES
# =============================================
print("\n" + "=" * 70)
print("3. ULTIMOS PARTIDOS JUGADOS DEL REAL MADRID")
print("=" * 70)
r3 = requests.get(f"{BASE_URL}/teams/{REAL_MADRID_ID}/matches?status=FINISHED&limit=5", headers=HEADERS)
print(f"Status: {r3.status_code}")
if r3.status_code == 200:
    data3 = r3.json()
    for m in data3.get("matches", [])[-5:]:
        score = m.get("score", {}).get("fullTime", {})
        print(f"  {m['utcDate'][:10]} | {m['homeTeam']['name']} {score.get('home','-')} - {score.get('away','-')} {m['awayTeam']['name']} | {m['competition']['name']}")
    
    # Estructura completa de un partido
    print("\n  ESTRUCTURA COMPLETA DE UN PARTIDO:")
    if data3.get("matches"):
        print(json.dumps(data3["matches"][-1], indent=2, default=str))

# =============================================
# 4. DETALLE COMPLETO DEL EQUIPO
# =============================================
print("\n" + "=" * 70)
print("4. DETALLE DEL EQUIPO REAL MADRID")
print("=" * 70)
r4 = requests.get(f"{BASE_URL}/teams/{REAL_MADRID_ID}", headers=HEADERS)
print(f"Status: {r4.status_code}")
if r4.status_code == 200:
    team = r4.json()
    print(f"  Nombre: {team.get('name')}")
    print(f"  Short: {team.get('shortName')}")
    print(f"  Venue: {team.get('venue')}")
    print(f"  Squad ({len(team.get('squad', []))} jugadores):")
    for p in team.get('squad', [])[:5]:
        print(f"    {p.get('name')} | {p.get('position')} | Nac: {p.get('nationality')}")

# =============================================
# 5. WEATHER API - CLIMA MADRID
# =============================================
print("\n" + "=" * 70)
print("5. WEATHERAPI - CLIMA EN MADRID")
print("=" * 70)
rw = requests.get(f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q=Madrid&days=3&lang=es")
print(f"Status: {rw.status_code}")
if rw.status_code == 200:
    weather = rw.json()
    for day in weather.get("forecast", {}).get("forecastday", []):
        d = day["day"]
        print(f"  {day['date']} | {d['avgtemp_c']}C | {d['condition']['text']} | Lluvia: {d['daily_chance_of_rain']}% | Viento: {d['maxwind_kph']} km/h")
    print("\n  ESTRUCTURA COMPLETA forecast day:")
    print(json.dumps(weather["forecast"]["forecastday"][0]["day"], indent=2, default=str))
