# Real Madrid AI Hub & Business Intelligence

**Por: MOISES MARTIN CONDORI YUJRA**

Bienvenido al repositorio del **Real Madrid AI Hub**, una plataforma integral de Business Intelligence impulsada por Inteligencia Artificial, Machine Learning y Modelos Econométricos. Este proyecto está diseñado para analizar, predecir y optimizar variables clave del club, como la asistencia al Estadio Santiago Bernabéu, usando técnicas estadísticas avanzadas y un asistente virtual multimodal.

---

## 📊 Descripción del Proyecto

El objetivo principal de esta plataforma es proporcionar al área directiva del Real Madrid herramientas cuantitativas y analíticas para tomar decisiones basadas en datos (Data-Driven). El sistema responde preguntas críticas como:
- *¿Por qué algunos partidos llenan el estadio al 100% y otros apenas superan el 55%?*
- *¿Cuál es el impacto financiero de la ausencia de jugadores estrella?*
- *¿Cuál es el precio óptimo para maximizar el revenue de taquilla en un partido específico?*

Para facilitar la interpretación de estos datos, el proyecto integra un **Asistente Virtual basado en IA (Gemini 2.5 Flash)** capaz de leer gráficas, analizar tendencias y responder preguntas de negocio de manera interactiva.

---

## 🚀 Características Principales

### 1. Modelos Econométricos de Asistencia (OLS)
Utilizamos **Regresión Lineal Múltiple (Ordinary Least Squares)** para entender los factores que determinan la asistencia al Bernabéu. Analizamos 9 variables clave divididas en tres categorías:
- **Económicas:** Efecto del precio (con comportamiento cuadrático y elasticidad precio-demanda).
- **Deportivas:** Importancia del partido, racha del equipo, disponibilidad de cracks y distancia del equipo rival.
- **Contextuales:** Horario y temperatura ambiente.

### 2. Forecasting y Series Temporales
Integración de modelos predictivos avanzados como:
- **SARIMAX**, **Prophet** y **Filtros de Kalman** para pronósticos precisos a largo plazo.
- Archivos como `prueva_sarimax.py` demuestran la validación rigurosa sobre los últimos intervalos de datos.

### 3. Asistente Virtual Inteligente (IA Multimodal)
- **Tecnología:** Google Gemini 2.5 Flash.
- **Multimodalidad:** Capaz de recibir imágenes (screenshots de los gráficos del dashboard) y devolver un análisis completo de picos, caídas y tendencias de las visualizaciones generadas por el sistema.
- **Integración Segura:** Configurado vía `.env` y encapsulado en la interfaz de usuario con Pop-overs inteligentes sobre cualquier pestaña del sistema sin perder el contexto visual.

### 4. Machine Learning & Deep Learning
El ecosistema incluye predictores de:
- **Fatiga de Jugadores** (vía XGBoost).
- **Sentimientos y emociones** de los fans usando Modelos de Deep Learning (Redes Neuronales / `.h5`).
- **Análisis de ventas, merchandising y modelos de colas.**

### 5. Integración Cloud
Sincronización automatizada con la base de datos y almacén Cloud de **Snowflake**, gestionando subida/descarga de *features* y exportación de modelos.

---

## 🛠️ Stack Tecnológico

| Componente | Herramienta/Tecnología |
| :--- | :--- |
| **Lenguaje Base** | Python 3 |
| **Frontend/UI** | Streamlit |
| **IA / LLM** | Google Gemini (google-generativeai SDK) |
| **Econometría & Estadísticas** | Statsmodels |
| **Machine Learning** | Scikit-Learn, XGBoost, Prophet |
| **Deep Learning** | TensorFlow / Keras |
| **Manipulación de Datos** | Pandas, Numpy |
| **Visualización** | Plotly (Express y Graph Objects) |
| **Base de Datos / Cloud** | Snowflake |

---

## ⚙️ Estructura del Proyecto

- `webapp/`: Contiene la aplicación principal de Streamlit, separada en módulos de Backend, Deep Learning, Forecasting y Pulso de Mercado.
- `validar_modelo/`: Scripts independientes para el entrenamiento, validación y predicción de diferentes KPIs (asistencia, ventas, merchandising, fatiga).
- `snowflake/`: Módulos de conexión, subida y descarga de *features* desde la nube.
- `modelos_exportados/`: Archivos `.pkl`, `.joblib` y `.h5` de modelos ya entrenados listos para predicción en tiempo real.

---

## 📈 Conclusión Ejecutiva

El **Real Madrid AI Hub** no es solo una herramienta de visualización de datos, sino un motor de recomendación integral. A través de este sistema, es posible fijar precios dinámicos, planificar estrategias operativas ante bajas de jugadores estrella, y mejorar significativamente la experiencia del fanético y los ingresos (Matchday Revenue) con respaldo puramente estadístico.
