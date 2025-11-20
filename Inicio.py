"""
streamlit_app_full_fixed.py

App Streamlit completa y corregida que:
- Lee correctamente CSV exportados desde InfluxDB (incluyendo filas de metadatos como #group, #datatype, #default).
- Detecta automáticamente la fila que contiene los nombres reales de columnas (time, temperature, humidity, device, location, lat, lon).
- Normaliza columnas (renombra a 'Time', 'temperatura', 'humedad'), parsea fechas y convierte valores numéricos.
- Conserva la interfaz: mapa, pestañas (Visualización, Estadísticas, Filtros, Información del Sitio).
- Muestra mensajes claros y ejemplos de valores problemáticos para que puedas arreglar el CSV si hace falta.
- Genera descarga de CSV filtrado.

Instrucciones de uso:
1) Reemplaza tu archivo anterior por este (o copia/pegalo en un nuevo .py).
2) Ejecuta: streamlit run streamlit_app_full_fixed.py
3) Sube el CSV exportado desde InfluxDB (o cualquier CSV con 'time' y columnas de temperatura/humedad).
"""
import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Análisis de Sensores - Mi Ciudad",
    page_icon="📊",
    layout="wide"
)

# Custom CSS (simple padding)
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title('📊 Análisis de datos de Sensores en Mi Ciudad')
st.markdown("""
    Esta aplicación permite analizar datos de temperatura y humedad
    recolectados por sensores en diferentes puntos de la ciudad.
    - Soporta CSV exportados desde InfluxDB (incluye eliminación automática de metadatos).
    - Detecta columna de tiempo y columnas de temperatura/humedad automáticamente.
""")

# Create map data for EAFIT (static)
eafit_location = pd.DataFrame({
    'lat': [6.2006],
    'lon': [-75.5783],
    'location': ['Universidad EAFIT']
})

# Display map
st.subheader("📍 Ubicación de los Sensores - Universidad EAFIT")
st.map(eafit_location, zoom=15)


# -------------------------
# Helper: robust CSV reader
# -------------------------
def read_influx_csv(uploaded_file) -> pd.DataFrame:
    """
    Lee un CSV que puede contener filas de metadatos (export de InfluxDB).
    - Detecta la línea que contiene los nombres reales de columnas (ej. 'time', 'temperature', 'humidity').
    - Lee el CSV usando esa línea como header.
    - Elimina columnas meta como '#group', '#datatype', '#default', 'result', 'table', 'Unnamed'.
    - Devuelve DataFrame o DataFrame vacío en caso de error.
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    # raw puede ser bytes o str
    if isinstance(raw, bytes):
        try:
            text = raw.decode('utf-8')
        except Exception:
            text = raw.decode('latin1', errors='replace')
    else:
        text = raw

    lines = text.splitlines()
    # patrón para detectar filas de header real
    header_pattern = re.compile(r'\b(time|timestamp|_time|temperature|temp|humidity|humedad|device|location|lat|lon)\b', flags=re.IGNORECASE)

    header_idx = None
    # buscar en las primeras 100 líneas para acelerar
    for i, line in enumerate(lines[:100]):
        # dividir por coma o punto y coma
        tokens = [t.strip().strip('"') for t in re.split(r',|;', line)]
        if any(header_pattern.search(tok) for tok in tokens):
            # asegurarnos que la línea no sea una fila de tipos (e.g. 'string,double,...')
            # heurística: si la línea tiene palabras (letras) y al menos una coincidencia con header_pattern
            if any(re.search(r'[A-Za-z]', tok) for tok in tokens):
                header_idx = i
                break

    # Si no encontramos, buscar en todo el archivo
    if header_idx is None:
        for i, line in enumerate(lines):
            tokens = [t.strip().strip('"') for t in re.split(r',|;', line)]
            if any(header_pattern.search(tok) for tok in tokens):
                if any(re.search(r'[A-Za-z]', tok) for tok in tokens):
                    header_idx = i
                    break

    if header_idx is None:
        st.error("No se pudo detectar la fila de encabezado con nombres de columna (no se encontró 'time'/'temperature'/'humidity').")
        return pd.DataFrame()

    # Intentar leer con pandas usando la línea detectada como header
    try:
        df = pd.read_csv(io.StringIO(text), header=header_idx)
    except Exception as e:
        st.error(f"Error leyendo CSV con header en la línea {header_idx}: {e}")
        return pd.DataFrame()

    # Eliminar columnas meta comunes
    cols_to_drop = [c for c in df.columns if isinstance(c, str) and (
        c.strip().startswith('#') or c.lower().startswith('result') or c.lower().startswith('table') or c.lower().startswith('unnamed')
    )]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')

    # Normalizar nombres de columnas: quitar espacios al inicio/fin
    df.columns = [str(c).strip() for c in df.columns]

    # Mostrar columnas detectadas
    st.info("Columnas detectadas en el CSV: " + ", ".join(list(df.columns)))

    return df


# -------------------------
# Process uploaded file
# -------------------------
uploaded_file = st.file_uploader('Seleccione archivo CSV (exportado desde InfluxDB o CSV estándar)', type=['csv'])

if uploaded_file is None:
    st.warning('Por favor, cargue un archivo CSV para comenzar el análisis.')
    st.stop()

# Try reading robustly
try:
    df1 = read_influx_csv(uploaded_file)
    if df1.empty:
        st.stop()
except Exception as e:
    st.error(f"Error al leer el CSV: {e}")
    st.stop()

# Normalize column names to lowercase for detection (but keep originals for display)
original_columns = list(df1.columns)
cols_lower = {c: c.lower() for c in original_columns}
df1.rename(columns=cols_lower, inplace=True)

# Detect time column: common names
time_candidates = [c for c in df1.columns if c in ('time', '_time', 'timestamp', 'date', 'fecha')]
chosen_time_col = time_candidates[0] if time_candidates else None

# If not found, search columns whose content looks like datetime
if not chosen_time_col:
    for c in df1.columns:
        # sample up to 10 non-null values
        sample = df1[c].dropna().astype(str).head(10).tolist()
        parsed = 0
        for s in sample:
            try:
                pd.to_datetime(s)
                parsed += 1
            except Exception:
                pass
        if parsed >= max(1, len(sample) // 2):
            chosen_time_col = c
            st.write(f"Columna de tiempo inferida por contenido: '{c}'")
            break

if not chosen_time_col:
    st.error("No se encontró columna de tiempo. Asegúrate de que el CSV incluya una columna con tiempos (time, _time, timestamp, date).")
    st.write("Columnas detectadas:", list(df1.columns))
    st.stop()

# Parse time column robustly (intentar formatos comunes)
df1[chosen_time_col] = pd.to_datetime(df1[chosen_time_col], errors='coerce', utc=True)

# If all NaT, try dayfirst True (for DD/MM/YYYY)
if df1[chosen_time_col].isna().all():
    df1[chosen_time_col] = pd.to_datetime(df1[chosen_time_col].astype(str), errors='coerce', dayfirst=True, utc=True)

# Report problematic rows if any
na_count = int(df1[chosen_time_col].isna().sum())
if na_count > 0:
    st.warning(f"{na_count} filas no pudieron parsearse en la columna de tiempo '{chosen_time_col}'. Se mostrarán solo las filas con tiempo válido.")
    bad_samples = df1[df1[chosen_time_col].isna()].head(5)
    if not bad_samples.empty:
        st.write("Ejemplos de filas con tiempo no parseable (primeras 5):")
        st.write(bad_samples)

# Filter invalid time rows
df1 = df1[df1[chosen_time_col].notna()].copy()
# rename to Time (capitalized) and set index
df1 = df1.rename(columns={chosen_time_col: 'Time'})
df1['Time'] = pd.to_datetime(df1['Time'], utc=True)
df1 = df1.set_index('Time')

# Detect temperature and humidity columns and normalize names
temp_col = None
hum_col = None
for c in df1.columns:
    lc = c.lower()
    if temp_col is None and ('temp' in lc or 'temperature' in lc or 'temperatura' in lc):
        temp_col = c
    if hum_col is None and ('hum' in lc):
        hum_col = c

# If not found, try to infer from values (e.g., numeric columns)
if temp_col is None or hum_col is None:
    # find numeric columns and guess by range/values
    numeric_cols = [c for c in df1.columns if pd.api.types.is_numeric_dtype(df1[c]) or df1[c].astype(str).str.replace(',','.', regex=False).str.replace('.','',1).str.isnumeric().any()]
    # Heuristic: temperature usually between -40 and 60, humidity 0-100
    for c in numeric_cols:
        vals = pd.to_numeric(df1[c].astype(str).str.replace(',','.'), errors='coerce')
        med = vals.median(skipna=True)
        if pd.notna(med):
            if -50 < med < 60 and temp_col is None:
                temp_col = c
            elif 0 <= med <= 100 and hum_col is None:
                hum_col = c

# Rename to standard Spanish names if found
rename_map = {}
if temp_col:
    rename_map[temp_col] = 'temperatura'
if hum_col:
    rename_map[hum_col] = 'humedad'
if rename_map:
    df1 = df1.rename(columns=rename_map)
    st.success(f"Columnas renombradas automáticamente: {rename_map}")

# If still not present, warn and list columns
if 'temperatura' not in df1.columns or 'humedad' not in df1.columns:
    st.warning("No se detectaron columnas llamadas exactamente 'temperatura' y 'humedad'. Columnas actuales:")
    st.write(list(df1.columns))
    st.info("Si tus columnas tienen otros nombres, renómbralas a 'temperatura' y 'humedad' o asegúrate que contienen 'temp'/'hum' en su nombre para que el renombrado automático funcione.")
    # proceed but later will block plotting if not present

# Convert numeric columns (replace comma decimal separators)
for col in ['temperatura', 'humedad']:
    if col in df1.columns:
        # replace comma with dot and convert
        df1[col] = pd.to_numeric(df1[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

# Drop rows where both temperatura and humedad are NaN (if both exist)
if 'temperatura' in df1.columns and 'humedad' in df1.columns:
    df1 = df1[~(df1['temperatura'].isna() & df1['humedad'].isna())]
else:
    # If at least one exists, drop rows where that one is NaN
    if 'temperatura' in df1.columns:
        df1 = df1[~df1['temperatura'].isna()]
    if 'humedad' in df1.columns:
        df1 = df1[~df1['humedad'].isna()]

# -------------------------
# Sidebar: filters & controls
# -------------------------
st.sidebar.title("Controles")
# Devices if present
devices = []
if 'device' in df1.columns:
    devices = sorted(df1['device'].dropna().unique().astype(str).tolist())

device = st.sidebar.selectbox("Seleccionar dispositivo (opcional)", options=['*'] + devices, index=0)

# Time range
min_time = df1.index.min()
max_time = df1.index.max()
st.sidebar.write(f"Rango disponible: {min_time} → {max_time}")
default_start = max_time - pd.Timedelta(hours=6) if (max_time - min_time) > pd.Timedelta(hours=1) else min_time
start = st.sidebar.datetime_input("Desde", value=default_start)
end = st.sidebar.datetime_input("Hasta", value=max_time)
if start is None:
    start = min_time
if end is None:
    end = max_time

# Refresh button: simply rerun
if st.sidebar.button("Refrescar datos (re-evaluar filtros)"):
    st.experimental_rerun()

# Apply filters
df_filtered = df1.copy()
df_filtered = df_filtered[(df_filtered.index >= pd.to_datetime(start)) & (df_filtered.index <= pd.to_datetime(end))]
if device != '*' and 'device' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['device'].astype(str) == device]

if df_filtered.empty:
    st.error("No hay datos para los filtros seleccionados. Revisa rango de tiempo o dispositivo.")
    st.stop()

# -------------------------
# UI: tabs and visualizations
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualización", "📊 Estadísticas", "🔍 Filtros", "🗺️ Información del Sitio"])

with tab1:
    st.subheader('Visualización de Datos')

    # determine available variables
    vars_available = [c for c in ['temperatura', 'humedad'] if c in df_filtered.columns]
    if not vars_available:
        st.error("No hay columnas de 'temperatura' ni 'humedad' detectadas para graficar.")
    else:
        options = vars_available.copy()
        if 'temperatura' in vars_available and 'humedad' in vars_available:
            options = ['temperatura', 'humedad', 'Ambas variables']
        variable = st.selectbox("Seleccione variable a visualizar", options)

        chart_type = st.selectbox("Seleccione tipo de gráfico", ["Línea", "Área", "Barra"])

        def plot_series(series, title, chart_type):
            st.write(f"### {title}")
            if chart_type == "Línea":
                st.line_chart(series)
            elif chart_type == "Área":
                st.area_chart(series)
            else:
                st.bar_chart(series)

        if variable == "Ambas variables":
            plot_series(df_filtered['temperatura'], "Temperatura (°C)", chart_type)
            plot_series(df_filtered['humedad'], "Humedad (%)", chart_type)
        else:
            plot_series(df_filtered[variable], variable.capitalize(), chart_type)

        if st.checkbox('Mostrar datos crudos'):
            st.write(df_filtered.head(200))

with tab2:
    st.subheader('Análisis Estadístico')

    stats_cols = [c for c in ['temperatura', 'humedad'] if c in df_filtered.columns]
    if not stats_cols:
        st.write("No hay columnas disponibles para calcular estadísticas.")
    else:
        # Show statistics for each available column
        for col in stats_cols:
            st.markdown(f"### {col.capitalize()}")
            s = df_filtered[col].dropna()
            if s.empty:
                st.write("No hay datos válidos.")
                continue
            maxv = float(s.max())
            minv = float(s.min())
            meanv = float(s.mean())
            stdv = float(s.std(ddof=0))
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Máximo", f"{maxv:.2f}" + (" °C" if col == 'temperatura' else " %"))
            col2.metric("Mínimo", f"{minv:.2f}" + (" °C" if col == 'temperatura' else " %"))
            col3.metric("Media", f"{meanv:.2f}" + (" °C" if col == 'temperatura' else " %"))
            col4.metric("Desviación estándar", f"{stdv:.2f}")

with tab3:
    st.subheader('Filtros de Datos (avance)')
    filter_variable = st.selectbox("Seleccione variable para filtrar", [c for c in ['temperatura', 'humedad'] if c in df_filtered.columns])
    col1, col2 = st.columns(2)
    with col1:
        min_val = st.slider(
            f'Valor mínimo de {filter_variable}',
            float(df_filtered[filter_variable].min()),
            float(df_filtered[filter_variable].max()),
            float(df_filtered[filter_variable].mean()),
            key="min_val"
        )
        filtrado_df_min = df_filtered[df_filtered[filter_variable] > min_val]
        st.write(f"Registros con {filter_variable} superior a {min_val}:")
        st.dataframe(filtrado_df_min.head(200))
    with col2:
        max_val = st.slider(
            f'Valor máximo de {filter_variable}',
            float(df_filtered[filter_variable].min()),
            float(df_filtered[filter_variable].max()),
            float(df_filtered[filter_variable].mean()),
            key="max_val"
        )
        filtrado_df_max = df_filtered[df_filtered[filter_variable] < max_val]
        st.write(f"Registros con {filter_variable} inferior a {max_val}:")
        st.dataframe(filtrado_df_max.head(200))

    # Download filtered data
    if st.button('Descargar datos filtrados (min filter)'):
        if not filtrado_df_min.empty:
            csv = filtrado_df_min.to_csv().encode('utf-8')
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name='datos_filtrados.csv',
                mime='text/csv',
            )
        else:
            st.info("No hay datos para descargar con ese filtro.")

with tab4:
    st.subheader("Información del Sitio de Medición")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Ubicación del Sensor")
        st.write("**Universidad EAFIT**")
        st.write("- Latitud: 6.2006")
        st.write("- Longitud: -75.5783")
        st.write("- Altitud: ~1,495 metros sobre el nivel del mar")
    with col2:
        st.write("### Detalles del Sensor")
        st.write("- Tipo: ESP32")
        st.write("- Variables medidas:")
        st.write("  * Temperatura (°C)")
        st.write("  * Humedad (%)")
        st.write("- Frecuencia de medición: según configuración")
        st.write("- Ubicación: Campus universitario")

# Final data table and download
st.markdown("---")
st.subheader("Datos procesados - vista previa")
st.dataframe(df_filtered.reset_index().head(500), height=300)
csv_all = df_filtered.reset_index().to_csv(index=False)
st.download_button("Descargar CSV procesado", data=csv_all, file_name="temphum_procesado.csv", mime="text/csv")

# Footer
st.markdown("""
---
Desarrollado para el análisis de datos de sensores urbanos.
Ubicación: Universidad EAFIT, Medellín, Colombia
""")
