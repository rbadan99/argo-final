"""
analisis_argo_final.py
======================
Taller integrador — Curso Python para el Análisis de Datos
Institut de Ciències del Mar (ICM-CSIC)

Análisis completo del dataset Argo del Mediterráneo:
  - Carga y limpieza de datos
  - Estadísticas descriptivas por cuenca y mes
  - Visualizaciones estáticas (Seaborn/Matplotlib) e interactivas (Plotly)
  - Exportación de resultados

Uso:
    python analisis_argo_final.py

Generado con ayuda de Microsoft Copilot.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------------------------------------------------------------------
# CONSTANTES — rutas de archivos y parámetros
# ---------------------------------------------------------------------------
RUTA_DATOS = "data/argo_mediterraneo.csv"
RUTA_ESTADISTICAS = "estadisticas_argo.csv"
RUTA_HEATMAP = "heatmap_temperatura.png"
RUTA_DIAGRAMA_TS = "diagrama_TS.png"
RUTA_MAPA = "mapa_argo_interactivo.html"

PROFUNDIDAD_SUPERFICIE = 10     # metros — umbral para datos superficiales
COLUMNAS_CLAVE = ["temperature", "salinity", "depth"]
MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
            "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------------

def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset Argo desde un archivo CSV.

    Args:
        ruta (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos crudos y la columna
            'date' parseada como datetime.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    logger.info(f"Cargando datos desde: {ruta}")
    df = pd.read_csv(ruta, parse_dates=["date"])
    logger.info(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def limpiar_datos(df: pd.DataFrame, columnas_clave: list = None) -> pd.DataFrame:
    """
    Limpia el DataFrame eliminando filas con NaN en columnas clave
    y añadiendo columnas temporales derivadas.

    Args:
        df (pd.DataFrame): DataFrame crudo del dataset Argo.
        columnas_clave (list, optional): Columnas donde se requieren valores
            no nulos. Por defecto: ['temperature', 'salinity', 'depth'].

    Returns:
        pd.DataFrame: DataFrame limpio con columnas 'month' y 'year' añadidas.
    """
    if columnas_clave is None:
        columnas_clave = COLUMNAS_CLAVE

    filas_antes = len(df)
    df_limpio = df.dropna(subset=columnas_clave).copy()
    filas_eliminadas = filas_antes - len(df_limpio)

    df_limpio["month"] = df_limpio["date"].dt.month
    df_limpio["year"] = df_limpio["date"].dt.year

    logger.info(
        f"Limpieza completada: {len(df_limpio):,} filas válidas "
        f"({filas_eliminadas:,} eliminadas)"
    )
    return df_limpio


def calcular_estadisticas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas de temperatura y salinidad por cuenca.

    Args:
        df (pd.DataFrame): DataFrame limpio con columnas 'basin',
            'temperature' y 'salinity'.

    Returns:
        pd.DataFrame: Tabla con estadísticas (media, std, min, max)
            por cuenca.
    """
    logger.info("Calculando estadísticas por cuenca...")
    stats = df.groupby("basin").agg(
        temp_media=("temperature", "mean"),
        temp_std=("temperature", "std"),
        temp_min=("temperature", "min"),
        temp_max=("temperature", "max"),
        sal_media=("salinity", "mean"),
        sal_std=("salinity", "std"),
        n_mediciones=("temperature", "count"),
    ).round(3)
    logger.info(f"Estadísticas calculadas para {len(stats)} cuencas")
    return stats


def generar_heatmap_temperatura(df: pd.DataFrame, ruta_salida: str) -> None:
    """
    Genera un heatmap de temperatura media superficial por mes y cuenca.

    Args:
        df (pd.DataFrame): DataFrame limpio con columnas 'depth', 'month',
            'basin' y 'temperature'.
        ruta_salida (str): Ruta donde guardar la imagen PNG.

    Returns:
        None
    """
    logger.info("Generando heatmap de temperatura...")
    pivot = pd.pivot_table(
        df[df["depth"] <= PROFUNDIDAD_SUPERFICIE],
        values="temperature",
        index="month",
        columns="basin",
        aggfunc="mean",
    ).round(1)

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(
        pivot.T,
        cmap="RdYlBu_r",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Temperatura (°C)"},
        ax=ax,
    )
    ax.set_title(
        "Temperatura media superficial (°C) por cuenca y mes",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cuenca")
    ax.set_xticklabels(MESES_ES, rotation=45)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap guardado en: {ruta_salida}")


def generar_diagrama_ts(df: pd.DataFrame, ruta_salida: str, n_muestra: int = 3000) -> None:
    """
    Genera un diagrama T-S (Temperatura vs Salinidad) coloreado por cuenca.

    El diagrama T-S es una visualización clásica en oceanografía que permite
    identificar masas de agua de diferente origen según sus propiedades
    termohalinas.

    Args:
        df (pd.DataFrame): DataFrame limpio con columnas 'salinity',
            'temperature' y 'basin'.
        ruta_salida (str): Ruta donde guardar la imagen PNG.
        n_muestra (int, optional): Número máximo de puntos a mostrar
            (submuestreo aleatorio). Default: 3000.

    Returns:
        None
    """
    logger.info("Generando diagrama T-S...")
    muestra = df.sample(n=min(n_muestra, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=muestra,
        x="salinity",
        y="temperature",
        hue="basin",
        alpha=0.5,
        s=20,
        ax=ax,
    )
    ax.set_xlabel("Salinidad (PSU)", fontsize=12)
    ax.set_ylabel("Temperatura (°C)", fontsize=12)
    ax.set_title(
        "Diagrama T-S — Masas de agua del Mediterráneo",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(title="Cuenca", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Diagrama T-S guardado en: {ruta_salida}")


def generar_mapa_interactivo(df: pd.DataFrame, ruta_salida: str) -> None:
    """
    Genera un mapa interactivo de distribución de boyas Argo en el Mediterráneo.

    Cada boya se representa con su posición media, temperatura media superficial
    y número de mediciones. El mapa se guarda como HTML interactivo.

    Args:
        df (pd.DataFrame): DataFrame limpio con columnas 'depth', 'float_id',
            'latitude', 'longitude', 'temperature', 'salinity' y 'basin'.
        ruta_salida (str): Ruta donde guardar el archivo HTML.

    Returns:
        None
    """
    logger.info("Generando mapa interactivo de boyas Argo...")
    mapa_data = (
        df[df["depth"] <= PROFUNDIDAD_SUPERFICIE]
        .groupby("float_id")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            temperature=("temperature", "mean"),
            salinity=("salinity", "mean"),
            basin=("basin", "first"),
            n_mediciones=("temperature", "count"),
        )
        .round(3)
        .reset_index()
    )

    fig = px.scatter_geo(
        mapa_data,
        lat="latitude",
        lon="longitude",
        color="temperature",
        size="n_mediciones",
        hover_name="float_id",
        hover_data={
            "latitude": ":.2f",
            "longitude": ":.2f",
            "temperature": ":.1f",
            "salinity": ":.2f",
            "basin": True,
        },
        color_continuous_scale="RdYlBu_r",
        title="Distribución geográfica de boyas Argo en el Mediterráneo",
        labels={"temperature": "Temp. media (°C)", "n_mediciones": "N mediciones"},
    )
    fig.update_geos(
        scope="europe",
        showland=True,
        landcolor="#e8e8e8",
        showocean=True,
        oceancolor="#d0e4f0",
        showcoastlines=True,
        coastlinecolor="#888888",
        center=dict(lat=38, lon=15),
        projection_scale=4,
    )
    fig.update_layout(height=550)
    fig.write_html(ruta_salida)
    logger.info(f"Mapa interactivo guardado en: {ruta_salida}")


def guardar_resultados(stats: pd.DataFrame, ruta_salida: str) -> None:
    """
    Exporta las estadísticas calculadas a un archivo CSV.

    Args:
        stats (pd.DataFrame): DataFrame con estadísticas por cuenca.
        ruta_salida (str): Ruta donde guardar el archivo CSV.

    Returns:
        None
    """
    stats.to_csv(ruta_salida)
    logger.info(f"Estadísticas exportadas a: {ruta_salida}")


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("ANÁLISIS ARGO — INICIO")
    logger.info("=" * 50)

    # 1. Cargar
    df_raw = cargar_datos(RUTA_DATOS)

    # 2. Limpiar
    df = limpiar_datos(df_raw)

    # 3. Estadísticas
    estadisticas = calcular_estadisticas(df)
    print("\nEstadísticas por cuenca:")
    print(estadisticas.to_string())

    # 4. Visualizaciones
    sns.set_theme(style="whitegrid", palette="colorblind")
    generar_heatmap_temperatura(df, RUTA_HEATMAP)
    generar_diagrama_ts(df, RUTA_DIAGRAMA_TS)
    generar_mapa_interactivo(df, RUTA_MAPA)

    # 5. Exportar
    guardar_resultados(estadisticas, RUTA_ESTADISTICAS)

    logger.info("=" * 50)
    logger.info("ANÁLISIS COMPLETADO ✓")
    logger.info(f"  Figuras:      {RUTA_HEATMAP}, {RUTA_DIAGRAMA_TS}")
    logger.info(f"  Mapa:         {RUTA_MAPA}")
    logger.info(f"  Estadísticas: {RUTA_ESTADISTICAS}")
    logger.info("=" * 50)
