"""Estilo de matplotlib de las figuras estáticas (portado del Trabajo 1, T4.1). Uso:

    from src.viz.style import aplicar_estilo, mapa_secuencial
    aplicar_estilo()        # antes de crear las figuras
"""
import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

from src.viz import paleta as p

DPI = 150


def parametros() -> dict:
    """rcParams del Design System: sans del sistema, líneas de 2 px, cuadrícula tenue, sin bordes superior/derecho."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": p.FUENTES,
        "font.size": 11,
        "text.color": p.TINTA,
        "axes.facecolor": p.SUPERFICIE,
        "axes.edgecolor": p.EJE,
        "axes.linewidth": 1,
        "axes.labelcolor": p.TINTA_SECUNDARIA,
        "axes.titleweight": "semibold",
        "axes.titlecolor": p.TINTA,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=p.CATEGORICA),
        "grid.color": p.CUADRICULA,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "xtick.color": p.TINTA_TENUE,
        "ytick.color": p.TINTA_TENUE,
        "lines.linewidth": p.ANCHO_LINEA,
        "lines.solid_capstyle": "round",
        "lines.markersize": p.TAMANO_MARCADOR,
        "legend.frameon": False,
        "legend.labelcolor": p.TINTA_SECUNDARIA,
        "figure.facecolor": p.SUPERFICIE,
        "figure.dpi": DPI,
        "savefig.facecolor": p.SUPERFICIE,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    }


def aplicar_estilo() -> None:
    mpl.rcParams.update(parametros())


def mapa_secuencial() -> LinearSegmentedColormap:
    """Rampa azul de un solo tono (claro → oscuro) para curvas de nivel y mapas de calor."""
    return LinearSegmentedColormap.from_list("secuencial_azul", p.SECUENCIAL_AZUL)
