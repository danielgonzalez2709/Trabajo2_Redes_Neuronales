"""Figuras de la pérdida esperada (Parte E, tarea E5; blog y notebook 05). Solo leen `results/data/el/summary.json`:
no recalculan la E[L] ni las bandas.

    el-bandas.png        por banda de puntaje, solo los créditos resueltos de test: E[L] frente a la pérdida realizada
                         (millones de USD; E §3.2). Los en mora no entran: su pérdida es provisional
    el-pd-vs-tasa.png    por banda de puntaje, todo el test: PD media predicha frente a la tasa de malos observada

Las bandas son cuantiles del puntaje de la scorecard (deciles con `bandas.n = 10`; banda 1 = menor puntaje, más
riesgo); el eje x muestra el
número de banda y su rango de puntaje. `generar(..., resumen=)` acepta el resumen en memoria (notebook 05 con una
corrida que no escribe en results/).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.common.results import load_json  # noqa: E402
from src.viz import paleta as p  # noqa: E402
from src.viz.style import aplicar_estilo  # noqa: E402

SALIDA = Path("blog/docs/figures")
ENTRADA = Path("results/data/el")
COMO_GENERAR = "E5: python run.py el --config configs/el/base.yaml"
MILLON = 1e6  # los montos se dibujan en millones de USD
ANCHO_BARRA = 0.4  # dos barras agrupadas por banda


def _etiquetas(bandas: list[dict]) -> list[str]:
    # número de banda y rango de puntaje, en dos líneas
    return [f"{b['banda']}\n{b['score_min']:.0f}–{b['score_max']:.0f}" for b in bandas]


def _subtitulo(resumen: dict) -> str:
    sc = resumen["scorecard"]
    return (f"PD: {p.NOMBRE_MODELO[resumen['modelo_pd']]} · LGD: {p.NOMBRE_MODELO[resumen['modelo_lgd']]} · "
            f"scorecard: base {sc['base']:g}, odds {sc['odds_base']:g}:1, PDO {sc['pdo']:g}")


def figura_bandas(resumen: dict, destino: Path) -> Path:
    bandas = resumen["por_banda"]
    x = np.arange(len(bandas))
    el = [b["resueltos"]["el"] / MILLON for b in bandas]
    obs = [b["resueltos"]["perdida_observada"] / MILLON for b in bandas]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - ANCHO_BARRA / 2, el, ANCHO_BARRA, color=p.COLOR_MODELO[resumen["modelo_pd"]],
           label="E[L] = PD · LGD · EAD")
    ax.bar(x + ANCHO_BARRA / 2, obs, ANCHO_BARRA, color=p.CUADRICULA, edgecolor=p.EJE, label="Pérdida realizada")
    ax.set_xticks(x, _etiquetas(bandas), fontsize=8)
    ax.set_xlabel("Banda de puntaje (1 = menor puntaje, más riesgo) y rango de puntaje")
    ax.set_ylabel("Millones de USD")
    ax.set_title("Pérdida esperada frente a realizada por banda de puntaje (créditos resueltos de test)", pad=18)
    ax.text(0.5, 1.01, _subtitulo(resumen), transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
            color=p.TINTA_SECUNDARIA)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _guardar(fig, destino / "el-bandas.png")


def figura_pd_vs_tasa(resumen: dict, destino: Path) -> Path:
    bandas = resumen["por_banda"]
    x = np.arange(len(bandas))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, [b["total"]["pd_media"] for b in bandas], marker="o", color=p.COLOR_MODELO[resumen["modelo_pd"]],
            label=f"PD media predicha ({p.NOMBRE_MODELO[resumen['modelo_pd']]})")
    ax.plot(x, [b["total"]["tasa_malos"] for b in bandas], marker="s", color=p.COLOR_TASA,
            label="Tasa de malos observada")
    ax.set_xticks(x, _etiquetas(bandas), fontsize=8)
    ax.set_xlabel("Banda de puntaje (1 = menor puntaje, más riesgo) y rango de puntaje")
    ax.set_ylabel("Fracción de los créditos de la banda")
    ax.set_title("PD predicha frente a tasa de malos observada por banda de puntaje (todo el test)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _guardar(fig, destino / "el-pd-vs-tasa.png")


def _leer_resumen(entrada: Path) -> dict:
    ruta = entrada / "summary.json"
    if not ruta.is_file():
        raise FileNotFoundError(f"falta {ruta.as_posix()}; genéralo con {COMO_GENERAR}")
    return load_json(ruta)


def _guardar(fig, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def generar(raiz: Path = Path("."), destino: Path | None = None, resumen: dict | None = None) -> list[Path]:
    """Regenera las dos figuras de la E[L] desde results/data/el/summary.json (o desde `resumen`, ya en memoria) y
    devuelve sus rutas. Sin summary.json: FileNotFoundError con el comando que lo genera (no dibuja nada)."""
    resumen = resumen if resumen is not None else _leer_resumen(raiz / ENTRADA)
    aplicar_estilo()
    destino = destino or raiz / SALIDA
    return [figura_bandas(resumen, destino), figura_pd_vs_tasa(resumen, destino)]
