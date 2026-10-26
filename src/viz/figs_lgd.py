"""Figuras de la Parte D (LGD; blog 08). Solo leen `results/data/lgd/`: no entrenan ni recalculan el preprocesador.

    lgd-histograma.png             LGD observada por split (fracción por bin) y fracción con LGD = 0 y = 1 (§3.2b)
    lgd-por-anio.png               fracción con LGD = 1 y LGD media por año de originación, coloreado por split (§3.2b)
    lgd-ead-plazo.png              EAD (mediana y p25–p75) y LGD media por plazo; EAD por año de originación (§3.2b)
    lgd-distribucion-predicha.png  test: distribución de la LGD observada frente a la lineal y el MLP, con las
                                   constantes de referencia como líneas verticales, y un panel con zoom en la cola alta
                                   con bins finos (§3.4; Anexo P2.4)
    lgd-qq.png                     test: cuantiles predichos frente a observados de la lineal y el MLP, con la diagonal
    lgd-error-por-banda.png        test: MAE por banda de LGD observada de la lineal, el MLP y la constante mediana
    lgd-semillas.png               MAE de test de cada semilla del MLP frente a la lineal y las dos constantes

Insumos: `descriptivo.json` (D1b), `summary.json`, `pred_test.parquet` (semilla mediana) y
`semillas/mlp_seed*.json` (D4).
`generar(..., datos=)` acepta esos insumos en memoria (notebook 04 con la corrida demo, sin escribir en results/).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.results import load_json  # noqa: E402
from src.viz import paleta as p  # noqa: E402
from src.viz.style import aplicar_estilo  # noqa: E402

SALIDA = Path("blog/docs/figures")
ENTRADA = Path("results/data/lgd")
RUTA_FEATURES = Path("configs/features.yaml")  # solo etiquetas en español (term.categorias_es)
COLOR_NEUTRO = p.TINTA_TENUE  # paneles sin split; el mismo gris que figs_datos usa para "fuera del modelado"
SPLITS = ("train", "val", "test")
MODELOS = ("lineal", "mlp")
# Bandas de LGD observada para el error por banda: la primera cerrada [0, 0.5]; las demás (a, b]. La última aísla la
# masa en 1 (LGD = 1 cae en (0.99, 1]). Sugeridas en el prompt de D6; no están en la Spec ni en la config.
BANDAS_LGD = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.99), (0.99, 1.0))
# Probabilidades del Q-Q: percentiles 1…99 (los extremos 0 y 100 son el mínimo y el máximo, muy ruidosos).
PROBS_QQ = np.linspace(0.01, 0.99, 99)
# Zoom de la distribución predicha: solo para la figura (no afecta a ningún modelo ni métrica). La LGD de test y las
# predicciones se concentran en [0.8, 1]; con bins de 0.01 se ve la masa en 1 (último bin [0.99, 1]) frente a la
# predicha.
ZOOM_LGD = (0.8, 1.0)
ANCHO_BIN_ZOOM = 0.01
# Insumos de results/data/lgd/ y cómo se generan si faltan.
COMO_GENERAR = {
    "descriptivo.json": "D1b: python -m src.lgd.descriptivo",
    "summary.json": "D4: python run.py lgd --config configs/lgd/base.yaml",
    "pred_test.parquet": "D4: python run.py lgd --config configs/lgd/base.yaml",
    "semillas": "D4: python run.py lgd --config configs/lgd/base.yaml",
}
ESTILO_CONSTANTE = {
    "constante_media": {"color": p.TINTA_SECUNDARIA, "linestyle": "--", "nombre": "Constante: media de train"},
    "constante_mediana": {"color": p.TINTA_TENUE, "linestyle": ":", "nombre": "Constante: mediana de train"},
}


# ---------------------------------------------------------------- funciones puras

def mae_por_banda(y: np.ndarray, pred: np.ndarray, bandas=BANDAS_LGD) -> list[dict]:
    """MAE de `pred` frente a `y` dentro de cada banda de `y`: la primera banda [a, b], las demás (a, b].

    Devuelve una lista de {"desde", "hasta", "n", "mae"}; una banda vacía tiene mae = NaN.
    """
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    tabla = []
    for i, (desde, hasta) in enumerate(bandas):
        mascara = (y <= hasta) & ((y >= desde) if i == 0 else (y > desde))
        n = int(mascara.sum())
        mae = float(np.mean(np.abs(y[mascara] - pred[mascara]))) if n else float("nan")
        tabla.append({"desde": desde, "hasta": hasta, "n": n, "mae": mae})
    return tabla


def cuantiles_qq(y: np.ndarray, pred: np.ndarray, probs: np.ndarray = PROBS_QQ) -> tuple[np.ndarray, np.ndarray]:
    """Para cada p en `probs`: (Q_y(p), Q_ŷ(p)), el cuantil p de y frente al cuantil p de ŷ (np.quantile, lineal)."""
    return np.quantile(np.asarray(y, dtype=float), probs), np.quantile(np.asarray(pred, dtype=float), probs)


def mae_semilla_mediana(semillas: list[dict], resumen: dict) -> tuple[int, float]:
    """(semilla, MAE de test) de la semilla mediana de summary.json; ValueError si no está entre las semillas."""
    mediana = resumen["mlp"].get("semilla_mediana")
    for r in semillas:
        if r["seed"] == mediana:
            return mediana, r["metricas"]["mae_test"]
    raise ValueError(f"La semilla mediana ({mediana}) de summary.json no está entre las semillas leídas.")


def _etiqueta_mae(mae: float) -> str:
    """Etiqueta de una barra de MAE: '—' si la banda está vacía (NaN)."""
    return "—" if np.isnan(mae) else f"{mae:.3f}"


def _bordes_zoom() -> np.ndarray:
    desde, hasta = ZOOM_LGD
    return np.linspace(desde, hasta, int(round((hasta - desde) / ANCHO_BIN_ZOOM)) + 1)


def _etiqueta_banda(desde: float, hasta: float, primera: bool) -> str:
    return f"{'[' if primera else '('}{desde:g}, {hasta:g}]"


# ---------------------------------------------------------------- figuras

def figura_histograma(obs: dict, destino: Path) -> Path:
    h = obs["histograma_lgd"]
    bordes = np.array(h["bordes"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    for s in SPLITS:
        conteos = np.array(h["por_split"][s], dtype=float)
        ax1.stairs(conteos / conteos.sum(), bordes, color=p.COLOR_SPLIT[s], linewidth=p.ANCHO_LINEA,
                   label=f"{p.NOMBRE_SPLIT[s]} (n = {obs['por_split'][s]['n']:,})".replace(",", " "))
    ax1.set_xlabel("LGD observada")
    ax1.set_ylabel("Fracción de los créditos del split")
    ax1.set_title(f"Histograma ({len(bordes) - 1} bins; el último incluye LGD = 1)")
    ax1.legend(loc="upper left")
    x = np.arange(2)
    ancho = 0.26
    for k, s in enumerate(SPLITS):
        valores = [100 * obs["por_split"][s]["fraccion_lgd_0"], 100 * obs["por_split"][s]["fraccion_lgd_1"]]
        barras = ax2.bar(x + (k - 1) * ancho, valores, width=ancho, color=p.COLOR_SPLIT[s], label=p.NOMBRE_SPLIT[s])
        ax2.bar_label(barras, fmt="%.1f", fontsize=7, color=p.TINTA_SECUNDARIA, padding=2)
    ax2.set_xticks(x, ["LGD = 0", "LGD = 1"])
    ax2.set_ylabel("% de los créditos del split")
    ax2.set_title("Masa en los extremos")
    fig.tight_layout()
    return _guardar(fig, destino / "lgd-histograma.png")


def figura_por_anio(obs: dict, destino: Path) -> Path:
    anios = sorted(obs["fraccion_lgd_1_por_anio"], key=int)
    colores = [p.COLOR_SPLIT.get(obs["split_por_anio"][a][0], p.TINTA_TENUE) for a in anios]
    fraccion = [100 * obs["fraccion_lgd_1_por_anio"][a] for a in anios]
    media = [obs["por_anio"][a]["lgd"]["media"] for a in anios]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.bar(anios, fraccion, color=colores)
    ax1.set_title("Créditos con LGD = 1")
    ax1.set_ylabel("% de los créditos del año")
    ax2.bar(anios, media, color=colores)
    ax2.set_title("LGD media")
    ax2.set_ylabel("LGD")
    ax2.set_ylim(min(media) - 0.05, 1.0)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Año de originación")
    manijas = [plt.Rectangle((0, 0), 1, 1, color=p.COLOR_SPLIT[s]) for s in SPLITS]
    fig.legend(manijas, [p.NOMBRE_SPLIT[s] for s in SPLITS], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return _guardar(fig, destino / "lgd-por-anio.png")


def figura_distribucion_predicha(pred: pd.DataFrame, bordes: np.ndarray, referencias: dict, destino: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    pesos = np.full(len(pred), 1 / len(pred))
    zoom = _bordes_zoom()
    for ax, b in ((ax1, bordes), (ax2, zoom)):
        ax.hist(pred["y"], bins=b, weights=pesos, color=p.CUADRICULA, edgecolor=p.EJE, label="Observada (test)")
        for m in MODELOS:
            ax.hist(pred[f"pred_{m}"], bins=b, weights=pesos, histtype="step", linewidth=p.ANCHO_LINEA,
                    color=p.COLOR_MODELO[m], label=f"Predicha: {p.NOMBRE_MODELO[m]}")
        for nombre, estilo in ESTILO_CONSTANTE.items():
            valor = referencias[nombre]["valor"]
            ax.axvline(valor, color=estilo["color"], linestyle=estilo["linestyle"], linewidth=1.5,
                       label=f"{estilo['nombre']} = {valor:.3f}")
        ax.set_xlabel("LGD")
        ax.set_ylabel("Fracción de los créditos de test")
    ax1.set_title(f"Test completo (n = {len(pred):,}; bins de {bordes[1] - bordes[0]:.2f})".replace(",", " "))
    ax1.legend(loc="upper left", fontsize=8)
    ax2.set_xlim(*ZOOM_LGD)
    ax2.set_title(f"Zoom en [{ZOOM_LGD[0]:g}, {ZOOM_LGD[1]:g}] (bins de {ANCHO_BIN_ZOOM:g}; el último incluye LGD = 1)")
    fig.tight_layout()
    return _guardar(fig, destino / "lgd-distribucion-predicha.png")


def figura_qq(pred: pd.DataFrame, destino: Path, probs: np.ndarray = PROBS_QQ) -> Path:
    """Q-Q en test: para cada p en `probs`, el punto (Q_y(p), Q_ŷ(p)), cuantil p de y frente a cuantil p de ŷ."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    extremos = []
    for m, marcador in zip(MODELOS, ("o", "s")):
        q_obs, q_pred = cuantiles_qq(pred["y"], pred[f"pred_{m}"], probs)
        ax.plot(q_obs, q_pred, marker=marcador, markersize=4, linewidth=1, alpha=0.8, color=p.COLOR_MODELO[m],
                label=p.NOMBRE_MODELO[m])
        extremos += [q_obs.min(), q_obs.max(), q_pred.min(), q_pred.max()]
    # Ejes ajustados al rango de los cuantiles dibujados (con un margen), no a [0, 1]: la LGD de test se concentra
    # arriba.
    margen = 0.02 * (max(extremos) - min(extremos))
    lim = (min(extremos) - margen, max(extremos) + margen)
    ax.plot(lim, lim, color=p.TINTA_TENUE, linestyle="--", linewidth=1, label="Diagonal (y = x)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Cuantil de la LGD observada (test)")
    ax.set_ylabel("Cuantil de la LGD predicha (test)")
    ax.set_title(f"Q-Q en test (percentiles {probs[0] * 100:g}–{probs[-1] * 100:g})")
    ax.legend(loc="upper left", fontsize=8)
    return _guardar(fig, destino / "lgd-qq.png")


def tabla_error_por_banda(pred: pd.DataFrame, referencias: dict) -> dict[str, list[dict]]:
    """MAE por banda en test de la lineal, el MLP y la constante mediana (su valor sale de summary.json)."""
    y = pred["y"].to_numpy()
    tabla = {m: mae_por_banda(y, pred[f"pred_{m}"].to_numpy()) for m in MODELOS}
    tabla["constante_mediana"] = mae_por_banda(y, np.full(len(y), referencias["constante_mediana"]["valor"]))
    return tabla


def figura_error_por_banda(pred: pd.DataFrame, referencias: dict, destino: Path) -> Path:
    tabla = tabla_error_por_banda(pred, referencias)
    series = [(m, p.COLOR_MODELO[m], p.NOMBRE_MODELO[m]) for m in MODELOS]
    series.append(("constante_mediana", p.TINTA_TENUE, ESTILO_CONSTANTE["constante_mediana"]["nombre"]))
    x = np.arange(len(BANDAS_LGD))
    ancho = 0.26
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (clave, color, nombre) in enumerate(series):
        barras = ax.bar(x + (k - 1) * ancho, [b["mae"] for b in tabla[clave]], width=ancho, color=color, label=nombre)
        ax.bar_label(barras, labels=[_etiqueta_mae(b["mae"]) for b in tabla[clave]], fontsize=6,
                     color=p.TINTA_SECUNDARIA, padding=2, rotation=90)
    etiquetas = [f"{_etiqueta_banda(b['desde'], b['hasta'], i == 0)}\nn = " + f"{b['n']:,}".replace(",", " ")
                 for i, b in enumerate(tabla["lineal"])]
    ax.set_xticks(x, etiquetas, fontsize=8)
    ax.set_xlabel("Banda de LGD observada (test)")
    ax.set_ylabel("MAE")
    ax.set_ylim(0, 1.15 * np.nanmax([b["mae"] for filas in tabla.values() for b in filas]))
    ax.set_title("Error absoluto medio por banda de LGD observada (test)")
    ax.legend(loc="upper right", fontsize=8)
    return _guardar(fig, destino / "lgd-error-por-banda.png")


def figura_semillas(semillas: list[dict], resumen: dict, destino: Path) -> Path:
    semillas = sorted(semillas, key=lambda r: r["seed"])
    seeds = [r["seed"] for r in semillas]
    mae = [r["metricas"]["mae_test"] for r in semillas]
    mediana, mae_mediana = mae_semilla_mediana(semillas, resumen)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(seeds, mae, color=p.COLOR_MODELO["mlp"], s=25, zorder=3,
               label=f"{p.NOMBRE_MODELO['mlp']} ({len(seeds)} semillas)")
    ax.scatter([mediana], [mae_mediana], s=110, facecolors="none", edgecolors=p.TINTA, linewidths=1.5, zorder=4,
               label=f"Semilla mediana ({mediana})")
    lineal = resumen["lineal"]["metricas"]["mae_test"]["media"]
    ax.axhline(lineal, color=p.COLOR_MODELO["lineal"], linewidth=1.5,
               label=f"{p.NOMBRE_MODELO['lineal']} = {lineal:.4f}")
    for nombre, estilo in ESTILO_CONSTANTE.items():
        valor = resumen["referencias"][nombre]["metricas"]["mae_test"]
        ax.axhline(valor, color=estilo["color"], linestyle=estilo["linestyle"], linewidth=1.5,
                   label=f"{estilo['nombre']} = {valor:.4f}")
    ax.set_xlabel("Semilla")
    ax.set_ylabel("MAE en test")
    ax.set_title("Variabilidad entre semillas del MLP (MAE en test)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    return _guardar(fig, destino / "lgd-semillas.png")


def _mediana_y_rango(resumenes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Mediana y distancias a p25 y p75 (formato yerr de matplotlib) de una lista de resúmenes de descriptivo.json."""
    mediana = np.array([r["mediana"] for r in resumenes])
    p25 = np.array([r["cuantiles"]["p25"] for r in resumenes])
    p75 = np.array([r["cuantiles"]["p75"] for r in resumenes])
    return mediana, np.vstack([mediana - p25, p75 - mediana])


def etiquetas_plazo(raiz: Path) -> dict[str, str]:
    """`term.categorias_es` de configs/features.yaml ("36 months" → "36 meses"); {} si no hay config o etiquetas."""
    ruta = raiz / RUTA_FEATURES
    if not ruta.exists():
        return {}
    from src.common.config import load_config

    for f in load_config(ruta)["features"]:
        if f["nombre"] == "term":
            return dict(f.get("categorias_es") or {})
    return {}


def etiqueta_plazo(codigo: str, etiquetas: dict[str, str]) -> str:
    """Etiqueta en español del plazo; si falta, el código tal cual."""
    return etiquetas.get(codigo, codigo)


def figura_ead_plazo(obs: dict, destino: Path, etiquetas: dict[str, str] | None = None) -> Path:
    """EAD y LGD por plazo (paneles 1 y 2, color neutro: no son splits) y EAD por año (panel 3, color por split)."""
    etiquetas = etiquetas or {}
    plazos = list(obs["por_plazo"])
    nombres = [etiqueta_plazo(t, etiquetas) for t in plazos]
    anios = sorted(obs["por_anio"], key=int)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1, 2]})
    x = np.arange(len(plazos))
    mediana, rango = _mediana_y_rango([obs["por_plazo"][t]["ead"] for t in plazos])
    ax1.bar(x, mediana, yerr=rango, color=COLOR_NEUTRO, ecolor=p.TINTA_SECUNDARIA, capsize=4)
    ax1.set_xticks(x, [f"{nombre}\nn = " + f"{obs['por_plazo'][t]['n']:,}".replace(",", " ")
                       for nombre, t in zip(nombres, plazos)], fontsize=8)
    ax1.set_title("EAD por plazo")
    ax1.set_ylabel("EAD (mediana; barra p25–p75)")
    media_lgd = [obs["por_plazo"][t]["lgd"]["media"] for t in plazos]
    barras = ax2.bar(x, media_lgd, color=COLOR_NEUTRO)
    ax2.bar_label(barras, fmt="%.3f", fontsize=7, color=p.TINTA_SECUNDARIA, padding=2)
    ax2.set_xticks(x, nombres, fontsize=8)
    ax2.set_ylim(min(media_lgd) - 0.05, 1.0)
    ax2.set_title("LGD media por plazo")
    ax2.set_ylabel("LGD")
    mediana, rango = _mediana_y_rango([obs["por_anio"][a]["ead"] for a in anios])
    colores = [p.COLOR_SPLIT.get(obs["split_por_anio"][a][0], p.TINTA_TENUE) for a in anios]
    ax3.bar(anios, mediana, yerr=rango, color=colores, ecolor=p.TINTA_SECUNDARIA, capsize=3)
    ax3.set_title("EAD por año de originación")
    ax3.set_ylabel("EAD (mediana; barra p25–p75)")
    ax3.set_xlabel("Año de originación")
    ax3.tick_params(axis="x", rotation=45)
    manijas = [plt.Rectangle((0, 0), 1, 1, color=p.COLOR_SPLIT[s]) for s in SPLITS]
    ax3.legend(manijas, [p.NOMBRE_SPLIT[s] for s in SPLITS], loc="upper left", fontsize=8)
    fig.tight_layout()
    return _guardar(fig, destino / "lgd-ead-plazo.png")


def _exigir(ruta: Path, insumo: str) -> Path:
    """Devuelve `ruta` si existe; si no, FileNotFoundError con la tarea y el comando que la genera."""
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta.as_posix()}; genéralo con {COMO_GENERAR[insumo]}")
    return ruta


def insumos_de_corrida(carpeta: Path, obs: dict) -> dict:
    """`datos` para `generar` desde la carpeta de una corrida de `run.py lgd` (p. ej. la demo del notebook 04 en una
    carpeta temporal): `summary.json`, `pred_test.parquet` y `semillas/mlp_seed*.json`; `obs` se pasa aparte
    (descriptivo["observado"], que la corrida no escribe)."""
    carpeta = Path(carpeta)
    resumen = load_json(_exigir(carpeta / "summary.json", "summary.json"))
    pred = pd.read_parquet(_exigir(carpeta / "pred_test.parquet", "pred_test.parquet"))
    sem = carpeta / "semillas"
    semillas = [load_json(r) for r in sorted(sem.glob("mlp_seed*.json"))] if sem.is_dir() else []
    return {"obs": obs, "resumen": resumen, "pred": pred, "semillas": semillas}


def _leer_insumos(entrada: Path) -> dict:
    obs = load_json(_exigir(entrada / "descriptivo.json", "descriptivo.json"))["observado"]
    return insumos_de_corrida(entrada, obs)


def _guardar(fig, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def generar(raiz: Path = Path("."), destino: Path | None = None, datos: dict | None = None) -> list[Path]:
    """Regenera las siete figuras de la Parte D desde results/data/lgd/ y devuelve sus rutas.

    `datos` (claves `obs` = descriptivo["observado"], `resumen` = summary.json, `pred` = pred_test como DataFrame,
    `semillas` = lista de mlp_seed*.json) reemplaza la lectura de archivos: el notebook 04 dibuja así la corrida demo
    sin escribir en results/ (usar un `destino` temporal).
    """
    aplicar_estilo()
    destino = destino or raiz / SALIDA
    datos = datos if datos is not None else _leer_insumos(raiz / ENTRADA)
    obs, resumen, pred, semillas = datos["obs"], datos["resumen"], datos["pred"], datos["semillas"]
    if not semillas:
        raise ValueError(f"No hay semillas del MLP (mlp_seed*.json); genéralas con {COMO_GENERAR['semillas']}")
    bordes = np.array(obs["histograma_lgd"]["bordes"])
    return [figura_histograma(obs, destino), figura_por_anio(obs, destino),
            figura_ead_plazo(obs, destino, etiquetas_plazo(raiz)),
            figura_distribucion_predicha(pred, bordes, resumen["referencias"], destino),
            figura_qq(pred, destino), figura_error_por_banda(pred, resumen["referencias"], destino),
            figura_semillas(semillas, resumen, destino)]
