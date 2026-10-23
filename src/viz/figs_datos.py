"""Figuras de la Parte A (blog 02 y 03). Solo leen `data/processed/resumen_datos.json` y `results/data/eda/*.json`.

    datos-anio.png            % sin resolver y tasa de malos por año de originación, con la partición
    eda-tasa-variables.png    tasa de malos por decil o categoría de cada variable (train)
    post-psi.png              PSI de cada variable, train → val y train → test (después del preprocesamiento)
    post-correlacion.png      correlaciones entre las variables numéricas transformadas (train)
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
from src.viz.style import aplicar_estilo, mapa_secuencial  # noqa: E402

SALIDA = Path("blog/docs/figures")
MAX_CATEGORIAS = 8


def _anio_split(anio: int, particion: dict) -> str:
    if anio <= pd.Timestamp(particion["train_hasta"]).year:
        return "train"
    if anio <= pd.Timestamp(particion["val_hasta"]).year:
        return "val"
    return "test" if anio <= pd.Timestamp(particion["test_hasta"]).year else "fuera"


def figura_anio(resumen: dict, destino: Path) -> Path:
    anios = sorted(int(a) for a in resumen["n_por_anio"])
    na = [100 * resumen["fraccion_na_por_anio"][str(a)] for a in anios]
    tasa = [100 * resumen["tasa_malos_por_anio"][str(a)] for a in anios]
    colores = [p.COLOR_SPLIT.get(_anio_split(a, resumen["particion"]), p.TINTA_TENUE) for a in anios]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.bar(anios, na, color=colores)
    ax1.set_title("Créditos sin resolver (objetivo NA)")
    ax1.set_ylabel("% de los créditos del año")
    ax2.bar(anios, tasa, color=colores)
    ax2.set_title("Tasa de malos entre los resueltos")
    ax2.set_ylabel("%")
    for ax in (ax1, ax2):
        ax.set_xticks(anios)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Año de originación")
    manijas = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [*p.COLOR_SPLIT.values(), p.TINTA_TENUE]]
    fig.legend(manijas, [*p.NOMBRE_SPLIT.values(), "Fuera del modelado"], loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return _guardar(fig, destino / "datos-anio.png")


def figura_tasa_variables(eda: dict, features: list[dict], destino: Path) -> Path:
    nombres = [f["nombre"] for f in features]
    etiquetas = {f["nombre"]: f["etiqueta_es"] for f in features}
    filas = int(np.ceil(len(nombres) / 3))
    fig, ejes = plt.subplots(filas, 3, figsize=(12, 2.6 * filas))
    base = 100 * eda["tasa_train"]
    for ax, nombre in zip(ejes.flat, nombres):
        info = eda["variables"][nombre]
        tabla = pd.DataFrame(info["tabla"])
        tabla = tabla[tabla["grupo"] != "vacío"]
        if info["tipo"] == "cat":
            tabla = tabla.head(MAX_CATEGORIAS).sort_values("tasa")
            ax.barh(tabla["grupo"], 100 * tabla["tasa"], color=p.CATEGORICA[0])
            ax.axvline(base, color=p.TINTA_SECUNDARIA, linewidth=1, linestyle="--")
            ax.tick_params(axis="y", labelsize=7)
        else:
            ax.plot(range(len(tabla)), 100 * tabla["tasa"], marker="o", color=p.CATEGORICA[0])
            ax.axhline(base, color=p.TINTA_SECUNDARIA, linewidth=1, linestyle="--")
            if tabla["grupo"].str.startswith(("(", "[")).any():  # deciles
                ax.set_xticks(range(len(tabla)), [str(i + 1) for i in range(len(tabla))])
                ax.set_xlabel("Decil" if len(tabla) == 10 else "Grupo por cuantiles", fontsize=8)
            else:  # conteo: valor exacto
                ax.set_xticks(range(len(tabla)), tabla["grupo"])
                ax.set_xlabel("Valor", fontsize=8)
        ax.set_title(etiquetas[nombre], fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(ejes.flat)[len(nombres):]:
        ax.axis("off")
    fig.supylabel("Tasa de malos (%) · línea discontinua = tasa de train")
    fig.tight_layout()
    return _guardar(fig, destino / "eda-tasa-variables.png")


def figura_psi(post: dict, features: list[dict], destino: Path) -> Path:
    etiquetas = {f["nombre"]: f["etiqueta_es"] for f in features}
    tabla = pd.DataFrame(post["psi_vs_train"]).sort_values("test")
    y = np.arange(len(tabla))
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(tabla) + 1.2))
    alto = 0.38
    ax.barh(y + alto / 2, tabla["val"], height=alto, color=p.COLOR_SPLIT["val"], label=p.NOMBRE_SPLIT["val"])
    ax.barh(y - alto / 2, tabla["test"], height=alto, color=p.COLOR_SPLIT["test"], label=p.NOMBRE_SPLIT["test"])
    ax.set_yticks(y, [etiquetas[n] for n in tabla.index], fontsize=8)
    ax.set_xlabel("PSI frente a entrenamiento")
    ax.legend(loc="lower right")
    return _guardar(fig, destino / "post-psi.png")


def figura_correlacion(post: dict, features: list[dict], destino: Path) -> Path:
    etiquetas = {f["nombre"]: f["etiqueta_es"] for f in features}
    nombres = post["correlacion_train"]["variables"]
    m = np.array(post["correlacion_train"]["matriz"])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(np.abs(m), cmap=mapa_secuencial(), vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(nombres)), [etiquetas[n] for n in nombres], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(nombres)), [etiquetas[n] for n in nombres], fontsize=7)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color=p.SUPERFICIE if abs(m[i, j]) > 0.6 else p.TINTA)
    fig.colorbar(im, ax=ax, label="|correlación de Pearson|")
    return _guardar(fig, destino / "post-correlacion.png")


def _guardar(fig, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def generar(raiz: Path = Path("."), destino: Path | None = None) -> list[Path]:
    """Regenera las cuatro figuras de la Parte A."""
    from src.common.config import load_config

    aplicar_estilo()
    destino = destino or raiz / SALIDA
    features = load_config(raiz / "configs" / "features.yaml")["features"]
    resumen = load_json(raiz / "data" / "processed" / "resumen_datos.json")
    eda = load_json(raiz / "results" / "data" / "eda" / "eda_inicial.json")
    post = load_json(raiz / "results" / "data" / "eda" / "post_preprocesamiento.json")
    return [figura_anio(resumen, destino), figura_tasa_variables(eda, features, destino),
            figura_psi(post, features, destino), figura_correlacion(post, features, destino)]
