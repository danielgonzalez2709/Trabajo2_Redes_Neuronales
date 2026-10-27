"""Descriptivo de la LGD observada y después del preprocesamiento (tarea D1b; Parte D §3.2b).

Uso: ``python -m src.lgd.descriptivo [--config configs/lgd/base.yaml] [--set clave=valor]`` (no usa internet).

Produce **solo datos** (``results/data/lgd/descriptivo.json``); las figuras son D6 (``src/viz/figs_lgd.py``) y se
dibujan desde este JSON sin recalcular (el histograma va con sus bordes y conteos).

- ``observado``: la LGD antes de modelar (histograma total y por split, masa exacta en 0 y en 1, casos recortados, y
  resumen de LGD y EAD por split, por año de originación y por plazo). La distribución distinta entre particiones se
  acepta y se documenta ([DECIDIDO, Camila, 24-oct]); aquí queda medida por split y por año.
- ``preprocesado``: ``src.datos.eda.descriptivo_post`` (Parte A §3.8) con ``objetivo="lgd"`` sobre
  ``loans_lgd.parquet`` y ``models/preprocesador_lgd.json`` tal cual (``Preprocesador.from_dict``; nunca se reajusta).
- ``evidencia``: ``{sha256_particion, sha256_preprocesador}`` de ``src/lgd/procedencia.py`` (núcleo §5.4 y §6; la
  misma definición que usan los JSON de D4).
- ``fuente``: rutas de los insumos y de ``resumen_objetivo.json`` (D2), con el que se comprueba que la
  masa en 0/1 y los recortes cuadran (no se duplica: si no cuadra, ``ValueError`` y no se escribe nada).

Desviación estándar: poblacional (``ddof=0``), igual que ``descriptivo_post`` y el ``Preprocesador``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.config import apply_overrides, load_config
from src.common.preprocesamiento import DESCONOCIDO, Preprocesador
from src.common.results import load_json, save_json
from src.datos.eda import CUANTILES, UMBRAL_MONOTONIA, descriptivo_post
from src.lgd.preparar import SPLITS
from src.lgd.procedencia import evidencia

RAIZ = Path(__file__).resolve().parents[2]
N_BINS_HISTOGRAMA = 20  # valor por defecto; la config (descriptivo.n_bins_histograma) manda en generar()
CLAVES_CUADRE = ("n_lgd_igual_0", "n_lgd_igual_1", "n_lgd_sin_recorte_menor_0", "n_lgd_sin_recorte_mayor_1")
NOTAS_OBJETIVO_CONTINUO = {
    "tasa_objetivo_por_split": "con objetivo continuo es la LGD media por split (el nombre viene de la Parte A, PD)",
    "relacion_con_objetivo_train": "tasa_grupo_bajo/alto y tasas_por_grupo son LGD media por grupo (decil o valor) "
                                   "de la variable transformada en train; spearman_grupos y monotona igual que en A",
    "monotona_con_dos_grupos": "con n_grupos == 2, monotona es siempre verdadera (ρ = signo de la diferencia entre "
                               "los dos grupos; src/datos/eda.py:74-75) y no es evidencia de relación (caso pub_rec)",
    "categoricas_sin_relacion": "descriptivo_post mide la relación con el objetivo solo para las numéricas; las "
                                "categóricas emp_length, home_ownership y purpose no están cubiertas (term se cubre "
                                "con observado.por_plazo)",
}


def _clave_cuantil(q: float) -> str:
    return f"p{round(q * 100, 6):g}"


def _resumen(x: pd.Series, cuantiles) -> dict:
    x = x.astype(float)
    return {"n": int(len(x)), "media": float(x.mean()), "mediana": float(x.median()), "desv": float(x.std(ddof=0)),
            "cuantiles": {_clave_cuantil(q): float(x.quantile(q)) for q in cuantiles}}


def _resumen_grupo(g: pd.DataFrame, cuantiles) -> dict:
    return {"n": int(len(g)), "lgd": _resumen(g["lgd"], cuantiles), "ead": _resumen(g["ead"], cuantiles),
            "fraccion_lgd_1": float((g["lgd"] == 1.0).mean()), "fraccion_lgd_0": float((g["lgd"] == 0.0).mean())}


def _limpiar_plazo(serie: pd.Series) -> pd.Series:
    """Replica la limpieza de categóricas de ``Preprocesador._limpiar_cat`` (``src/common/preprocesamiento.py``; núcleo
    §5.2): strip; vacío o NA → ``DESCONOCIDO``. Se copia porque ``_limpiar_cat`` es privada."""
    return serie.astype("string").str.strip().fillna(DESCONOCIDO).replace("", DESCONOCIDO).astype(str)


def _histograma(df: pd.DataFrame, bordes: np.ndarray, splits: list[str]) -> dict:
    """Conteos de la LGD con los mismos ``bordes`` para el total y para cada split."""
    return {
        "bordes": bordes.tolist(),
        "conteos": np.histogram(df["lgd"].astype(float), bins=bordes)[0].tolist(),
        "por_split": {s: np.histogram(df.loc[df["split"] == s, "lgd"].astype(float), bins=bordes)[0].tolist()
                      for s in splits},
        "nota": "bins [b_i, b_{i+1}) y el último cerrado [b_{n-1}, 1] (np.histogram): la masa en 1 cae en el "
                "último bin y la masa en 0 en el primero; sus conteos exactos están en `masa`",
    }


def descriptivo_observado(df: pd.DataFrame, n_bins: int = N_BINS_HISTOGRAMA, cuantiles=CUANTILES) -> dict:
    """LGD observada antes de modelar (función pura). ``df`` necesita ``issue_d``, ``split``, ``term``, ``ead``,
    ``lgd`` y ``lgd_sin_recorte``."""
    lgd = df["lgd"].astype(float).to_numpy()
    splits = [s for s in SPLITS if (df["split"] == s).any()]
    n = int(len(df))
    n0, n1 = int((lgd == 0.0).sum()), int((lgd == 1.0).sum())
    sin = df["lgd_sin_recorte"].astype(float)

    anio = df["issue_d"].dt.year.astype(int)
    plazo = _limpiar_plazo(df["term"])
    por_anio = {str(a): _resumen_grupo(g, cuantiles) for a, g in df.groupby(anio, sort=True)}
    por_plazo = {str(p): _resumen_grupo(g, cuantiles) for p, g in df.groupby(plazo, sort=True)}
    return {
        "n": n,
        "histograma_lgd": _histograma(df, np.linspace(0.0, 1.0, int(n_bins) + 1), splits),
        "masa": {"n_lgd_igual_0": n0, "n_lgd_igual_1": n1,
                 "fraccion_lgd_0": n0 / n if n else 0.0, "fraccion_lgd_1": n1 / n if n else 0.0},
        "recortes": {"n_lgd_sin_recorte_menor_0": int((sin < 0).sum()),
                     "n_lgd_sin_recorte_mayor_1": int((sin > 1).sum())},
        "por_split": {s: _resumen_grupo(df[df["split"] == s], cuantiles) for s in splits},
        "por_anio": por_anio,
        "split_por_anio": {str(a): sorted(g["split"].unique().tolist()) for a, g in df.groupby(anio, sort=True)},
        "por_plazo": por_plazo,
        # atajo para D6: lo mismo que por_anio[a]["fraccion_lgd_1"], plano para dibujarlo directo
        "fraccion_lgd_1_por_anio": {a: v["fraccion_lgd_1"] for a, v in por_anio.items()},
        "desv": "poblacional (ddof=0)",
    }


def descriptivo_preprocesado(df: pd.DataFrame, pre: Preprocesador, umbral: float = UMBRAL_MONOTONIA) -> dict:
    """Las funciones de la Parte A (A §3.8) con ``objetivo="lgd"``; ``pre`` se usa tal cual (no se reajusta)."""
    out = descriptivo_post(df, pre, objetivo="lgd", umbral=float(umbral))
    out["notas_objetivo_continuo"] = dict(NOTAS_OBJETIVO_CONTINUO)
    return out


def _cuadrar_con_resumen(observado: dict, resumen: dict) -> None:
    propio = {**observado["masa"], **observado["recortes"]}
    distintos = {k: (propio[k], resumen.get(k)) for k in CLAVES_CUADRE if propio[k] != resumen.get(k)}
    if observado["n"] != resumen.get("n_lgd_valida"):
        distintos["n_lgd_valida"] = (observado["n"], resumen.get("n_lgd_valida"))
    if distintos:
        raise ValueError(f"El descriptivo no cuadra con resumen_objetivo.json (D2) (propio, resumen): {distintos}")


def generar(cfg: dict, raiz: Path = Path(".")) -> dict:
    """Lee los insumos de D2, arma ``{observado, preprocesado, fuente, evidencia}`` y escribe ``descriptivo.salida``."""
    raiz = Path(raiz)
    datos, desc = cfg["datos"], cfg["descriptivo"]
    salida = datos["salida"]
    df = pd.read_parquet(raiz / salida["loans_lgd"])
    pre_dict = load_json(raiz / salida["preprocesador"])
    pre = Preprocesador.from_dict(pre_dict)
    resumen = load_json(raiz / salida["resumen"])
    umbral = float(load_config(raiz / datos["config_datos"])["eda"]["umbral_monotonia"])

    observado = descriptivo_observado(df, n_bins=int(desc["n_bins_histograma"]),
                                      cuantiles=[float(q) for q in desc["cuantiles"]])
    _cuadrar_con_resumen(observado, resumen)
    out = {
        "observado": observado,
        "preprocesado": descriptivo_preprocesado(df, pre, umbral=umbral),
        "fuente": {"loans_lgd": str(salida["loans_lgd"]), "preprocesador": str(salida["preprocesador"]),
                   "resumen_objetivo": str(salida["resumen"]), "umbral_monotonia": umbral},
        "evidencia": evidencia(df, raiz / salida["preprocesador"]),
    }
    save_json(raiz / desc["salida"], out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/lgd/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(RAIZ / args.config), args.overrides)
    print("Configuración final resuelta:")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), end="")
    inicio = time.perf_counter()
    r = generar(cfg, RAIZ)
    o = r["observado"]
    print(f"LGD: n = {o['n']:,} · masa en 0: {o['masa']['n_lgd_igual_0']} · en 1: {o['masa']['n_lgd_igual_1']}")
    print(f"Fracción LGD = 1 por split: { {s: round(v['fraccion_lgd_1'], 4) for s, v in o['por_split'].items()} }")
    print(f"Escrito: {cfg['descriptivo']['salida']}")
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
