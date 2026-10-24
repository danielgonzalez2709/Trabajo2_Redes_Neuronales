"""Análisis descriptivo (Parte A §3.7 y §3.8). La Parte D reutiliza estas funciones con `objetivo="lgd"`.

- EDA inicial (§3.7): sobre los datos con objetivo, ANTES de transformar, solo en train (para no mirar val/test al
  formular hipótesis). Tasa del objetivo por decil (numéricas) o por categoría.
- Después del preprocesamiento (§3.8): sobre la matriz que ven los modelos, por split: efecto de cada transformación
  (incluida la distribución antes vs. después), verificación de la estandarización, estabilidad entre splits (PSI),
  correlaciones y relación con el objetivo.

PSI (índice de estabilidad poblacional) de una variable, con los cortes de los deciles de la referencia (train):
    PSI = Σ_i (a_i − e_i) · ln(a_i / e_i) = KL(a‖e) + KL(e‖a)
donde e_i y a_i son las proporciones de la referencia y de la población comparada en el bin i. La segunda igualdad
(divergencia de Kullback–Leibler simetrizada) es álgebra: Σ a ln(a/e) + Σ e ln(e/a); `tests/test_eda.py` la comprueba.
Umbrales de interpretación: NO se usan (no hay fuente verificada; ver blog/references.bib).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

from src.common.preprocesamiento import DESCONOCIDO, OTRA, Preprocesador

EPS_PSI = 1e-6  # evita log(0) en bins vacíos
# Relación "monótona": |Spearman| entre el orden del grupo y su tasa >= este umbral (configurable en
# configs/datos.yaml, eda.umbral_monotonia; Parte A §3.7). La monotonía estricta es demasiado frágil: con datos reales
# una oscilación de 0,001 entre deciles vecinos la rompería.
UMBRAL_MONOTONIA = 0.9
CUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
N_BINS_HISTOGRAMA = 30
MAX_VALORES_DISCRETA = 10


# ------------------------------------------------------------------ EDA inicial (§3.7)
def tasa_por_grupo(x: pd.Series, y: pd.Series, tipo: str, n_bins: int = 10) -> pd.DataFrame:
    """Tasa media de `y` por decil de `x` (num) o por categoría (cat). Los vacíos van en su propio grupo."""
    if tipo == "num":
        bins = pd.qcut(x, q=n_bins, duplicates="drop")
        if len(bins.cat.categories) >= 3:
            grupo = bins.astype("string").fillna("vacío")
            orden = {str(c): i for i, c in enumerate(bins.cat.categories)} | {"vacío": len(bins.cat.categories)}
        else:  # conteo con muchos ceros (pub_rec, delinq_2yrs): valor exacto y la cola desde el p99 junta
            tope = float(x.quantile(0.99))
            valores = sorted(x.dropna().clip(upper=tope).unique())
            etiqueta = {v: (f"≥{v:g}" if v >= tope and v > valores[0] else f"{v:g}") for v in valores}
            grupo = x.clip(upper=tope).map(etiqueta).astype("string").fillna("vacío")
            orden = {etiqueta[v]: i for i, v in enumerate(valores)} | {"vacío": len(valores)}
    else:
        grupo = x.astype("string").str.strip().fillna("vacío")
        orden = None
    tabla = (pd.DataFrame({"grupo": grupo, "y": y}).groupby("grupo", observed=True)["y"]
             .agg(n="size", tasa="mean").reset_index())
    if orden is not None:
        tabla = tabla.sort_values("grupo", key=lambda s: s.map(orden)).reset_index(drop=True)
    else:
        tabla = tabla.sort_values("n", ascending=False).reset_index(drop=True)
    return tabla


def grupos_ordenados(x: pd.Series, n_bins: int = 10) -> pd.Series:
    """Índice de grupo ordenado por valor: deciles; si quedan menos de 3 (variables de conteo con muchos ceros, como
    `pub_rec`), el valor exacto con la cola agrupada desde el percentil 99."""
    g = pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
    if g.nunique() >= 3:
        return g
    return (x.clip(upper=x.quantile(0.99)).rank(method="dense") - 1).astype(int)


def tendencia_numerica(x: pd.Series, y: pd.Series, n_bins: int = 10, umbral: float = UMBRAL_MONOTONIA) -> dict:
    """Spearman entre el orden del grupo y su tasa, y tasas del primer y del último grupo (`grupos_ordenados`)."""
    validos = x.notna()
    grupos = grupos_ordenados(x[validos], n_bins)
    tasas = y[validos].groupby(grupos).mean()
    rho = float(spearmanr(tasas.index, tasas.values).statistic) if len(tasas) > 2 else float(
        np.sign(tasas.iloc[-1] - tasas.iloc[0]))
    return {"spearman_grupos": rho, "tasa_grupo_bajo": float(tasas.iloc[0]),
            "tasa_grupo_alto": float(tasas.iloc[-1]), "n_grupos": int(len(tasas)),
            "monotona": bool(abs(rho) >= umbral), "tasas_por_grupo": [float(v) for v in tasas]}


def evaluar_hipotesis(train: pd.DataFrame, hipotesis: list[dict], objetivo: str = "target",
                      umbral: float = UMBRAL_MONOTONIA) -> list[dict]:
    """Primera evidencia (EDA en train) de cada hipótesis. Veredicto provisional: lo cierra la Parte C con SHAP."""
    salida = []
    for h in hipotesis:
        x, y = train[h["variable"]], train[objetivo].astype(float)
        if "direccion" in h:
            t = tendencia_numerica(x, y, umbral=umbral)
            signo = 1 if h["direccion"] == "+" else -1
            apoya = signo * (t["tasa_grupo_alto"] - t["tasa_grupo_bajo"]) > 0 and signo * t["spearman_grupos"] > 0
            evidencia = t
            if apoya and not t["monotona"]:
                salida.append({**h, "evidencia_eda": evidencia, "veredicto_eda": "la apoya, pero no es monótona"})
                continue
        else:
            tasas = y.groupby(x.astype("string").str.strip()).mean()
            a, b = float(tasas.get(h["mas_riesgosa"], np.nan)), float(tasas.get(h["referencia"], np.nan))
            apoya = bool(a > b)
            evidencia = {"tasa_mas_riesgosa": a, "tasa_referencia": b}
        salida.append({**h, "evidencia_eda": evidencia,
                       "veredicto_eda": "la apoya" if apoya else "no la apoya"})
    return salida


def eda_inicial(df: pd.DataFrame, features: list[dict], objetivo: str = "target") -> dict:
    """Resumen por variable en train: faltantes, estadísticos y tasa del objetivo por grupo."""
    train = df[df["split"] == "train"]
    y = train[objetivo].astype(float)
    out = {"n_train": int(len(train)), "tasa_train": float(y.mean()), "variables": {}}
    for f in features:
        x = train[f["nombre"]]
        info = {"tipo": f["tipo"], "faltantes": float(x.isna().mean()),
                "tabla": tasa_por_grupo(x, y, f["tipo"]).to_dict(orient="records")}
        if f["tipo"] == "num":
            q = x.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            info["estadisticos"] = {"media": float(x.mean()), "desv": float(x.std()), "min": float(x.min()),
                                    "max": float(x.max()), **{f"p{int(k * 100)}": float(v) for k, v in q.items()}}
        out["variables"][f["nombre"]] = info
    return out


# ------------------------------------------------- después del preprocesamiento (§3.8)
def psi(referencia: np.ndarray, comparada: np.ndarray, n_bins: int = 10) -> float:
    """PSI con los cortes de los deciles de la referencia (ver la fórmula en el docstring del módulo).

    Si la referencia tiene ≤ n_bins valores distintos (variables de conteo recortadas, p. ej. `pub_rec` ∈ {0, 1}), cada
    valor es su propio bin: con cuantiles los cortes colapsarían en uno solo y el PSI daría 0 sin medir nada (H-01).
    """
    valores = np.unique(referencia)
    if len(valores) <= n_bins:
        bordes = (valores[:-1] + valores[1:]) / 2  # cortes entre valores consecutivos
    else:
        bordes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(bordes, referencia, side="right"), minlength=len(bordes) + 1) / len(referencia)
    a = np.bincount(np.searchsorted(bordes, comparada, side="right"), minlength=len(bordes) + 1) / len(comparada)
    e, a = np.clip(e, EPS_PSI, None), np.clip(a, EPS_PSI, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _efecto_transformaciones(df: pd.DataFrame, pre: Preprocesador) -> dict:
    out = {}
    for v in pre.variables:
        x = df[v["nombre"]]
        if v["tipo"] == "num":
            num = pd.to_numeric(x, errors="raise").astype(float)
            lleno = num.fillna(v["mediana"])
            out[v["nombre"]] = {"imputado": float(num.isna().mean()),
                                "recortado_abajo": float((lleno < v["recorte"][0]).mean()),
                                "recortado_arriba": float((lleno > v["recorte"][1]).mean())}
        else:
            limpio = Preprocesador._limpiar_cat(x)
            out[v["nombre"]] = {"desconocido": float((limpio == DESCONOCIDO).mean()),
                                "a_otra": float((~limpio.isin(v["categorias"]) | (limpio == OTRA)).mean())}
    return out


def _resumen_distribucion(x: np.ndarray) -> dict:
    """Cuantiles, media, desviación, asimetría e histograma (30 bins) de un vector sin vacíos."""
    conteos, bordes = np.histogram(x, bins=N_BINS_HISTOGRAMA)
    return {"cuantiles": {f"p{round(q * 100)}": float(np.quantile(x, q)) for q in CUANTILES},
            "media": float(np.mean(x)), "desv": float(np.std(x)), "asimetria": float(skew(x)),
            "histograma": {"bordes": [float(b) for b in bordes], "conteos": [int(c) for c in conteos]}}


def distribucion_antes_despues(train: pd.DataFrame, pre: Preprocesador) -> dict:
    """A §3.8.1: distribución de cada variable numérica de train antes (cruda, sin vacíos) y después del preprocesador."""
    X = pre.transform(train)
    nombres = pre.nombres_salida()
    out = {}
    for v in pre.variables:
        if v["tipo"] != "num":
            continue
        crudo = pd.to_numeric(train[v["nombre"]], errors="raise").astype(float).dropna().to_numpy()
        out[v["nombre"]] = {"log1p": v["log1p"], "recorte": v["recorte"],
                            "proporcion_cero_cruda": float(np.mean(crudo == 0)),
                            "antes": _resumen_distribucion(crudo),
                            "despues": _resumen_distribucion(X[:, nombres.index(v["nombre"])])}
    return out


def distribucion_discretas(df: pd.DataFrame, pre: Preprocesador, splits: list[str]) -> dict:
    """Proporción de cada valor (ya recortado) de las variables numéricas con ≤ 10 valores distintos, por split.

    Da cifras como "con al menos un registro público: 4,1 % en train y 19,5 % en test" sin tener que leer el parquet.
    """
    out = {}
    for v in pre.variables:
        if v["tipo"] != "num":
            continue
        x = pd.to_numeric(df[v["nombre"]], errors="raise").astype(float).fillna(v["mediana"]).clip(*v["recorte"])
        if x.nunique() > MAX_VALORES_DISCRETA:
            continue
        out[v["nombre"]] = {s: {f"{k:g}": float(p) for k, p in
                                x[df["split"] == s].value_counts(normalize=True).sort_index().items()}
                            for s in splits}
    return out


def descriptivo_post(df: pd.DataFrame, pre: Preprocesador, objetivo: str = "target",
                     umbral: float = UMBRAL_MONOTONIA) -> dict:
    """Los cinco puntos de A §3.8 sobre la salida del preprocesador, por split."""
    nombres = pre.nombres_salida()
    numericas = [v["nombre"] for v in pre.variables if v["tipo"] == "num"]
    idx_num = [nombres.index(n) for n in numericas]
    splits = [s for s in ("train", "val", "test") if (df["split"] == s).any()]
    X = {s: pre.transform(df[df["split"] == s]) for s in splits}
    y = {s: df.loc[df["split"] == s, objetivo].astype(float).to_numpy() for s in splits}

    efecto = {s: _efecto_transformaciones(df[df["split"] == s], pre) for s in splits}
    estandar = {s: {n: {"media": float(X[s][:, i].mean()), "desv": float(X[s][:, i].std())}
                    for n, i in zip(numericas, idx_num)} for s in splits}
    psi_tabla = {s: {n: psi(X["train"][:, i], X[s][:, i]) for n, i in zip(numericas, idx_num)}
                 for s in splits if s != "train"}
    # PSI de las categóricas: proporción de cada columna one-hot
    for s in psi_tabla:
        for v in pre.variables:
            if v["tipo"] == "cat":
                cols = [nombres.index(f"{v['nombre']}={c}") for c in v["categorias"]]
                e = np.clip(X["train"][:, cols].mean(axis=0), EPS_PSI, None)
                a = np.clip(X[s][:, cols].mean(axis=0), EPS_PSI, None)
                psi_tabla[s][v["nombre"]] = float(np.sum((a - e) * np.log(a / e)))
    corr = np.corrcoef(X["train"][:, idx_num], rowvar=False)
    relacion = {n: tendencia_numerica(pd.Series(X["train"][:, i]), pd.Series(y["train"]), umbral=umbral)
                for n, i in zip(numericas, idx_num)}
    return {
        "splits": splits,
        "n_por_split": {s: int(len(y[s])) for s in splits},
        "tasa_objetivo_por_split": {s: float(y[s].mean()) for s in splits},
        "efecto_transformaciones": efecto,
        "distribucion_antes_despues_train": distribucion_antes_despues(df[df["split"] == "train"], pre),
        "distribucion_discretas": distribucion_discretas(df, pre, splits),
        "estandarizacion": estandar,
        "psi_vs_train": psi_tabla,
        "correlacion_train": {"variables": numericas, "matriz": corr.round(6).tolist()},
        "relacion_con_objetivo_train": relacion,
    }
