"""Crudo de Kaggle → data/processed/loans_pd.parquet + resumen_datos.json + models/preprocesador_pd.json (Parte A §3.6).

Pasos: leer el CSV → objetivo con la tabla del enunciado → descartar los NA del objetivo → variables derivadas →
partición temporal → ajustar el preprocesador SOLO con train. Las cifras del blog salen de `resumen_datos.json`.
No usa internet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.preprocesamiento import Preprocesador
from src.common.results import save_json
from src.datos.cargar import agregar_derivadas, leer_crudo, sha256
from src.datos.objetivo import construir_objetivo
from src.datos.particion import SPLITS, asignar_split

DERIVADAS = {"antiguedad_credito_anios": ["issue_d", "earliest_cr_line"]}
BASICAS = ["id", "issue_d", "loan_status"]


def columnas_crudas(features: list[dict]) -> list[str]:
    """Columnas del CSV que hacen falta para el objetivo, la partición y las variables (incluidas las derivadas)."""
    cols = list(BASICAS)
    for f in features:
        for c in DERIVADAS.get(f["nombre"], [f["nombre"]]):
            if c not in cols:
                cols.append(c)
    return cols


def _tasas_por(df: pd.DataFrame, clave: str) -> dict:
    return {str(k): float(v) for k, v in df.groupby(clave)["target"].mean().items()}


def preparar(cfg: dict, raiz: Path = Path(".")) -> dict:
    """Ejecuta la preparación completa y devuelve el resumen que también escribe en disco."""
    features = load_config(raiz / cfg["features"])["features"]
    nombres = [f["nombre"] for f in features]
    ruta_csv = raiz / cfg["crudo"]["csv"]

    n_variables = len(pd.read_csv(ruta_csv, nrows=0).columns)
    crudo = leer_crudo(ruta_csv, columnas_crudas(features))
    crudo["target"] = construir_objetivo(crudo["loan_status"])
    crudo["anio_origen"] = crudo["issue_d"].dt.year

    por_anio = crudo.groupby("anio_origen")["target"]
    fraccion_na = {str(a): float(v) for a, v in (1 - por_anio.count() / por_anio.size()).items()}

    corte = cfg["particion"]
    df = crudo[crudo["target"].notna()].copy()
    posteriores = df["issue_d"] > pd.Timestamp(corte["test_hasta"])
    df = df[~posteriores]
    df = agregar_derivadas(df)
    df["split"] = asignar_split(df["issue_d"], corte["train_hasta"], corte["val_hasta"])
    df["target"] = df["target"].astype("int8")
    df["id"] = df["id"].astype(str)
    df = df[["id", "issue_d", "anio_origen", "target", "split", *nombres]].reset_index(drop=True)

    pre = Preprocesador(tuple(cfg["preprocesamiento"]["percentiles_recorte"]))
    pre.fit(df[df["split"] == "train"], features)

    salida = raiz / cfg["salida"]["carpeta"]
    salida.mkdir(parents=True, exist_ok=True)
    df.to_parquet(salida / "loans_pd.parquet", index=False, compression="zstd")
    save_json(raiz / cfg["salida"]["preprocesador"], pre.to_dict())

    resumen = {
        "archivo_crudo": cfg["crudo"]["csv"],
        "url": cfg["crudo"]["url"],
        "sha256": sha256(ruta_csv),
        "fecha_descarga": cfg["crudo"]["fecha_descarga"],
        "n_registros_crudo": int(len(crudo)),
        "n_variables_crudo": int(n_variables),
        "periodo": {"min": crudo["issue_d"].min().strftime("%Y-%m"), "max": crudo["issue_d"].max().strftime("%Y-%m")},
        "conteo_por_loan_status": {str(k): int(v) for k, v in crudo["loan_status"].value_counts().items()},
        "n_objetivo_na": int(crudo["target"].isna().sum()),
        "n_resueltos_posteriores_a_test": int(posteriores.sum()),
        "particion": dict(corte),
        "n_modelado": int(len(df)),
        "n_por_split": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "tasa_malos_por_split": {s: float(df.loc[df["split"] == s, "target"].mean()) for s in SPLITS},
        "tasa_malos_por_anio": _tasas_por(crudo.dropna(subset=["target"]), "anio_origen"),
        "fraccion_na_por_anio": fraccion_na,
        "n_por_anio": {str(k): int(v) for k, v in crudo["anio_origen"].value_counts().sort_index().items()},
        "variables": nombres,
        "faltantes_por_variable_train": {
            n: float(df.loc[df["split"] == "train", n].isna().mean()) for n in nombres
        },
    }
    save_json(salida / "resumen_datos.json", resumen)
    return resumen
