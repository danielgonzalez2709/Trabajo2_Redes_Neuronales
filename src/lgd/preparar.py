"""``loans_lgd.parquet`` y ``preprocesador_lgd.json`` (tarea D2; Parte D §3.1–§3.2; núcleo §5.1, §5.2, §5.4).

Uso: ``python -m src.lgd.preparar [--config configs/lgd/base.yaml] [--set clave=valor]`` (no usa internet).

Entradas (Parte A): ``data/processed/loans_pd.parquet`` y ``data/processed/montos.parquet`` (mismas filas y orden).
La población y la fórmula salen de ``src/lgd/objetivo.py`` (D1): Charged Off con EAD > 0 y sin NA en los campos de la
fórmula ([DECIDIDO, Camila, 24-oct]: lo excluido se cuenta en el resumen; ``lgd_valida`` filtra y no se escribe). La
distribución distinta de la LGD entre particiones se acepta y se documenta: no se filtra por año
([DECIDIDO, Camila, 24-oct]); ``lgd_por_split`` en el resumen la deja medida.

Preprocesador de la LGD — **[DECIDIDO, Camila, 24-oct]**: ``models/preprocesador_lgd.json`` es el
mismo ``Preprocesador`` de la Parte A (mismas variables de ``configs/features.yaml`` y mismos percentiles de recorte, que
se leen de ``configs/datos.yaml``), **ajustado solo con las filas ``split == "train"`` de la población LGD**. Lo usan la
regresión lineal y el MLP de LGD (misma tarea → mismo objeto, núcleo §5.2) y también se aplica a todo el test de
``loans_pd`` para ``pred_pd_test.parquet`` en D4.

Los montos posteriores al desembolso (``funded_amnt``, ``total_rec_prncp``, ``recoveries``, ``collection_recovery_fee``)
y lo derivado de ellos (``ead``, ``lgd``, ``lgd_sin_recorte``) van en el parquet solo para construir y auditar el
objetivo; nunca son variables del preprocesador ni de un modelo (núcleo §5.7).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from src.common.config import apply_overrides, load_config
from src.common.preprocesamiento import Preprocesador
from src.common.results import save_json, to_jsonable
from src.lgd.objetivo import calcular_lgd, mascara_poblacion_lgd, resumen_objetivo

CAMPOS_FORMULA = ("funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee")
COLUMNAS_OBJETIVO = ("ead", "lgd", "lgd_sin_recorte")
POSTERIORES = frozenset({"loan_status", *CAMPOS_FORMULA, *COLUMNAS_OBJETIVO})
SPLITS = ("train", "val", "test")
RAIZ = Path(__file__).resolve().parents[2]


def columnas_contrato(nombres_features: list[str]) -> list[str]:
    """Orden de columnas de ``loans_lgd.parquet`` (núcleo §5.1)."""
    return ["id", "issue_d", "split", *nombres_features, *CAMPOS_FORMULA, *COLUMNAS_OBJETIVO]


def rechazar_posteriores(nombres: list[str]) -> None:
    """``ValueError`` si alguna variable es un monto posterior al desembolso o derivada de él (núcleo §5.7)."""
    prohibidas = [n for n in nombres if n in POSTERIORES]
    if prohibidas:
        raise ValueError(f"Variables posteriores al desembolso no pueden ser features (núcleo §5.7): {prohibidas}")


def _lgd_por_split(df: pd.DataFrame) -> dict:
    """Por split: ``n`` (única fuente del conteo por split) y, si no está vacío, la forma de la LGD."""
    por_split = {}
    for s in SPLITS:
        g = df[df["split"] == s]
        por_split[s] = {"n": int(len(g))}
        if len(g):
            por_split[s].update({"media": float(g["lgd"].mean()), "mediana": float(g["lgd"].median()),
                                 "fraccion_lgd_1": float((g["lgd"] == 1.0).mean()),
                                 "fraccion_lgd_0": float((g["lgd"] == 0.0).mean())})
    return por_split


def construir_loans_lgd(loans_pd: pd.DataFrame, montos: pd.DataFrame,
                        nombres_features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Subconjunto elegible para LGD con las columnas del contrato, y su resumen (``resumen_objetivo`` + por split).

    ``nombres_features``: variables de ``configs/features.yaml`` en su orden. ``ValueError`` si ``id`` no coincide fila a
    fila entre ``loans_pd`` y ``montos`` o si una variable es un monto posterior al desembolso.
    """
    rechazar_posteriores(nombres_features)
    ids_pd = loans_pd["id"].astype(str).to_numpy()
    ids_m = montos["id"].astype(str).to_numpy()
    if len(ids_pd) != len(ids_m) or not (ids_pd == ids_m).all():
        raise ValueError("loans_pd y montos deben tener el mismo `id` fila a fila (mismas filas y mismo orden)")

    loans_pd = loans_pd.reset_index(drop=True)
    montos = montos.reset_index(drop=True)
    # calcular_lgd corre también dentro de mascara_poblacion_lgd y resumen_objetivo (D1); repetirlo evita duplicar su lógica.
    mascara = mascara_poblacion_lgd(montos)
    objetivo = calcular_lgd(montos[mascara])
    df = pd.concat([loans_pd.loc[mascara, ["id", "issue_d", "split", *nombres_features]],
                    montos.loc[mascara, list(CAMPOS_FORMULA)].astype(float),
                    objetivo[list(COLUMNAS_OBJETIVO)]], axis=1)
    df = df[columnas_contrato(nombres_features)].reset_index(drop=True)

    resumen = resumen_objetivo(montos)
    resumen["lgd_por_split"] = _lgd_por_split(df)
    return df, resumen


def ajustar_preprocesador(loans_lgd: pd.DataFrame, features: list[dict],
                          percentiles_recorte: tuple[float, float]) -> Preprocesador:
    """``Preprocesador`` de A ajustado SOLO con las filas ``split == "train"`` de la población LGD."""
    rechazar_posteriores([f["nombre"] for f in features])
    return Preprocesador(percentiles_recorte).fit(loans_lgd[loans_lgd["split"] == "train"], features)


def preparar(cfg: dict, raiz: Path = Path(".")) -> dict:
    """Lee las entradas, construye ``loans_lgd``, ajusta el preprocesador con train y escribe los tres artefactos."""
    raiz = Path(raiz)
    datos = cfg["datos"]
    cfg_a = load_config(raiz / datos["config_datos"])
    features = load_config(raiz / cfg_a["features"])["features"]
    nombres = [f["nombre"] for f in features]

    loans_pd = pd.read_parquet(raiz / datos["loans_pd"])
    montos = pd.read_parquet(raiz / datos["montos"])
    df, resumen = construir_loans_lgd(loans_pd, montos, nombres)
    pre = ajustar_preprocesador(df, features, tuple(cfg_a["preprocesamiento"]["percentiles_recorte"]))
    pre_dict = pre.to_dict()
    # Valida los dos JSON (tipos y sin NaN/inf) ANTES de escribir nada: si algo falla no quedan artefactos mezclados.
    for contenido in (pre_dict, resumen):
        json.dumps(to_jsonable(contenido), allow_nan=False)

    salida = datos["salida"]
    ruta_parquet = raiz / salida["loans_lgd"]
    ruta_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta_parquet, index=False, compression="zstd")
    save_json(raiz / salida["preprocesador"], pre_dict)
    save_json(raiz / salida["resumen"], resumen)
    return resumen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/lgd/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(RAIZ / args.config), args.overrides)
    print("Configuración final resuelta:")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), end="")
    inicio = time.perf_counter()
    r = preparar(cfg, RAIZ)
    print(f"Población LGD: {r['n_lgd_valida']:,} de {r['n_total']:,} · por split: { {s: v['n'] for s, v in r['lgd_por_split'].items()} }")
    print(f"Excluidos: por estado {r['excluidos_por_estado']} · EAD ≤ 0: {r['n_ead_no_positiva']} · "
          f"NA: {r['na_por_campo']}")
    print(f"LGD sin recorte < 0: {r['n_lgd_sin_recorte_menor_0']} · > 1: {r['n_lgd_sin_recorte_mayor_1']} · "
          f"media {r['lgd_media']:.4f} · mediana {r['lgd_mediana']:.4f}")
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
