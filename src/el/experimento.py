"""`python run.py el --config configs/el/base.yaml`: pérdida esperada de cada crédito de test (Parte E, tarea E5).

Lee la PD de B (`pd.predicciones`), la LGD de D para todo el test (`lgd.predicciones`), los montos de A y escribe en
`salida`: `el_test.parquet` (por crédito, con su puntaje `score` y su `banda`) y `summary.json` (E[L] de todo el test;
contraste con la pérdida realizada solo en los créditos resueltos, total y por banda de puntaje; créditos en mora con su
pérdida provisional aparte; anomalía de los pagados; `evidencia`). El puntaje sale de la scorecard de la Parte C con el
bloque `scorecard` de la config: score = offset − factor · logit(PD).
Si todavía falta la entrega de B o de D, lo dice y termina con código 2 sin escribir nada.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.common.results import save_json
from src.el.perdida_esperada import bandas_de_puntaje, calcular_el, resumen_el
from src.pd.experimento import huella_archivo, huella_particion
from src.scorecard.scorecard import parametros_scorecard, score_desde_pd


def criterio_bandas(n_bandas: int) -> str:
    """Texto del criterio de bandas de summary.json, con el número de bandas pedido."""
    return f"cuantiles del puntaje de la scorecard en {n_bandas} bandas; banda 1 = menor puntaje (más riesgo)"


def evidencia(loans: pd.DataFrame, ruta_pd: Path, ruta_lgd: Path) -> dict:
    """Núcleo §6, "Mismos datos": huella de la partición (todas las filas de `loans`, en el orden del parquet) y de los
    bytes de los dos archivos de predicciones que consume la E[L] (la E[L] no usa preprocesador)."""
    return {"sha256_particion": huella_particion(loans["id"], loans["split"]),
            "sha256_pred_pd": huella_archivo(ruta_pd), "sha256_pred_lgd": huella_archivo(ruta_lgd)}


def correr_el(cfg: dict, raiz: Path = Path(".")) -> dict:
    loans = pd.read_parquet(raiz / cfg["datos"]["loans"], columns=["id", "target", "split"])
    test = loans[loans["split"] == "test"].reset_index(drop=True)
    montos = pd.read_parquet(raiz / cfg["datos"]["montos"])
    ruta_pd, ruta_lgd = raiz / cfg["pd"]["predicciones"], raiz / cfg["lgd"]["predicciones"]
    el = calcular_el(test, pd.read_parquet(ruta_pd), pd.read_parquet(ruta_lgd), montos, cfg["pd"]["modelo"],
                     cfg["lgd"]["modelo"])
    params = parametros_scorecard(cfg["scorecard"])  # base, odds_base, pdo, factor, offset (Parte C §3.1)
    el["score"] = score_desde_pd(el["pd"].to_numpy(), params)  # mayor puntaje = menor riesgo
    n_bandas = int(cfg["bandas"]["n"])
    el["banda"] = bandas_de_puntaje(el["score"], n_bandas)
    resumen_bandas = resumen_el(el)
    resumen = {"modelo_pd": cfg["pd"]["modelo"], "modelo_lgd": cfg["lgd"]["modelo"], "ead": "funded_amnt (originación)",
               "scorecard": params, "bandas": criterio_bandas(n_bandas), "n_bandas_pedidas": n_bandas,
               # con empates en el puntaje, qcut fusiona bordes repetidos y puede dar menos bandas
               "n_bandas_obtenidas": len(resumen_bandas["por_banda"]),
               **resumen_bandas, "evidencia": evidencia(loans, ruta_pd, ruta_lgd)}
    salida = raiz / cfg["salida"]
    salida.mkdir(parents=True, exist_ok=True)
    el.to_parquet(salida / "el_test.parquet", index=False)
    save_json(salida / "summary.json", resumen)
    return resumen


def _texto_razon(bloque: dict) -> str:
    razon = bloque["razon_el_sobre_perdida_observada"]
    return "—" if razon is None else f"{razon:.3f}"


def manejador(cfg: dict | None) -> int:
    inicio = time.perf_counter()
    faltan = [(parte, cfg[clave]["predicciones"]) for parte, clave in (("B", "pd"), ("D", "lgd"))
              if not Path(cfg[clave]["predicciones"]).is_file()]
    if faltan:
        for parte, ruta in faltan:
            print(f"Falta {ruta} (lo entrega la Parte {parte}; núcleo §5.4).")
        return 2
    r = correr_el(cfg)
    t, res, mora, anom = r["total"], r["resueltos"], r["en_mora"], r["anomalia_pagados"]
    print(f"Todo el test: {t['n']:,} créditos · EAD {t['ead']:,.0f} USD · E[L] {t['el']:,.0f} USD")
    print(f"Contraste con los resueltos (n = {res['n']:,}): E[L] {res['el']:,.0f} USD · pérdida realizada "
          f"{res['perdida_observada']:,.0f} USD · razón {_texto_razon(res)}")
    print(f"En mora, cobranza abierta (n = {mora['n']:,}): E[L] {mora['el']:,.0f} USD · pérdida provisional "
          f"(LGD = 1 sobre el pendiente) {mora['perdida_provisional']:,.0f} USD")
    print(f"Anomalía: {anom['n']:,} pagados con {anom['principal_no_registrado']:,.0f} USD de principal no registrado "
          "(no se suma a la pérdida)")
    print("Por banda, contraste de los resueltos:")
    print("Banda  puntaje          n resueltos  E[L] (USD)    realizada (USD)  razón  n mora  provisional (USD)")
    for b in r["por_banda"]:
        res_b, mora_b = b["resueltos"], b["en_mora"]
        print(f"{b['banda']:>5}  {b['score_min']:5.1f}–{b['score_max']:5.1f}  {res_b['n']:>11,}  "
              f"{res_b['el']:>12,.0f}  "
              f"{res_b['perdida_observada']:>15,.0f}  {_texto_razon(res_b):>5}  {mora_b['n']:>6,}  "
              f"{mora_b['perdida_provisional']:>17,.0f}")
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0
