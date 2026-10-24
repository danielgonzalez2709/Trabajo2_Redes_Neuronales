"""`python run.py el --config configs/el/base.yaml`: pérdida esperada de cada crédito de test (Parte E, tarea E5).

Lee la PD de B (`pd.predicciones`), la LGD de D para todo el test (`lgd.predicciones`), los montos de A y escribe en
`salida`: `el_test.parquet` (por crédito) y `summary.json` (totales, por banda y contraste con la pérdida observada).
Si todavía falta la entrega de B o de D, lo dice y termina con código 2 sin escribir nada.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.common.results import save_json
from src.el.perdida_esperada import calcular_el, resumen_el


def correr_el(cfg: dict, raiz: Path = Path(".")) -> dict:
    loans = pd.read_parquet(raiz / cfg["datos"]["loans"], columns=["id", "target", "split"])
    test = loans[loans["split"] == "test"].reset_index(drop=True)
    montos = pd.read_parquet(raiz / cfg["datos"]["montos"])
    pred_pd = pd.read_parquet(raiz / cfg["pd"]["predicciones"])
    pred_lgd = pd.read_parquet(raiz / cfg["lgd"]["predicciones"])
    el = calcular_el(test, pred_pd, pred_lgd, montos, cfg["pd"]["modelo"], cfg["lgd"]["modelo"])
    resumen = {"modelo_pd": cfg["pd"]["modelo"], "modelo_lgd": cfg["lgd"]["modelo"], "ead": "funded_amnt (originación)",
               **resumen_el(el, n_bandas=int(cfg["bandas"]["n"]))}
    salida = raiz / cfg["salida"]
    salida.mkdir(parents=True, exist_ok=True)
    el.to_parquet(salida / "el_test.parquet", index=False)
    save_json(salida / "summary.json", resumen)
    return resumen


def manejador(cfg: dict | None) -> int:
    inicio = time.perf_counter()
    faltan = [(parte, cfg[clave]["predicciones"]) for parte, clave in (("B", "pd"), ("D", "lgd"))
              if not Path(cfg[clave]["predicciones"]).is_file()]
    if faltan:
        for parte, ruta in faltan:
            print(f"Falta {ruta} (lo entrega la Parte {parte}; núcleo §5.4).")
        return 2
    r = correr_el(cfg)
    t = r["total"]
    print(f"Créditos de test: {t['n']:,} · EAD total: {t['ead']:,.0f} USD")
    print(f"E[L] total: {t['el']:,.0f} USD · pérdida observada: {t['perdida_observada']:,.0f} USD")
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0
