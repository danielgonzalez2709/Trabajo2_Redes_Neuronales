"""`python run.py eda --config configs/datos.yaml`: EDA inicial, hipótesis y descriptivo después del preprocesamiento.

Lee `data/processed/loans_pd.parquet` y `models/preprocesador_pd.json` (no reentrena nada) y escribe en
`results/data/eda/`: `eda_inicial.json`, `hipotesis_eda.json`, `post_preprocesamiento.json`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json, save_json
from src.datos.eda import descriptivo_post, eda_inicial, evaluar_hipotesis


def correr_eda(cfg: dict, raiz: Path = Path(".")) -> dict:
    features = load_config(raiz / cfg["features"])["features"]
    hipotesis = load_config(raiz / cfg["eda"]["hipotesis"])["hipotesis"]
    df = pd.read_parquet(raiz / cfg["salida"]["carpeta"] / "loans_pd.parquet")
    pre = Preprocesador.from_dict(load_json(raiz / cfg["salida"]["preprocesador"]))
    salida = raiz / cfg["eda"]["salida"]

    inicial = eda_inicial(df, features)
    hips = evaluar_hipotesis(df[df["split"] == "train"], hipotesis)
    post = descriptivo_post(df, pre)
    save_json(salida / "eda_inicial.json", inicial)
    save_json(salida / "hipotesis_eda.json", hips)
    save_json(salida / "post_preprocesamiento.json", post)
    return {"inicial": inicial, "hipotesis": hips, "post": post}


def manejador(cfg: dict | None) -> int:
    inicio = time.perf_counter()
    r = correr_eda(cfg)
    for h in r["hipotesis"]:
        print(f"{h['id']}: {h['veredicto_eda']:>12} · {h['enunciado']}")
    psi_test = r["post"]["psi_vs_train"].get("test", {})
    if psi_test:
        peor = max(psi_test, key=psi_test.get)
        print(f"PSI train→test más alto: {peor} = {psi_test[peor]:.3f}")
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0
