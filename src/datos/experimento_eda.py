"""`python run.py eda --config configs/datos.yaml`: EDA inicial, hipótesis y descriptivo después del preprocesamiento.

- `eda_en_memoria(cfg)`: parte de `data/processed/loans_pd.parquet`, aplica la partición y el recorte de `cfg` y
  ajusta el preprocesador SOLO con train, todo en memoria. No escribe nada: es lo que usa el notebook 01 para la
  modificación en vivo (`SET = {"particion.train_hasta": ...}`) sin tocar los archivos que consumen B, C y D.
  Con la config del repositorio, el preprocesador que ajusta es idéntico a `models/preprocesador_pd.json`.
- `correr_eda(cfg)`: lo mismo y además escribe en `eda.salida` (`results/data/eda/` por defecto) `eda_inicial.json`,
  `hipotesis_eda.json` y `post_preprocesamiento.json`.

La partición en memoria solo puede reasignar los créditos que ya están en el parquet (originados hasta
`particion.test_hasta` del repositorio); para incluir otros años hay que rehacer los datos con `scripts/preparar_datos.py`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.preprocesamiento import Preprocesador
from src.common.results import save_json
from src.datos.eda import descriptivo_post, eda_inicial, evaluar_hipotesis
from src.datos.particion import SPLITS, asignar_split


def eda_en_memoria(cfg: dict, raiz: Path = Path(".")) -> dict:
    features = load_config(raiz / cfg["features"])["features"]
    hipotesis = load_config(raiz / cfg["eda"]["hipotesis"])["hipotesis"]
    umbral = float(cfg["eda"]["umbral_monotonia"])
    corte = cfg["particion"]

    df = pd.read_parquet(raiz / cfg["salida"]["carpeta"] / "loans_pd.parquet")
    df = df[df["issue_d"] <= pd.Timestamp(corte["test_hasta"])].copy()
    df["split"] = asignar_split(df["issue_d"], corte["train_hasta"], corte["val_hasta"])
    pre = Preprocesador(tuple(cfg["preprocesamiento"]["percentiles_recorte"]))
    pre.fit(df[df["split"] == "train"], features)

    return {
        "particion": {"cortes": dict(corte),
                      "n_por_split": {s: int((df["split"] == s).sum()) for s in SPLITS},
                      "tasa_malos_por_split": {s: float(df.loc[df["split"] == s, "target"].mean()) for s in SPLITS}},
        "inicial": eda_inicial(df, features),
        "hipotesis": evaluar_hipotesis(df[df["split"] == "train"], hipotesis, umbral=umbral),
        "post": descriptivo_post(df, pre, umbral=umbral),
        "preprocesador": pre,
    }


def correr_eda(cfg: dict, raiz: Path = Path(".")) -> dict:
    r = eda_en_memoria(cfg, raiz)
    salida = raiz / cfg["eda"]["salida"]
    save_json(salida / "eda_inicial.json", r["inicial"])
    save_json(salida / "hipotesis_eda.json", r["hipotesis"])
    save_json(salida / "post_preprocesamiento.json", r["post"])
    return r


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
