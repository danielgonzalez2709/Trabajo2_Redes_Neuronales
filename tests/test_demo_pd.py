# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo de la PD en < 2 min en CPU (Parte B §4, tarea B6)
#
# `python run.py pd --config configs/pd/demo.yaml` corre en otro proceso, como en la terminal, con los datos reales de A
# y las salidas en `tmp_path` (`--set salida=… --set modelos=…`): nunca escribe en `results/` ni en `models/` reales.
# El límite es 115 s, el corte de `correr_demo` del notebook 00 (`src/common/notebooks.py`), por debajo de los 2 min del
# enunciado. Lentas (marcador `lento`, la CI no las corre): `pytest -m lento tests/test_demo_pd.py`. Se omiten si faltan
# `data/processed/loans_pd.parquet` o `models/preprocesador_pd.json`.

# %%
"""B6: la demo de PD corre por CLI en < 115 s en CPU, deja sus artefactos y no toca los reales (Parte B §4)."""

import hashlib  # SHA-256 de los artefactos reales
import subprocess  # la demo corre en otro proceso, como en la terminal
import sys  # mismo intérprete del entorno
import time  # reloj de pared de la corrida
from pathlib import Path  # rutas

import numpy as np  # rangos de las PD
import pandas as pd  # pred_test.parquet y filas para predecir
import pytest  # marcador lento, skip y parametrización

from src.common.preprocesamiento import Preprocesador  # preprocesador guardado de A (sin reajustar)
from src.common.results import load_json  # summary.json
from src.pd.mlp import MLPPD  # carga de pd_mlp.pt

# %% [markdown]
# ## Constantes

# %%
RAIZ = Path(__file__).resolve().parents[1]  # raíz del repositorio (cwd de run.py)
DEMO = RAIZ / "configs" / "pd" / "demo.yaml"  # config de la demo
PARQUET = RAIZ / "data" / "processed" / "loans_pd.parquet"  # datos reales de A
PREPROCESADOR = RAIZ / "models" / "preprocesador_pd.json"  # preprocesador de A
LIMITE_S = 115  # corte de correr_demo (src/common/notebooks.py, timeout_s=115) < 2 min del enunciado
REALES = [RAIZ / "results" / "data" / "pd" / "summary.json", RAIZ / "results" / "data" / "pd" / "pred_test.parquet",
          RAIZ / "models" / "pd_mlp.pt"]  # artefactos reales que usan C y E (no deben cambiar)
FILAS_DEMO = "train 19005 · val 15397 · test 15598"  # reparto de 50 000 por el mayor resto (90 347/73 194/74 154)
N_FILAS_PREDICCION = 5  # filas del parquet para comprobar que pd_mlp.pt predice
CAMBIOS_EN_VIVO = ["mlp.capas=[64,32]", "mlp.dropout=0.3", "mlp.lr=1e-2", "mlp.max_epocas=5", "datos.muestra=20000",
                   "semillas.n=5", "mlp.lr=1e-4"]  # núcleo §4.1 + el peor caso medido (lr = 1e-4)

pytestmark = [  # todas las pruebas del archivo
    pytest.mark.lento,  # tardan (la CI corre -m "not lento")
    pytest.mark.skipif(not (PARQUET.is_file() and PREPROCESADOR.is_file()),
                       reason="faltan los datos procesados reales de A"),  # clon sin datos
]


# %% [markdown]
# ## Ayudas

# %%
def _sha(ruta: Path) -> str | None:
    """SHA-256 de ``ruta`` o None si no existe."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest() if ruta.is_file() else None  # huella o ausencia


def _demo(carpeta: Path, extra=()) -> tuple[subprocess.CompletedProcess, float]:
    """(proceso, segundos de pared) de run.py pd con la demo, salidas en ``carpeta`` y los --set de ``extra``."""
    argv = [sys.executable, str(RAIZ / "run.py"), "pd", "--config", str(DEMO),  # comando de la demo
            "--set", f"salida={(carpeta / 'res').as_posix()}", "--set", f"modelos={(carpeta / 'models').as_posix()}"]
    for cambio in extra:  # modificaciones en vivo
        argv += ["--set", cambio]  # una por una
    inicio = time.perf_counter()  # reloj de pared
    proceso = subprocess.run(argv, cwd=RAIZ, capture_output=True, encoding="utf-8", check=False)  # como la terminal
    return proceso, time.perf_counter() - inicio  # resultado y duración


# %% [markdown]
# ## L1. La demo completa

# %%
def test_l1_demo_pd(tmp_path):
    """Código 0 en < 115 s; summary con logistica y mlp (fuera de la búsqueda); pred_test con las dos PD en [0, 1];
    pd_mlp.pt carga y predice; los artefactos reales no cambian; la consola trae las filas de la muestra."""
    antes = [_sha(r) for r in REALES]  # huellas de los artefactos reales
    proceso, segundos = _demo(tmp_path)  # la demo
    assert proceso.returncode == 0, proceso.stderr  # terminó bien
    assert segundos < LIMITE_S  # < 115 s de pared
    assert FILAS_DEMO in proceso.stdout and "Tiempo de ejecución" in proceso.stdout  # muestra y tiempo
    summary = load_json(tmp_path / "res" / "summary.json")  # resultados de la demo
    assert set(summary) == {"logistica", "mlp"} and summary["mlp"]["config_de_busqueda"] is False  # dos modelos
    pred = pd.read_parquet(tmp_path / "res" / "pred_test.parquet")  # PD de test
    for columna in ("pred_logistica", "pred_mlp"):  # ambos modelos
        assert pred[columna].between(0.0, 1.0).all()  # probabilidades
    pre = Preprocesador.from_dict(load_json(PREPROCESADOR))  # preprocesador de A (sin reajustar)
    X = pre.transform(pd.read_parquet(PARQUET).head(N_FILAS_PREDICCION))  # unas filas reales
    p = MLPPD.cargar(tmp_path / "models" / "pd_mlp.pt").predict_proba(X)  # red de la demo
    assert p.shape == (N_FILAS_PREDICCION,) and np.all((p >= 0.0) & (p <= 1.0))  # predice PD
    assert [_sha(r) for r in REALES] == antes  # nada real cambió


# %% [markdown]
# ## L2. Modificaciones en vivo (núcleo §4.1) y el peor caso

# %%
@pytest.mark.parametrize("cambio", CAMBIOS_EN_VIVO, ids=["capas", "dropout", "lr_alta", "max_epocas", "muestra",
                                                         "semillas", "lr_baja"])
def test_l2_demo_pd_en_vivo(tmp_path, cambio):
    """Cada --set de la sustentación termina con código 0 en < 115 s, en su carpeta temporal."""
    proceso, segundos = _demo(tmp_path, [cambio])  # demo modificada
    assert proceso.returncode == 0, proceso.stderr  # terminó bien
    assert segundos < LIMITE_S  # < 115 s de pared
