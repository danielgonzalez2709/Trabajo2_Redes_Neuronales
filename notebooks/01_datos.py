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
# # Parte A — Datos, partición y análisis descriptivo
#
# Capa delgada sobre `src/` (núcleo §5.8): el notebook no calcula nada por su cuenta, solo llama funciones del
# repositorio y muestra sus resultados. **Se edita este `.py`; el `.ipynb` lo genera `python run.py notebooks`.**
#
# Todo se calcula **en memoria** a partir de `data/processed/loans_pd.parquet`: el notebook no escribe en
# `data/processed/`, `models/`, `results/` ni `blog/` (esos archivos los consumen las demás partes). Para la
# modificación en vivo, cambia `SET` (mismas claves que `--set`) y corre todo. Ejemplos:
#
# - otra partición: `SET = {"particion.train_hasta": "2011-12-31", "particion.val_hasta": "2012-12-31"}`
# - otro recorte: `SET = {"preprocesamiento.percentiles_recorte": [5.0, 95.0]}`
# - otro umbral de monotonía: `SET = {"eda.umbral_monotonia": 0.8}`
#
# Rehacer los datos desde el CSV crudo de Kaggle es otro comando: `python scripts/preparar_datos.py`.

# %% tags=["parameters"]
CONFIG = "configs/datos.yaml"
SET = {}

# %%
import tempfile
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from src.common.notebooks import config_notebook
from src.common.results import load_json

cfg = config_notebook(CONFIG, SET)

# %% [markdown]
# ## 1. Datos procesados (versionados en el repositorio)

# %%
resumen = load_json("data/processed/resumen_datos.json")
print(f"Registros crudos: {resumen['n_registros_crudo']:,} · variables: {resumen['n_variables_crudo']} · "
      f"periodo {resumen['periodo']['min']} a {resumen['periodo']['max']} · descarga {resumen['fecha_descarga']}")
pd.Series(resumen["conteo_por_loan_status"], name="créditos").to_frame()

# %% [markdown]
# ## 2. Partición, análisis descriptivo, hipótesis y datos después del preprocesamiento (en memoria)

# %%
from src.datos.experimento_eda import eda_en_memoria

eda = eda_en_memoria(cfg)
pd.DataFrame({"créditos": eda["particion"]["n_por_split"], "tasa de malos": eda["particion"]["tasa_malos_por_split"]})

# %%
pd.DataFrame([{"id": h["id"], "hipótesis": h["enunciado"], "lectura": h["veredicto_eda"]} for h in eda["hipotesis"]])

# %%
pd.DataFrame(eda["post"]["psi_vs_train"]).sort_values("test", ascending=False).round(3)

# %% [markdown]
# ## 3. Figuras (las mismas del blog, dibujadas con esta configuración en una carpeta temporal)

# %%
from src.viz.figs_datos import generar

for ruta in generar(destino=Path(tempfile.mkdtemp()), datos=eda):
    display(Image(filename=str(ruta)))
