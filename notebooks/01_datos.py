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
# Capa delgada sobre `src/` (núcleo §5.8): este notebook no calcula nada por su cuenta, solo llama funciones del
# repositorio y muestra sus resultados. **Se edita este `.py`; el `.ipynb` lo genera `python run.py notebooks`.**
#
# Para la modificación en vivo, cambia la celda de parámetros y corre todo. Ejemplos:
# `SET = {"particion.train_hasta": "2011-12-31", "particion.val_hasta": "2012-12-31"}` (otra partición; requiere el
# CSV crudo) o `SET = {"preprocesamiento.percentiles_recorte": [5.0, 95.0]}`.

# %% tags=["parameters"]
CONFIG = "configs/datos.yaml"
SET = {}
REPREPARAR = False  # True: rehace data/processed desde el CSV crudo de Kaggle (debe estar en data/raw/)

# %%
from IPython.display import Image, display
import pandas as pd

from src.common.notebooks import config_notebook
from src.common.results import load_json

cfg = config_notebook(CONFIG, SET)

# %% [markdown]
# ## 1. Datos procesados
# Con `REPREPARAR = True` se corre la preparación completa (objetivo, variables, partición y preprocesador ajustado
# solo con train). Sin el CSV crudo, se usan los datos ya versionados en `data/processed/`.

# %%
if REPREPARAR:
    from src.datos.preparar import preparar
    resumen = preparar(cfg)
else:
    resumen = load_json("data/processed/resumen_datos.json")
print(f"Registros crudos: {resumen['n_registros_crudo']:,} · variables: {resumen['n_variables_crudo']} · "
      f"periodo {resumen['periodo']['min']} a {resumen['periodo']['max']} · descarga {resumen['fecha_descarga']}")
pd.DataFrame({"créditos": resumen["n_por_split"], "tasa de malos": resumen["tasa_malos_por_split"]})

# %%
pd.Series(resumen["conteo_por_loan_status"], name="créditos").to_frame()

# %% [markdown]
# ## 2. Análisis descriptivo, hipótesis y datos después del preprocesamiento

# %%
from src.datos.experimento_eda import correr_eda

eda = correr_eda(cfg)
pd.DataFrame([{"id": h["id"], "hipótesis": h["enunciado"], "lectura": h["veredicto_eda"]} for h in eda["hipotesis"]])

# %%
pd.DataFrame(eda["post"]["psi_vs_train"]).sort_values("test", ascending=False).round(3)

# %% [markdown]
# ## 3. Figuras (las mismas del blog)

# %%
from src.viz.figs_datos import generar

for ruta in generar():
    display(Image(filename=str(ruta)))
