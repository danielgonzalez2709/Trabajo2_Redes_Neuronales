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
# # Parte E — Pérdida esperada E[L] por crédito y por banda de puntaje
#
# Capa delgada sobre `src/` (núcleo §5.8): el notebook no calcula nada por su cuenta, solo llama funciones del
# repositorio y muestra sus resultados. **Se edita este `.py`; el `.ipynb` lo genera `python run.py notebooks`.**
#
# Para cada crédito de test, **E[L] = PD · LGD · EAD** (Parte E §3.2; enunciado, Parte 2, punto 5):
#
# - **PD**: la probabilidad de incumplimiento que predice la Parte B (`results/data/pd/pred_test.parquet`, columna
#   `pred_<pd.modelo>`).
# - **LGD**: la pérdida dado el incumplimiento que predice la Parte D para **todo** el test, también para los créditos
#   que no incumplieron (`results/data/lgd/pred_pd_test.parquet`, columna `pred_<lgd.modelo>`).
# - **EAD**: el monto desembolsado en la originación (`funded_amnt`). Es **conservadora**: supone que el incumplimiento
#   ocurre antes de amortizar nada.
#
# Las bandas son cuantiles del puntaje de la scorecard de la Parte C (score = offset − factor · logit(PD); banda 1 =
# menor puntaje, más riesgo). El contraste con lo observado se hace **solo con los créditos resueltos** (castigados y
# pagados); los que siguen en mora van aparte con una pérdida provisional.
#
# La corrida escribe en una **carpeta temporal**: el notebook no escribe en `data/`, `models/`, `results/` ni `blog/`
# (`results/data/el/` lo consumen el blog y la app). Para la modificación en vivo, cambia `SET` (mismas claves que
# `--set`) y corre todo. Ejemplos:
#
# - otra escala de la scorecard: `SET = {"scorecard.pdo": 40}` (cambia los rangos de puntaje de cada banda, no qué
#   créditos caen en cada una)
# - la LGD de la regresión lineal de la Parte D: `SET = {"lgd.modelo": "lineal"}`
# - menos bandas: `SET = {"bandas.n": 5}`
#
# Los resultados versionados son otro comando: `python run.py el --config configs/el/base.yaml`.

# %% tags=["parameters"]
CONFIG = "configs/el/base.yaml"
SET = {}

# %%
import tempfile
import time
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from src.common.notebooks import config_notebook

pd.set_option("display.width", 160)

# La salida va siempre a una carpeta temporal (aunque se cambie en SET); se imprime un marcador en vez de su ruta local
carpeta = Path(tempfile.mkdtemp(prefix="el_"))
MARCADOR_SALIDA = "<carpeta temporal>"
cfg = config_notebook(CONFIG, {**SET, "salida": MARCADOR_SALIDA})
cfg["salida"] = str(carpeta)

# %% [markdown]
# ## 1. Corrida de la E[L] con esta configuración (carpeta temporal)
#
# `correr_el` lee la PD de B, la LGD de D y los montos de A, calcula la E[L] de cada crédito, su puntaje y su banda, y
# escribe `el_test.parquet` y `summary.json` en `salida`, que aquí es **siempre** la carpeta temporal (el marcador de la
# config impresa arriba), aunque se cambie en `SET`.

# %%
from src.el.experimento import correr_el, texto_resumen

inicio = time.perf_counter()
resumen = correr_el(cfg)
print(f"Tiempo de la corrida: {time.perf_counter() - inicio:.1f} s")
scorecard = resumen["scorecard"]
print(f"PD: {resumen['modelo_pd']} · LGD: {resumen['modelo_lgd']} · EAD: {resumen['ead']}")
print(f"Scorecard: base {scorecard['base']:g}, odds {scorecard['odds_base']:g}:1, PDO {scorecard['pdo']:g} "
      f"(factor {scorecard['factor']:.4f}, offset {scorecard['offset']:.4f})")
print(f"Bandas: {resumen['bandas']} · pedidas {resumen['n_bandas_pedidas']}, obtenidas {resumen['n_bandas_obtenidas']}")

# %%
print(texto_resumen(resumen))
print("Fracción del monto que seguía pendiente al castigo (media de los castigados): "
      f"{resumen['fraccion_principal_pendiente_media_castigados']:.3f}")

# %% [markdown]
# ## 2. Totales y por banda de puntaje (del resumen de esta corrida)
#
# Montos en USD; tasa de malos, PD y LGD como fracciones. Lo que no aplica a una fila (la pérdida realizada de los
# créditos en mora, la provisional de los resueltos) aparece como NaN.

# %%
from src.el.tablas import tabla_bandas, tabla_totales

tabla_totales(resumen).round(3)

# %%
tabla_bandas(resumen).round(3)

# %% [markdown]
# ## 3. Figuras (las del blog, dibujadas con esta corrida en la carpeta temporal)

# %%
from src.viz.figs_el import generar

for ruta in generar(destino=carpeta / "figuras", resumen=resumen):
    display(Image(filename=str(ruta)))

# %% [markdown]
# ## 4. Por crédito (`el_test.parquet` de esta corrida)
#
# Unas filas de cada estado. La **pérdida realizada** (`perdida_observada`) solo existe en los créditos resueltos: en
# los castigados es el principal pendiente menos las recuperaciones netas, y en los pagados es 0; en los que siguen
# **en mora** es NaN, porque la cobranza no terminó. Para ellos `perdida_provisional` aplica la misma cuenta sin
# recuperaciones, es decir, **supone LGD = 1** sobre el principal pendiente: no es una pérdida observada. Por eso el
# contraste E[L] frente a lo realizado se hace solo con los resueltos.

# %%
FILAS_POR_ESTADO = 3  # solo para mostrar
COLUMNAS = ["id", "target", "estado", "pd", "score", "banda", "lgd", "ead", "el", "perdida_observada",
            "perdida_provisional"]

el_test = pd.read_parquet(carpeta / "el_test.parquet")
el_test.groupby("estado").head(FILAS_POR_ESTADO)[COLUMNAS].round(3)

# %% [markdown]
# ## 5. Qué parte de la razón pone cada factor (blog, 9.5)
# La razón E[L] / realizada de los resueltos es el producto de un factor PD, uno LGD y uno EAD
# (`src/el/tablas.py::descomposicion_razon`); cada factor vale 1 si su pieza es perfecta.

# %%
from src.el.tablas import descomposicion_razon

descomposicion_razon(el_test).round(3)

# %% [markdown]
# ## 6. Cómo se lee
#
# - **Razón E[L] / realizada (resueltos):** mayor que 1, la E[L] sobreestima la pérdida que de verdad ocurrió; menor
#   que 1, la subestima. Se lee en total (sección 1) y por banda (tabla y primera figura).
# - **Orden por banda:** si la PD ordena bien el riesgo, la E[L] y la pérdida realizada se concentran en las bandas de
#   menor puntaje y las barras de las dos series bajan juntas al subir de banda.
# - **Calibración de la PD:** la segunda figura compara la PD media predicha con la tasa de malos observada en cada
#   banda; la distancia entre las dos curvas es el error de nivel de la PD, que se traslada tal cual a la E[L].
# - **La EAD de originación es conservadora:** la E[L] multiplica por todo `funded_amnt`, mientras que la pérdida
#   realizada parte del principal que seguía pendiente (la fracción pendiente media de los castigados está impresa en
#   la sección 1). Por eso una razón cercana a 1 no prueba que la PD y la LGD estén bien por separado: una EAD alta
#   puede compensar una PD o una LGD bajas. Hay que leer la razón junto con la segunda figura.
# - **Créditos en mora:** su pérdida provisional no entra en el contraste; supone LGD = 1 y no es una pérdida observada.
