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
# # Sustentación — guion de los 10 minutos
#
# | Momento | Duración | Qué hacemos aquí |
# |---|---|---|
# | Demostración | 2 min | Celda 1: corre una demo con un solo comando (el mismo que se escribiría en la terminal) |
# | Modificación en vivo | 3 min | Celda 2: el profesor pide un cambio, **escribimos la predicción antes** y lo corremos |
# | Preguntas | 5 min | Sección 3: dónde está cada cosa para responder (cualquiera de los cinco) |
#
# Capa delgada sobre `src/` (núcleo §5.8): las demos son comandos de `run.py` y escriben en carpetas temporales, así que
# nada de lo que se haga aquí cambia los resultados versionados. **Se edita este `.py`; el `.ipynb` lo genera
# `python run.py notebooks`.**

# %% tags=["parameters"]
CONFIG = "configs/sustentacion.yaml"
SET = {}            # p. ej. {"demo": "pd"} para mostrar otra demo
CAMBIO = {}         # lo que pide el profesor, con las claves de --set, p. ej. {"mlp.dropout": 0.5}
PREDICCION = ""     # se escribe ANTES de correr la celda 2: qué esperamos que pase y por qué

# %%
import pandas as pd

from src.common.notebooks import config_notebook, correr_demo, estado_demos

cfg = config_notebook(CONFIG, SET)
pd.DataFrame(estado_demos(cfg))

# %% [markdown]
# ## 1. Demostración (2 min)
# Un solo comando, con la configuración resuelta impresa al inicio y el tiempo al final.

# %%
demo = correr_demo(cfg, cfg["demo"])
print(f"Estado: {demo['estado']} · {demo.get('segundos', '–')} s")

# %% [markdown]
# ## 2. Modificación en vivo (3 min)
# 1. El profesor pide el cambio → lo escribimos en `CAMBIO` (celda de parámetros) con las mismas claves de `--set`.
# 2. **Antes de correr**, escribimos en `PREDICCION` qué va a pasar y por qué. La celda se niega a correr sin predicción.
# 3. Corremos y comparamos con lo que dijimos.

# %%
if not CAMBIO:
    print("Sin cambio pedido todavía: escribe CAMBIO y PREDICCION en la celda de parámetros y vuelve a correr.")
else:
    assert PREDICCION.strip(), "Primero escribe la predicción (qué esperamos y por qué); después se corre el cambio."
    print(f"Predicción: {PREDICCION}\n")
    modificada = correr_demo(cfg, cfg["demo"], CAMBIO)
    print(f"Estado: {modificada['estado']} · {modificada.get('segundos', '–')} s")

# %% [markdown]
# ### Predicciones de referencia
# Verificadas corriendo el cambio (24-oct-2026, demo `eda`, Parte A). Las de B, C y D las agrega el `tutor-sustentacion`
# (tarea T5.4) cuando sus demos estén conectadas; una predicción sin correr se marca "no verificada".
#
# | Cambio (`CAMBIO`) | Predicción | Por qué | ¿Se cumplió? |
# |---|---|---|---|
# | `{"preprocesamiento.percentiles_recorte": [5.0, 95.0]}` | Cada cola de `dti` pasa de ~1 % a ~5 % recortado; desaparece el grupo aislado de saldo rotativo 0 (asimetría de −3,1 a −0,8); el PSI casi no cambia | El percentil 5 del saldo es mayor que 0, así que los ceros se suben al recorte; el PSI usa deciles de train y recortar no cambia el orden | Sí |
# | `{"particion.train_hasta": "2011-12-31", "particion.val_hasta": "2012-12-31"}` | Train baja de 90 347 a ~40 500 créditos y prueba sube (2013 + 2014); la tasa de malos de prueba queda cerca de 25 % | Se corre un año cada corte; los años recientes tienen más malos entre los resueltos (sesgo de selección, blog 2.4) | Sí: 40 534 / 49 813 / 147 348; 15,8 % / 16,7 % / 24,6 % |
# | `{"eda.umbral_monotonia": 0.7}` | H6 (historial) y H10 (monto) pasan de "no es monótona" a "la apoya" | Su \|Spearman\| es 0,73 y 0,76: superan 0,7 pero no 0,9 | Sí |

# %% [markdown]
# ## 3. Preguntas (5 min): dónde está cada cosa
#
# | Tema | Dónde | Quién lo hizo |
# |---|---|---|
# | Enunciado y reglas | `docs/spec/SPEC.md` (Anexo = enunciado literal) | todos |
# | Datos, objetivo, variables, partición, análisis descriptivo | `docs/spec/partes/A-datos.md`, `src/datos/`, blog 1–3, notebook 01 | Daniel |
# | PD: logística, MLP, búsqueda, métricas | `docs/spec/partes/B-pd.md`, `src/pd/`, blog 4–6, notebook 02 | Andrés y Juan José |
# | Scorecard, SHAP, hipótesis | `docs/spec/partes/C-scorecard.md`, `src/scorecard/`, blog 7, notebook 03 | Andrés y Juan José |
# | LGD | `docs/spec/partes/D-lgd.md`, `src/lgd/`, blog 8, notebook 04 | Camila y José |
# | Pérdida esperada, app, video | `docs/spec/partes/E-producto.md`, `src/el/`, `app/`, `video/`, blog 9 | Daniel y Camila |
# | Cacería de la alucinación | `alucinaciones/registro.md`, blog 12 | todos |
# | Fichas de estudio y preguntas probables | `docs/sustentacion/` (tarea T5.4) | todos |
