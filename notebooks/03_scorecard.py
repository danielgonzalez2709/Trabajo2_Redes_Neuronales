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
# # Parte C — Scorecard: probabilidad → puntaje, puntos por variable, SHAP e hipótesis
#
# Capa delgada sobre `src/` (núcleo §5.8): el notebook no calcula nada por su cuenta, solo llama funciones del
# repositorio y muestra sus resultados o los resultados guardados. **Se edita este `.py`; el `.ipynb` lo genera
# `python run.py notebooks`.**
#
# **Convención (C §3.1):** $y = 1$ malo, $p = \text{PD} = P(y = 1)$; odds de **buen** pagador $o = (1 - p)/p$;
# $\text{score} = \text{offset} - \text{factor}\cdot\text{logit}(p)$ con $\text{factor} = \text{PDO}/\ln 2$ y
# $\text{offset} = \text{base} - \text{factor}\cdot\ln(\text{odds\_base})$: **mayor puntaje = menor riesgo**.
#
# **Qué lee:** la config `configs/scorecard/base.yaml`; la logística de la Parte B (`models/pd_logistica.joblib`), el
# preprocesador y los datos de la Parte A (`models/preprocesador_pd.json`, `data/processed/loans_pd.parquet`); los
# resultados guardados de C4 y C6 (`results/data/scorecard/shap_mlp.json`, `scorecard_mlp.json`, `hipotesis.json`) y
# las figuras del blog (`blog/docs/figures/sc-*.png`). **No escribe** en `data/`, `models/`, `results/`, `configs/` ni
# `blog/`: la única figura que dibuja va a una carpeta temporal.
#
# **Qué responde a `SET`:** las secciones 1 a 4 se calculan **en vivo** con la config elegida; las secciones 5 y 6
# leen resultados guardados con la config por defecto (base 600, odds 50:1, PDO 20) y **no cambian** con `SET`.
#
# Para la modificación en vivo, cambia `SET` (mismas claves que `--set`) y corre todo. Ejemplos:
#
# - `SET = {"scorecard.pdo": 40}`: los puntos de cada variable y la separación entre bandas se **duplican** alrededor
#   del ancla (los puntajes se alejan del valor base); el orden de los clientes no cambia, así que cada banda tiene los
#   mismos créditos.
# - `SET = {"scorecard.base": 500}`: todos los puntajes (puntos base, cortes de las bandas) **bajan 100**; los puntos
#   por variable no cambian.
# - `SET = {"scorecard.odds_base": 20}`: se **desplaza el offset** ($\text{factor}\cdot\ln(50/20)$ más): todos los
#   puntajes suben esa cantidad y los puntos por variable no cambian.

# %% [markdown]
# ## 1. Parámetros y configuración resuelta
#
# `config_notebook` aplica `SET` con el mismo cargador del CLI (una clave inexistente es un error) e imprime la config.

# %% tags=["parameters"]
CONFIG = "configs/scorecard/base.yaml"
SET = {}

# %%
import tempfile
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from src.common.notebooks import config_notebook

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

cfg = config_notebook(CONFIG, SET)
carpeta = Path(tempfile.mkdtemp(prefix="scorecard_"))  # solo para la figura en vivo de la sección 2; no se imprime

# %% [markdown]
# ## 2. Probabilidad → puntaje (en vivo)
#
# `parametros_scorecard` (C1) calcula `factor` y `offset` con la config elegida; `score_desde_pd` convierte unas PD de
# ejemplo en puntaje. La segunda fila es la **PD del ancla** (odds `odds_base`:1), que `pd_desde_score` obtiene del
# puntaje `base`: por construcción, su puntaje es exactamente `base`.

# %%
from src.scorecard.scorecard import parametros_scorecard, pd_desde_score, score_desde_pd

params = parametros_scorecard(cfg["scorecard"])
print(f"base {params['base']:g} · odds_base {params['odds_base']:g}:1 · PDO {params['pdo']:g} · "
      f"factor {params['factor']:.4f} · offset {params['offset']:.4f}")

PD_EJEMPLO = [0.01, pd_desde_score(params["base"], params), 0.1, 0.5]  # solo para mostrar
pd.DataFrame({"PD": PD_EJEMPLO, "puntaje": score_desde_pd(PD_EJEMPLO, params)}).round(4)

# %% [markdown]
# La figura `sc-transformacion` dibujada con estos parámetros en la carpeta temporal. La **segunda curva es siempre
# PDO 40** (fija en `src/viz/figs_scorecard.py`, con la misma base y odds); si `SET` pide PDO 40, las dos curvas
# coinciden.

# %%
from src.viz.figs_scorecard import datos_transformacion, figura_transformacion
from src.viz.style import aplicar_estilo

aplicar_estilo()
display(Image(filename=str(figura_transformacion(datos_transformacion(params), carpeta))))

# %% [markdown]
# ## 3. Scorecard de la logística (en vivo)
#
# `preparar_splits` carga los datos, el preprocesador (ajustado solo con train por la Parte A) y la logística de B3;
# `construir_scorecard` (C3) pasa el logit a puntaje con la config elegida, reparte los puntos por variable y comprueba
# que puntos base + Σ puntos = puntaje en los tres splits. No escribe nada. El comando equivalente en la terminal (ese
# sí escribe `scorecard.json` en `results/data/scorecard/`, salvo que se añada `--set salida=<otra carpeta>`):
#
# `python run.py scorecard --config configs/scorecard/base.yaml --set scorecard.pdo=40`

# %%
from src.scorecard.puntos import construir_scorecard, preparar_splits

preparado = preparar_splits(cfg)
sc = construir_scorecard(cfg, preparado)
print(f"Puntos base: {sc['puntos_base']:.4f} (factor {sc['factor']:.4f}, offset {sc['offset']:.4f})")
print(f"Error máximo de aditividad: {sc['aditividad']['error_max']:.3e} (tolerancia {sc['aditividad']['tolerancia']:.0e})")

# %% [markdown]
# Puntos por valor crudo de dos numéricas (percentiles de **train**) y por categoría del plazo. En una numérica los
# puntos valen 0 en la media transformada de train y son planos fuera del recorte $[p_1, p_{99}]$.

# %%
entradas = {e["nombre"]: e for e in sc["puntos_por_variable"]}
for nombre in ("annual_inc", "revol_util"):
    print(f"{nombre} — {entradas[nombre]['etiqueta_es']}")
    display(pd.DataFrame(entradas[nombre]["tabla"]).round(2))
print(f"term — {entradas['term']['etiqueta_es']}")
pd.DataFrame(entradas["term"]["categorias"]).round(4)

# %% [markdown]
# Rango de puntos de las 15 variables (numéricas: $|\text{puntos}(p_{99}) - \text{puntos}(p_1)|$; categóricas: entre las
# categorías con al menos `fraccion_min_categoria` de train). Los rangos de numéricas y categóricas **no son
# comparables** entre sí: miden cosas distintas (C §3.2).

# %%
(pd.DataFrame(sc["puntos_por_variable"])[["nombre", "etiqueta_es", "tipo", "rango_puntos"]]
 .sort_values("rango_puntos", ascending=False).round(2).reset_index(drop=True))

# %% [markdown]
# Bandas de test (deciles del puntaje de **train**; banda 0 = menor puntaje = más riesgo): rango de puntaje
# $[\text{desde}, \text{hasta})$, número de créditos, PD media predicha y tasa de malos observada.

# %%
pd.DataFrame(sc["distribucion_scores"]["test"]["bandas"]).round(4)

# %% [markdown]
# ## 4. Un cliente: puntos base + puntos por variable = su puntaje (en vivo)
#
# El crédito de test es el **primero de los explicados por SHAP** (`explicados.ids[0]` de `shap_mlp.json`), el mismo
# de la sección 5. `puntos_cliente` (C8) toma su fila del preparado, reparte sus puntos por variable con la scorecard de
# la sección 3 y lee sus valores crudos del parquet de la Parte A, antes del preprocesador: las categóricas como vienen
# en el crudo (el preprocesador les quita los espacios y un vacío, NaN, pasa a la categoría `DESCONOCIDO`) y un vacío
# numérico se imputa con la mediana de train.

# %%
from src.common.results import load_json
from src.scorecard.puntos import puntos_cliente

shap = load_json("results/data/scorecard/shap_mlp.json")
ID_CLIENTE = shap["explicados"]["ids"][0]

cliente = puntos_cliente(preparado, params, ID_CLIENTE)
print(f"Crédito {cliente['id']} ({cliente['split']})")
print(f"Puntos base: {cliente['puntos_base']:.2f} · Σ puntos por variable: {cliente['variables']['puntos'].sum():.2f} · "
      f"puntaje: {cliente['score']:.2f} (error de aditividad {cliente['error_aditividad']:.1e})")
cliente["variables"].round(2)

# %% [markdown]
# ## 5. SHAP del MLP (resultados guardados; no cambia con `SET`)
#
# Valores de Shapley intervencionales del **logit** del MLP de PD (C4): $\varphi_0 + \sum_j \varphi_j = \text{logit}$
# de cada crédito explicado. No se recalcula aquí: es la parte lenta de C4 (sus tiempos, impresos abajo, están en
# `tiempos_s` del JSON). Para regenerarlo (escribe en
# `results/data/scorecard/`; con `--set salida=<otra carpeta>` no pisa lo versionado):
#
# `python run.py scorecard --config configs/scorecard/mlp.yaml`

# %%
sc_mlp = load_json("results/data/scorecard/scorecard_mlp.json")
print(f"φ₀ (logit medio del fondo): {shap['phi0']:.4f}")
print(f"Eficiencia (φ₀ + Σφ = logit): error máx {shap['eficiencia']['error_max']:.1e} "
      f"(tolerancia {shap['eficiencia']['tolerancia']:.0e})")
v = shap["verificacion_logistica"]
print(f"Verificación con la logística (forma cerrada): error máx {v['error_max']:.1e} en {v['n']} créditos")
print(f"Estabilidad entre semillas (Spearman de la importancia): {shap['estabilidad']['spearman_importancia']:.4f}")
print("Tiempos de C4 (s):", {k: round(s, 1) for k, s in shap["tiempos_s"].items()})
print(f"Scorecard del MLP: puntos base {sc_mlp['puntos_base']:.4f}, aditividad {sc_mlp['aditividad']['error_max']:.1e}")

# %% [markdown]
# Importancia global (media de $|\varphi|$ en test) con su dirección (numéricas: $\rho$ de Spearman entre el valor y
# $\varphi$, positivo = más valor, más riesgo; NaN en las categóricas, que van en la tabla siguiente) y la caída media de AUC al
# permutar la variable en test, para los dos modelos.

# %%
from src.scorecard.puntos import direccion_shap
from src.viz.figs_scorecard import SALIDA, datos_importancia

imp = datos_importancia(shap)
direccion = direccion_shap(shap, imp["nombres"])
pd.DataFrame({
    "variable": imp["nombres"],
    "media_abs_phi": imp["media_abs_phi"],
    "spearman": direccion["spearman"],
    "caida_auc_mlp": imp["caida"]["mlp"]["media"],
    "caida_auc_logistica": imp["caida"]["logistica"]["media"],
}).round(4)

# %% [markdown]
# Dirección de las categóricas: $\bar\varphi$ (media de $\varphi$ en el logit del MLP) de cada categoría entre los
# créditos explicados de test y cuántos hay; positivo = esa categoría sube el logit de la PD (más riesgo).

# %%
direccion["categorias"].round(4)

# %% [markdown]
# $\varphi$ del mismo cliente de la sección 4 (logit del MLP = $\varphi_0 + \sum_j \varphi_j$).

# %%
print(f"Crédito {shap['explicados']['ids'][0]} · logit del MLP: {shap['explicados']['logit'][0]:.4f}")
pd.DataFrame({"jugador": shap["metodo"]["jugadores"], "phi": shap["explicados"]["phi"][0]}).round(4)

# %% [markdown]
# Figuras del blog (generadas con los resultados guardados).

# %%
for nombre in ("sc-importancia.png", "sc-dependencia-shap.png", "sc-puntos-variable.png"):
    display(Image(filename=str(SALIDA / nombre)))

# %% [markdown]
# ## 6. Cierre de las hipótesis (resultados guardados; no cambia con `SET`)
#
# Las 11 hipótesis de riesgo con sus tres señales (EDA de train, SHAP del MLP y test fuera de tiempo), la compuerta de
# la permutación (`relevante`) y el veredicto (C6). Es una asociación observada en los datos, no un efecto causal. Para
# regenerarlo: `python -m src.scorecard.hipotesis`.

# %%
hip = load_json("results/data/scorecard/hipotesis.json")
for veredicto, regla in hip["criterio"]["reglas"].items():
    print(f"{veredicto}: {regla}")
pd.DataFrame([{
    "id": h["id"], "enunciado": h["enunciado"], "eda": h["eda"]["senal"], "shap": h["shap"]["senal"],
    "test": h["test"]["senal"], "relevante": h["shap"]["relevante"], "veredicto": h["veredicto"],
    "cambia_con_sensibilidad": h["cambia_con_sensibilidad"],
} for h in hip["hipotesis"]])

# %%
display(Image(filename=str(SALIDA / "sc-bandas.png")))
