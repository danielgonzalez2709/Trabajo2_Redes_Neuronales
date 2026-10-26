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
# # Parte D — LGD: variable objetivo, preprocesamiento, lineal frente a MLP y distribuciones
#
# Capa delgada sobre `src/` (núcleo §5.8): el notebook no calcula nada por su cuenta, solo llama funciones del
# repositorio y muestra sus resultados. **Se edita este `.py`; el `.ipynb` lo genera `python run.py notebooks`.**
#
# Lee `results/data/lgd/resumen_objetivo.json`, `descriptivo.json` y `summary.json`, los datos procesados de la LGD
# (`data/processed/loans_lgd.parquet`, `models/preprocesador_lgd.json`, tal cual) y las figuras ya versionadas de
# `blog/docs/figures/lgd-*.png`. La demo en vivo entrena en una **carpeta temporal** (resultados y modelos): el notebook
# no escribe en `data/processed/`, `models/`, `results/`, `configs/lgd/mlp_final.yaml` ni `blog/`.
#
# Para la modificación en vivo, cambia `SET` (mismas claves que `--set`) y corre todo. Ejemplos:
#
# - otra arquitectura: `SET = {"mlp.capas": [128, 64, 32]}`
# - más regularización: `SET = {"mlp.dropout": 0.3}`
# - otra pérdida: `SET = {"mlp.perdida": "mse"}`
# - más semillas o más datos: `SET = {"semillas.n": 5, "datos.muestra": 5000}`
#
# Los resultados reportados (30 semillas, configuración de la búsqueda) son otro comando:
# `python run.py lgd --config configs/lgd/base.yaml`.

# %% tags=["parameters"]
CONFIG = "configs/lgd/demo.yaml"
SET = {}

# %%
import tempfile
import time
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from src.common.config import load_config
from src.common.notebooks import config_notebook
from src.common.results import load_json
from src.viz.figs_lgd import SALIDA as FIGURAS_BLOG

CONFIG_OFICIAL = "configs/lgd/base.yaml"  # solo para leer dónde están los resultados reportados (sección 5)
pd.set_option("display.width", 160)

cfg = config_notebook(CONFIG, SET)

# %% [markdown]
# ## 1. Variable objetivo observada (resultados versionados de D1 y D1b)
#
# LGD = 1 − (`recoveries` − `collection_recovery_fee`) / EAD, con EAD = `funded_amnt` − `total_rec_prncp`, recortada a
# [0, 1], sobre los créditos castigados (Parte D §3.2). Población, exclusiones, masa en 0 y en 1 y casos recortados:

# %%
from src.lgd.tablas import tabla_poblacion, tabla_por_anio, tabla_por_plazo, tabla_por_split

resumen_objetivo = load_json(cfg["datos"]["salida"]["resumen"])
descriptivo = load_json(cfg["descriptivo"]["salida"])
obs = descriptivo["observado"]
tabla_poblacion(resumen_objetivo)

# %%
display(Image(filename=str(FIGURAS_BLOG / "lgd-histograma.png")))

# %% [markdown]
# ### LGD por partición y por año de originación

# %%
tabla_por_split(obs).round(3)

# %%
tabla_por_anio(obs).round(3)

# %%
display(Image(filename=str(FIGURAS_BLOG / "lgd-por-anio.png")))

# %% [markdown]
# La distribución de la LGD cambia entre particiones (la fracción con LGD = 1 crece en validación y en prueba). Se
# acepta y se documenta como limitación (Parte D §3.2): en 2013–2014 es **consistente con censura** (recuperaciones
# todavía abiertas en los castigos recientes), no una causa probada; tampoco se descartan composición ni cambios de
# cobranza.
#
# ### EAD y LGD por plazo

# %%
tabla_por_plazo(obs).round(3)

# %%
display(Image(filename=str(FIGURAS_BLOG / "lgd-ead-plazo.png")))

# %% [markdown]
# ## 2. Después del preprocesamiento (funciones de la Parte A sobre la población de la LGD)
#
# Efecto de las transformaciones (fracción de filas imputadas o recortadas en las numéricas; desconocidas o llevadas a
# OTRA en las categóricas), estabilidad entre particiones (PSI frente a train) y relación de cada variable con la LGD.

# %%
from src.lgd.tablas import tabla_efecto_transformaciones, tabla_psi, tabla_relacion_objetivo

pre = descriptivo["preprocesado"]
tabla_efecto_transformaciones(pre).round(3)

# %%
tabla_psi(pre).round(3)

# %%
tabla_relacion_objetivo(pre).round(3)

# %%
for clave, nota in pre["notas_objetivo_continuo"].items():
    print(f"- {clave}: {nota}")

# %% [markdown]
# Dos cautelas que trae el propio `descriptivo.json` (notas de arriba): con `n_grupos == 2`, "monótona" es
# siempre verdadera y **no es evidencia** de relación; y las categóricas no tienen relación medida con la LGD (el
# plazo se ve en la tabla por plazo de la sección 1).

# %% [markdown]
# ## 3. Demo en vivo: lineal frente a MLP (config demo, carpeta temporal)
#
# Muestra, semillas y épocas reducidas para que corra en < 2 min: **no son los resultados reportados** (sección 5).
# La salida y los modelos **siempre van a la carpeta temporal**, aunque se cambien `salida` o `modelos.*` en `SET`.

# %%
from src.lgd.experimento import entrenar_semillas
from src.lgd.tablas import redirigir_salidas, tabla_metricas_compacta

carpeta_demo = Path(tempfile.mkdtemp(prefix="demo_lgd_"))
inicio = time.perf_counter()
resumen_demo = entrenar_semillas(redirigir_salidas(cfg, carpeta_demo))
print(f"Tiempo de la demo: {time.perf_counter() - inicio:.1f} s · créditos: {resumen_demo['mlp']['n']} · "
      f"configuración del MLP: {cfg['busqueda']['origen']}")
tabla_metricas_compacta(resumen_demo)

# %% [markdown]
# ## 4. Distribuciones de la demo (las figuras del blog, dibujadas con esta corrida en la carpeta temporal)

# %%
from src.viz.figs_lgd import generar, insumos_de_corrida

figuras_demo = {r.name: r for r in generar(destino=carpeta_demo / "figuras",
                                           datos=insumos_de_corrida(carpeta_demo, obs))}
for nombre in ("lgd-distribucion-predicha.png", "lgd-qq.png", "lgd-error-por-banda.png"):
    display(Image(filename=str(figuras_demo[nombre])))

# %% [markdown]
# ## 5. Resultados oficiales (leídos de `summary.json` de la config oficial, sin entrenar)

# %%
from src.lgd.tablas import comparacion, describir_mlp, tabla_referencias

resumen = load_json(Path(load_config(CONFIG_OFICIAL)["salida"]) / "summary.json")
mlp = resumen["mlp"]
print(describir_mlp(mlp))
tabla_metricas_compacta(resumen)

# %%
for metrica in ("mae_test", "rmse_test"):
    c = comparacion(resumen, metrica)
    print(f"{metrica} del MLP (media de {mlp['metricas'][metrica]['n']} semillas): "
          f"{c['valores']['mlp']:.4f} ± {c['desv_mlp']:.4f}")
    for otro, diferencia in c["mlp_menos"].items():
        print(f"  MLP − {otro} ({c['valores'][otro]:.4f}): {diferencia:+.5f}")
    print(f"  menor {metrica}: {c['mejor']}")

# %%
tabla_referencias(resumen).round(4)

# %% [markdown]
# Lectura: las diferencias de arriba (MLP menos cada alternativa) dicen si la red aportó algo frente a la regresión
# lineal y frente a predecir una constante (media o mediana de la LGD de train, `summary.json` → `referencias`); un
# signo negativo es a favor del MLP. Dos hechos para leerlas, sin afirmar causas:
#
# - Entre las constantes, la **mediana** es la que minimiza el error absoluto esperado (por eso es la referencia natural
#   para el MAE), mientras que el MLP se entrena con la pérdida impresa arriba; con `bce_fraccional` su minimizador
#   poblacional es la **media** condicional (`src/lgd/perdidas.py`).
# - La LGD de test está desplazada hacia 1 frente a train (tabla por partición de la sección 1), lo que favorece a una
#   constante alta (los valores de las dos constantes están en la tabla de arriba).

# %%
for nombre in ("lgd-semillas.png", "lgd-error-por-banda.png", "lgd-distribucion-predicha.png"):
    display(Image(filename=str(FIGURAS_BLOG / nombre)))
