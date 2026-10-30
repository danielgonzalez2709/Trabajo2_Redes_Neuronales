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
# # Parte B — PD: regresión logística frente a un MLP
#
# Capa delgada sobre `src/` (núcleo §5.8): el notebook no calcula nada por su cuenta, solo llama funciones del
# repositorio y muestra lo que devuelven. **Se edita este `.py`; el `.ipynb` lo genera `python run.py notebooks`.**
#
# - **Sección 1 (demo en vivo):** entrena la logística y el MLP con `configs/pd/demo.yaml` en una **carpeta temporal**.
#   Es lo único que cambia con `SET`.
# - **Secciones 2–4 (repaso de la Parte B):** solo leen lo versionado: `data/processed/resumen_datos.json`,
#   `results/data/eda/post_preprocesamiento.json`, `configs/features.yaml`, `results/data/pd/` y las figuras de
#   `blog/docs/figures/`.
# - **Sección 5 (de la PD a la decisión):** solo lee `results/data/pd_umbral/` (`umbral.json`) y dos figuras del blog.
#
# El notebook **nunca escribe** en `data/processed/`, `models/`, `results/` (incluida `results/data/pd_umbral/`),
# `configs/pd/mlp_final.yaml` ni `blog/`.
#
# **Cómo usar los moldes:** copia al `SET` de la sección 1.0 las claves que quieras cambiar (con su nuevo valor) y corre
# todo (*Run All*). Una clave mal escrita es un error, igual que en `--set`.

# %% [markdown]
# ## Moldes de los parámetros de la demo
#
# Todas las claves de `configs/pd/demo.yaml` que se pueden cambiar en vivo, con su valor actual. `salida`, `modelos` y
# `busqueda` no van: el notebook manda la salida y los modelos a la carpeta temporal, y la demo no viene de la
# búsqueda.
#
# ### Claves comunes
#
# ```python
# {
#     "base_seed": 0,  # semilla del muestreo de datos.muestra y base de las semillas del MLP (base_seed + k)
#     "datos.muestra": 50000,  # filas repartidas entre train, val y test en proporción a su tamaño; None = todas
#     "semillas.n": 3,  # número de semillas del MLP (base_seed + 0, …, n − 1)
# }
# ```
#
# ### Logística
#
# ```python
# {
#     "logistica.rejilla_C": [0.001, 0.0031623, 0.01],  # valores de C que se prueban (gana el de menor log-loss de val)
#     "logistica.max_iter": 1000,  # iteraciones máximas de lbfgs
#     "logistica.tol": 1e-06,  # tolerancia de lbfgs
# }
# ```
#
# ### MLP
#
# ```python
# {
#     "mlp.capas": [32, 16, 32],  # neuronas de cada capa oculta (una entrada por capa)
#     "mlp.activacion": "relu",  # activación de las capas ocultas: "relu", "leaky_relu" o "tanh"
#     "mlp.dropout": 0.09416553239584849,  # probabilidad de dropout
#     "mlp.l2": 4.451065531349953e-05,  # λ de la penalización (λ/2)‖W‖² (weight_decay; sin sesgos)
#     "mlp.optimizador": "adam",  # optimizador: "adam" o "sgd"
#     "mlp.lr": 0.004965184800549709,  # tasa de aprendizaje
#     "mlp.momento": 0.9,  # momento de sgd (adam no lo usa)
#     "mlp.batch": 512,  # tamaño del minilote
#     "mlp.max_epocas": 20,  # máximo de épocas
#     "mlp.paciencia": 5,  # épocas sin mejorar la pérdida de val antes de parar
#     "mlp.device": "cpu",  # dispositivo de entrenamiento
#     "mlp.pos_weight": 1.0,  # peso de la clase positiva en la BCE (debe ser 1, B-pd §3.2)
# }
# ```
#
# ### Comando de terminal equivalente
#
# La primera línea es la demo; la segunda, un ejemplo de modificación. Con otra muestra hay que cambiar también
# `salida` (si no, el pre-chequeo falla porque `pred_test.parquet` tiene otros ids, B-pd §3.5) y, para no pisar los
# modelos de la demo, `modelos`; `results/data/*_demo/` y `models/*_demo*` están ignoradas por git.
#
# ```bash
# python run.py pd --config configs/pd/demo.yaml
# python run.py pd --config configs/pd/demo.yaml --set datos.muestra=20000 --set 'mlp.capas=[64, 32]' --set salida=results/data/pd2_demo --set modelos=models/pd2_demo
# ```

# %% [markdown]
# ## 1. Demo en vivo
#
# El mismo código que la corrida reportada, con menos presupuesto (muestra, semillas, rejilla y épocas reducidas):
# sirve para ver el flujo completo en menos de dos minutos, **no** para reportar cifras (esas están en la sección 4).
# Desarrollo completo en el [blog 06](../blog/secciones/06-resultados-pd.md).
#
# ### 1.0 Parámetros
#
# `CONFIG` y `SET` (mismas claves que `--set`; copia las de los moldes de arriba).

# %% tags=["parameters"]
CONFIG = "configs/pd/demo.yaml"  # config de la demo (B6)
SET = {}  # cambios en vivo: {"clave.con.punto": valor}

# %% [markdown]
# ### 1.1 Configuración resuelta
#
# Se carga con el mismo cargador del CLI (una clave inexistente es un error) y se imprime: es la que se usa para
# predecir qué cambiará antes de correr.

# %%
import tempfile  # carpeta temporal de la demo
import time  # duración de la demo
from pathlib import Path  # rutas

import pandas as pd  # tablas
from IPython.display import Image, display  # figuras PNG

from src.common.config import load_config  # cargador de YAML (features.yaml)
from src.common.notebooks import config_notebook  # config del notebook = config del CLI
from src.common.results import load_json  # lectura de los JSON versionados
from src.pd import tablas  # tablas y textos de este notebook (B9)
from src.pd import tablas_umbral  # tablas de la sección 5 (B11)
from src.pd.experimento import correr_pd  # run.py pd: logística + MLP (B5)
from src.viz.figs_pd import ENTRADA, generar, insumos_de_corrida  # figuras de la PD (B7)
from src.viz.figs_pd import SALIDA as FIGURAS_BLOG  # blog/docs/figures (solo lectura)

RESUMEN_DATOS = "data/processed/resumen_datos.json"  # resumen de la Parte A
POST_PREPROCESAMIENTO = "results/data/eda/post_preprocesamiento.json"  # descriptivo después del preprocesamiento
FEATURES = "configs/features.yaml"  # las 15 variables de originación
UMBRAL = "results/data/pd_umbral/umbral.json"  # resultados de B11 (sección 5)
FIGURAS_DEMO = ("pd-roc.png", "pd-ks.png", "pd-calibracion.png", "pd-aprendizaje.png", "pd-semillas.png",
                "pd-busqueda-logistica.png")  # la demo no tiene hpo_ensayos.json: no hay pd-busqueda-mlp
pd.set_option("display.width", 160)  # tablas anchas
pd.set_option("display.max_colwidth", 200)  # citas del diccionario completas

cfg = config_notebook(CONFIG, SET)  # config resuelta e impresa

# %% [markdown]
# ### 1.2 Entrenamiento en una carpeta temporal
#
# `correr_pd` hace el pre-chequeo, la búsqueda de C de la logística y las semillas del MLP. La salida y los modelos
# van **siempre** a la carpeta temporal, aunque `SET` cambie `salida` o `modelos`. Debajo: tiempo, filas por
# partición, el MLP de esta corrida y la tabla logística frente a MLP (val y test; media y desviación entre semillas).

# %%
carpeta = Path(tempfile.mkdtemp(prefix="demo_pd_"))  # carpeta temporal (no se imprime)
inicio = time.perf_counter()  # reloj
demo = correr_pd(tablas.redirigir_salidas(cfg, carpeta))  # logística y MLP, salidas en la carpeta temporal
segundos = time.perf_counter() - inicio  # duración
insumos_demo = insumos_de_corrida(carpeta / tablas.SUBCARPETA_RESULTADOS)  # JSON y predicciones de esta corrida
print(f"Tiempo de la demo: {segundos:.1f} s")  # tiempo
print("Filas por partición:", tablas.filas_por_particion(insumos_demo["semillas"]))  # train, val, test
print(tablas.describir_mlp(demo["mlp"]))  # arquitectura, parámetros y semilla mediana
tablas.tabla_metricas(demo).T  # una columna por modelo

# %% [markdown]
# ### 1.3 Figuras de esta corrida
#
# Las mismas funciones que dibujan las figuras del blog, alimentadas con esta corrida y escritas en la carpeta
# temporal: ROC, KS, calibración, aprendizaje de la semilla mediana, semillas y búsqueda de C.

# %%
figuras_demo = {r.name: r for r in generar(destino=carpeta / "figuras", datos=insumos_demo)}  # nombre → ruta
for nombre in FIGURAS_DEMO:  # en orden
    display(Image(filename=str(figuras_demo[nombre])))  # figura

# %% [markdown]
# ## 2. Datos que consume la PD (Parte A, solo lectura)
#
# La PD se entrena con los datos que entrega la Parte A, sin cambiarlos. El análisis completo está en el notebook
# `01_datos` y en el [blog 02](../blog/secciones/02-datos.md) y el [blog 03](../blog/secciones/03-descriptivo.md).
#
# ### 2.1 Partición temporal y tasa de malos
#
# La partición es por fecha de originación (train, val y test en ese orden), para medir como se usaría el modelo:
# con créditos posteriores a los de entrenamiento. Tasa de malos por partición y por año; los años posteriores a test
# quedan fuera. La tasa de malos sube de train a test: esa deriva reaparece en la sección 5.5, cuando un corte de PD
# elegido en val se aplica tal cual en test.

# %%
resumen_datos = load_json(RESUMEN_DATOS)  # resumen de la Parte A
display(tablas.tabla_particion(resumen_datos))  # cortes, n y tasa por partición
tablas.tabla_por_anio(resumen_datos)  # por año de originación

# %%
display(Image(filename=str(FIGURAS_BLOG / "datos-anio.png")))  # figura de la Parte A

# %% [markdown]
# ### 2.2 Las variables de originación
#
# Solo entran variables conocidas antes del desembolso (núcleo §5.7); el significado es la cita literal del
# diccionario de Kaggle. El preprocesador de A se ajustó solo con train y B lo carga sin reajustar. La figura muestra
# la tasa de malos por grupo de cada variable (línea discontinua = tasa de train).

# %%
tablas.tabla_variables(load_config(FEATURES))  # tipo, etiqueta, transformación y cita

# %%
display(Image(filename=str(FIGURAS_BLOG / "eda-tasa-variables.png")))  # figura de la Parte A

# %% [markdown]
# ### 2.3 Estabilidad entre particiones (PSI frente a train)
#
# El PSI compara la distribución de cada variable en val y en test con la de train (después del preprocesamiento):
# más alto = más distinta. No se aplican umbrales de interpretación: su fuente sigue marcada [VERIFICAR] en la
# spec de la Parte A.

# %%
post = load_json(POST_PREPROCESAMIENTO)  # descriptivo después del preprocesamiento
pd.concat([tablas.tabla_psi(post, por="test"), tablas.tabla_psi(post, por="val")], axis=1)  # ordenado por test

# %% [markdown]
# ## 3. Búsqueda de hiperparámetros
#
# Las dos búsquedas eligen con la **log-loss de validación** y nunca miran test; su presupuesto (configuraciones y
# tiempo) se reporta para comparar con justicia. Detalle en el [blog 06](../blog/secciones/06-resultados-pd.md) y
# los fundamentos en el [blog 04](../blog/secciones/04-fundamentos.md).
#
# ### 3.1 Logística: rejilla de C
#
# C es el inverso de la fuerza de la regularización L2 de la logística. Se prueba cada C de la rejilla y se elige el
# de menor log-loss de val (empate → menor C); `en_borde` avisa si el elegido está en un extremo de la rejilla.

# %%
oficial = insumos_de_corrida(ENTRADA)  # resultados versionados de B3–B5 (solo lectura)
resumen, hpo = oficial["resumen"], oficial["hpo"]  # summary.json y hpo_ensayos.json
print(f"Criterio: {oficial['busqueda']['criterio']} · C elegido: {oficial['busqueda']['C_elegido']} · "
      f"en el borde: {oficial['busqueda']['en_borde']}")  # resumen de la búsqueda de C
tablas.tabla_busqueda_c(oficial["busqueda"])  # rejilla ordenada por C

# %%
display(Image(filename=str(FIGURAS_BLOG / "pd-busqueda-logistica.png")))  # log-loss y AUC de val contra C

# %% [markdown]
# ### 3.2 MLP: búsqueda con TPE
#
# Optuna propone configuraciones del espacio de abajo con el muestreador TPE; cada ensayo entrena un MLP y se mide en
# val. El presupuesto cuenta todos los ensayos y la suma de los tiempos de entrenamiento (B-pd §3.3).

# %%
print("Muestreador:", hpo["muestreador"], "· podador:", hpo["podador"])  # TPE registrado
print(f"Criterio: {hpo['criterio']} · ensayos: {hpo['n_ensayos']} · estados: {hpo['estados']} · "
      f"Σ fit: {hpo['tiempo_total_s']:.1f} s · reloj: {hpo['tiempo_reloj_s']:.1f} s")  # presupuesto
tablas.tabla_espacio(hpo)  # espacio de búsqueda

# %% [markdown]
# ### 3.3 Trayectoria de la búsqueda
#
# Log-loss de val de cada ensayo completo, el mejor acumulado, el fin de los ensayos aleatorios iniciales y la
# logística como referencia.

# %%
display(Image(filename=str(FIGURAS_BLOG / "pd-busqueda-mlp.png")))  # trayectoria del TPE

# %% [markdown]
# ### 3.4 Mejores ensayos
#
# Los ensayos completos con menor log-loss de val (empate → menor número de ensayo). `en_borde` indica si lr, l2 o
# dropout del mejor quedaron cerca de un extremo de su rango (`null` = no).

# %%
print("en_borde del mejor ensayo:", hpo["en_borde"])  # lr, l2, dropout
tablas.tabla_mejores_ensayos(hpo)  # los mejores

# %% [markdown]
# ### 3.5 Resumen de las dos búsquedas
#
# Método, configuraciones probadas, criterio, mejor valor del criterio y tiempo (suma de los entrenamientos) de cada
# búsqueda, lado a lado.

# %%
tablas.tabla_resumen_busquedas(oficial["busqueda"], hpo)  # logística y MLP

# %% [markdown]
# ## 4. Modelo final y comparación justa
#
# Misma partición, mismas variables y mismo preprocesador para los dos modelos; el MLP se repite con varias semillas
# para separar la diferencia entre modelos de la variación por semilla. Detalle en el
# [blog 06](../blog/secciones/06-resultados-pd.md) y las métricas en el [blog 05](../blog/secciones/05-metricas.md).
#
# ### 4.1 Arquitectura ganadora
#
# La config del mejor ensayo, que `pd-hpo` escribió en `configs/pd/mlp_final.yaml` y `summary.json` copia, con sus
# parámetros entrenables frente a los de la logística.

# %%
print(tablas.describir_mlp(resumen["mlp"]))  # arquitectura y procedencia
tablas.tabla_arquitectura(resumen)  # config y parámetros

# %% [markdown]
# ### 4.2 Variación entre semillas
#
# Media, desviación, mejor y peor del MLP en todas sus semillas, con el valor de la logística al lado. El MLP que se
# guarda es el de la semilla mediana, elegida con validación (regla impresa abajo).

# %%
print(f"Semilla mediana: {resumen['mlp']['semilla_mediana']} · "  # semilla de models/pd_mlp.pt
      f"regla: {resumen['mlp']['criterio_semilla_mediana']}")  # regla de B5
tablas.tabla_semillas(resumen)  # por métrica

# %%
display(Image(filename=str(FIGURAS_BLOG / "pd-semillas.png")))  # AUC, KS y Brier por semilla

# %% [markdown]
# ### 4.3 MLP − logística
#
# Diferencia de medias con signo y quién es mejor según la métrica (AUC, Gini y KS: mayor; Brier y log-loss: menor), y
# en cuántas semillas el MLP supera a la logística (un empate no cuenta).

# %%
display(tablas.comparacion(resumen))  # diferencia por métrica
tablas.victorias_por_semilla(oficial["semillas"], resumen)  # victorias por semilla

# %% [markdown]
# ### 4.4 Discriminación en prueba: ROC y KS
#
# Las curvas de los dos modelos en test (el MLP es la semilla mediana): cuánto separan a los malos de los buenos.

# %%
display(Image(filename=str(FIGURAS_BLOG / "pd-roc.png")))  # ROC
display(Image(filename=str(FIGURAS_BLOG / "pd-ks.png")))  # KS

# %% [markdown]
# ### 4.5 Calibración en prueba
#
# Lectura (B-pd §3.6): los dos modelos ordenan el riesgo pero lo **subestiman**: cuando la tasa observada supera a la
# PD media en un bin, el punto queda **por encima** de la diagonal (la tabla cuenta en cuántos bins pasa). En la
# sección 2.1 la tasa de malos sube de train a test; eso es consistente con la subestimación, pero aquí no se prueba
# que sea su causa. La sección 5 muestra qué implica esta calibración al elegir un corte para aprobar o rechazar.

# %%
display(tablas.resumen_calibracion(load_json(FIGURAS_BLOG / "pd-calibracion.json")))  # PD media frente a tasa
display(Image(filename=str(FIGURAS_BLOG / "pd-calibracion.png")))  # curva de calibración

# %% [markdown]
# ### 4.6 Curva de aprendizaje y parada temprana
#
# BCE de train y de val por época de la semilla mediana. El entrenamiento se detiene tras `paciencia` épocas sin
# mejorar la pérdida de val y se quedan los pesos de la mejor época (B-pd §3.2).

# %%
print(tablas.ficha_aprendizaje(oficial["semillas"], resumen))  # épocas, mejor época y paciencia
display(Image(filename=str(FIGURAS_BLOG / "pd-aprendizaje.png")))  # curva de aprendizaje


# %% [markdown]
# ## 5. De la PD a la decisión: el umbral
#
# La PD se vuelve una decisión al fijar un corte: se rechaza a quien queda por encima. Positivo = malo; FN = malo
# aprobado (el error caro), FP = bueno rechazado. La sección **ilustra** cómo cambia la decisión con el corte y con el
# costo relativo `L/G`; **no fija una política**, que depende del apetito de riesgo del banco. Solo lee
# `umbral.json` (generado con `python scripts/umbral_pd.py --config configs/pd/umbral.yaml`). Definiciones en el
# [blog 05](../blog/secciones/05-metricas.md) (§5.8) y resultados en el [blog 06](../blog/secciones/06-resultados-pd.md)
# (§6.8).
#
# ### 5.1 Frente al modelo trivial
#
# El modelo trivial asigna a todos la misma PD: la tasa de malos de train (y, como piso teórico que usa etiquetas de
# prueba, la de test). No ordena (su AUC es la del azar) y su matriz es degenerada: aprueba o rechaza a todos. El
# Brier dice cuánto mejoran los modelos sobre esa PD constante.

# %%
umbral = load_json(UMBRAL)  # resultados de B11 (solo lectura)
print("Tasa de malos por partición:", umbral["trivial"]["tasas"])  # train, val y test
tablas_umbral.tabla_trivial(umbral)  # Brier del trivial y de los modelos

# %% [markdown]
# ### 5.2 Matriz de confusión con el umbral inicial
#
# Con el umbral inicial de la config (impreso abajo), VP, FN, FP y VN en val y test, la sensibilidad (TPR, malos
# rechazados entre los malos), la especificidad (TNR, buenos aprobados entre los buenos) y la tasa de rechazo, que
# muestra a cuántos rechaza ese corte.

# %%
print("Umbral inicial:", umbral["matriz_inicial"]["umbral"])  # de configs/pd/umbral.yaml
tablas_umbral.tabla_matriz_inicial(umbral)  # una fila por (modelo, partición)

# %% [markdown]
# ### 5.3 Sensibilidad al umbral
#
# TPR, TNR y tasa de rechazo en toda la rejilla de umbrales, en val y test: bajar el corte rechaza a más malos pero
# también a más buenos. Ningún umbral es "el correcto" sin decir cuánto cuesta cada error.

# %%
display(Image(filename=str(FIGURAS_BLOG / "pd-umbral-sensibilidad.png")))  # TPR, TNR y rechazo contra el umbral

# %% [markdown]
# ### 5.4 Cuota por costo y metas de sensibilidad
#
# Para cada razón `L/G` se elige en val la cuota (fracción de créditos de mayor PD que se rechaza) de menor costo
# `(L/G)·FN + FP`, y se aplica en test sobre la cohorte: es una política **por lotes o de cuota**, no la decisión
# individual en tiempo real de la app. Las metas fijan la menor cuota con TPR de val por encima de la meta. Al final,
# la variación entre las semillas del MLP (reentrenadas con la config final) y el control de la semilla mediana.

# %%
display(Image(filename=str(FIGURAS_BLOG / "pd-umbral-costo.png")))  # cuota y costo contra L/G
display(tablas_umbral.tabla_escenarios(umbral))  # escenarios L/G: cuota de val y medidas de prueba
display(tablas_umbral.tabla_metas(umbral))  # metas de sensibilidad
print("Semillas reentrenadas:", umbral["semillas"]["n"])  # n del resumen
display(tablas_umbral.tabla_semillas(umbral))  # media, desv., mejor y peor por política
tablas_umbral.ficha_control(umbral)  # semilla mediana reentrenada = models/pd_mlp.pt

# %% [markdown]
# ### 5.5 PD absoluta frente a cuota
#
# Si la PD estuviera calibrada, el corte de menor costo esperado sería $p^* = 1/(1 + L/G)$. La tabla lo pone junto al
# umbral de val de la cuota elegida (el corte observado) y su brecha, que mide la descalibración; y compara la tasa de
# rechazo en test por cuota con la de aplicar ese umbral tal cual: el mismo corte de PD rechaza en test otra fracción
# de créditos. Eso es consistente con la deriva de la sección 2.1, pero aquí no se prueba que sea su causa.

# %%
print("Tasa de malos por partición:", umbral["trivial"]["tasas"])  # deriva (sección 2.1)
tablas_umbral.tabla_pd_absoluta(umbral)  # p*, corte observado y tasas de rechazo
