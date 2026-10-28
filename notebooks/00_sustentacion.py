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
SET = {}            # p. ej. {"demo": "eda"} para mostrar otra demo (por defecto, la de PD)
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
# Verificadas corriendo el cambio con `correr_demo` (carpeta temporal). Las de `eda` (Parte A, 24-oct-2026) se corren con
# `SET = {"demo": "eda"}`; las de PD (B), scorecard (C) y LGD (D), verificadas el 28-oct-2026, con la demo indicada en la
# primera columna (la demo por defecto es la de PD, B6). Cifras de la demo `pd`: muestra de 50 000, 3 semillas; de `lgd`: muestra de
# 20 000, 3 semillas; no son los resultados reportados. Las predicciones que fallaron o se cumplieron a medias se dejan:
# son material de la sustentación. Evidencia completa (predicciones escritas antes de correr): `alucinaciones/evidencias/T5.4_predicciones.md`.
#
# | Cambio (`CAMBIO`) | Predicción | Por qué | ¿Se cumplió? |
# |---|---|---|---|
# | `{"preprocesamiento.percentiles_recorte": [5.0, 95.0]}` | Cada cola de `dti` pasa de ~1 % a ~5 % recortado; desaparece el grupo aislado de saldo rotativo 0 (asimetría de −3,1 a −0,8); el PSI casi no cambia | El percentil 5 del saldo es mayor que 0, así que los ceros se suben al recorte; el PSI usa deciles de train y recortar no cambia el orden | Sí |
# | `{"particion.train_hasta": "2011-12-31", "particion.val_hasta": "2012-12-31"}` | Train baja de 90 347 a ~40 500 créditos y prueba sube (2013 + 2014); la tasa de malos de prueba queda cerca de 25 % | Se corre un año cada corte; los años recientes tienen más malos entre los resueltos (sesgo de selección, blog 2.4) | Sí: 40 534 / 49 813 / 147 348; 15,8 % / 16,7 % / 24,6 % |
# | `{"eda.umbral_monotonia": 0.7}` | H6 (historial) y H10 (monto) pasan de "no es monótona" a "la apoya" | Su \|Spearman\| es 0,73 y 0,76: superan 0,7 pero no 0,9 | Sí |
# | `SET = {"demo": "pd"}` + `{"mlp.lr": 0.05}` (×10) | La logística no cambia; el MLP empeora: AUC de test menor que 0,6545, más desviación entre semillas y Brier cerca de 0,19 | Adam da pasos 10 veces mayores con minilotes de 512 y oscila; la parada temprana conserva la mejor época de val | En parte: AUC 0,6545 → 0,6496 y desviación 0,0050 → 0,0103, logística igual (0,6563); pero el Brier solo pasa de 0,1839 a 0,1846 (no a ~0,19). 11,4 s |
# | `SET = {"demo": "pd"}` + `{"mlp.max_epocas": 2}` (de 20) | El MLP queda sin entrenar: AUC entre 0,60 y 0,65, Brier mayor que 0,19 y el tiempo del MLP cae ~10×; mejor_epoca = 2 en las 3 semillas | ~76 pasos de Adam no alcanzan a calibrar la salida (la tasa de malos de train es ~0,16) | En parte: AUC 0,6450 y Brier 0,1839 → 0,1870 (no pasa de 0,19); el MLP tarda 2,4 s (de 5,2), no ~10× menos; mejor_epoca 1, 2, 2. La base ya paraba sola antes de la época 20 (mejor época 5, 7, 3). 9,3 s |
# | `SET = {"demo": "pd"}` + `{"mlp.dropout": 0.5}` (de 0,094) | AUC menor, pero cae menos de 0,01; el Brier sube un poco; sin colapso | La red es pequeña (2 673 parámetros) y no sobreajusta: el dropout solo quita capacidad y añade ruido | En parte: AUC 0,6545 → 0,6540 (menos de 0,01) pero el Brier no cambia (0,1839) y el KS sube (0,2236 → 0,2272); con 3 semillas no se distingue del ruido. 11,6 s |
# | `SET = {"demo": "scorecard"}` + `{"scorecard.pdo": 40}` (de 20) | factor 28,85 → 57,71, offset 487,12 → 374,25 y puntos base 525,96 → ~451,9; todos los rangos de puntos se duplican (annual_inc 45,86 → ~91,7); la PD y la tasa de malos de cada banda no cambian | factor = pdo/ln 2; el puntaje nuevo es 600 + 2·(puntaje − 600) y el orden de los clientes no cambia | Sí: puntos base 451,9232; annual_inc 91,71, revol_util 72,84, inq_last_6mths 58,52; primer corte de banda 423,5; pd_media por banda idéntica. 6,4 s |
# | `SET = {"demo": "scorecard"}` + `{"scorecard.odds_base": 100}` (de 50) | Todo el puntaje baja exactamente un PDO (20 puntos): puntos base ~505,96, cortes de banda −20; los rangos por variable no cambian | offset = base − factor·ln(odds_base) y factor·ln 2 = pdo, así que duplicar odds_base resta un PDO | Sí: offset 467,1229 y puntos base 505,9616 (−20,0000), primer corte 491,8, rangos idénticos. 6,6 s |
# | `SET = {"demo": "scorecard"}` + `{"scorecard.base": 700}` (de 600) | Se traslada la escala: puntos base ~625,96 y cortes de banda +100; rangos y PD por banda iguales | `base` solo mueve el offset; ni el factor ni los betas dependen de ella | Sí: puntos base 625,9616, offset 587,1229, primer corte 611,8. 6,3 s |
# | `SET = {"demo": "lgd"}` + `{"mlp.lr": 0.01}` (×10, de 0,001) | El MAE de test del MLP casi no se mueve (\|Δ\| < 0,003) y no le gana a la constante mediana; la lineal y las constantes quedan iguales | La LGD es casi impredecible con las variables de originación (MLP ≈ lineal en la corrida real) | Sí: MAE test 0,0772 → 0,0787 ± 0,0015; la mediana constante sigue ganando (0,0701); lineal 0,0778. 9,5 s |
# | `SET = {"demo": "lgd"}` + `{"mlp.max_epocas": 2}` (de 30) | El MAE de test del MLP sube a más de 0,12 (estimación ~0,2) y el tiempo del MLP baja ~15× | La sigmoide arranca cerca de 0,5 y la LGD media es ~0,9; 2 épocas de ~26 pasos de Adam no llegan | No: sube (0,0772 → 0,0899 ± 0,0186; semillas 0,1113, 0,0768, 0,0818) pero no pasa de 0,12, y el MLP tarda 2,3 s (de 4,4), no ~15× menos; la red llega cerca de la constante en 2 épocas y lo que crece es la dispersión entre semillas. 8,1 s |
# | `SET = {"demo": "lgd"}` + `{"mlp.l2": 0.1}` (de 1e-4) | El MLP se vuelve casi una constante: MAE de test a menos de 0,003 del de la constante media | (λ/2)‖W‖² empuja los pesos a 0 y el sesgo (sin L2) conserva la media de train | No: MAE test 0,0838 ± 0,0008, a 0,0059 de la constante media (0,0779) y peor que ella. En val sí es casi constante (0,0757–0,0759 frente a 0,0765); por qué en test se separa no se investigó (predicción no verificada). 11,4 s |

# %% [markdown]
# ## 3. Preguntas (5 min): dónde está cada cosa
#
# | Tema | Dónde | Quién lo hizo |
# |---|---|---|
# | Enunciado y reglas | `docs/spec/SPEC.md` (Anexo = enunciado literal) | todos |
# | Datos, objetivo, variables, partición, análisis descriptivo | `docs/spec/partes/A-datos.md`, `src/datos/`, blog 1–3, notebook 01 | Daniel |
# | PD: logística, MLP, búsqueda, métricas | `docs/spec/partes/B-pd.md`, `src/pd/`, blog 4–6, notebook 02 | Andrés |
# | Scorecard, SHAP, hipótesis | `docs/spec/partes/C-scorecard.md`, `src/scorecard/`, blog 7, notebook 03 | Juan José |
# | LGD | `docs/spec/partes/D-lgd.md`, `src/lgd/`, blog 8, notebook 04 | Camila y José |
# | Pérdida esperada, video, blog, integración | `docs/spec/partes/E-producto.md`, `src/el/`, `video/`, blog 0 y 9–13, notebook 05 | Daniel |
# | App web | `app/` (https://trabajo2-redes-neuronales.vercel.app) | José |
# | Cacería de la alucinación | `alucinaciones/registro.md`, blog 12 | todos |
# | Fichas de estudio y preguntas probables | `docs/sustentacion/` (tarea T5.4) | todos |
