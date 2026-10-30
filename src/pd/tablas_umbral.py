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
# # Tablas de B11: de la PD a la decisión (B-pd §3.9; notebook 02 §5 y blog 06 §6.8)
#
# Funciones **puras** que convierten `umbral.json` (ya leído) en tablas de pandas: no leen archivos ni calculan nada
# del modelo; cada celda es un valor del JSON. Este módulo es además el **único lugar** del contrato de `umbral.json`
# (`ARCHIVO_UMBRAL`, `MODELOS`, `COMANDO_UMBRAL` y `bloque`): lo importan la orquestación y las figuras (es liviano:
# pandas y la paleta, sin torch ni matplotlib, así que no hay import circular). Lo que no aplica (umbral de val con
# cuota 0, una meta inalcanzable, un resumen nulo) queda como texto vacío, nunca NaN. Un bloque que falta →
# `ValueError` que lo nombra.

# %%
"""Tablas de B11 desde umbral.json: trivial, matriz inicial, escenarios L/G, PD absoluta, metas, semillas y control."""

import pandas as pd  # tablas

from src.viz.paleta import NOMBRE_MODELO as _NOMBRES_PALETA  # nombres de los modelos (constantes, sin matplotlib)

# %% [markdown]
# ## Constantes
#
# El contrato de `umbral.json` (nombre del archivo, orden de los modelos y comando que lo genera), los nombres de
# filas y columnas en español y el texto de lo que no aplica.

# %%
ARCHIVO_UMBRAL = "umbral.json"  # resultados de B11 (en salida)
COMANDO_UMBRAL = "python scripts/umbral_pd.py --config configs/pd/umbral.yaml"  # genera umbral.json
MODELOS = ("logistica", "mlp")  # orden fijo (= experimento.MODELO_LOGISTICA, MODELO_MLP; lo exige una prueba)
NOMBRE_MODELO = {m: _NOMBRES_PALETA[m] for m in MODELOS}  # Regresión logística, Perceptrón multicapa
VACIO = ""  # celda sin dato (no NaN)
NO_APLICA = "no aplica: la semilla mediana no está entre las reentrenadas"  # control null (demo)
FILAS_TRIVIAL = {"principal": "Trivial (PD = tasa de train)", "piso_teorico": "Trivial (PD = tasa de prueba, piso)",
                 **NOMBRE_MODELO}  # filas de la tabla del trivial
COLUMNAS_MATRIZ = {"vp": "VP", "fn": "FN", "fp": "FP", "vn": "VN", "tpr": "TPR", "tnr": "TNR",  # matriz y tasas
                   "tasa_rechazo": "rechazo"}  # tasa de rechazo
ETIQUETA_TEST_SEMILLA = {"tpr_test": "TPR prueba", "tnr_test": "TNR prueba",  # métricas de las semillas
                         "tasa_aprobacion_test": "aprobación prueba"}  # aprobación
COLUMNAS_RESUMEN = ("media", "desv", "mejor", "peor", "n")  # de experimento.resumir
COLUMNAS_PD_ABSOLUTA = ("p*", "umbral val", "p* − umbral val", "rechazo val", "rechazo prueba (cuota)",  # orden
                        "rechazo prueba (umbral val)")  # tabla de la PD absoluta frente a la cuota


# %% [markdown]
# Un bloque obligatorio de `umbral.json`; si falta → `ValueError` que dice cómo generarlo (la usan también las
# figuras de `src/viz/umbral_pd.py`).

# %%
def bloque(umbral: dict, clave: str):
    """umbral[clave] o ValueError que nombra la clave y el comando."""
    if not isinstance(umbral, dict) or clave not in umbral:  # falta
        raise ValueError(f"{ARCHIVO_UMBRAL} no tiene el bloque {clave!r}; genéralo con {COMANDO_UMBRAL}")  # cómo
    return umbral[clave]  # bloque


# %% [markdown]
# ## Trivial frente a los modelos
#
# PD constante $q$ (tasa de train; tasa de prueba como piso teórico $\pi(1-\pi)$) y los dos modelos: Brier de val y
# de test.

# %%
def tabla_trivial(umbral: dict) -> pd.DataFrame:
    """Filas FILAS_TRIVIAL; columnas PD constante, Brier val, Brier test."""
    t = bloque(umbral, "trivial")  # bloque
    filas = {}  # por fila
    for clave in ("principal", "piso_teorico"):  # los dos triviales
        filas[FILAS_TRIVIAL[clave]] = {"PD constante": t[clave]["q"], "Brier val": t[clave]["brier_val"],  # q
                                       "Brier test": t[clave]["brier_test"]}  # Brier
    for m in MODELOS:  # los dos modelos
        filas[FILAS_TRIVIAL[m]] = {"PD constante": VACIO, "Brier val": t["modelos"][m]["brier_val"],  # sin q
                                   "Brier test": t["modelos"][m]["brier_test"]}  # Brier
    return pd.DataFrame.from_dict(filas, orient="index")  # una fila por clave


# %% [markdown]
# ## Matriz de confusión con el umbral inicial
#
# Una fila por (modelo, split): VP, FN, FP, VN, TPR, TNR y tasa de rechazo.

# %%
def tabla_matriz_inicial(umbral: dict) -> pd.DataFrame:
    """Índice (modelo, split); columnas COLUMNAS_MATRIZ."""
    mi = bloque(umbral, "matriz_inicial")  # bloque
    filas = [{"modelo": NOMBRE_MODELO[m], "split": s, **{v: d[k] for k, v in COLUMNAS_MATRIZ.items()}}  # celdas
             for m in MODELOS for s, d in mi["modelos"][m].items()]  # val y test
    return pd.DataFrame(filas).set_index(["modelo", "split"])  # conteos enteros


# %% [markdown]
# ## Escenarios $L/G$
#
# Una fila por ($L/G$, modelo): $p^\*$, la cuota elegida en val, su umbral de val y las medidas de prueba por cuota
# (la política) y por el umbral de val (contraste de PD absoluta).

# %%
def _o_vacio(valor):
    """VACIO si ``valor`` es None."""
    return VACIO if valor is None else valor  # nunca NaN


def tabla_escenarios(umbral: dict) -> pd.DataFrame:
    """Índice (L/G, modelo); p*, cuota, umbral val y medidas de prueba."""
    filas = []  # una por (razón, modelo)
    for e in bloque(umbral, "escenarios"):  # cada razón
        for m in MODELOS:  # cada modelo
            d = e["modelos"][m]  # política
            tc, t_umbral = d["test_cuota"], d["test_umbral"]  # prueba por cuota y con el umbral de val
            filas.append({"L/G": e["razon"], "modelo": NOMBRE_MODELO[m], "p*": e["p_estrella"],  # referencia
                          "cuota": d["cuota"], "umbral val": _o_vacio(d["umbral_val"]),  # política
                          "TPR prueba (cuota)": tc["tpr"], "TNR prueba (cuota)": tc["tnr"],  # cuota
                          "rechazo prueba (cuota)": tc["tasa_rechazo"],  # cuota
                          "costo/crédito prueba (cuota)": tc["costo_por_credito"],  # cuota
                          "rechazo prueba (umbral val)": t_umbral["tasa_rechazo"],  # PD absoluta
                          "TPR prueba (umbral val)": t_umbral["tpr"],  # PD absoluta
                          "costo/crédito prueba (umbral val)": t_umbral["costo_por_credito"]})  # PD absoluta
    return pd.DataFrame(filas).set_index(["L/G", "modelo"])  # tabla


# %% [markdown]
# ## PD absoluta frente a cuota
#
# Una fila por ($L/G$, modelo): $p^\*$ (el corte si la PD estuviera calibrada), el umbral de val de la cuota elegida
# (el corte observado), su brecha $p^\* - t_{\text{val}}$ (mide la descalibración, B-pd §3.9) y la tasa de rechazo en
# val, en prueba por cuota y en prueba con el umbral de val (efecto de la deriva). Umbral nulo → umbral y brecha vacíos.

# %%
def tabla_pd_absoluta(umbral: dict) -> pd.DataFrame:
    """Índice (L/G, modelo); columnas COLUMNAS_PD_ABSOLUTA."""
    filas = []  # una por (razón, modelo)
    for e in bloque(umbral, "escenarios"):  # cada razón
        for m in MODELOS:  # cada modelo
            d = e["modelos"][m]  # política
            t_val = d["umbral_val"]  # corte observado en val (None con cuota 0)
            brecha = VACIO if t_val is None else e["p_estrella"] - t_val  # p* − t_val
            valores = (e["p_estrella"], _o_vacio(t_val), brecha, d["val"]["tasa_rechazo"],  # val
                       d["test_cuota"]["tasa_rechazo"], d["test_umbral"]["tasa_rechazo"])  # prueba
            filas.append({"L/G": e["razon"], "modelo": NOMBRE_MODELO[m],  # índice
                          **dict(zip(COLUMNAS_PD_ABSOLUTA, valores, strict=True))})  # celdas
    return pd.DataFrame(filas).set_index(["L/G", "modelo"])  # tabla


# %% [markdown]
# ## Metas de sensibilidad
#
# Una fila por (meta, modelo): la menor cuota con $\mathrm{TPR}_{\text{val}} \ge$ meta, su umbral de val y la medida
# en val y prueba. Meta inalcanzable → fila vacía.

# %%
def tabla_metas(umbral: dict) -> pd.DataFrame:
    """Índice (meta, modelo); cuota, umbral val, TPR val y medidas de prueba."""
    columnas = ("cuota", "umbral val", "TPR val", "TPR prueba", "TNR prueba", "rechazo prueba")  # orden
    filas = []  # una por (meta, modelo)
    for meta in bloque(umbral, "metas"):  # cada meta
        for m in MODELOS:  # cada modelo
            d = meta["modelos"][m]  # política o nulos
            valores = dict.fromkeys(columnas, VACIO) if d["cuota"] is None else {  # inalcanzable
                "cuota": d["cuota"], "umbral val": _o_vacio(d["umbral_val"]), "TPR val": d["val"]["tpr"],  # val
                "TPR prueba": d["test_cuota"]["tpr"], "TNR prueba": d["test_cuota"]["tnr"],  # prueba
                "rechazo prueba": d["test_cuota"]["tasa_rechazo"]}  # prueba
            filas.append({"meta": meta["meta"], "modelo": NOMBRE_MODELO[m], **valores})  # fila
    return pd.DataFrame(filas).set_index(["meta", "modelo"])  # tabla


# %% [markdown]
# ## Variación entre semillas
#
# Una fila por (política, métrica de prueba) con media, desviación muestral, mejor, peor y $n$ de
# `experimento.resumir`. Política = "L/G = r" (escenario) o "TPR val ≥ m" (meta); un resumen nulo → fila vacía.

# %%
def tabla_semillas(umbral: dict) -> pd.DataFrame:
    """Índice (política, métrica); columnas COLUMNAS_RESUMEN."""
    sem = bloque(umbral, "semillas")  # bloque
    politicas = [(f"L/G = {e['razon']:g}", e["resumen"]) for e in sem["escenarios"]] + [  # escenarios
        (f"TPR val ≥ {m['meta']:g}", m["resumen"]) for m in sem["metas"]]  # metas
    filas = []  # una por (política, métrica)
    for nombre, resumen in politicas:  # cada política
        for clave, etiqueta in ETIQUETA_TEST_SEMILLA.items():  # cada métrica
            valores = dict.fromkeys(COLUMNAS_RESUMEN, VACIO) if resumen is None else {  # nulo
                c: resumen[clave][c] for c in COLUMNAS_RESUMEN}  # resumen
            filas.append({"política": nombre, "métrica": etiqueta, **valores})  # fila
    return pd.DataFrame(filas).set_index(["política", "métrica"])  # tabla


# %% [markdown]
# ## Control de la semilla mediana
#
# El bloque tal cual, o `{control: NO_APLICA}` si es `null` (demo con pocas semillas).

# %%
def ficha_control(umbral: dict) -> dict:
    """Control de la semilla mediana o la ficha "no aplica"."""
    control = bloque(umbral, "control_semilla_mediana")  # bloque (puede ser None)
    return {"control": NO_APLICA} if control is None else dict(control)  # copia
