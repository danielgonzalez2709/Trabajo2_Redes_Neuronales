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
# # Cierre de hipótesis (Parte C §3.4, tarea C6; enunciado P1.2)
#
# El enunciado pide formular hipótesis explícitas sobre qué hace más riesgosa a una persona y, al final, decir cuáles se
# confirmaron y cuáles no. Las 11 hipótesis están en `configs/hipotesis.yaml` (formuladas antes de modelar); la Parte A
# dio la primera evidencia con el EDA de train (`results/data/eda/hipotesis_eda.json`). Aquí se cierran reuniendo, por
# hipótesis, **cuatro fuentes** y una regla de decisión fija:
#
# 1. **EDA de train** (efecto *marginal*): la evidencia de A, copiada tal cual.
# 2. **SHAP del MLP** (efecto *condicional*, C4: `shap_mlp.json`): dirección de los valores de Shapley del logit.
# 3. **Test fuera de tiempo** (efecto *marginal*): tasas de malos OBSERVADAS en test, con los grupos/cortes de TRAIN.
# 4. **Compuerta de relevancia**: importancia por permutación del MLP en test (C4).
#
# Como contraste, sin voto: la logística (β y puntos de C3, `scorecard.json`) y el PSI de test frente a train (A).
#
# **Convención:** $y = 1$ malo. Una hipótesis numérica tiene `direccion` "+" (a mayor valor, más riesgo) o "−"; se
# escribe $s = +1$ o $s = -1$. Una categórica dice que `mas_riesgosa` tiene más riesgo que `referencia`.
#
# **Comando:** `python -m src.scorecard.hipotesis [--salida CARPETA]` lee `shap_mlp.json` y `scorecard.json` de la
# carpeta (por defecto `results/data/scorecard`) y escribe allí `hipotesis.json`. No corre SHAP ni modelos: tarda pocos
# segundos. `run.py scorecard --config configs/scorecard/mlp.yaml` lo llama al final si ya existe `scorecard.json`.
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable sin
# efectos al importar.

# %%
"""Cierre de hipótesis: EDA, SHAP del MLP, test fuera de tiempo y permutación (Parte C §3.4, tarea C6)."""

import argparse  # línea de comandos (--salida)
import math  # isfinite: señales con valores faltantes
import sys  # salida estándar en UTF-8 (Windows usa cp1252)
import time  # tiempo del comando
from pathlib import Path  # rutas de entrada y salida

import numpy as np  # bordes, searchsorted y signos
import pandas as pd  # parquet de datos, qcut/cut y medias por grupo
from scipy.stats import spearmanr  # tendencia entre el orden del grupo y su tasa (como A)

from src.common.config import load_config  # configs/hipotesis.yaml con el cargador común
from src.common.results import load_json, save_json  # lectura sin NaN y escritura UTF-8, sangría 2, sin NaN/inf
from src.datos.eda import UMBRAL_MONOTONIA  # |ρ| mínimo para "monótona" (A §3.7; configs/datos.yaml)
from src.pd.experimento import huella_archivo, huella_particion  # evidencia del núcleo §6
from src.scorecard.puntos import ARCHIVO_SCORECARD  # "scorecard.json" (C3)

__all__ = [  # interfaz pública (tarea C6)
    "ARCHIVO_HIPOTESIS", "ARCHIVO_SHAP", "ARCHIVO_SCORECARD", "RUTAS_POR_DEFECTO", "UMBRAL_RHO_SHAP",  # constantes
    "Z_PERMUTACION", "UMBRALES_SENSIBILIDAD", "N_MIN_CATEGORIA", "VEREDICTOS", "SENALES", "AVISO_CEGUERA",
    "NOTA_CAUSALIDAD", "CRITERIO",
    "senal_direccion", "senal_categorica", "senal_shap", "relevante", "veredicto",  # reglas
    "cortes_de_train", "asignar_grupos", "grupos_test_con_cortes_de_train", "tendencia_por_grupos",  # grupos
    "comprobar_train", "entrada_hipotesis", "matiz", "construir_hipotesis", "escribir", "main",  # orquestación
]

ARCHIVO_HIPOTESIS = "hipotesis.json"  # resultado (núcleo §5.4)
ARCHIVO_SHAP = "shap_mlp.json"  # SHAP y permutación del MLP (C4; mismo nombre que shap_pd.ARCHIVO_SHAP)
RUTAS_POR_DEFECTO = {  # entradas, relativas a la raíz del repositorio
    "hipotesis_yaml": "configs/hipotesis.yaml",  # enunciados y direcciones (A §3.7)
    "hipotesis_eda": "results/data/eda/hipotesis_eda.json",  # evidencia del EDA de train (A)
    "post_preprocesamiento": "results/data/eda/post_preprocesamiento.json",  # PSI de test frente a train (A)
    "shap": f"results/data/scorecard/{ARCHIVO_SHAP}",  # SHAP y permutación (C4)
    "scorecard": f"results/data/scorecard/{ARCHIVO_SCORECARD}",  # β y puntos de la logística (C3)
    "datos": "data/processed/loans_pd.parquet",  # id, split, target y variables crudas (A)
    "preprocesador": "models/preprocesador_pd.json",  # solo para la huella (núcleo §6)
}
CARPETA_POR_DEFECTO = "results/data/scorecard"  # carpeta de shap_mlp.json, scorecard.json e hipotesis.json

# %% [markdown]
# ## Umbrales
#
# - `UMBRAL_RHO_SHAP` $u = 0{,}5$: la señal de SHAP de una numérica es $s\,\rho$ con $\rho$ = Spearman entre el valor
#   crudo y su $\varphi$ en los explicados (C4). No se usa el 0,9 de A: en las variables de conteo (`pub_rec`,
#   `delinq_2yrs`, casi todo 0) los empates acotan $\rho$ por debajo de 1 aunque la relación sea perfectamente
#   creciente; $u = 0{,}5$ pide una relación clara sin castigar los empates. La sensibilidad repite el veredicto con
#   $u = 0{,}3$ y $u = 0{,}7$.
# - `Z_PERMUTACION` $= 2$: la variable es relevante si la caída media de AUC al permutarla supera $2$ desviaciones de las
#   $k$ repeticiones. Es conservador: usa la desviación de las repeticiones, no el error estándar ($\text{desv}/\sqrt{k}$).
# - `UMBRAL_MONOTONIA` (0,9, de A): solo describe la forma de la relación (matiz); la hipótesis afirma una dirección,
#   no monotonía, así que la no monotonía no quita el voto.
# - `N_MIN_CATEGORIA` = 30: por debajo, el $\bar\varphi$ de una categoría entre los explicados se avisa como poco preciso.

# %%
UMBRAL_RHO_SHAP = 0.5  # |s·ρ| mínimo de SHAP (ver la celda de arriba)
Z_PERMUTACION = 2  # caída media − Z·desv > 0 ⇒ relevante
UMBRALES_SENSIBILIDAD = (0.3, 0.7)  # otros u de SHAP para la sensibilidad
N_MIN_CATEGORIA = 30  # n mínimo de una categoría entre los explicados de SHAP sin aviso
N_DECILES = 10  # grupos del EDA de A (tendencia_numerica, n_bins por defecto)
MIN_GRUPOS_DECILES = 3  # con menos deciles distintos, A agrupa por valor exacto (grupos_ordenados)
CUANTIL_TOPE = 0.99  # recorte de la cola de los conteos (grupos_ordenados)
TOL_TASAS_TRAIN = 1e-12  # |Δ| máximo entre las tasas de train rehechas y las de hipotesis_eda.json
VEREDICTOS = ("confirmada", "refutada", "no concluyente")  # veredictos permitidos
SENALES = ("a_favor", "en_contra", "nula")  # señales de cada fuente
FUENTES = ("eda", "shap", "test")  # fuentes con voto
NOMBRE_FUENTE = {"eda": "EDA", "shap": "SHAP", "test": "test"}  # nombre legible de cada fuente
AVISO_CEGUERA = ("Las cifras del EDA, de SHAP y de la permutación ya se conocían (C4) cuando se fijó la regla; la "
                 "regla solo es ciega frente a las tasas de test, que no se habían calculado.")  # aviso literal
NOTA_CAUSALIDAD = "Es una asociación observada en los datos, no un efecto causal."  # cierre de todo matiz

# %% [markdown]
# ## Señales de cada fuente
#
# Cada fuente vota `a_favor`, `en_contra` o `nula`:
#
# - **EDA y test, numéricas** (como `evaluar_hipotesis` de A): $a = s\,(\text{tasa}_\text{alto} - \text{tasa}_\text{bajo})$
#   y $b = s\,\rho_\text{grupos}$; a favor si $a > 0$ y $b > 0$; en contra si $a < 0$ y $b < 0$; si no, nula.
# - **EDA y test, categóricas**: a favor si la tasa de `mas_riesgosa` supera la de `referencia`; en contra si es menor;
#   nula si son iguales o falta una.
# - **SHAP, numéricas**: a favor si $s\,\rho \ge u$; en contra si $s\,\rho \le -u$; si no, nula.
# - **SHAP, categóricas**: $\Delta\varphi = \bar\varphi(\text{mas\_riesgosa}) - \bar\varphi(\text{referencia})$; a favor
#   si $> 0$, en contra si $< 0$, nula si $= 0$ o falta una categoría entre los explicados.
# - **Compuerta**: relevante si $\overline{\Delta\text{AUC}} - Z\cdot\text{desv} > 0$.

# %%
def _finito(x) -> bool:
    """True si x es un número finito (None y NaN no lo son)."""
    return x is not None and math.isfinite(x)  # None, NaN e inf → False


def senal_direccion(tasa_bajo, tasa_alto, rho, signo: int) -> str:
    """Señal de una numérica en el EDA o en test (regla de `evaluar_hipotesis` de A).

    Parámetros:
        tasa_bajo, tasa_alto: tasa de malos del primer y del último grupo.
        rho: Spearman entre el orden del grupo y su tasa.
        signo: +1 si la hipótesis dice "+", −1 si dice "-".
    Retorna:
        str: "a_favor", "en_contra" o "nula".
    """
    if not (_finito(tasa_bajo) and _finito(tasa_alto) and _finito(rho)):  # falta un dato
        return "nula"  # sin evidencia
    a, b = signo * (tasa_alto - tasa_bajo), signo * rho  # diferencia de extremos y tendencia, orientadas
    if a > 0 and b > 0:  # las dos en la dirección de la hipótesis
        return "a_favor"  # apoya
    if a < 0 and b < 0:  # las dos en contra
        return "en_contra"  # contradice
    return "nula"  # mixtas o cero


def senal_categorica(valor_mas, valor_ref) -> str:
    """Señal de una categórica: compara una cantidad de `mas_riesgosa` con la de `referencia`.

    Parámetros:
        valor_mas: tasa (o φ medio) de la categoría más riesgosa según la hipótesis.
        valor_ref: la misma cantidad de la categoría de referencia.
    Retorna:
        str: "a_favor" si valor_mas > valor_ref, "en_contra" si <, "nula" si = o falta alguno.
    """
    if not (_finito(valor_mas) and _finito(valor_ref)):  # categoría ausente
        return "nula"  # sin evidencia
    if valor_mas > valor_ref:  # más riesgo en la categoría señalada
        return "a_favor"  # apoya
    if valor_mas < valor_ref:  # menos riesgo
        return "en_contra"  # contradice
    return "nula"  # iguales


def _signo(h: dict) -> int:
    """+1 si la hipótesis numérica dice "+", −1 si dice "-"."""
    if h["direccion"] not in ("+", "-"):  # solo dos valores válidos
        raise ValueError(f"{h['id']}: direccion debe ser '+' o '-'; se recibió {h['direccion']!r}")  # dato
    return 1 if h["direccion"] == "+" else -1  # s


def senal_shap(direccion_var: dict, h: dict, umbral: float = UMBRAL_RHO_SHAP) -> str:
    """Señal de SHAP del MLP para la variable de la hipótesis.

    Parámetros:
        direccion_var: entrada ``direccion[variable]`` de ``shap_mlp.json`` (num: spearman; cat: phi_medio).
        h: hipótesis de ``configs/hipotesis.yaml``.
        umbral: u de las numéricas (s·ρ ≥ u a favor; ≤ −u en contra).
    Retorna:
        str: "a_favor", "en_contra" o "nula".
    """
    if "direccion" in h:  # numérica
        rho = direccion_var.get("spearman")  # ρ(valor, φ)
        if not _finito(rho):  # dirección no definida
            return "nula"  # sin evidencia
        s_rho = _signo(h) * rho  # ρ orientado según la hipótesis
        if s_rho >= umbral:  # relación clara en la dirección afirmada
            return "a_favor"  # apoya
        if s_rho <= -umbral:  # relación clara en la dirección contraria
            return "en_contra"  # contradice
        return "nula"  # débil
    phi = direccion_var.get("phi_medio", {})  # categoría → φ medio
    return senal_categorica(phi.get(h["mas_riesgosa"]), phi.get(h["referencia"]))  # Δφ > 0 a favor


def relevante(caida_media: float, caida_desv: float, z: float = Z_PERMUTACION) -> bool:
    """Compuerta de relevancia: caída media de AUC − z·desv > 0.

    Parámetros:
        caida_media: caída media de AUC al permutar la variable (k repeticiones, test).
        caida_desv: desviación de las k caídas.
        z: número de desviaciones exigidas.
    Retorna:
        bool: True si la variable es relevante para el MLP en test.
    """
    return bool(caida_media - z * caida_desv > 0)  # estrictamente positiva


# %% [markdown]
# ## Regla de decisión
#
# En este orden:
#
# 1. **confirmada**: EDA, SHAP y test a favor **y** la permutación del MLP es relevante;
# 2. **refutada**: al menos 2 de {EDA, SHAP, test} en contra y ninguna a favor;
# 3. **no concluyente**: todo lo demás.
#
# El `motivo` (plantilla) dice qué se cumplió o qué condición falló: fuentes que discrepan, señal nula de alguna o
# permutación no relevante.

# %%
def _lista(nombres: list[str]) -> str:
    """'A', 'A y B', 'A, B y C'."""
    return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]  # español


def veredicto(senales: dict, es_relevante: bool, usar_compuerta: bool = True) -> tuple[str, str]:
    """Veredicto y motivo a partir de las tres señales con voto y de la compuerta.

    Parámetros:
        senales: {"eda", "shap", "test"} → "a_favor" | "en_contra" | "nula".
        es_relevante: resultado de la compuerta de permutación.
        usar_compuerta: False desactiva la compuerta (sensibilidad).
    Retorna:
        tuple[str, str]: (veredicto en VEREDICTOS, motivo).
    """
    for f in FUENTES:  # señales válidas
        if senales[f] not in SENALES:  # valor desconocido
            raise ValueError(f"Señal inválida de {f}: {senales[f]!r}")  # dato
    favor = [NOMBRE_FUENTE[f] for f in FUENTES if senales[f] == "a_favor"]  # fuentes a favor
    contra = [NOMBRE_FUENTE[f] for f in FUENTES if senales[f] == "en_contra"]  # fuentes en contra
    nulas = [NOMBRE_FUENTE[f] for f in FUENTES if senales[f] == "nula"]  # fuentes sin señal
    pasa = es_relevante or not usar_compuerta  # compuerta superada o desactivada
    if len(favor) == len(FUENTES) and pasa:  # regla 1
        cola = ("la permutación del MLP en test es relevante" if es_relevante
                else "la compuerta de permutación está desactivada")  # por qué pasa la compuerta
        return "confirmada", f"EDA, SHAP y test a favor y {cola}."  # confirmada
    if len(contra) >= 2 and not favor:  # regla 2
        return "refutada", f"{_lista(contra)} en contra y ninguna fuente a favor."  # refutada
    partes = []  # condiciones que fallaron
    if favor and contra:  # votos opuestos
        partes.append(f"las fuentes discrepan (a favor: {_lista(favor)}; en contra: {_lista(contra)})")  # discrepancia
    if nulas:  # alguna sin señal
        partes.append(f"señal nula de {_lista(nulas)}")  # nula
    if usar_compuerta and not es_relevante:  # la compuerta también falló (se nombra siempre)
        partes.append("la permutación del MLP en test no es relevante")  # compuerta
    return "no concluyente", "No concluyente: " + "; ".join(partes) + "."  # no concluyente


# %% [markdown]
# ## Grupos de test con los cortes de TRAIN (sin fuga)
#
# Para comparar test con el EDA se repiten los grupos de A (`grupos_ordenados`), pero sus **cortes salen solo de
# train** y test solo se evalúa (núcleo §5.7):
#
# - **Deciles** (si train da al menos 3 deciles distintos): bordes = `pd.qcut(x_train sin vacíos, 10, retbins=True,
#   duplicates="drop")`; los bordes extremos pasan a $-\infty$ y $+\infty$ para que entre todo valor de test; los grupos
#   de test son `pd.cut(x_test, bordes)` (mismo cierre a la derecha que `qcut`).
# - **Conteos** (menos de 3 deciles, p. ej. `pub_rec`): tope = percentil 99 de TRAIN y valores únicos de train ya
#   recortados; un valor de test se recorta al tope y va al último valor de train $\le$ él (`searchsorted`). Un valor
#   menor que el mínimo de train iría al primer grupo (en los conteos no ocurre: el mínimo es 0).
# - Los vacíos se excluyen (como A).
#
# La tendencia (`tendencia_por_grupos`) es la de `tendencia_numerica` de A: Spearman entre el orden del grupo y su
# tasa si hay más de 2 grupos; si no, el signo de la diferencia. Antes de usar test se comprueba que los grupos rehechos
# en train reproducen las `tasas_por_grupo` de `hipotesis_eda.json` ($|\Delta| \le 10^{-12}$).

# %%
def cortes_de_train(x_train: pd.Series) -> dict:
    """Cortes de los grupos de A calculados solo con train.

    Parámetros:
        x_train: valores crudos de train (puede tener vacíos).
    Retorna:
        dict: {"tipo": "deciles", "bordes": [-inf, …, +inf]} o {"tipo": "conteo", "tope", "valores"}.
    """
    x = x_train.dropna()  # vacíos fuera (como A)
    g, bordes = pd.qcut(x, q=N_DECILES, labels=False, retbins=True, duplicates="drop")  # deciles de train
    if g.nunique() >= MIN_GRUPOS_DECILES:  # misma condición que grupos_ordenados
        bordes = np.asarray(bordes, dtype=np.float64).copy()  # bordes de train
        bordes[0], bordes[-1] = -np.inf, np.inf  # todo valor de test cae en algún decil
        return {"tipo": "deciles", "bordes": bordes.tolist()}  # deciles
    tope = float(x.quantile(CUANTIL_TOPE))  # p99 de train
    valores = np.unique(x.clip(upper=tope).to_numpy(dtype=np.float64))  # valores de train recortados, ordenados
    return {"tipo": "conteo", "tope": tope, "valores": valores.tolist()}  # conteo


def asignar_grupos(x: pd.Series, cortes: dict) -> pd.Series:
    """Índice de grupo (0 = valores más bajos) de cada fila con los cortes de train; vacíos → NaN.

    Parámetros:
        x: valores crudos (train o test).
        cortes: salida de ``cortes_de_train``.
    Retorna:
        pd.Series: float con el índice de grupo y NaN en los vacíos (mismo índice que x).
    """
    if cortes["tipo"] == "deciles":  # cortes de los deciles de train
        return pd.cut(x, np.asarray(cortes["bordes"]), labels=False).astype(float)  # (b_i, b_{i+1}]
    valores = np.asarray(cortes["valores"], dtype=np.float64)  # valores de train
    xv = x.clip(upper=cortes["tope"]).to_numpy(dtype=np.float64)  # recorte en el p99 de train
    idx = np.clip(np.searchsorted(valores, xv, side="right") - 1, 0, None).astype(float)  # último valor ≤ x
    idx[np.isnan(xv)] = np.nan  # vacíos fuera
    return pd.Series(idx, index=x.index)  # grupo por fila


def grupos_test_con_cortes_de_train(x_train: pd.Series, x_test: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Grupos de train y de test, los dos con los cortes de train.

    Parámetros:
        x_train: valores crudos de train.
        x_test: valores crudos de test.
    Retorna:
        tuple[pd.Series, pd.Series]: (grupos de train, grupos de test); NaN en los vacíos.
    """
    cortes = cortes_de_train(x_train)  # solo train
    return asignar_grupos(x_train, cortes), asignar_grupos(x_test, cortes)  # train y test con los mismos cortes


def tendencia_por_grupos(grupos: pd.Series, y: pd.Series, umbral: float = UMBRAL_MONOTONIA) -> dict:
    """Tasa de malos por grupo y tendencia (misma fórmula que ``tendencia_numerica`` de A).

    Parámetros:
        grupos: índice de grupo por fila (NaN = vacío, se excluye).
        y: objetivo (1 = malo) con el mismo índice.
        umbral: |ρ| mínimo para "monótona".
    Retorna:
        dict: {spearman_grupos, tasa_grupo_bajo, tasa_grupo_alto, n_grupos, monotona, tasas_por_grupo, n_por_grupo,
        grupos}; spearman_grupos es None si no está definido (tasas constantes).
    """
    validos = grupos.notna()  # vacíos fuera
    g = grupos[validos].astype(int)  # índice entero
    agregado = y[validos].astype(float).groupby(g).agg(["mean", "size"])  # tasa y n por grupo presente
    tasas = agregado["mean"]  # tasa de malos por grupo
    if len(tasas) > 2:  # como A: Spearman orden-vs-tasa
        rho = float(spearmanr(tasas.index, tasas.values).statistic)  # tendencia
    else:  # dos grupos: signo de la diferencia
        rho = float(np.sign(tasas.iloc[-1] - tasas.iloc[0]))  # +1, −1 o 0
    rho = rho if math.isfinite(rho) else None  # tasas constantes: ρ no definido
    return {"spearman_grupos": rho, "tasa_grupo_bajo": float(tasas.iloc[0]),
            "tasa_grupo_alto": float(tasas.iloc[-1]), "n_grupos": int(len(tasas)),
            "monotona": bool(rho is not None and abs(rho) >= umbral),
            "tasas_por_grupo": [float(v) for v in tasas], "n_por_grupo": [int(n) for n in agregado["size"]],
            "grupos": [int(i) for i in tasas.index]}  # misma forma que A más n y el índice de cada grupo


def comprobar_train(tendencia_train: dict, evidencia_eda: dict, id_h: str) -> None:
    """Exige que los grupos rehechos en train reproduzcan las tasas de ``hipotesis_eda.json``.

    Parámetros:
        tendencia_train: ``tendencia_por_grupos`` de train con los cortes de ``cortes_de_train``.
        evidencia_eda: ``evidencia_eda`` de A para la hipótesis.
        id_h: identificador (para el mensaje).
    Retorna:
        None; lanza ValueError si el número de grupos o alguna tasa difiere más de TOL_TASAS_TRAIN.
    """
    nuevas, de_a = tendencia_train["tasas_por_grupo"], evidencia_eda["tasas_por_grupo"]  # tasas a comparar
    if len(nuevas) != len(de_a) or max(abs(a - b) for a, b in zip(nuevas, de_a)) > TOL_TASAS_TRAIN:  # difieren
        raise ValueError(f"{id_h}: los grupos rehechos en train no reproducen tasas_por_grupo de "
                         f"hipotesis_eda.json ({nuevas} frente a {de_a})")  # los cortes no son los de A


# %% [markdown]
# ## Evidencia de test por hipótesis
#
# Numéricas: tasas y n por grupo de test, extremos, Spearman, monotonía y señal. Categóricas: tasa y n de las dos
# categorías en test y señal. En las categóricas también se comprueba que las tasas de train coinciden con A.

# %%
def _test_numerica(h: dict, train: pd.DataFrame, test: pd.DataFrame, evidencia_eda: dict) -> dict:
    """Bloque ``test`` de una hipótesis numérica (grupos con los cortes de train)."""
    v = h["variable"]  # variable cruda
    cortes = cortes_de_train(train[v])  # solo train
    t_train = tendencia_por_grupos(asignar_grupos(train[v], cortes), train["target"])  # train rehecho
    comprobar_train(t_train, evidencia_eda, h["id"])  # mismos grupos que A
    t = tendencia_por_grupos(asignar_grupos(test[v], cortes), test["target"])  # test con cortes de train
    del t["n_grupos"]  # implícito en las listas
    return {"agrupacion": cortes["tipo"], **t,
            "senal": senal_direccion(t["tasa_grupo_bajo"], t["tasa_grupo_alto"], t["spearman_grupos"],
                                     _signo(h))}  # mismo criterio que el EDA


def _tasas_categoricas(df: pd.DataFrame, h: dict) -> dict:
    """Tasa y n de `mas_riesgosa` y `referencia` en un split (categorías limpias como A)."""
    x = df[h["variable"]].astype("string").str.strip()  # mismo limpiado que evaluar_hipotesis
    agregado = df["target"].astype(float).groupby(x).agg(["mean", "size"])  # por categoría
    salida = {}  # rol → {tasa, n}
    for rol in ("mas_riesgosa", "referencia"):  # las dos categorías de la hipótesis
        cat = h[rol]  # categoría
        presente = cat in agregado.index  # ¿aparece en el split?
        salida[rol] = {"categoria": cat, "tasa": float(agregado.loc[cat, "mean"]) if presente else None,
                       "n": int(agregado.loc[cat, "size"]) if presente else 0}  # tasa y n
    return salida  # dos categorías


def _test_categorica(h: dict, train: pd.DataFrame, test: pd.DataFrame, evidencia_eda: dict) -> dict:
    """Bloque ``test`` de una hipótesis categórica (y comprobación de las tasas de train)."""
    tr = _tasas_categoricas(train, h)  # train rehecho
    for rol, clave in (("mas_riesgosa", "tasa_mas_riesgosa"), ("referencia", "tasa_referencia")):  # contra A
        a, b = tr[rol]["tasa"], evidencia_eda.get(clave)  # rehecha y de A
        if not (_finito(a) and _finito(b) and abs(a - b) <= TOL_TASAS_TRAIN):  # difieren
            raise ValueError(f"{h['id']}: la tasa de train de {tr[rol]['categoria']!r} no reproduce {clave} de "
                             f"hipotesis_eda.json ({a} frente a {b})")  # datos distintos
    te = _tasas_categoricas(test, h)  # test
    return {**te, "senal": senal_categorica(te["mas_riesgosa"]["tasa"], te["referencia"]["tasa"])}  # señal


# %% [markdown]
# ## Entrada de una hipótesis
#
# Reúne la evidencia copiada de cada fuente, las señales, el veredicto, la sensibilidad ($u = 0{,}3$, $u = 0{,}7$ y
# compuerta desactivada) y el matiz. El veredicto oficial es el de los umbrales fijados.

# %%
def _bloque_shap(h: dict, shap_json: dict, d: dict) -> dict:
    """Bloque ``shap``: dirección, importancia global, permutación del MLP y compuerta (sin la señal).

    Parámetros:
        h: hipótesis.
        shap_json: contenido de ``shap_mlp.json``.
        d: ``direccion[variable]`` de ``shap_mlp.json``.
    Retorna:
        dict: {direccion, media_abs_phi, puesto, n_variables, permutacion_mlp, relevante}.
    """
    v = h["variable"]  # variable
    if "direccion" in h:  # numérica
        direccion = {"spearman": d["spearman"], "n": d["n"]}  # ρ(valor, φ) y n explicados
    else:  # categórica
        phi, n = d["phi_medio"], d["n"]  # φ medio y n por categoría
        pm, pr = phi.get(h["mas_riesgosa"]), phi.get(h["referencia"])  # φ̄ de las dos
        direccion = {"phi_mas_riesgosa": pm, "phi_referencia": pr,
                     "n_mas_riesgosa": n.get(h["mas_riesgosa"], 0), "n_referencia": n.get(h["referencia"], 0),
                     "delta_phi": pm - pr if _finito(pm) and _finito(pr) else None}  # Δφ
    imp = {e["nombre"]: e for e in shap_json["importancia_global"]}[v]  # importancia global
    perm = {e["nombre"]: e for e in shap_json["permutacion"]["mlp"]["variables"]}[v]  # permutación del MLP
    permutacion = {"caida_auc_media": perm["caida_auc_media"], "caida_auc_desv": perm["caida_auc_desv"],
                   "k": shap_json["permutacion"]["k"]}  # copia
    return {"direccion": direccion, "media_abs_phi": imp["media_abs_phi"], "puesto": imp["puesto"],
            "n_variables": len(shap_json["importancia_global"]), "permutacion_mlp": permutacion,
            "relevante": relevante(perm["caida_auc_media"], perm["caida_auc_desv"])}  # compuerta


def _bloque_logistica(h: dict, var_logistica: dict) -> dict:
    """Bloque ``logistica`` (contraste sin voto): β y puntos de C3; coincide con la dirección de la hipótesis."""
    if "direccion" in h:  # numérica
        tabla = {f["percentil"]: f["puntos"] for f in var_logistica["tabla"]}  # percentil → puntos
        beta = var_logistica["beta"]  # β estandarizado (β > 0: más riesgo)
        return {"beta": beta, "puntos_p1": tabla.get(1), "puntos_p99": tabla.get(99),
                "coincide": bool(np.sign(beta) == _signo(h))}  # signo de β = s
    cats = {c["categoria"]: c for c in var_logistica["categorias"]}  # categoría → β y puntos
    m, r = cats.get(h["mas_riesgosa"], {}), cats.get(h["referencia"], {})  # las dos
    bm, br = m.get("beta"), r.get("beta")  # β de cada una
    return {"beta_mas_riesgosa": bm, "beta_referencia": br, "puntos_mas_riesgosa": m.get("puntos"),
            "puntos_referencia": r.get("puntos"),
            "coincide": bool(_finito(bm) and _finito(br) and bm > br)}  # más β ⇒ más riesgo


def entrada_hipotesis(h: dict, eda_h: dict, shap_json: dict, var_logistica: dict, test: dict, psi: float) -> dict:
    """Entrada de ``hipotesis.json`` para una hipótesis.

    Parámetros:
        h: hipótesis de ``configs/hipotesis.yaml``.
        eda_h: entrada de ``hipotesis_eda.json`` (evidencia_eda, veredicto_eda).
        shap_json: contenido de ``shap_mlp.json`` (direccion, importancia_global, permutacion).
        var_logistica: entrada de ``puntos_por_variable`` de ``scorecard.json`` para la variable.
        test: bloque de test (con ``senal``) de ``_test_numerica`` o ``_test_categorica``.
        psi: PSI de test frente a train de la variable.
    Retorna:
        dict: {id, enunciado, variable, direccion | mas_riesgosa + referencia, eda, shap, logistica, test, psi_test,
        veredicto, motivo, sensibilidad, cambia_con_sensibilidad, matiz}.
    """
    ev = eda_h["evidencia_eda"]  # evidencia de A
    if "direccion" in h:  # numérica
        senal_eda = senal_direccion(ev["tasa_grupo_bajo"], ev["tasa_grupo_alto"], ev["spearman_grupos"], _signo(h))
    else:  # categórica
        senal_eda = senal_categorica(ev.get("tasa_mas_riesgosa"), ev.get("tasa_referencia"))  # tasas de train
    d_var = shap_json["direccion"][h["variable"]]  # dirección de C4
    sh = _bloque_shap(h, shap_json, d_var)  # SHAP y permutación
    sh["senal"] = senal_shap(d_var, h)  # u oficial
    senales = {"eda": senal_eda, "shap": sh["senal"], "test": test["senal"]}  # votos
    v, motivo = veredicto(senales, sh["relevante"])  # oficial
    sensibilidad = {f"rho_{u}": veredicto({**senales, "shap": senal_shap(d_var, h, umbral=u)}, sh["relevante"])[0]
                    for u in UMBRALES_SENSIBILIDAD}  # otros u
    sensibilidad["sin_compuerta"] = veredicto(senales, sh["relevante"], usar_compuerta=False)[0]  # sin compuerta
    entrada = {**h,  # id, enunciado, variable y dirección (o categorías) del YAML
               "eda": {**ev, "veredicto_eda": eda_h["veredicto_eda"], "senal": senal_eda},  # copia de A
               "shap": sh, "logistica": _bloque_logistica(h, var_logistica), "test": dict(test), "psi_test": psi,
               "veredicto": v, "motivo": motivo, "sensibilidad": sensibilidad,
               "cambia_con_sensibilidad": any(x != v for x in sensibilidad.values())}  # sin matiz aún
    entrada["matiz"] = matiz(entrada)  # plantillas
    return entrada  # entrada completa


# %% [markdown]
# ## Matiz (solo plantillas)
#
# El matiz se arma solo con plantillas a partir de los números (nada escrito a mano por hipótesis): EDA no monótona;
# test no monótono o en otra dirección; discrepancia entre el efecto **marginal** (EDA y test: la tasa de malos por grupo
# de una sola variable) y el **condicional** (SHAP: lo que la variable mueve el logit del MLP con las demás presentes);
# permutación no relevante junto al PSI de test; logística en contra; categoría con pocos explicados; veredicto que
# cambia con la sensibilidad. Siempre termina con la nota de que es una asociación, no causalidad.

# %%
def _num(x, decimales: int = 3) -> str:
    """Número con coma decimal; None → 'n/d'."""
    return "n/d" if not _finito(x) else f"{x:.{decimales}f}".replace(".", ",")  # español


def matiz(entrada: dict) -> str:
    """Texto del matiz de una entrada (plantillas a partir de sus números).

    Parámetros:
        entrada: entrada de ``entrada_hipotesis`` (sin ``matiz``).
    Retorna:
        str: frases separadas por espacio; termina con NOTA_CAUSALIDAD.
    """
    frases = []  # frases del matiz
    eda, sh, te, lg = entrada["eda"], entrada["shap"], entrada["test"], entrada["logistica"]  # bloques
    numerica = "direccion" in entrada  # tipo de hipótesis
    if numerica and not eda["monotona"]:  # forma de la relación en train
        rho = eda["spearman_grupos"]  # ρ de grupos de A (puede faltar)
        rho_abs = abs(rho) if _finito(rho) else None  # |ρ|; None se imprime "n/d"
        frases.append(f"En el EDA de train la relación no es monótona (|ρ| de grupos = "
                      f"{_num(rho_abs, 2)} < {_num(UMBRAL_MONOTONIA, 1)}).")  # A
    if numerica:  # test con cortes de train
        if te["senal"] != "a_favor":  # dirección no sostenida
            frases.append(f"En test (cortes de train) la dirección no se sostiene (señal {te['senal']}; ρ de grupos "
                          f"= {_num(te['spearman_grupos'], 2)}).")  # test
        elif not te["monotona"]:  # dirección sí, forma no
            frases.append(f"En test (cortes de train) la relación no es monótona (ρ de grupos = "
                          f"{_num(te['spearman_grupos'], 2)}).")  # test
    elif te["senal"] != "a_favor":  # categórica en test
        frases.append(f"En test la tasa de {te['mas_riesgosa']['categoria']} ({_num(te['mas_riesgosa']['tasa'])}) "
                      f"no supera la de {te['referencia']['categoria']} ({_num(te['referencia']['tasa'])}).")  # test
    if sh["senal"] != eda["senal"] and sh["senal"] != te["senal"]:  # SHAP discrepa de las dos marginales
        frases.append(f"El efecto marginal (EDA: {eda['senal']}; test: {te['senal']}) y el condicional del MLP "
                      f"(SHAP: {sh['senal']}) no coinciden: SHAP mide el efecto con las demás variables presentes; el "
                      f"EDA y el test miran la variable sola.")  # discrepancia
    elif eda["senal"] != te["senal"]:  # las dos marginales no coinciden entre sí
        frases.append(f"El efecto marginal cambia entre train y test (EDA: {eda['senal']}; test: {te['senal']}); "
                      f"SHAP: {sh['senal']}.")  # la relación marginal no se sostiene en el tiempo
    if not sh["relevante"]:  # compuerta
        p = sh["permutacion_mlp"]  # permutación del MLP
        frases.append(f"La permutación del MLP en test no es relevante (caída de AUC {_num(p['caida_auc_media'], 4)} "
                      f"± {_num(p['caida_auc_desv'], 4)}, k = {p['k']}; se exige media − {Z_PERMUTACION}·desv > 0); "
                      f"PSI de test frente a train = {_num(entrada['psi_test'])}.")  # compuerta y PSI
    if not lg["coincide"]:  # logística en contra
        detalle = (f"β = {_num(lg['beta'])}" if numerica else
                   f"β de {entrada['mas_riesgosa']} = {_num(lg['beta_mas_riesgosa'])}, de {entrada['referencia']} = "
                   f"{_num(lg['beta_referencia'])}")  # β
        frases.append(f"La logística no coincide con la dirección de la hipótesis ({detalle}).")  # contraste
    if not numerica:  # categorías con pocos explicados
        for rol in ("mas_riesgosa", "referencia"):  # las dos
            n = sh["direccion"][f"n_{rol}"]  # n entre los explicados
            if n < N_MIN_CATEGORIA:  # poco preciso
                frases.append(f"La categoría {entrada[rol]} tiene solo n = {n} clientes entre los explicados de SHAP; "
                              f"su φ medio es poco preciso.")  # aviso
    if entrada["cambia_con_sensibilidad"]:  # veredicto frágil
        cambios = [f"{k} → {x}" for k, x in entrada["sensibilidad"].items() if x != entrada["veredicto"]]  # cambios
        frases.append(f"El veredicto cambia con la sensibilidad: {', '.join(cambios)}.")  # sensibilidad
    frases.append(NOTA_CAUSALIDAD)  # cierre fijo
    return " ".join(frases)  # texto


# %% [markdown]
# ## Criterio y construcción de `hipotesis.json`
#
# `construir_hipotesis` lee las entradas, exige que la evidencia (`sha256_particion`, `sha256_preprocesador`) de
# `shap_mlp.json`, de `scorecard.json` y la recalculada con el parquet y el preprocesador coincidan, y arma una entrada
# por hipótesis en el orden del YAML. El `criterio` del JSON repite las reglas de este módulo.

# %%
CRITERIO = {  # reglas y umbrales que se escriben en hipotesis.json
    "reglas": {
        "confirmada": "EDA, SHAP y test a favor y permutación del MLP relevante.",
        "refutada": "Al menos 2 de {EDA, SHAP, test} en contra y ninguna a favor.",
        "no concluyente": "Todo lo demás (fuentes que discrepan, señal nula o permutación no relevante).",
    },
    "senales": {
        "eda": ("Train, efecto marginal (evidencia de A copiada tal cual). Numéricas: a = s·(tasa_alto − tasa_bajo), "
                "b = s·ρ de grupos; a favor si a > 0 y b > 0, en contra si a < 0 y b < 0, si no nula. Categóricas: "
                "a favor si tasa(mas_riesgosa) > tasa(referencia), en contra si <, nula si = o falta. La no "
                "monotonía no quita el voto (la hipótesis afirma una dirección)."),
        "shap": ("MLP, efecto condicional. Numéricas: a favor si s·ρ(valor, φ) ≥ u, en contra si ≤ −u, si no nula. "
                 "Categóricas: Δφ = φ̄(mas_riesgosa) − φ̄(referencia); a favor si > 0, en contra si < 0, nula si = 0 "
                 "o falta una categoría."),
        "test": ("Test fuera de tiempo, efecto marginal: mismo criterio que el EDA con las tasas de malos observadas "
                 "en test y los grupos/cortes de TRAIN (deciles de train con extremos ±∞; conteos: valores de train "
                 "recortados en su p99); vacíos excluidos."),
        "permutacion": "Compuerta: relevante si caída media de AUC − Z·desv de las k repeticiones > 0 (MLP, test).",
    },
    "contraste_sin_voto": "Logística (β y puntos de C3) y PSI de test frente a train (sin umbral, como A).",
    "umbrales": {"umbral_rho_shap": UMBRAL_RHO_SHAP, "z_permutacion": Z_PERMUTACION,
                 "umbral_monotonia": UMBRAL_MONOTONIA, "sensibilidad_rho": list(UMBRALES_SENSIBILIDAD),
                 "n_min_categoria": N_MIN_CATEGORIA, "tol_tasas_train": TOL_TASAS_TRAIN},
    "sensibilidad": ("Veredicto con u = 0,3 y u = 0,7 y con la compuerta desactivada; el oficial es el de los "
                     "umbrales fijados."),
    "marginal_vs_condicional": ("EDA y test miden la tasa de malos por grupo de una sola variable (marginal); SHAP "
                                "mide cuánto mueve la variable el logit del MLP con las demás presentes "
                                "(condicional). Pueden discrepar si la variable está correlacionada con otras."),
    "causalidad": NOTA_CAUSALIDAD,
    "aviso": AVISO_CEGUERA,
}


def _evidencia(rutas: dict, datos: pd.DataFrame, shap_json: dict, scorecard_json: dict) -> dict:
    """Evidencia recalculada; ValueError si no coincide con la de shap_mlp.json y la de scorecard.json."""
    propia = {"sha256_particion": huella_particion(datos["id"].tolist(), datos["split"].tolist()),
              "sha256_preprocesador": huella_archivo(rutas["preprocesador"])}  # núcleo §6
    for nombre, otra in ((ARCHIVO_SHAP, shap_json.get("evidencia")),
                         (ARCHIVO_SCORECARD, scorecard_json.get("evidencia"))):  # las dos fuentes de C
        if otra != propia:  # otra partición u otro preprocesador
            raise ValueError(f"La evidencia de {nombre} ({otra}) no coincide con la recalculada ({propia})")  # dato
    return dict(shap_json["evidencia"])  # la de shap_mlp.json


def construir_hipotesis(rutas: dict) -> dict:
    """Construye el contenido de ``hipotesis.json``.

    Parámetros:
        rutas: claves de RUTAS_POR_DEFECTO (hipotesis_yaml, hipotesis_eda, post_preprocesamiento, shap, scorecard,
            datos, preprocesador).
    Retorna:
        dict: {criterio, hipotesis: [...], evidencia}.
    """
    hipotesis = load_config(rutas["hipotesis_yaml"])["hipotesis"]  # orden y enunciados
    eda = {e["id"]: e for e in load_json(rutas["hipotesis_eda"])}  # evidencia de A por id
    psi = load_json(rutas["post_preprocesamiento"])["psi_vs_train"]["test"]  # PSI de test
    shap_json, scorecard_json = load_json(rutas["shap"]), load_json(rutas["scorecard"])  # C4 y C3
    variables = sorted({h["variable"] for h in hipotesis})  # columnas necesarias
    datos = pd.read_parquet(rutas["datos"], columns=["id", "split", "target", *variables])  # todas las filas
    evidencia = _evidencia(rutas, datos, shap_json, scorecard_json)  # huellas iguales
    train = datos[datos["split"] == "train"]  # cortes y grupos
    test = datos[datos["split"] == "test"]  # solo se evalúa
    log = {e["nombre"]: e for e in scorecard_json["puntos_por_variable"]}  # logística por variable
    salida = []  # entradas en el orden del YAML
    for h in hipotesis:  # una por hipótesis
        e = eda.get(h["id"])  # evidencia de A
        if e is None or e["variable"] != h["variable"]:  # A no la evaluó o es otra variable
            raise ValueError(f"{h['id']}: falta en hipotesis_eda.json o tiene otra variable")  # dato
        bloque_test = (_test_numerica if "direccion" in h else _test_categorica)(h, train, test,
                                                                                 e["evidencia_eda"])  # test
        salida.append(entrada_hipotesis(h, e, shap_json, log[h["variable"]], bloque_test,
                                        psi[h["variable"]]))  # entrada
    return {"criterio": CRITERIO, "hipotesis": salida, "evidencia": evidencia}  # contenido


def escribir(rutas: dict, salida) -> Path:
    """Escribe ``<salida>/hipotesis.json`` (UTF-8, sangría 2, sin NaN/inf) y devuelve la ruta.

    Parámetros:
        rutas: como en ``construir_hipotesis``.
        salida: carpeta de destino.
    Retorna:
        Path: ruta escrita.
    """
    return save_json(Path(salida) / ARCHIVO_HIPOTESIS, construir_hipotesis(rutas))  # atómico


# %% [markdown]
# ## Línea de comandos
#
# Desde `run.py scorecard` con `modelo: mlp`, `hipotesis.json` se escribe DESPUÉS de guardar `scorecard_mlp.json` y
# `shap_mlp.json`: si `scorecard.json` es de otra partición o de otro preprocesador, el ValueError de la evidencia salta
# con los JSON de SHAP ya guardados.
#
# `python -m src.scorecard.hipotesis [--salida CARPETA]`: lee `shap_mlp.json` y `scorecard.json` de la carpeta,
# escribe allí `hipotesis.json` e imprime una tabla breve (señales, veredicto y si cambia con la sensibilidad) y el
# tiempo.

# %%
def main(argv=None) -> int:
    """Escribe ``hipotesis.json`` e imprime un resumen; devuelve 0.

    Parámetros:
        argv: argumentos (None → sys.argv).
    Retorna:
        int: 0 si todo salió bien.
    """
    for flujo in (sys.stdout, sys.stderr):  # como src/common/cli.py: → y ρ en la consola de Windows
        if hasattr(flujo, "reconfigure"):  # flujos de texto reales (no los de pytest)
            flujo.reconfigure(encoding="utf-8")  # UTF-8
    parser = argparse.ArgumentParser(description="Cierre de hipótesis (Parte C §3.4, C6)")  # CLI
    parser.add_argument("--salida", default=CARPETA_POR_DEFECTO,
                        help="carpeta con shap_mlp.json y scorecard.json; allí se escribe hipotesis.json")  # carpeta
    args = parser.parse_args(argv)  # argumentos
    inicio = time.perf_counter()  # reloj
    carpeta = Path(args.salida)  # carpeta
    rutas = {**RUTAS_POR_DEFECTO, "shap": carpeta / ARCHIVO_SHAP, "scorecard": carpeta / ARCHIVO_SCORECARD}  # entradas
    resultado = construir_hipotesis(rutas)  # contenido
    ruta = save_json(carpeta / ARCHIVO_HIPOTESIS, resultado)  # UTF-8, sangría 2, sin NaN/inf
    for h in resultado["hipotesis"]:  # una línea por hipótesis
        print(f"{h['id']:<4} {h['variable']:<26} EDA {h['eda']['senal']:<9} SHAP {h['shap']['senal']:<9} "
              f"test {h['test']['senal']:<9} relevante {str(h['shap']['relevante']):<5} → {h['veredicto']}"
              f"{' (cambia con la sensibilidad)' if h['cambia_con_sensibilidad'] else ''}")  # resumen
    print(f"Escrito: {ruta.as_posix()} ({time.perf_counter() - inicio:.1f} s)")  # ruta y tiempo
    return 0  # éxito


if __name__ == "__main__":  # python -m src.scorecard.hipotesis
    raise SystemExit(main())  # código de salida
