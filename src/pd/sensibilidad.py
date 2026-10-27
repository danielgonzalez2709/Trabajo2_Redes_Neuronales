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
# # Sensibilidad de la PD con `grade` e `int_rate` (B10, B-pd §3.8)
#
# `python run.py pd-sensibilidad --config configs/pd/sensibilidad.yaml` mide cuánto mejoraría la PD si el modelo
# viera la evaluación de riesgo de Lending Club: `grade` ("LC assigned loan grade") e `int_rate` ("Interest Rate on
# the loan"), citadas de `configs/features_sensibilidad.yaml`. Es un análisis **aparte**: no va a la app ni reemplaza
# al modelo principal.
#
# 1. **Datos** (puros): une `loans_pd.parquet` con `sensibilidad_lc.parquet` por `id` (mismas filas y orden) y ajusta
#    un preprocesador propio (el `Preprocesador` de A) **solo con train** sobre las 15 + 2 variables.
# 2. **Rutas y config** (puras): un `RutasPD` por modo que apunta al parquet unido y a ese preprocesador, y la config
#    que recibe `correr_pd` en cada modo (`reusar` = misma arquitectura que el modelo principal; `optuna` =
#    arquitectura buscada con las 17 variables).
# 3. **Comparación** (pura): principal frente a cada modo, con Δ, presupuesto, calibración global y la prueba de que
#    la partición es la misma.
# 4. **Orquestación**: las lecturas, escrituras y piezas del pipeline de B (`correr_pd`, `hpo.buscar`,
#    `comprobar_antes`) se inyectan; el pipeline no se copia, así que la diferencia se atribuye a las dos variables.
#
# Nunca se escribe en `results/data/pd/`, `models/pd_*` ni `configs/pd/mlp_final.yaml` (lo usan C y E).
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable
# sin efectos al importar.

# %%
"""Sensibilidad de la PD con grade e int_rate (B10): unión, preprocesador propio, modos reusar/optuna y comparación."""

import copy  # copias de las configs (las funciones no mutan la del llamador)
import functools  # partial de cargar_datos, correr_mlp y comprobar_antes con los argumentos del modo
import json  # validación del preprocesador antes de escribir (sin NaN)
import sys  # stderr de los errores de run.py pd-sensibilidad
import time  # tiempo total del subcomando
from pathlib import Path  # rutas

import numpy as np  # medias de la calibración global
import optuna  # nivel de log de la búsqueda (como run.py pd-hpo)
import pandas as pd  # parquets de A y pred_test

from src.common.config import load_config  # mismo cargador que el CLI
from src.common.preprocesamiento import Preprocesador  # preprocesador de A (núcleo §5.2)
from src.common.results import load_json, save_json, to_jsonable  # JSON sin NaN y escritura atómica (núcleo §5.4)
from src.pd import experimento as ex  # pipeline de B que se reutiliza (B3, B5, B6)
from src.pd import hpo  # búsqueda Optuna que se reutiliza (B4)
from src.pd.logistica import bloque_logistica  # validación del bloque logistica (B3)

__all__ = [  # interfaz pública (B10); lo demás es interno
    "MODOS", "MODO_REUSAR", "MODO_OPTUNA", "PROHIBIDAS", "PREFIJOS_PROHIBIDOS", "ADICIONALES_PERMITIDAS",  # reglas
    "ARCHIVO_COMPARACION", "ARCHIVO_MLP_FINAL", "PREFIJO_ERROR",  # nombres
    "unir_sensibilidad", "features_sensibilidad", "ajustar_preprocesador",  # datos
    "rutas_sensibilidad", "cfg_modo", "cfg_hpo_sensibilidad", "validar_cfg_sensibilidad",  # rutas y config
    "comparacion",  # comparación
    "leer_archivo", "escribir_archivo", "preparar", "escribir_comparacion", "correr_sensibilidad", "manejador",  # E/S
]


# %% [markdown]
# ## Constantes: modos y nombres
#
# - `MODOS`: `reusar` (misma arquitectura que el modelo principal: rejilla de C de B3 y `mlp` de `mlp_final.yaml`) y
#   `optuna` (arquitectura buscada con las 17 variables: TPE nuevo con el presupuesto y el espacio de B-pd §3.3);
# - claves exactas de la raíz, del bloque `sensibilidad` y del bloque `hpo` de `configs/pd/sensibilidad.yaml` (el
#   `destino` y la `salida` de la búsqueda se fijan dentro de `optuna/`, nunca en `configs/pd/mlp_final.yaml`);
# - modelos y métricas de test que se comparan (B-pd §3.8) y el presupuesto del núcleo §6.

# %%
MODO_REUSAR = "reusar"  # misma arquitectura que el modelo principal
MODO_OPTUNA = "optuna"  # arquitectura buscada con las 17 variables
MODOS = (MODO_REUSAR, MODO_OPTUNA)  # modos válidos, en el orden en que se informan
CLAVES_RAIZ = ("base_seed", "datos", "logistica", "mlp", "busqueda", "semillas", "hpo", "sensibilidad")  # raíz
CLAVES_SENSIBILIDAD = ("modos", "loans_pd", "features", "datos_lc", "adicionales", "config_datos", "union",  # entradas
                       "preprocesador", "salida", "modelos", "principal")  # salidas y principal
CLAVES_RUTA = CLAVES_SENSIBILIDAD[1:]  # todas menos modos son rutas (texto no vacío)
CLAVES_HPO_SENSIBILIDAD = ("n_ensayos", "espacio")  # destino y salida los fija cfg_hpo_sensibilidad
ARCHIVO_COMPARACION = "comparacion.json"  # {salida}/comparacion.json (B-pd §3.8)
ARCHIVO_MLP_FINAL = "mlp_final.yaml"  # {salida}/optuna/mlp_final.yaml (nunca configs/pd/mlp_final.yaml)
FUENTE_PRINCIPAL = "principal"  # nombre del modelo principal en comparacion.json
MODELOS_COMPARADOS = (ex.MODELO_LOGISTICA, ex.MODELO_MLP)  # logística y MLP
METRICAS_COMPARADAS = ("auc_test", "gini_test", "ks_test", "brier_test")  # métricas de test (B-pd §3.8)
CLAVES_PRESUPUESTO = ("n_parametros", "n_configuraciones_busqueda", "tiempo_total_s")  # núcleo §6
CLAVES_RESUMEN = ("media", "desv", "n")  # de cada métrica de summary.json que pasan a la comparación
PREFIJO_ERROR = "run.py pd-sensibilidad: error: "  # inicio de la línea de error en stderr (como run.py pd)


# %% [markdown]
# ## Constantes: variables prohibidas (núcleo §5.7 y Parte A §3.4)
#
# `PROHIBIDAS` y `PREFIJOS_PROHIBIDOS` repiten la lista de `tests/test_fuga.py` (posteriores a la originación del
# núcleo §5.7, identificadores, texto libre, puntaje y precio de Lending Club y variables sensibles): `src/` no importa
# de `tests/` y una prueba exige que coincidan. `ADICIONALES_PERMITIDAS` son las **únicas** prohibidas del modelo
# principal que este análisis admite, y solo como adicionales (B-pd §3.8): `sub_grade` (redundante con `grade`) e
# `installment` (depende de la tasa) no entran (A §3.4).

# %%
PROHIBIDAS = frozenset({  # = tests/test_fuga.py::PROHIBIDAS
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",  # pagos
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d",  # pagos
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "pymnt_plan",  # pagos posteriores
    "id", "member_id", "url", "desc", "title", "emp_title",  # identificadores y texto libre
    "grade", "sub_grade", "int_rate", "installment", "funded_amnt", "funded_amnt_inv", "initial_list_status",  # LC
    "zip_code", "addr_state",  # sensibles
})
PREFIJOS_PROHIBIDOS = ("total_pymnt", "total_rec_", "out_prncp", "last_pymnt_")  # = tests/test_fuga.py (núcleo §5.7)
ADICIONALES_PERMITIDAS = ("grade", "int_rate")  # las de configs/features_sensibilidad.yaml (B-pd §3.8)


# %% [markdown]
# ## Datos: unión con las variables de Lending Club
#
# `sensibilidad_lc.parquet` debe tener **los mismos ids en el mismo orden** que `loans_pd.parquet` (A lo garantiza y
# aquí se comprueba): la unión es por posición, y por eso conserva el orden y el índice de `loans_pd`. Error
# (`ValueError`) si los ids difieren en número, conjunto u orden, si hay ids repetidos, si una adicional falta en
# `sens`, ya existe en `loans_pd` o tiene vacíos (NA o texto en blanco). No muta las entradas.

# %%
def _vacios(serie: pd.Series) -> int:
    """Número de vacíos: NA y, si no es numérica, texto en blanco."""
    if pd.api.types.is_numeric_dtype(serie):  # int_rate
        return int(serie.isna().sum())  # NaN
    texto = serie.astype("string")  # grade (texto)
    return int((texto.isna() | (texto.str.strip() == "")).sum())  # NA o en blanco


def unir_sensibilidad(loans_pd: pd.DataFrame, sens: pd.DataFrame, adicionales: list[str]) -> pd.DataFrame:
    """loans_pd con las columnas ``adicionales`` de ``sens`` al final (mismas filas y orden); ValueError si no casan."""
    if not adicionales:  # sin adicionales no hay análisis
        raise ValueError("unir_sensibilidad necesita al menos una variable adicional")  # vacío
    faltan = [c for c in [ex.COLUMNA_ID, *adicionales] if c not in sens.columns]  # columnas de sens
    if faltan:  # p. ej. un parquet ajeno
        raise ValueError(f"sensibilidad_lc no tiene las columnas {faltan}")  # cuáles
    repetidas = [c for c in adicionales if c in loans_pd.columns]  # ya en loans_pd
    if repetidas:  # se pisaría una columna
        raise ValueError(f"las columnas {repetidas} ya existen en loans_pd")  # cuáles
    ids_pd = loans_pd[ex.COLUMNA_ID].astype(str).to_numpy()  # ids de A, en orden
    ids_lc = sens[ex.COLUMNA_ID].astype(str).to_numpy()  # ids de sensibilidad_lc, en orden
    if len(ids_pd) != len(ids_lc) or not (ids_pd == ids_lc).all():  # número, conjunto u orden
        raise ValueError("loans_pd y sensibilidad_lc deben tener el mismo id fila a fila (mismas filas y orden)")  # A
    if pd.Series(ids_pd).duplicated().any():  # id repetido
        raise ValueError("hay ids repetidos en loans_pd/sensibilidad_lc")  # unión ambigua
    con_vacios = {c: _vacios(sens[c]) for c in adicionales if _vacios(sens[c])}  # vacíos por columna
    if con_vacios:  # A entrega sin vacíos
        raise ValueError(f"variables adicionales con vacíos: {con_vacios}")  # cuántos
    union = loans_pd.copy()  # no muta loans_pd
    for c in adicionales:  # en el orden pedido
        union[c] = sens[c].to_numpy()  # por posición (ids ya comprobados)
    return union  # mismas filas, orden e índice que loans_pd


# %% [markdown]
# ## Datos: las 17 variables
#
# Las 15 de `configs/features.yaml` y luego las adicionales de `configs/features_sensibilidad.yaml`, en ese orden
# (copias). Se validan como las principales: `disponible_en_originacion` debe ser exactamente `true`, sin nombres
# repetidos y ninguna prohibida del núcleo §5.7 (por nombre o por prefijo); las principales no pueden ser ninguna de
# `PROHIBIDAS` y las adicionales solo pueden ser `grade` e `int_rate`. Cualquier falla → `ValueError`.

# %%
def _prohibida(nombre: str, permitidas=()) -> bool:
    """True si ``nombre`` está en PROHIBIDAS (salvo ``permitidas``) o empieza por un prefijo prohibido."""
    return (nombre in PROHIBIDAS and nombre not in permitidas) or nombre.startswith(PREFIJOS_PROHIBIDOS)  # §5.7


def features_sensibilidad(features: list[dict], adicionales: list[dict]) -> list[dict]:
    """Copias de las principales y luego las adicionales, validadas; ValueError ante cualquier falla."""
    if not adicionales:  # sin adicionales no hay análisis
        raise ValueError("features_sensibilidad necesita las variables adicionales (grade, int_rate)")  # vacío
    ajenas = [a["nombre"] for a in adicionales if a["nombre"] not in ADICIONALES_PERMITIDAS]  # p. ej. sub_grade
    if ajenas:  # B-pd §3.8: solo grade e int_rate
        raise ValueError(f"adicionales fuera de {list(ADICIONALES_PERMITIDAS)}: {ajenas} (B-pd §3.8, A §3.4)")  # cuáles
    todas = [*features, *adicionales]  # 15 + 2, en orden
    nombres = [f["nombre"] for f in todas]  # nombres en orden
    repetidos = sorted({n for n in nombres if nombres.count(n) > 1})  # nombres repetidos
    if repetidos:  # una variable dos veces
        raise ValueError(f"variables repetidas: {repetidos}")  # cuáles
    no_disponibles = [f["nombre"] for f in todas if f.get("disponible_en_originacion") is not True]  # no en origen
    if no_disponibles:  # núcleo §5.7
        raise ValueError(f"variables no disponibles en la originación: {no_disponibles}")  # cuáles
    prohibidas = [f["nombre"] for f in features if _prohibida(f["nombre"])]  # principales prohibidas
    prohibidas += [a["nombre"] for a in adicionales if _prohibida(a["nombre"], ADICIONALES_PERMITIDAS)]  # adicionales
    if prohibidas:  # fuga de información
        raise ValueError(f"variables prohibidas (núcleo §5.7): {prohibidas}")  # cuáles
    return copy.deepcopy(todas)  # copias: el llamador no comparte los dicts


# %% [markdown]
# ## Datos: preprocesador propio, ajustado solo con train
#
# El `Preprocesador` de A (núcleo §5.2) con los percentiles de recorte de `configs/datos.yaml`, ajustado **solo** con
# las filas `split == "train"` (como `src/lgd/preparar.py::ajustar_preprocesador`): val y test no influyen. Sin filas
# de train → `ValueError`.

# %%
def ajustar_preprocesador(df: pd.DataFrame, features: list[dict], percentiles: tuple[float, float]) -> Preprocesador:
    """Preprocesador de A ajustado SOLO con las filas split == "train" de ``df``."""
    train = df[df[ex.COLUMNA_SPLIT] == "train"]  # solo train
    if train.empty:  # se ajustaría con nada (o con val/test por error)
        raise ValueError("ajustar_preprocesador: no hay filas con split == 'train'")  # rechazo
    return Preprocesador(percentiles).fit(train, features)  # ajuste (núcleo §5.2)


# %% [markdown]
# ## Rutas: validación del modo
#
# Solo `reusar` y `optuna` (`MODOS`); cualquier otro → `ValueError`.

# %%
def _validar_modo(modo) -> str:
    """``modo`` si está en MODOS; si no, ValueError."""
    if modo not in MODOS:  # p. ej. "pd"
        raise ValueError(f"modo de sensibilidad desconocido {modo!r}; válidos: {list(MODOS)}")  # rechazo
    return modo  # válido


# %% [markdown]
# ## Rutas: protección del modelo principal
#
# Ninguna ruta del modo puede coincidir con las del principal (`RUTAS_PD`, resueltas desde el directorio actual): los
# resultados no pueden ser `results/data/pd` ni estar dentro, los modelos no pueden ser `models` (se pisarían
# `pd_logistica.joblib` y `pd_mlp.pt`), y el parquet unido y el preprocesador no pueden ser los de A ni los insumos de
# la unión. Si no → `ValueError`.

# %%
def _rechazar_protegidas(rutas: ex.RutasPD, insumos: list[Path]) -> None:
    """ValueError si ``rutas`` pisaría o anidaría las del modelo principal o los insumos de la unión."""
    principal = ex.RUTAS_PD  # rutas del modelo principal
    res, res_pd = rutas.resultados.resolve(), principal.resultados.resolve()  # carpetas de resultados
    if res == res_pd or res_pd in res.parents:  # results/data/pd o dentro
        raise ValueError(f"la salida {rutas.resultados} pisaría o anidaría {principal.resultados}")  # rechazo
    if rutas.modelos.resolve() == principal.modelos.resolve():  # models: pisaría pd_*.{joblib,pt}
        raise ValueError(f"los modelos {rutas.modelos} pisarían los del modelo principal")  # rechazo
    protegidas = {p.resolve() for p in (principal.parquet, principal.preprocesador, *insumos)}  # entradas
    for ruta in (rutas.parquet, rutas.preprocesador):  # lo que escribe preparar
        if ruta.resolve() in protegidas:  # pisaría un archivo de A
            raise ValueError(f"{ruta} pisaría un archivo de entrada (loans_pd, sensibilidad_lc o preprocesador)")  # A


# %% [markdown]
# ## Rutas de cada modo
#
# `RutasPD(union, preprocesador, {salida}/{modo}, {modelos}/{modo})`: el mismo objeto que usa todo el pipeline de B,
# con el parquet unido y el preprocesador de sensibilidad como entradas.

# %%
def rutas_sensibilidad(cfg: dict, modo: str) -> ex.RutasPD:
    """RutasPD del modo con el parquet unido y el preprocesador de sensibilidad; ValueError si pisa el principal."""
    _validar_modo(modo)  # reusar u optuna
    s = cfg["sensibilidad"]  # bloque de rutas
    rutas = ex.RutasPD(Path(s["union"]), Path(s["preprocesador"]), Path(s["salida"]) / modo,
                       Path(s["modelos"]) / modo)  # carpetas por modo
    _rechazar_protegidas(rutas, [Path(s["loans_pd"]), Path(s["datos_lc"])])  # nunca las del principal
    return rutas  # rutas del modo


# %% [markdown]
# ## Config de cada modo
#
# Copia de la config sin `hpo` ni `sensibilidad` (lo que recibe `correr_pd`) con `salida` y `modelos` del modo:
#
# - `reusar`: `busqueda = {tipo: sin_busqueda}` (la búsqueda del principal se hizo con otros datos: con su bloque la
#   huella no coincidiría y `presupuesto_busqueda` daría `ValueError`); `mlp` = el de la config (= `mlp_final.yaml`);
# - `optuna`: `mlp` y `busqueda` del `mlp_final.yaml` de su búsqueda (`mlp_final_optuna`, obligatorio), así que el
#   presupuesto cuenta los ensayos y el tiempo de esa búsqueda.
#
# `reusar` con `mlp_final_optuna` → `ValueError` (se ignoraría en silencio). No muta `cfg`.

# %%
def cfg_modo(cfg: dict, modo: str, mlp_final_optuna: dict | None = None) -> dict:
    """Config que recibe correr_pd en ``modo`` (copia, con salida y modelos del modo)."""
    _validar_modo(modo)  # reusar u optuna
    c = {k: copy.deepcopy(v) for k, v in cfg.items() if k not in ("hpo", "sensibilidad")}  # solo lo de correr_pd
    if modo == MODO_REUSAR:  # misma arquitectura que el principal
        if mlp_final_optuna is not None:  # no se usaría
            raise ValueError("el modo reusar no recibe mlp_final_optuna")  # rechazo
        c["busqueda"] = {"tipo": ex.TIPO_SIN_BUSQUEDA}  # sin presupuesto de búsqueda
    else:  # optuna
        if mlp_final_optuna is None:  # falta la búsqueda del modo
            raise ValueError("el modo optuna necesita el mlp_final.yaml de su búsqueda")  # rechazo
        c["mlp"] = copy.deepcopy(mlp_final_optuna["mlp"])  # ganador de la búsqueda con 17 variables
        c["busqueda"] = copy.deepcopy(mlp_final_optuna["busqueda"])  # su procedencia y presupuesto
    rutas = rutas_sensibilidad(cfg, modo)  # carpetas del modo
    c["salida"], c["modelos"] = rutas.resultados.as_posix(), rutas.modelos.as_posix()  # = las inyectadas
    return c  # config del modo


# %% [markdown]
# ## Config de la búsqueda del modo optuna
#
# La config para `hpo.buscar`: el bloque `hpo` con los `n_ensayos` y el `espacio` de la config (= `hpo.yaml`) y
# `destino` = `{salida}/optuna/mlp_final.yaml`, `salida` = `{salida}/optuna`. Nunca `configs/pd/mlp_final.yaml`.

# %%
def cfg_hpo_sensibilidad(cfg: dict) -> dict:
    """Config de hpo.buscar con destino y salida dentro de optuna/ (copia)."""
    rutas = rutas_sensibilidad(cfg, MODO_OPTUNA)  # carpetas de optuna
    destino = rutas.resultados / ARCHIVO_MLP_FINAL  # mlp_final.yaml del modo
    if destino.resolve() == ex.RUTA_MLP_FINAL.resolve():  # nunca el del principal
        raise ValueError(f"hpo.destino no puede ser {ex.RUTA_MLP_FINAL}")  # rechazo
    c = {k: copy.deepcopy(v) for k, v in cfg.items() if k != "sensibilidad"}  # copia sin el bloque propio
    c["hpo"] = {"n_ensayos": c["hpo"]["n_ensayos"], "destino": destino.as_posix(),  # presupuesto y YAML
                "salida": rutas.resultados.as_posix(), "espacio": c["hpo"]["espacio"]}  # JSON y espacio
    return c  # config de la búsqueda


# %% [markdown]
# ## Validación de la config (antes de leer datos)
#
# Raíz con **exactamente** `CLAVES_RAIZ`; `sensibilidad` con exactamente `CLAVES_SENSIBILIDAD` (rutas = texto no
# vacío); `modos` = lista no vacía, sin repetidos, ⊆ `MODOS`; `hpo` con exactamente `n_ensayos` y `espacio`;
# `busqueda` = `{tipo: sin_busqueda}`; y las mismas reglas del pipeline: `datos.muestra`, bloques `mlp`, `semillas`,
# `base_seed` y `logistica`, rutas de cada modo y, si corre optuna, el bloque `hpo` armado. Devuelve los modos.

# %%
def _claves_exactas(bloque, claves, nombre: str) -> None:
    """ValueError si ``bloque`` no es un mapeo con exactamente ``claves``."""
    if not isinstance(bloque, dict):  # bloque reemplazado por un valor
        raise ValueError(f"{nombre} debe ser un mapeo; se recibió {bloque!r}")  # tipo
    faltan, sobran = set(claves) - set(bloque), set(bloque) - set(claves)  # diferencias
    if faltan or sobran:  # el bloque debe ser exacto
        raise ValueError(f"{nombre} inválido: faltan {sorted(faltan)}, sobran {sorted(sobran)}")  # cuáles


def validar_cfg_sensibilidad(cfg: dict) -> list[str]:
    """Modos validados; cualquier falla de la config → ValueError (antes de leer datos)."""
    _claves_exactas(cfg, CLAVES_RAIZ, "La config de pd-sensibilidad")  # raíz
    s = cfg["sensibilidad"]  # bloque propio
    _claves_exactas(s, CLAVES_SENSIBILIDAD, "El bloque sensibilidad")  # claves
    modos = s["modos"]  # modos pedidos
    if not isinstance(modos, list) or not modos or len(set(map(str, modos))) != len(modos):  # lista sin repetidos
        raise ValueError(f"sensibilidad.modos debe ser una lista no vacía y sin repetidos; se recibió {modos!r}")  # forma
    for modo in modos:  # cada modo
        _validar_modo(modo)  # reusar u optuna
    for clave in CLAVES_RUTA:  # rutas
        if not isinstance(s[clave], str) or not s[clave].strip():  # '' o un número no son rutas
            raise ValueError(f"sensibilidad.{clave} debe ser una ruta (texto no vacío); se recibió {s[clave]!r}")  # tipo
    _claves_exactas(cfg["hpo"], CLAVES_HPO_SENSIBILIDAD, "El bloque hpo de pd-sensibilidad")  # sin destino/salida
    if cfg["busqueda"] != {"tipo": ex.TIPO_SIN_BUSQUEDA}:  # la del principal no vale aquí
        raise ValueError(f"busqueda debe ser {{tipo: {ex.TIPO_SIN_BUSQUEDA}}} (B-pd §3.8); se recibió "
                         f"{cfg['busqueda']!r}")  # rechazo
    ex.muestra_de(cfg)  # datos.muestra y base_seed (B6)
    ex.validar_cfg_mlp_final(cfg_modo(cfg, MODO_REUSAR))  # semillas, base_seed, mlp y busqueda (B5)
    bloque_logistica(cfg)  # logística (B3)
    for modo in modos:  # rutas de cada modo
        rutas_sensibilidad(cfg, modo)  # ValueError si pisan el principal
    if MODO_OPTUNA in modos:  # la búsqueda se valida antes de leer datos
        hpo.validar_cfg(cfg_hpo_sensibilidad(cfg))  # reglas de B4
    return list(modos)  # modos validados


# %% [markdown]
# ## Comparación: una fuente
#
# Una fuente es `{"summary": dict, "pred_test": DataFrame}` (el principal o un modo). De cada modelo se toman las
# métricas de test (`media`, `desv`, `n`), el presupuesto del núcleo §6, la huella de la partición y la calibración
# global de test:
#
# $$\bar p_{\text{test}} = \frac1{n}\sum_{i \in \text{test}} \hat p_i, \qquad \bar y_{\text{test}} = \frac1{n}
# \sum_{i \in \text{test}} y_i,$$
#
# con $\hat p_i$ la columna `pred_<modelo>` de `pred_test.parquet` (MLP = semilla mediana). Falta algo → `ValueError`.

# %%
def _entrada_modelo(summary, modelo: str, fuente: str) -> dict:
    """Entrada ``modelo`` de ``summary`` con las claves que se comparan; ValueError si falta algo."""
    if not isinstance(summary, dict) or modelo not in summary:  # modelo ausente
        raise ValueError(f"{fuente}: summary.json no tiene el modelo {modelo!r}")  # cuál
    entrada = summary[modelo]  # entrada del modelo
    faltan = [k for k in (*CLAVES_PRESUPUESTO, "metricas", "evidencia") if k not in entrada]  # claves
    faltan += [m for m in METRICAS_COMPARADAS if m not in entrada.get("metricas", {})]  # métricas de test
    if "sha256_particion" not in entrada.get("evidencia", {}):  # huella de la partición
        faltan.append("evidencia.sha256_particion")  # se nombra
    if faltan:  # summary incompleto
        raise ValueError(f"{fuente}: la entrada {modelo!r} de summary.json no tiene {faltan}")  # cuáles
    return entrada  # entrada completa


def _calibracion(pred_test: pd.DataFrame, modelo: str, fuente: str) -> dict:
    """{pd_media_test, tasa_test} de la columna pred_<modelo> y de y; ValueError si faltan."""
    columna = ex.PREFIJO_PRED + modelo  # pred_logistica o pred_mlp
    if columna not in pred_test.columns or ex.COLUMNA_Y not in pred_test.columns:  # columnas
        raise ValueError(f"{fuente}: pred_test.parquet no tiene {columna!r} o {ex.COLUMNA_Y!r}")  # cuál
    return {"pd_media_test": float(np.mean(pred_test[columna])),  # p̄ de test
            "tasa_test": float(np.mean(pred_test[ex.COLUMNA_Y]))}  # ȳ de test


# %% [markdown]
# ## Comparación: principal frente a cada modo
#
# Para cada modelo $m$ y métrica de test $k$, con $\bar x$ la media entre semillas (n = 1 en la logística, desv 0):
#
# $$\Delta_{m,k}^{\text{modo}} = \bar x_{m,k}^{\text{modo}} - \bar x_{m,k}^{\text{principal}}$$
#
# (positivo en AUC, Gini y KS = mejor; negativo en Brier = mejor). Además, el presupuesto (`n_parametros`,
# `n_configuraciones_busqueda`, `tiempo_total_s`), la calibración global y `misma_particion` = todas las
# `sha256_particion` (de los dos modelos de cada fuente) iguales a la del principal. Solo los modos presentes, en el
# orden de `MODOS`; modo desconocido o ninguno → `ValueError`. Pura: no lee ni escribe ni muta las entradas.

# %%
def comparacion(principal: dict, por_modo: dict) -> dict:
    """Contenido de comparacion.json (B-pd §3.8): métricas con Δ, presupuesto, calibración y misma_particion."""
    if not por_modo:  # nada que comparar
        raise ValueError("comparacion necesita al menos un modo")  # vacío
    for modo in por_modo:  # modos recibidos
        _validar_modo(modo)  # reusar u optuna
    modos = [m for m in MODOS if m in por_modo]  # en el orden de MODOS
    fuentes = {FUENTE_PRINCIPAL: principal, **{m: por_modo[m] for m in modos}}  # principal primero
    metricas, presupuesto, calibracion, huellas = {}, {}, {}, {}  # bloques del JSON
    for modelo in MODELOS_COMPARADOS:  # logística y MLP
        entradas = {f: _entrada_modelo(d["summary"], modelo, f) for f, d in fuentes.items()}  # validadas
        metricas[modelo] = {}  # por métrica
        for k in METRICAS_COMPARADAS:  # métricas de test
            base = entradas[FUENTE_PRINCIPAL]["metricas"][k]["media"]  # media del principal
            metricas[modelo][k] = {f: {c: e["metricas"][k][c] for c in CLAVES_RESUMEN}  # media, desv y n
                                   for f, e in entradas.items()}  # de cada fuente
            for m in modos:  # Δ = sensibilidad − principal
                metricas[modelo][k][m]["delta"] = float(entradas[m]["metricas"][k]["media"] - base)  # con signo
        presupuesto[modelo] = {f: {c: e[c] for c in CLAVES_PRESUPUESTO} for f, e in entradas.items()}  # núcleo §6
        calibracion[modelo] = {f: _calibracion(d["pred_test"], modelo, f) for f, d in fuentes.items()}  # p̄ y ȳ
        huellas[modelo] = {f: e["evidencia"]["sha256_particion"] for f, e in entradas.items()}  # partición
    todas = {h for por_fuente in huellas.values() for h in por_fuente.values()}  # huellas distintas
    return copy.deepcopy({  # copia: nada comparte objetos con las entradas
        "modos": modos,  # modos comparados
        "modelos": list(MODELOS_COMPARADOS),  # logística y MLP
        "metricas_test": list(METRICAS_COMPARADAS),  # AUC, Gini, KS, Brier de test
        "misma_particion": len(todas) == 1,  # comparación justa: misma partición que el principal
        "sha256_particion": huellas,  # {modelo: {fuente: huella}}
        "metricas": metricas,  # {modelo: {métrica: {fuente: {media, desv, n[, delta]}}}}
        "presupuesto": presupuesto,  # {modelo: {fuente: {n_parametros, n_configuraciones_busqueda, tiempo_total_s}}}
        "calibracion_global": calibracion,  # {modelo: {fuente: {pd_media_test, tasa_test}}}
    })


# %% [markdown]
# ## E/S por defecto: lectura
#
# Por la extensión: `.parquet` → `pd.read_parquet`, `.yaml`/`.yml` → `load_config` (el cargador del CLI), `.json` →
# `load_json`. Otra → `ValueError`.

# %%
def leer_archivo(ruta):
    """Contenido de ``ruta`` según su extensión (parquet, yaml o json)."""
    sufijo = Path(ruta).suffix.lower()  # extensión
    if sufijo == ".parquet":  # datos
        return pd.read_parquet(ruta)  # DataFrame
    if sufijo in (".yaml", ".yml"):  # configs
        return load_config(ruta)  # mismo cargador que el CLI
    if sufijo == ".json":  # resultados
        return load_json(ruta)  # JSON
    raise ValueError(f"no sé leer {ruta} (extensión {sufijo!r})")  # extensión desconocida


# %% [markdown]
# ## E/S por defecto: escritura atómica
#
# Un `DataFrame` va a parquet con la escritura atómica de `src/pd/experimento.py` (temporal + `os.replace`, permisos
# 0o644); un `dict` va a JSON con `save_json` (atómico, sin NaN). Otro tipo → `ValueError`.

# %%
def escribir_archivo(ruta, obj) -> None:
    """Escribe ``obj`` en ``ruta`` de forma atómica (DataFrame → parquet, dict → JSON)."""
    if isinstance(obj, pd.DataFrame):  # la unión
        ex._escribir_parquet_atomico(Path(ruta), obj)  # temporal + os.replace (B3)
    elif isinstance(obj, dict):  # preprocesador y comparación
        save_json(ruta, obj)  # atómico, sin NaN
    else:  # tipo inesperado
        raise ValueError(f"no sé escribir un {type(obj).__name__} en {ruta}")  # rechazo


# %% [markdown]
# ## Preparación: unión y preprocesador
#
# Lee con `leer` los dos parquets de A y los tres YAML (15 variables, adicionales y percentiles de
# `configs/datos.yaml`), valida las 17 variables, une, ajusta el preprocesador **solo con train**, comprueba que su
# JSON es serializable sin NaN (antes de escribir nada) y escribe con `escribir` la unión y el preprocesador.

# %%
def preparar(cfg: dict, leer=None, escribir=None) -> dict:
    """Escribe la unión y el preprocesador de sensibilidad; devuelve {filas, variables, union, preprocesador}."""
    leer = leer_archivo if leer is None else leer  # al llamar
    escribir = escribir_archivo if escribir is None else escribir  # al llamar
    s = cfg["sensibilidad"]  # rutas
    adicionales = leer(s["adicionales"])["adicionales"]  # grade e int_rate (A)
    features = features_sensibilidad(leer(s["features"])["features"], adicionales)  # 15 + 2 validadas
    percentiles = tuple(leer(s["config_datos"])["preprocesamiento"]["percentiles_recorte"])  # de A
    union = unir_sensibilidad(leer(s["loans_pd"]), leer(s["datos_lc"]), [a["nombre"] for a in adicionales])  # unión
    pre = ajustar_preprocesador(union, features, percentiles).to_dict()  # solo train
    json.dumps(to_jsonable(pre), allow_nan=False)  # sin NaN antes de escribir nada
    escribir(s["union"], union)  # parquet unido (atómico)
    escribir(s["preprocesador"], pre)  # preprocesador (atómico)
    filas = {sp: int((union[ex.COLUMNA_SPLIT] == sp).sum()) for sp in ex.SPLITS}  # filas por split
    return {"filas": filas, "variables": [f["nombre"] for f in features], "union": s["union"],
            "preprocesador": s["preprocesador"]}  # resumen


# %% [markdown]
# ## Comparación en disco
#
# Lee con `leer` el `summary.json` del principal y el `pred_test.parquet` de su misma carpeta, y los de cada modo;
# arma `comparacion` y la escribe con `escribir` en `{salida}/comparacion.json`. Devuelve el contenido.

# %%
def _fuente(leer, carpeta_o_summary: Path) -> dict:
    """{summary, pred_test} de la carpeta del summary.json dado."""
    summary = Path(carpeta_o_summary)  # ruta del summary
    return {"summary": leer(summary), "pred_test": leer(summary.parent / ex.ARCHIVO_PRED_TEST)}  # fuente


def escribir_comparacion(cfg: dict, modos: list[str], leer=None, escribir=None) -> dict:
    """Lee el principal y los modos, escribe {salida}/comparacion.json y devuelve su contenido."""
    leer = leer_archivo if leer is None else leer  # al llamar
    escribir = escribir_archivo if escribir is None else escribir  # al llamar
    s = cfg["sensibilidad"]  # rutas
    principal = _fuente(leer, Path(s["principal"]))  # modelo principal (solo lectura)
    por_modo = {m: _fuente(leer, rutas_sensibilidad(cfg, m).resultados / ex.ARCHIVO_SUMMARY) for m in modos}  # modos
    comp = comparacion(principal, por_modo)  # pura
    escribir(Path(s["salida"]) / ARCHIVO_COMPARACION, comp)  # atómico
    return comp  # contenido escrito


# %% [markdown]
# ## Modo optuna
#
# 1. `buscar_fn` (por defecto `hpo.buscar`) con la config de `cfg_hpo_sensibilidad`, las rutas de optuna y `cargar` =
#    `cargar_datos` con la `datos.muestra` de la config (así la huella de la búsqueda es la de las filas que después
#    se entrenan; en la corrida reportada no hay muestra);
# 2. relee y valida su `mlp_final.yaml` (`leer_mlp_final`);
# 3. `correr_pd_fn` con la config del modo y `correr_red` y `comprobar` con `ruta_mlp_final` = ese YAML: el
#    presupuesto del MLP cuenta los ensayos y el tiempo de esa búsqueda.

# %%
def _correr_optuna(cfg: dict, rutas: ex.RutasPD, buscar_fn, correr_pd_fn):
    """Búsqueda en optuna/ y luego correr_pd con su mlp_final.yaml."""
    cfg_hpo = cfg_hpo_sensibilidad(cfg)  # destino y salida en optuna/
    cargar = functools.partial(ex.cargar_datos, **ex.muestra_de(cfg))  # mismas filas que correr_pd
    buscar_fn(cfg_hpo, rutas=rutas, cargar=cargar)  # TPE con las 17 variables
    destino = Path(cfg_hpo["hpo"]["destino"])  # mlp_final.yaml del modo
    mlp, busqueda = ex.leer_mlp_final(destino)  # ganador y procedencia validados
    cfg_opt = cfg_modo(cfg, MODO_OPTUNA, {"mlp": mlp, "busqueda": busqueda})  # config del modo
    return correr_pd_fn(cfg_opt, rutas=rutas,  # mismas rutas
                        correr_red=functools.partial(ex.correr_mlp, ruta_mlp_final=destino),  # su procedencia
                        comprobar=functools.partial(ex.comprobar_antes, ruta_mlp_final=destino))  # ídem


# %% [markdown]
# ## Orquestación
#
# Valida (antes de leer datos) → `preparar_fn` (unión y preprocesador) → `comprobar_fn` de **cada** modo antes de
# entrenar nada (con la config de reusar: salidas previas ajenas o una config rota no cuestan la búsqueda) → cada modo
# en el orden de la config (reusar: `correr_pd_fn` con sus rutas; optuna: `_correr_optuna`) → `comparar_fn`. Todas las
# piezas se inyectan y se resuelven al llamar.

# %%
def correr_sensibilidad(cfg: dict, preparar_fn=None, correr_pd_fn=None, buscar_fn=None, comprobar_fn=None,
                        comparar_fn=None) -> dict:
    """{preparacion, modos {modo: resultado de correr_pd}, comparacion}."""
    modos = validar_cfg_sensibilidad(cfg)  # antes de leer datos
    preparar_fn = preparar if preparar_fn is None else preparar_fn  # al llamar
    correr_pd_fn = ex.correr_pd if correr_pd_fn is None else correr_pd_fn  # al llamar
    buscar_fn = hpo.buscar if buscar_fn is None else buscar_fn  # al llamar
    comprobar_fn = ex.comprobar_antes if comprobar_fn is None else comprobar_fn  # al llamar
    comparar_fn = escribir_comparacion if comparar_fn is None else comparar_fn  # al llamar
    preparacion = preparar_fn(cfg)  # unión y preprocesador (solo train)
    for modo in modos:  # pre-chequeo de todos los modos antes de entrenar
        comprobar_fn(cfg_modo(cfg, MODO_REUSAR), rutas_sensibilidad(cfg, modo))  # salidas previas y config
    corridas = {}  # resultado por modo
    for modo in modos:  # en el orden de la config
        rutas = rutas_sensibilidad(cfg, modo)  # rutas del modo
        if modo == MODO_REUSAR:  # misma arquitectura que el principal
            corridas[modo] = correr_pd_fn(cfg_modo(cfg, MODO_REUSAR), rutas=rutas)  # logística + MLP
        else:  # arquitectura buscada con las 17 variables
            corridas[modo] = _correr_optuna(cfg, rutas, buscar_fn, correr_pd_fn)  # búsqueda + logística + MLP
    return {"preparacion": preparacion, "modos": corridas, "comparacion": comparar_fn(cfg, modos)}  # todo


# %% [markdown]
# ## `run.py pd-sensibilidad`: corrida e informe
#
# Corre todo e imprime las filas por split, una línea por modo y modelo (AUC, KS y Brier de test, parámetros,
# configuraciones y tiempo: `_linea_modelo` de `run.py pd`), el ΔAUC de test frente al principal, si la partición es
# la misma, dónde quedaron los archivos y el tiempo total (núcleo §4.1). La config resuelta ya la imprime el CLI.

# %%
def _correr_e_informar(cfg: dict) -> int:
    """Corre todo e imprime el informe; devuelve 0."""
    inicio = time.perf_counter()  # tiempo total del subcomando
    res = correr_sensibilidad(cfg)  # corrida completa (valida la config)
    comp = res["comparacion"]  # comparación escrita
    print(f"Filas de la unión: {res['preparacion']['filas']}")  # por split
    for modo, r in res["modos"].items():  # cada modo
        for modelo in MODELOS_COMPARADOS:  # logística y MLP
            print(f"[{modo}] " + ex._linea_modelo(modelo, r[modelo]))  # misma línea que run.py pd
        deltas = " · ".join(f"{m} {comp['metricas'][m]['auc_test'][modo]['delta']:+.4f}"
                            for m in MODELOS_COMPARADOS)  # Δ por modelo
        print(f"[{modo}] ΔAUC test frente al principal: {deltas}")  # sensibilidad − principal
    print(f"Misma partición que el principal: {'sí' if comp['misma_particion'] else 'no'}")  # comparación justa
    print(f"Resultados: {Path(cfg['sensibilidad']['salida']).as_posix()} · "
          f"Modelos: {Path(cfg['sensibilidad']['modelos']).as_posix()}")  # archivos
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")  # núcleo §4.1
    return 0  # éxito


# %% [markdown]
# ## `run.py pd-sensibilidad`
#
# Como `run.py pd` (B-pd §3.5): los errores esperados (`ERRORES_ESPERADOS` de `src/pd/experimento.py`) salen como una
# línea `run.py pd-sensibilidad: error: <Tipo>: <mensaje>` en stderr y código 2, sin traza; los errores de programación
# (`NotImplementedError`, `RecursionError`) y cualquier otro se propagan con su traza. El log de Optuna queda en
# WARNING (como `run.py pd-hpo`).

# %%
def manejador(cfg: dict | None) -> int:
    """`run.py pd-sensibilidad`: corrida e informe; error esperado → una línea en stderr y código 2."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # sin una línea INFO por ensayo
    try:  # solo los errores esperados se informan sin traza
        return _correr_e_informar(cfg)  # corrida completa
    except ex.ERRORES_DE_PROGRAMACION:  # bugs que heredan de RuntimeError
        raise  # con su traza
    except ex.ERRORES_ESPERADOS as exc:  # config, datos, fit o archivo
        print(f"{PREFIJO_ERROR}{type(exc).__name__}: {ex._una_linea(exc)}", file=sys.stderr)  # una línea
        return ex.CODIGO_ERROR  # como run.py pd
