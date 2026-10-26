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
# # Búsqueda Optuna TPE del MLP de PD (Parte B §3.3, tarea B4)
#
# `python run.py pd-hpo --config configs/pd/hpo.yaml` busca la arquitectura y los hiperparámetros del MLP de PD
# (enunciado, Parte 1, tarea 4) con el *Tree-structured Parzen Estimator* (TPE) de Optuna [bergstra2011; akiba2019]:
#
# 1. valida la config **antes** de leer datos (toda falla → `ValueError`);
# 2. carga train y val con el preprocesador **ya ajustado por A** (test nunca llega a la búsqueda, núcleo §5.7);
# 3. corre `hpo.n_ensayos` ensayos; cada uno entrena `MLPPD` con `{**mlp heredado, **parámetros del ensayo}` y la
#    misma semilla `base_seed`, y se puntúa con la **log-loss de validación**;
# 4. elige el mejor, escribe `{hpo.salida}/hpo_ensayos.json` y reescribe `hpo.destino` (`configs/pd/mlp_final.yaml`),
#    que hereda `configs/pd/base.yaml`: B5 usa el ganador sin copiar nada a mano.
#
# **TPE multivariado.** En Optuna 5.0.0, `TPESampler()` con sus valores por defecto es multivariado en un estudio de
# un objetivo (`_is_multivariate`), no el TPE univariado de Bergstra et al. (2011). Se mantienen los valores por
# defecto y se registra `multivariate: true` en el JSON. TODO(cita): Falkner et al. (2018), BOHB, para el TPE
# multivariado [VERIFICAR].
#
# Las dependencias (`cargar`, `modelo_cls`, `reloj`, `ahora`, `optuna_mod`, `criterio`, `auc_fn`) se inyectan: las
# pruebas sustituyen cualquiera sin tocar el código. `buscar` solo compone las piezas.
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable
# sin efectos al importar.

# %%
"""Búsqueda Optuna TPE del MLP de PD: valida, busca, elige y escribe hpo_ensayos.json y mlp_final.yaml (tarea B4)."""

import contextlib  # borrar el temporal si la escritura falla
import copy  # copias de la config validada
import inspect  # valor por defecto de n_startup_trials en la firma de TPESampler
import math  # isfinite y log10 de la regla del borde
import os  # os.replace y permisos de la escritura atómica
import tempfile  # temporal en la misma carpeta del destino
import time  # reloj por defecto
from collections import Counter  # conteo de ensayos por estado
from datetime import datetime  # fecha de la búsqueda
from numbers import Real  # comprobación de que un valor es numérico
from pathlib import Path  # rutas de salida

import numpy as np  # bool de numpy en la validación
import optuna  # estudio, muestreador TPE y podador
import torch  # versión e hilos registrados en el JSON
import yaml  # texto de mlp_final.yaml

from src.common.metricas import auc  # AUC de validación (B1)
from src.common.results import save_json, to_jsonable  # JSON sin NaN y escritura atómica (núcleo §5.4)
from src.common.seeds import validar_entero  # enteros de la config (regla única del proyecto)
from src.pd.entrenar import OPTIMIZADORES  # adam | sgd
from src.pd.experimento import RUTAS_PD, cargar_datos, evidencia  # datos de A y huellas (núcleo §6)
from src.pd.logistica import logloss  # criterio de la búsqueda (el mismo de B3)
from src.pd.mlp import ACTIVACIONES, CLAVES_MLP, MLPPD, validar_cfg_mlp  # modelo y reglas del bloque mlp (B2)

__all__ = [  # interfaz pública acordada (tarea B4); lo demás es interno
    "CLAVES_HPO", "CLAVES_ESPACIO", "CLAVES_BUSCADAS", "CLAVES_PROHIBIDAS", "CRITERIO", "ARCHIVO_HPO",  # nombres
    "FRACCION_BORDE", "PARAMS_BORDE", "CABECERA_MLP_FINAL",  # constantes de las reglas
    "validar_cfg_hpo", "validar_cfg", "mlp_fijo",  # validación
    "sugerir", "crear_estudio", "describir_muestreador",  # muestreo
    "medir_ensayo", "crear_objetivo", "registro_ensayo", "elegir_mejor", "en_borde",  # ensayos y elección
    "armar_resumen", "armar_busqueda", "texto_mlp_final", "escribir_texto_atomico",  # salidas
    "buscar", "manejador_hpo",  # búsqueda completa y CLI
]

CLAVES_HPO = ("n_ensayos", "destino", "salida", "espacio")  # claves exactas del bloque hpo
CLAVES_ESPACIO = ("n_capas", "neuronas", "activacion", "dropout", "l2", "optimizador", "lr", "batch")  # del espacio
RANGOS = {"n_capas": "entero", "dropout": "lineal", "l2": "log", "lr": "log"}  # claves [mín, máx] y su escala
CLAVES_BUSCADAS = ("capas", "activacion", "dropout", "l2", "optimizador", "lr", "batch")  # claves mlp que se buscan
CLAVES_PROHIBIDAS = ("pos_weight", "device")  # nunca se buscan (calibración y dispositivo)
CLAVES_METRICAS = ("logloss_val", "auc_val", "perdida_val_entrenar", "epocas", "mejor_epoca", "n_parametros")  # ensayo
CLAVES_MEJOR = ("numero", "parametros", "logloss_val", "auc_val", "n_parametros")  # bloque mejor del JSON
CRITERIO = "logloss_val"  # criterio de la búsqueda (Parte B §3.3)
ARCHIVO_HPO = "hpo_ensayos.json"  # resultados de la búsqueda (núcleo §5.4)
ATRIBUTO_SEMILLA = "semilla_muestreador"  # user_attr del estudio con la semilla del TPE
ESTADO_COMPLETO = optuna.trial.TrialState.COMPLETE.name  # "COMPLETE"
TIPO_BUSQUEDA = "optuna"  # busqueda.tipo de mlp_final.yaml
FRACCION_BORDE = 0.05  # 5 % del ancho del rango (decisión de Andrés, 25-oct)
PARAMS_BORDE = {"lr": "log", "l2": "log", "dropout": "lineal"}  # parámetros revisados en el borde y su escala
CODIFICACION = "utf-8"  # texto del YAML
PERMISOS_ARCHIVO = 0o644  # rw-r--r--: los mismos que deja save_json
CABECERA_MLP_FINAL = (  # dos líneas de comentario al inicio de mlp_final.yaml
    "# Generado por `python run.py pd-hpo` (src/pd/hpo.py). No se edita a mano.\n"  # origen
    "# Configuración final del MLP de PD (Parte B §3.2–§3.3) y su procedencia; la hereda configs/pd/base.yaml.\n"  # uso
)


# %% [markdown]
# ## Validación: tipos numéricos
#
# - **entero:** se reutiliza `seeds.validar_entero` (int de Python o de numpy, nunca bool); su `TypeError` se convierte
#   en `ValueError`, porque toda falla de la config es `ValueError`;
# - **real:** número real y finito (nunca bool).

# %%
def _entero(valor, nombre: str, minimo: int) -> int:
    """``seeds.validar_entero`` con su TypeError convertido en ValueError."""
    try:  # regla única de enteros del proyecto
        return validar_entero(valor, nombre, minimo=minimo)  # ValueError si es < minimo
    except TypeError as exc:  # flotante, bool, texto…
        raise ValueError(str(exc)) from exc  # mismo mensaje, tipo de error del contrato


def _es_real(valor) -> bool:
    """True si ``valor`` es un número real finito (los bool no cuentan)."""
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, Real):  # bool es subclase de int
        return False  # no es un número de la config
    return math.isfinite(valor)  # ±inf y NaN tampoco valen


# %% [markdown]
# ## Validación: rangos `[mín, máx]`
#
# `n_capas`, `dropout`, `l2` y `lr` son rangos: una lista de 2 valores con mín ≤ máx (mín = máx fija el valor).
#
# | Escala | Regla |
# |---|---|
# | `entero` (`n_capas`) | enteros $\ge 1$ |
# | `log` (`l2`, `lr`) | reales finitos $> 0$ (el muestreo es log-uniforme) |
# | `lineal` (`dropout`) | reales finitos en $[0, 1)$ (`validar_cfg_mlp` exige dropout $< 1$) |

# %%
def _validar_rango(espacio: dict, clave: str, escala: str) -> None:
    """[mín, máx] reales finitos (enteros si es n_capas), mín ≤ máx, > 0 en log, dropout en [0, 1)."""
    rango = espacio[clave]  # valor declarado
    if not isinstance(rango, list) or len(rango) != 2:  # [mín, máx]
        raise ValueError(f"hpo.espacio.{clave} debe ser una lista [mín, máx]; se recibió {rango!r}")  # forma
    if escala == "entero":  # número de capas
        bajo, alto = (_entero(v, f"hpo.espacio.{clave}", minimo=1) for v in rango)  # enteros ≥ 1
    elif all(_es_real(v) for v in rango):  # reales finitos
        bajo, alto = rango  # extremos
    else:  # bool, texto, inf o NaN
        raise ValueError(f"hpo.espacio.{clave} debe tener dos reales finitos; se recibió {rango!r}")  # tipo
    if bajo > alto:  # rango vacío
        raise ValueError(f"hpo.espacio.{clave}: mín > máx ({rango!r})")  # orden
    if escala == "log" and bajo <= 0:  # log10 de 0 o de un negativo
        raise ValueError(f"hpo.espacio.{clave} es log-uniforme: sus límites deben ser > 0 ({rango!r})")  # dominio
    if escala == "lineal" and not (0 <= bajo and alto < 1):  # probabilidad de dropout
        raise ValueError(f"hpo.espacio.{clave} debe estar en [0, 1); se recibió {rango!r}")  # dominio


# %% [markdown]
# ## Validación: listas de opciones
#
# `neuronas`, `activacion`, `optimizador` y `batch` son listas no vacías, **sin duplicados** (un duplicado sesgaría el
# muestreo hacia ese valor) y con cada valor admitido. Una lista de un elemento fija el valor.

# %%
def _anchura_valida(valor) -> bool:
    """True si ``valor`` es un entero ≥ 1 (no bool): anchura de capa o tamaño de minilote."""
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, (int, np.integer)):  # 8.0 o True no valen
        return False  # no es un entero de la config
    return valor >= 1  # al menos una neurona o una fila


def _opcion_de(opciones):
    """Predicado: el valor es texto y está en ``opciones``."""
    return lambda valor: isinstance(valor, str) and valor in opciones  # texto: evita comparar listas con el dict


ADMITIDOS = {  # lista → predicado de sus valores
    "neuronas": _anchura_valida,  # anchuras enteras ≥ 1
    "activacion": _opcion_de(ACTIVACIONES),  # relu | leaky_relu | tanh
    "optimizador": _opcion_de(OPTIMIZADORES),  # adam | sgd
    "batch": _anchura_valida,  # tamaños enteros ≥ 1
}


def _validar_lista(espacio: dict, clave: str, es_admitido) -> None:
    """Lista no vacía, sin duplicados y con cada valor admitido."""
    lista = espacio[clave]  # valor declarado
    if not isinstance(lista, list) or len(lista) == 0:  # un texto se iteraría letra por letra
        raise ValueError(f"hpo.espacio.{clave} debe ser una lista no vacía; se recibió {lista!r}")  # forma
    malos = [v for v in lista if not es_admitido(v)]  # valores no admitidos
    if malos:  # p. ej. 'sigmoid', 0 o True
        raise ValueError(f"hpo.espacio.{clave}: valores no admitidos {malos!r}")  # nombra los malos
    if len(set(lista)) != len(lista):  # los admitidos son hashables
        raise ValueError(f"hpo.espacio.{clave} tiene valores duplicados: {lista!r}")  # sesgo del muestreo


# %% [markdown]
# ## Validación del bloque `hpo`
#
# Claves exactas `n_ensayos`, `destino`, `salida` y `espacio`; `n_ensayos` entero $\ge 1$; `destino` y `salida`
# texto no vacío; el espacio con exactamente sus 8 claves. `pos_weight` y `device` en el espacio se rechazan con un
# mensaje propio: buscar `pos_weight` descalibraría la PD (Parte B §3.2) y el dispositivo no es un hiperparámetro.

# %%
def _validar_claves(bloque, claves, nombre: str) -> None:
    """ValueError si ``bloque`` no es un mapeo con exactamente ``claves``."""
    if not isinstance(bloque, dict):  # bloque reemplazado por un valor
        raise ValueError(f"{nombre} debe ser un mapeo; se recibió {bloque!r}")  # tipo
    faltan, sobran = set(claves) - set(bloque), set(bloque) - set(claves)  # diferencias
    if faltan or sobran:  # el bloque debe ser exacto
        raise ValueError(f"{nombre} inválido: faltan {sorted(faltan)}, sobran {sorted(sobran)}")  # ambas listas


def _validar_ruta(valor, nombre: str) -> None:
    """ValueError si ``valor`` no es texto no vacío."""
    if not isinstance(valor, str) or not valor.strip():  # '' o un número no son rutas
        raise ValueError(f"{nombre} debe ser una ruta (texto no vacío); se recibió {valor!r}")  # tipo


def validar_cfg_hpo(cfg_hpo: dict) -> dict:
    """Valida el bloque hpo (claves exactas, rangos, listas) y devuelve una copia; cualquier falla → ValueError."""
    _validar_claves(cfg_hpo, CLAVES_HPO, "El bloque hpo")  # primero las claves: lo demás las indexa
    _entero(cfg_hpo["n_ensayos"], "hpo.n_ensayos", minimo=1)  # al menos un ensayo
    _validar_ruta(cfg_hpo["destino"], "hpo.destino")  # YAML que se reescribe
    _validar_ruta(cfg_hpo["salida"], "hpo.salida")  # carpeta del JSON
    espacio = cfg_hpo["espacio"]  # espacio de búsqueda
    prohibidas = [c for c in CLAVES_PROHIBIDAS if isinstance(espacio, dict) and c in espacio]  # claves vetadas
    if prohibidas:  # mensaje propio, antes del genérico de claves
        raise ValueError(f"hpo.espacio no puede buscar {prohibidas}: pos_weight queda fijo para no descalibrar la "
                         "PD y device no es un hiperparámetro")  # por qué
    _validar_claves(espacio, CLAVES_ESPACIO, "hpo.espacio")  # las 8 claves
    for clave, escala in RANGOS.items():  # rangos [mín, máx]
        _validar_rango(espacio, clave, escala)  # forma, tipo, orden y dominio
    for clave, es_admitido in ADMITIDOS.items():  # listas de opciones
        _validar_lista(espacio, clave, es_admitido)  # no vacía, admitida, sin duplicados
    return copy.deepcopy(cfg_hpo)  # copia: mutarla no altera la config del llamador


# %% [markdown]
# ## Validación de toda la config
#
# Antes de leer datos: el bloque `hpo`, el bloque `mlp` heredado (pasa `validar_cfg_mlp`, B2) y `base_seed`
# entero $\ge 0$. Las demás claves de la raíz (`logistica`, `busqueda`) se ignoran.

# %%
def validar_cfg(cfg: dict) -> tuple[dict, dict, int]:
    """(hpo validado, mlp heredado validado, base_seed) antes de leer datos; cualquier falla → ValueError."""
    if not isinstance(cfg, dict):  # sin config no hay búsqueda
        raise ValueError(f"La config de pd-hpo debe ser un mapeo; se recibió {type(cfg).__name__}")  # tipo
    faltan = [c for c in ("hpo", "mlp", "base_seed") if c not in cfg]  # bloques necesarios
    if faltan:  # p. ej. una config de otro subcomando
        raise ValueError(f"La config de pd-hpo no tiene {faltan}")  # cuáles
    if not isinstance(cfg["mlp"], dict):  # validar_cfg_mlp indexa el bloque
        raise ValueError(f"El bloque mlp debe ser un mapeo; se recibió {cfg['mlp']!r}")  # tipo
    cfg_hpo = validar_cfg_hpo(cfg["hpo"])  # bloque de la búsqueda
    cfg_mlp = validar_cfg_mlp(cfg["mlp"])  # mlp heredado (B2)
    base_seed = _entero(cfg["base_seed"], "base_seed", minimo=0)  # semilla del muestreador y de los ensayos
    return cfg_hpo, cfg_mlp, base_seed  # todo validado


# %% [markdown]
# ## Claves fijas del bloque `mlp`
#
# `mlp_base` del JSON guarda **solo** las claves que no se buscan (`momento`, `max_epocas`, `paciencia`, `device`,
# `pos_weight`): la herencia `hpo.yaml → base.yaml → mlp_final.yaml` (= `hpo.destino`) haría cambiar el JSON entre
# corridas si se guardara el bloque completo, porque las claves buscadas vendrían del ganador anterior.

# %%
def mlp_fijo(cfg_mlp: dict) -> dict:
    """Claves del bloque mlp que no se buscan (``mlp_base`` del JSON)."""
    return {k: cfg_mlp[k] for k in CLAVES_MLP if k not in CLAVES_BUSCADAS}  # en el orden de CLAVES_MLP


# %% [markdown]
# ## Parámetros de un ensayo
#
# Nombres de Optuna: `n_capas` (entero), `neuronas_capa{k}` para $k = 1, \dots,$ `n_capas` (cada capa elige su
# anchura por separado: sin forma de pirámide), `activacion`, `dropout` (uniforme), `l2` y `lr` (log-uniformes),
# `optimizador` y `batch`. Se devuelven ya armados: `capas` es la lista de anchuras.

# %%
def sugerir(trial, espacio: dict) -> dict:
    """Parámetros del ensayo ya armados (``capas`` = lista), con los nombres n_capas y neuronas_capa{k}."""
    n_capas = trial.suggest_int("n_capas", *espacio["n_capas"])  # número de capas ocultas
    capas = [trial.suggest_categorical(f"neuronas_capa{k}", espacio["neuronas"])  # anchura de la capa k
             for k in range(1, n_capas + 1)]  # k desde 1
    return {  # en el orden de CLAVES_BUSCADAS
        "capas": capas,  # anchuras
        "activacion": trial.suggest_categorical("activacion", espacio["activacion"]),  # φ
        "dropout": trial.suggest_float("dropout", *espacio["dropout"]),  # uniforme
        "l2": trial.suggest_float("l2", *espacio["l2"], log=True),  # log-uniforme
        "optimizador": trial.suggest_categorical("optimizador", espacio["optimizador"]),  # adam | sgd
        "lr": trial.suggest_float("lr", *espacio["lr"], log=True),  # log-uniforme
        "batch": trial.suggest_categorical("batch", espacio["batch"]),  # tamaño de minilote
    }


# %% [markdown]
# ## Estudio
#
# Minimiza la log-loss de validación con `TPESampler(seed=base_seed)` (valores por defecto) y `NopPruner` (sin
# poda: cada ensayo entrena hasta su parada temprana), en memoria. La semilla se guarda como atributo del estudio
# (`semilla_muestreador`): el `LazyRandomState` del muestreador de Optuna 5 no la conserva.

# %%
def crear_estudio(base_seed: int, optuna_mod=optuna):
    """Estudio de minimización con TPESampler(seed) y NopPruner, en memoria."""
    estudio = optuna_mod.create_study(  # sin storage: en memoria
        direction="minimize",  # log-loss: menor es mejor
        sampler=optuna_mod.samplers.TPESampler(seed=base_seed),  # TPE con sus valores por defecto
        pruner=optuna_mod.pruners.NopPruner(),  # sin poda
    )
    estudio.set_user_attr(ATRIBUTO_SEMILLA, base_seed)  # el muestreador no conserva la semilla
    return estudio  # listo para optimize


# %% [markdown]
# ## Muestreador efectivo
#
# `{nombre, semilla, n_startup_trials, multivariate}`. `n_startup_trials` sale de la firma de `TPESampler` (se usa el
# valor por defecto; 10 en Optuna 5.0.0) y `multivariate` de `sampler._is_multivariate(estudio)`: es API **privada a
# propósito** (decisión de Andrés), porque es la única forma de registrar el comportamiento efectivo; en Optuna 5.0.0
# vale `True` y la prueba 6 lo fija para que un cambio de versión no pase en silencio.

# %%
def describir_muestreador(estudio) -> dict:
    """{nombre, semilla, n_startup_trials, multivariate} efectivos del muestreador del estudio."""
    muestreador = estudio.sampler  # TPESampler
    firma = inspect.signature(type(muestreador))  # parámetros del constructor
    return {  # va al JSON
        "nombre": type(muestreador).__name__,  # "TPESampler"
        "semilla": estudio.user_attrs[ATRIBUTO_SEMILLA],  # la que guardó crear_estudio
        "n_startup_trials": firma.parameters["n_startup_trials"].default,  # ensayos aleatorios iniciales
        "multivariate": bool(muestreador._is_multivariate(estudio)),  # API privada a propósito (ver arriba)
    }


# %% [markdown]
# ## Medición de un ensayo
#
# Con $p_i$ la PD del modelo entrenado en validación, el criterio es la log-loss
# $-\frac1n\sum_i [y_i\log p_i + (1-y_i)\log(1-p_i)]$ (`logistica.logloss`) y al lado la AUC. Del resultado del
# entrenamiento se guardan la pérdida de validación de la mejor época (`perdida_val_entrenar`, la misma BCE con
# $w_+ = 1$, así que coincide con la log-loss salvo la precisión float32), las épocas corridas, la mejor época y los
# parámetros entrenables.

# %%
def medir_ensayo(modelo, X_val, y_val, criterio=logloss, auc_fn=auc) -> dict:
    """logloss_val, auc_val, perdida_val_entrenar, epocas, mejor_epoca y n_parametros del modelo entrenado."""
    p = modelo.predict_proba(X_val)  # PD de validación (sin dropout)
    resultado = modelo.resultado  # ResultadoEntrenamiento del fit
    return {  # en el orden de CLAVES_METRICAS
        "logloss_val": float(criterio(y_val, p)),  # criterio de la búsqueda
        "auc_val": float(auc_fn(y_val, p)),  # discriminación, al lado
        "perdida_val_entrenar": float(resultado.mejor_perdida_val),  # BCE de la mejor época
        "epocas": int(resultado.epocas_corridas),  # épocas entrenadas
        "mejor_epoca": int(resultado.mejor_epoca),  # época de los pesos restaurados
        "n_parametros": int(modelo.n_parametros()),  # parámetros entrenables
    }


# %% [markdown]
# ## Función objetivo
#
# Para cada ensayo: sugiere los parámetros (se registran **antes** del fit), entrena `modelo_cls()` con
# `{"mlp": {**mlp heredado, **parámetros}}` y la semilla `seed`, cronometra **solo el fit** con `reloj` y mide en
# validación. Todo va a `trial.set_user_attr`. Si el entrenamiento diverge (`FloatingPointError`, B2) se registran
# también el tiempo y el mensaje (Optuna 5 no guarda el mensaje de un ensayo FAIL) y se relanza: la búsqueda lo
# marca FAIL y sigue. El objetivo solo recibe `(X_tr, y_tr)` y `(X_val, y_val)`: test nunca llega aquí.

# %%
def crear_objetivo(entreno, validacion, mlp_heredado, espacio, seed, modelo_cls=MLPPD, reloj=time.perf_counter):
    """Función objetivo de Optuna: sugiere, cronometra el fit y mide en val; guarda todo en user_attrs."""
    X_tr, y_tr = entreno  # entrenamiento
    X_val, y_val = validacion  # validación (decide la parada y el criterio)

    def objetivo(trial) -> float:
        parametros = sugerir(trial, espacio)  # parámetros armados
        trial.set_user_attr("parametros", parametros)  # antes del fit: un FAIL también los registra
        modelo = modelo_cls()  # modelo nuevo por ensayo
        inicio = reloj()  # comienzo del fit
        try:  # solo el fit se cronometra
            modelo.fit(X_tr, y_tr, X_val, y_val, {"mlp": {**mlp_heredado, **parametros}}, seed=seed)  # entrena
        except FloatingPointError as exc:  # divergencia: el ensayo queda FAIL
            trial.set_user_attr("error", str(exc))  # Optuna 5 no guarda el mensaje
            raise  # la búsqueda lo captura (catch) y sigue
        finally:  # también si falló
            trial.set_user_attr("tiempo_s", reloj() - inicio)  # cuenta en el presupuesto
        metricas = medir_ensayo(modelo, X_val, y_val)  # métricas de validación
        for clave, valor in metricas.items():  # una por una
            trial.set_user_attr(clave, valor)  # van al JSON
        return metricas[CRITERIO]  # Optuna minimiza la log-loss de val

    return objetivo  # cierre sobre los datos y la config


# %% [markdown]
# ## Fila de un ensayo
#
# A partir de un `FrozenTrial`: `numero`, `estado` (nombre del `TrialState`), `parametros`, las métricas (`None` si
# el ensayo falló), `tiempo_s` y, si falló, `error`.

# %%
def registro_ensayo(trial) -> dict:
    """Fila del JSON a partir de un FrozenTrial (estado, parámetros, métricas, tiempo, error)."""
    attrs = trial.user_attrs  # lo que guardó el objetivo
    fila = {"numero": trial.number, "estado": trial.state.name, "parametros": attrs.get("parametros")}  # identidad
    fila.update({clave: attrs.get(clave) for clave in CLAVES_METRICAS})  # None si el ensayo falló
    fila["tiempo_s"] = attrs.get("tiempo_s")  # tiempo del fit
    if "error" in attrs:  # solo los FAIL
        fila["error"] = attrs["error"]  # mensaje de la divergencia
    return fila  # una fila de "ensayos"


# %% [markdown]
# ## Elección del mejor
#
# $$k^{*} = \arg\min_{k \,:\, \text{estado}_k = \texttt{COMPLETE}} \ \text{logloss\_val}_k,$$
#
# con empate → menor `numero` (no el primero de la lista). Sin ensayos completos → `RuntimeError`.

# %%
def elegir_mejor(ensayos: list[dict]) -> dict:
    """Argmin de logloss_val entre los completos; empate → menor número; ninguno → RuntimeError."""
    completos = [e for e in ensayos if e["estado"] == ESTADO_COMPLETO]  # ignora FAIL
    if not completos:  # nada que elegir
        raise RuntimeError("Ningún ensayo de la búsqueda terminó (todos fallaron): no hay configuración que elegir")
    return min(completos, key=lambda e: (e[CRITERIO], e["numero"]))  # criterio y desempate


# %% [markdown]
# ## ¿El ganador quedó en el borde del espacio?
#
# Con el rango $[a, b]$ de un parámetro, su valor $v$ y $f$ = `FRACCION_BORDE` = 0.05 (decisión de Andrés), en la
# escala $s$ del parámetro ($s = \log_{10}$ para `lr` y `l2`, la identidad para `dropout`):
#
# $$\text{borde} = \begin{cases} \texttt{"min"} & s(v) - s(a) \le f\,[s(b) - s(a)] \\
# \texttt{"max"} & s(b) - s(v) \le f\,[s(b) - s(a)] \\ \texttt{None} & \text{en otro caso o si } a = b. \end{cases}$$
#
# Un ganador en el borde sugiere que el espacio debió ser más amplio (se reporta como limitación).

# %%
def en_borde(parametros: dict, espacio: dict, fraccion=FRACCION_BORDE) -> dict:
    """{lr, l2, dropout: "min"/"max"/None} según la regla del 5 %."""
    resultado = {}  # parámetro → borde
    for clave, escala in PARAMS_BORDE.items():  # lr, l2, dropout
        a, b = espacio[clave]  # rango
        v = parametros[clave]  # valor del ganador
        if escala == "log":  # distancias en décadas
            a, b, v = math.log10(a), math.log10(b), math.log10(v)  # escala log10
        tolerancia = fraccion * (b - a)  # f · ancho
        if a == b:  # rango degenerado: valor fijo, no es un borde
            resultado[clave] = None  # sin borde
        elif v - a <= tolerancia:  # cerca del mínimo
            resultado[clave] = "min"  # borde inferior
        elif b - v <= tolerancia:  # cerca del máximo
            resultado[clave] = "max"  # borde superior
        else:  # en el interior
            resultado[clave] = None  # sin borde
    return resultado  # tres claves


# %% [markdown]
# ## Contenido de `hpo_ensayos.json`
#
# Presupuesto de la regla de comparación justa (núcleo §6): `n_ensayos` configuraciones y
#
# $$\texttt{tiempo\_total\_s} = \sum_{k} t^{\text{fit}}_k,$$
#
# la suma del tiempo de **todos** los fit (incluidos los FAIL); `tiempo_reloj_s` es todo `estudio.optimize`
# (incluye el muestreo de Optuna y la medición en val). `mlp_base` = solo las claves fijas; versiones e hilos de
# torch para el determinismo (núcleo §7.1); `evidencia` = huellas de la partición y del preprocesador (núcleo §6).

# %%
def armar_resumen(estudio, ensayos: list[dict], mejor: dict, espacio: dict, mlp_heredado: dict, base_seed: int,
                  tiempo_reloj_s: float, huellas: dict) -> dict:
    """Contenido de hpo_ensayos.json (claves de la tarea + evidencia)."""
    return {  # claves exactas (decisión de Andrés)
        "criterio": CRITERIO,  # log-loss de validación
        "muestreador": describir_muestreador(estudio),  # TPE efectivo
        "podador": type(estudio.pruner).__name__,  # "NopPruner"
        "semilla_ensayos": base_seed,  # misma semilla en todos los ensayos
        "versiones": {"optuna": optuna.__version__, "torch": str(torch.__version__)},  # reproducibilidad
        "hilos_torch": torch.get_num_threads(),  # hilos de CPU (determinismo)
        "espacio": espacio,  # espacio validado
        "mlp_base": mlp_fijo(mlp_heredado),  # solo las claves que no se buscan
        "n_ensayos": len(ensayos),  # configuraciones evaluadas
        "estados": dict(Counter(e["estado"] for e in ensayos)),  # conteo por estado presente
        "tiempo_total_s": sum((e["tiempo_s"] for e in ensayos if e["tiempo_s"] is not None), 0.0),  # Σ fit
        "tiempo_reloj_s": tiempo_reloj_s,  # todo optimize
        "mejor": {clave: mejor[clave] for clave in CLAVES_MEJOR},  # ganador
        "en_borde": en_borde(mejor["parametros"], espacio),  # regla del 5 %
        "ensayos": ensayos,  # una fila por ensayo
        "evidencia": huellas,  # núcleo §6
    }


# %% [markdown]
# ## Procedencia en `mlp_final.yaml`
#
# Bloque `busqueda`: `tipo: optuna`, el ensayo ganador, el presupuesto (`n_ensayos`, `tiempo_s` = `tiempo_total_s`),
# la evidencia y la fecha. B5 lo lee para reportar el presupuesto de la búsqueda.

# %%
def armar_busqueda(resumen: dict, evidencia: dict, fecha: str) -> dict:
    """Bloque ``busqueda`` de mlp_final.yaml {tipo: optuna, ensayo, n_ensayos, tiempo_s, evidencia, fecha}."""
    return {  # procedencia del ganador
        "tipo": TIPO_BUSQUEDA,  # "optuna"
        "ensayo": resumen["mejor"]["numero"],  # ensayo ganador
        "n_ensayos": resumen["n_ensayos"],  # configuraciones evaluadas
        "tiempo_s": resumen["tiempo_total_s"],  # presupuesto de tiempo
        "evidencia": evidencia,  # huellas de los datos de la búsqueda
        "fecha": fecha,  # ISO 8601
    }


# %% [markdown]
# ## Texto de `mlp_final.yaml`
#
# Cabecera "no se edita a mano" + los bloques `mlp` y `busqueda` en YAML de bloque. Los valores pasan por
# `to_jsonable` (tipos nativos) y PyYAML escribe los flotantes con `repr`, así que `load_config` los lee sin pérdida.

# %%
def texto_mlp_final(mlp: dict, busqueda: dict) -> str:
    """Cabecera "lo genera run.py pd-hpo, no se edita a mano" + YAML de los bloques mlp y busqueda."""
    cuerpo = yaml.safe_dump(  # YAML de bloque, en el orden de las claves
        to_jsonable({"mlp": mlp, "busqueda": busqueda}),  # tipos nativos (sin numpy ni tuplas)
        sort_keys=False, allow_unicode=True, default_flow_style=False,  # legible
    )
    return CABECERA_MLP_FINAL + cuerpo  # texto completo


# %% [markdown]
# ## Escritura atómica del YAML
#
# Crea las carpetas que falten, escribe en un temporal de la **misma** carpeta y lo mueve con `os.replace`
# (atómico): si algo falla, se borra el temporal y el archivo anterior queda intacto. Permisos 0o644 como los JSON.

# %%
def escribir_texto_atomico(ruta, texto: str) -> Path:
    """Escribe en un temporal de la misma carpeta y hace os.replace (permisos 0o644)."""
    destino = Path(ruta)  # acepta str o Path
    destino.parent.mkdir(parents=True, exist_ok=True)  # carpetas que falten
    fd, temporal = tempfile.mkstemp(prefix=f".{destino.name}.", suffix=".tmp", dir=destino.parent)  # misma carpeta
    try:  # cualquier falla borra el temporal
        with os.fdopen(fd, "w", encoding=CODIFICACION, newline="\n") as f:  # texto UTF-8
            f.write(texto)  # contenido
            f.flush()  # al sistema operativo
            os.fsync(f.fileno())  # al disco
        os.replace(temporal, destino)  # atómico
    except BaseException:  # incluso KeyboardInterrupt
        with contextlib.suppress(FileNotFoundError):  # puede no existir ya
            os.remove(temporal)  # sin temporales sueltos
        raise  # la falla se propaga
    os.chmod(destino, PERMISOS_ARCHIVO)  # mkstemp crea 0o600
    return destino  # ruta escrita


# %% [markdown]
# ## Partes de `buscar`
#
# - `_datos_busqueda`: train y val (nunca test) y las huellas del mismo DataFrame;
# - `_correr_estudio`: crea el estudio y corre los ensayos; solo `FloatingPointError` se captura (cualquier otro
#   error es un bug y sale de la búsqueda); devuelve las filas y el tiempo de reloj;
# - `_escribir`: arma en memoria el JSON **y** el texto del YAML, escribe el JSON y por último el YAML. Si falla la
#   serialización de cualquiera de los dos, no queda ninguno nuevo.

# %%
def _datos_busqueda(rutas, cargar) -> tuple:
    """((X_tr, y_tr), (X_val, y_val), huellas) con una sola lectura de los datos."""
    df, datos = cargar(rutas)  # (df, {split: (X, y, ids)})
    X_tr, y_tr, _ = datos["train"]  # entrenamiento
    X_val, y_val, _ = datos["val"]  # validación
    return (X_tr, y_tr), (X_val, y_val), evidencia(df, rutas.preprocesador)  # test no se lee


def _correr_estudio(objetivo, n_ensayos: int, base_seed: int, reloj) -> tuple:
    """(estudio, filas de los ensayos, tiempo de reloj de optimize)."""
    estudio = crear_estudio(base_seed)  # TPE + NopPruner
    inicio = reloj()  # comienzo de la búsqueda
    estudio.optimize(objetivo, n_trials=n_ensayos, n_jobs=1, catch=(FloatingPointError,))  # solo divergencias
    tiempo_reloj_s = reloj() - inicio  # todo optimize
    return estudio, [registro_ensayo(t) for t in estudio.trials], tiempo_reloj_s  # en orden de número


def _escribir(cfg_hpo: dict, resumen: dict, mlp: dict, busqueda: dict) -> None:
    """Arma el texto del YAML, escribe el JSON y por último el YAML."""
    texto = texto_mlp_final(mlp, busqueda)  # en memoria: si falla, no se escribe nada
    save_json(Path(cfg_hpo["salida"]) / ARCHIVO_HPO, resumen)  # valida NaN antes de escribir
    escribir_texto_atomico(cfg_hpo["destino"], texto)  # al final: B5 nunca hereda un ganador sin su JSON


# %% [markdown]
# ## Búsqueda completa
#
# Valida, carga, corre, elige, arma y escribe. `rutas`, `cargar` y `ahora` se resuelven **al llamar** (los globales
# `RUTAS_PD` y `cargar_datos` de este módulo; `datetime.now`), para que el CLI se pueda probar con `monkeypatch`.
# `destino` y `salida` son relativos al directorio actual. No toca el logging global de Optuna.

# %%
def buscar(cfg, rutas=None, cargar=None, modelo_cls=MLPPD, reloj=time.perf_counter, ahora=None) -> dict:
    """Valida, carga train/val, corre el estudio, elige, escribe el JSON y mlp_final.yaml, y devuelve el resumen."""
    cfg_hpo, mlp, base_seed = validar_cfg(cfg)  # antes de leer datos
    rutas = RUTAS_PD if rutas is None else rutas  # al llamar
    cargar = cargar_datos if cargar is None else cargar  # al llamar
    ahora = datetime.now if ahora is None else ahora  # al llamar
    entreno, validacion, huellas = _datos_busqueda(rutas, cargar)  # train, val y evidencia
    objetivo = crear_objetivo(entreno, validacion, mlp, cfg_hpo["espacio"], base_seed, modelo_cls, reloj)  # f(trial)
    estudio, ensayos, tiempo_reloj_s = _correr_estudio(objetivo, cfg_hpo["n_ensayos"], base_seed, reloj)  # ensayos
    mejor = elegir_mejor(ensayos)  # RuntimeError si ninguno completó
    resumen = armar_resumen(estudio, ensayos, mejor, cfg_hpo["espacio"], mlp, base_seed, tiempo_reloj_s,
                            huellas)  # JSON
    busqueda = armar_busqueda(resumen, huellas, ahora().isoformat(timespec="seconds"))  # procedencia
    _escribir(cfg_hpo, resumen, {**mlp, **mejor["parametros"]}, busqueda)  # JSON y luego YAML
    return resumen  # lo mismo que se escribió


# %% [markdown]
# ## `run.py pd-hpo`
#
# Fija el log de Optuna en WARNING (sin una línea por ensayo), corre la búsqueda e imprime los ensayos completos y
# fallidos, el mejor con su log-loss y AUC de validación, el presupuesto, dónde quedaron los archivos y el tiempo
# total (núcleo §4.1). La config resuelta ya la imprime el CLI.

# %%
def manejador_hpo(cfg: dict | None) -> int:
    """`run.py pd-hpo`: ensayos completos y fallidos, mejor con log-loss y AUC de val, rutas y tiempo total."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # sin una línea INFO por ensayo
    inicio = time.perf_counter()  # tiempo total del subcomando
    resumen = buscar(cfg)  # búsqueda completa (valida la config)
    estados, mejor = resumen["estados"], resumen["mejor"]  # atajos
    print(f"Ensayos: {resumen['n_ensayos']} ({estados.get(ESTADO_COMPLETO, 0)} completos, "
          f"{resumen['n_ensayos'] - estados.get(ESTADO_COMPLETO, 0)} fallidos)")  # estados
    print(f"Mejor ensayo: #{mejor['numero']} · log-loss val {mejor['logloss_val']:.5f} · "
          f"AUC val {mejor['auc_val']:.4f} · {mejor['n_parametros']} parámetros")  # ganador
    print(f"Presupuesto: {resumen['n_ensayos']} configuraciones, {resumen['tiempo_total_s']:.1f} s de "
          "entrenamiento")  # regla de comparación justa
    print(f"Resultados: {(Path(cfg['hpo']['salida']) / ARCHIVO_HPO).as_posix()}")  # JSON
    print(f"Configuración final: {Path(cfg['hpo']['destino']).as_posix()}")  # YAML
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")  # núcleo §4.1
    return 0  # éxito
