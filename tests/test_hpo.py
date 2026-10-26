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
# # Pruebas de la búsqueda Optuna TPE del MLP de PD (Parte B §3.3, tarea B4)
#
# Solo se usa la interfaz pública de `src.pd.hpo` (más `cargar_datos` y `evidencia` de `src.pd.experimento`, que B4
# vuelve públicas). No se prueba Optuna (que el TPE muestree bien) ni PyTorch: se prueban nuestras decisiones
# (Andrés, 25-oct): validación de la config antes de leer datos, nombres de los parámetros, criterio log-loss de
# validación, desempate por menor número, solo `FloatingPointError` se captura, presupuesto de tiempo con los
# ensayos fallidos, regla de `en_borde`, `mlp_base` con las claves fijas, orden y atomicidad de la escritura, test
# nunca se toca y el `mlp_final.yaml` que hereda `configs/pd/base.yaml`.
#
# **Datos:** sintéticos, generados aquí con una logística conocida (núcleo §0 regla 8: solo para desarrollar). El
# doble `cargar` devuelve `(df con id/split, {split: (X float32, y, ids)})`, como `experimento.cargar_datos`. El
# preprocesador es un JSON falso en `tmp_path`: solo sirve para su huella SHA-256. La única excepción es la prueba
# 23 (marcada `lento`), que corre la búsqueda con los datos reales de A.
#
# **Nunca** se escribe en `configs/` ni en `results/` reales: destino y salida van siempre a `tmp_path`.
#
# Pruebas añadidas a las 23 del plan aprobado (cada una atrapa un fallo de nuestra infraestructura que ninguna otra
# ve):
#
# - `test_a01_validar_cfg_hpo_devuelve_copia`: la copia validada no comparte listas con la config (una mutación
#   posterior alteraría la config del CLI) y las claves ajenas de la raíz (`logistica`, `busqueda`) se toleran.
# - `test_a02_sugerir_declara_las_distribuciones`: `l2` y `lr` log-uniformes, `dropout` lineal, `n_capas` entero y
#   las anchuras categóricas por capa (un `log=False` en lr pasaría la prueba 3, que usa `FixedTrial`).
# - `test_a03_mlp_fijo_y_buscadas_parten_el_bloque`: las claves fijas y las buscadas son disjuntas y juntas son
#   `CLAVES_MLP` (si B2 añade una clave al bloque mlp, esta prueba avisa antes de que el JSON la pierda).
# - `test_a04_medir_ensayo_usa_lo_inyectado`: `medir_ensayo` mide con el criterio y la AUC que recibe, sobre
#   `X_val`, y no cruza `epocas` con `mejor_epoca` (DIP; un modelo falso con valores distintos en cada campo).
# - `test_a05_buscar_no_muta_la_config`: el CLI y B5 reutilizan la config después de la búsqueda.
# - `test_a06_destino_relativo_al_directorio_actual`: `hpo.destino` y `hpo.salida` relativos se resuelven contra el
#   directorio actual (decisión de Andrés), no contra la carpeta del YAML ni la del módulo.

# %%
"""Pruebas de src/pd/hpo.py: búsqueda Optuna TPE del MLP de PD (Parte B §3.3, tarea B4)."""

import copy  # copias profundas de la config de prueba
import hashlib  # SHA-256 de referencia para la evidencia
import inspect  # valor por defecto de n_startup_trials en la firma de TPESampler
import itertools  # reloj falso que avanza a paso fijo y contadores de llamadas
import json  # comparar el resumen devuelto con el JSON escrito
import math  # inf y nan en los rangos inválidos
import os  # os.replace (para simular una falla de escritura) y permisos
import shutil  # copiar la cadena de YAML versionados a tmp_path
import stat  # S_IMODE: bits de permisos
import time  # reloj real (prueba 15)
from collections.abc import Mapping  # diccionario de matrices que prohíbe leer test
from datetime import datetime  # fecha fija inyectada en `ahora`
from pathlib import Path  # rutas de los archivos de prueba

import numpy as np  # datos sintéticos
import optuna  # tipos del estudio y FixedTrial
import pandas as pd  # parquet sintético de la prueba 21
import pytest  # parametrización, fixtures y pytest.raises
import torch  # versión e hilos registrados en el JSON
import yaml  # error de serialización simulado (prueba 19)
from scipy.special import expit  # σ de la logística generadora

from src.common import cli  # CLI de run.py (prueba 22)
from src.common.config import apply_overrides, cargar_yaml, load_config  # mismo cargador que el CLI (--set)
from src.common.metricas import auc  # AUC de referencia
from src.common.preprocesamiento import Preprocesador  # preprocesador común (prueba 21)
from src.common.results import load_json, save_json, to_jsonable  # JSON del núcleo §5.4
from src.pd import experimento, hpo  # módulos bajo prueba (hpo resuelve RUTAS_PD y cargar_datos al llamarse)
from src.pd.experimento import RutasPD, cargar_datos, cargar_matrices, evidencia  # públicas desde B4
from src.pd.hpo import (  # interfaz pública de B4
    CLAVES_BUSCADAS,  # claves del bloque mlp que se buscan
    FRACCION_BORDE,  # 5 % del ancho del rango
    buscar,  # búsqueda completa
    crear_estudio,  # estudio TPE + NopPruner en memoria
    describir_muestreador,  # muestreador efectivo para el JSON
    elegir_mejor,  # argmin de logloss_val entre los completos
    en_borde,  # regla del 5 %
    escribir_texto_atomico,  # escritura atómica del YAML
    medir_ensayo,  # métricas de un ensayo entrenado
    mlp_fijo,  # claves del bloque mlp que no se buscan
    sugerir,  # parámetros de un ensayo
    texto_mlp_final,  # texto de mlp_final.yaml
    validar_cfg,  # validación de toda la config antes de cargar datos
    validar_cfg_hpo,  # validación del bloque hpo
)
from src.pd.logistica import logloss  # criterio de la búsqueda (el mismo de B3)
from src.pd.mlp import CLAVES_MLP, MLPPD, validar_cfg_mlp  # modelo y reglas del bloque mlp (B2)

# Los parámetros deprecados o experimentales de Optuna son error en todo el módulo (decisión de Andrés).
pytestmark = [  # marcas aplicadas a todas las pruebas del módulo
    pytest.mark.filterwarnings("error::FutureWarning"),  # Optuna marca lo deprecado con FutureWarning
    pytest.mark.filterwarnings("error::optuna.exceptions.ExperimentalWarning"),  # API experimental = error
]

# %% [markdown]
# ## Constantes de las pruebas
#
# Son valores de prueba, no constantes del dominio: tamaños pequeños para que cada búsqueda tarde poco.

# %%
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio (configs versionadas y datos reales)
SEMILLA_DATOS = 20261025  # semilla de los datos sintéticos (fecha de la tarea)
N_FILAS = {"train": 200, "val": 60, "test": 40}  # filas por split (≈ 300)
N_COLUMNAS = 6  # columnas de X
PESOS_GEN = np.array([1.5, -1.0, 0.8, 0.0, 0.0, 0.5])  # logística generadora (dos columnas sin señal)
INTERCEPTO_GEN = -1.0  # tasa de malos ≈ 0.3
N_ENSAYOS = 3  # ensayos de las búsquedas sintéticas
PASO_RELOJ = 1.0  # avance del reloj falso por llamada (exacto en binario)
FECHA_FIJA = datetime(2026, 10, 25, 12, 0, 0)  # `ahora` inyectado (sin microsegundos)
MENSAJE_DIVERGE = "Pérdida de train no finita (nan) en la época 1"  # FloatingPointError simulado
CABECERA = "# Generado por `python run.py pd-hpo` (src/pd/hpo.py). No se edita a mano."  # 1.ª línea del YAML
CLAVES_FIJAS = {"momento", "max_epocas", "paciencia", "device", "pos_weight"}  # mlp_base del JSON
CLAVES_JSON = {  # claves exactas de hpo_ensayos.json (decisión de Andrés)
    "criterio", "muestreador", "podador", "semilla_ensayos", "versiones", "hilos_torch", "espacio", "mlp_base",
    "n_ensayos", "estados", "tiempo_total_s", "tiempo_reloj_s", "mejor", "en_borde", "ensayos", "evidencia",
}
CLAVES_MEJOR = {"numero", "parametros", "logloss_val", "auc_val", "n_parametros"}  # bloque mejor
CLAVES_ENSAYO = {  # claves de cada ensayo (+ "error" si FAIL)
    "numero", "estado", "parametros", "logloss_val", "auc_val", "perdida_val_entrenar", "epocas", "mejor_epoca",
    "n_parametros", "tiempo_s",
}
CLAVES_METRICAS_ENSAYO = {  # lo que mide medir_ensayo
    "logloss_val", "auc_val", "perdida_val_entrenar", "epocas", "mejor_epoca", "n_parametros",
}
CLAVES_BUSQUEDA = {"tipo", "ensayo", "n_ensayos", "tiempo_s", "evidencia", "fecha"}  # bloque busqueda del YAML
CLAVES_EVIDENCIA = {"sha256_particion", "sha256_preprocesador"}  # núcleo §6
PERMISOS_LEGIBLES = 0o644  # rw-r--r--: los de save_json
TIEMPOS = {"tiempo_total_s", "tiempo_reloj_s"}  # claves de tiempo que cambian entre corridas

MLP_PRUEBA = {  # bloque mlp heredado: válido y rápido (max_epocas pequeño)
    "capas": [8], "activacion": "relu", "dropout": 0.0, "l2": 1.0e-4, "optimizador": "adam", "lr": 1.0e-2,
    "momento": 0.9, "batch": 64, "max_epocas": 3, "paciencia": 2, "device": "cpu", "pos_weight": 1.0,
}
ESPACIO_PRUEBA = {  # espacio pequeño para las búsquedas sintéticas
    "n_capas": [1, 2], "neuronas": [4, 8], "activacion": ["relu", "tanh"], "dropout": [0.0, 0.3],
    "l2": [1.0e-6, 1.0e-2], "optimizador": ["adam", "sgd"], "lr": [1.0e-4, 1.0e-2], "batch": [32, 64],
}
ESPACIO_B33 = {  # espacio de la Parte B §3.3 (el de configs/pd/hpo.yaml)
    "n_capas": [1, 4], "neuronas": [16, 32, 64, 128, 256], "activacion": ["relu", "leaky_relu", "tanh"],
    "dropout": [0.0, 0.5], "l2": [1.0e-6, 1.0e-2], "optimizador": ["adam", "sgd"], "lr": [1.0e-4, 1.0e-2],
    "batch": [256, 512, 1024, 2048],
}
ESPACIO_PEQUENO_SET = [  # --set que achican el espacio versionado en las pruebas 18 y 22
    "hpo.espacio.n_capas=[1, 2]", "hpo.espacio.neuronas=[4, 8]", "hpo.espacio.batch=[32, 64]",
    "mlp.max_epocas=2",  # pocas épocas: rápido
]

# %% [markdown]
# ## Datos sintéticos y dobles
#
# - `_datos`: DataFrame con `id`/`split` (para la huella) y matrices `{split: (X float32, y, ids)}` de una logística
#   conocida, en el formato de `experimento.cargar_datos`.
# - `_Cargar`: doble de `cargar` que cuenta sus llamadas (la validación debe fallar **antes** de llamarlo).
# - `_MatricesSinTest`: diccionario de matrices cuyo acceso a `"test"` lanza un error propio (prueba 11).
# - `_reloj`: reloj falso que avanza `PASO_RELOJ` en cada llamada (tiempos exactos).

# %%
def _datos():
    """(df con id/split en el orden del "parquet", {split: (X float32, y int64, ids)}) sintéticos."""
    rng = np.random.default_rng(SEMILLA_DATOS)  # generador de los datos sintéticos
    filas, matrices = [], {}  # filas del df y matrices por split
    for split, n in N_FILAS.items():  # train, val, test
        X = rng.normal(size=(n, N_COLUMNAS)).astype(np.float32)  # variables ya "estandarizadas"
        y = rng.binomial(1, expit(X @ PESOS_GEN + INTERCEPTO_GEN)).astype(np.int64)  # etiquetas 0/1
        ids = np.array([f"{split}{i:04d}" for i in range(n)], dtype=object)  # ids str
        matrices[split] = (X, y, ids)  # tripleta del split
        filas += [{"id": i, "split": split} for i in ids]  # líneas del df
    return pd.DataFrame(filas), matrices  # mismo formato que experimento.cargar_datos


class _Cargar:
    """Doble de ``cargar``: devuelve (df, matrices) y registra cada llamada."""

    def __init__(self, df, matrices):
        self.df, self.matrices = df, matrices  # lo que devuelve
        self.llamadas = []  # argumentos de cada llamada

    def __call__(self, *args, **kwargs):
        self.llamadas.append((args, kwargs))  # registra la llamada
        return self.df, self.matrices  # (df, {split: (X, y, ids)})


class _AccesoATest(Exception):
    """Error propio: alguien leyó las matrices de test (no se confunde con los errores que captura la búsqueda)."""


class _MatricesSinTest(Mapping):
    """Matrices por split que lanzan ``_AccesoATest`` si se lee ``"test"`` y registran las claves leídas."""

    def __init__(self, matrices):
        self._matrices = matrices  # train, val y test
        self.leidas = []  # claves leídas, en orden

    def __getitem__(self, clave):
        self.leidas.append(clave)  # registra la lectura
        if clave == "test":  # test nunca llega a la búsqueda
            raise _AccesoATest("la búsqueda leyó las matrices de test")  # fuga de información
        return self._matrices[clave]  # train o val

    def __iter__(self):
        return iter(self._matrices)  # las tres claves existen: la prueba es sobre la lectura

    def __len__(self):
        return len(self._matrices)  # tres splits


def _reloj(paso=PASO_RELOJ):
    """Reloj falso: 0, paso, 2·paso, … en llamadas sucesivas."""
    contador = itertools.count(start=0.0, step=paso)  # flotantes exactos con paso 1.0
    return lambda: next(contador)  # misma firma que time.perf_counter


def _ahora():
    """`ahora` inyectado: siempre FECHA_FIJA."""
    return FECHA_FIJA  # fecha fija: el YAML es reproducible


def _rutas(base: Path) -> RutasPD:
    """RutasPD con un preprocesador falso (solo para la huella); parquet, resultados y modelos no existen."""
    base.mkdir(parents=True, exist_ok=True)  # carpeta de la prueba
    pre = base / "preprocesador_falso.json"  # JSON cualquiera
    pre.write_text('{"falso": true}\n', encoding="utf-8")  # contenido fijo: huella fija
    return RutasPD(base / "no_existe.parquet", pre, base / "rutas_resultados", base / "rutas_modelos")  # rutas


def _cfg(base: Path, n_ensayos=N_ENSAYOS, espacio=None, base_seed=0) -> dict:
    """Config completa (como la deja load_config) con destino y salida en ``base``."""
    return {  # raíz de la config
        "base_seed": base_seed,  # semilla del muestreador y de los ensayos
        "logistica": {"rejilla_C": [1.0], "max_iter": 100, "tol": 1.0e-6},  # bloque ajeno: se ignora
        "mlp": copy.deepcopy(MLP_PRUEBA),  # bloque mlp heredado
        "busqueda": {"tipo": "sin_busqueda"},  # procedencia heredada de mlp_final.yaml
        "hpo": {  # bloque de la búsqueda
            "n_ensayos": n_ensayos,  # presupuesto
            "destino": str(base / "mlp_final.yaml"),  # YAML que se reescribe
            "salida": str(base / "resultados"),  # carpeta de hpo_ensayos.json
            "espacio": copy.deepcopy(ESPACIO_PRUEBA if espacio is None else espacio),  # espacio de búsqueda
        },
    }


def _correr(base: Path, cfg=None, df_matrices=None, **kwargs):
    """Corre ``buscar`` sobre datos sintéticos; devuelve (resumen, json escrito, ruta del yaml, cargar, rutas)."""
    df, matrices = _datos() if df_matrices is None else df_matrices  # datos sintéticos
    cargar = _Cargar(df, matrices)  # doble de cargar
    rutas = _rutas(base / "entradas")  # preprocesador falso
    cfg = _cfg(base) if cfg is None else cfg  # config por defecto
    kwargs.setdefault("reloj", _reloj())  # reloj falso salvo que la prueba pida otro
    kwargs.setdefault("ahora", _ahora)  # fecha fija
    resumen = buscar(cfg, rutas=rutas, cargar=cargar, **kwargs)  # búsqueda completa
    escrito = load_json(Path(cfg["hpo"]["salida"]) / "hpo_ensayos.json")  # JSON escrito
    return resumen, escrito, Path(cfg["hpo"]["destino"]), cargar, rutas  # todo lo que inspeccionan las pruebas


def _sin_tiempos(resumen: dict) -> dict:
    """Copia del JSON sin los campos de tiempo (cambian entre corridas)."""
    r = {k: v for k, v in copy.deepcopy(resumen).items() if k not in TIEMPOS}  # sin tiempos globales
    for e in r["ensayos"]:  # y sin el tiempo de cada ensayo
        e.pop("tiempo_s")  # cronometrado
    return r  # comparable entre corridas


def _sha256_particion(df):
    """SHA-256 de referencia: texto UTF-8 de las líneas f"{id},{split}\\n" en el orden del df."""
    texto = "".join(f"{i},{s}\n" for i, s in zip(df["id"], df["split"]))  # líneas id,split
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()  # hexadecimal


def _modelo_que_falla(error, fallan):
    """Subclase de MLPPD cuyo fit lanza ``error`` en las llamadas (desde 0) de ``fallan``; las demás entrenan."""
    llamadas = itertools.count()  # una llamada a fit por ensayo (n_jobs = 1)

    class _Modelo(MLPPD):
        def fit(self, *args, **kwargs):
            if next(llamadas) in fallan:  # ensayo que debe fallar
                raise error  # FloatingPointError o ValueError
            return super().fit(*args, **kwargs)  # entrenamiento real

    return _Modelo  # modelo_cls inyectable


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    """Una búsqueda sintética de N_ENSAYOS ensayos con reloj falso; la comparten las pruebas 9, 10 y 17."""
    base = tmp_path_factory.mktemp("hpo_corrida")  # carpeta de la corrida
    df, matrices = _datos()  # datos sintéticos
    resumen, escrito, destino, cargar, rutas = _correr(base, df_matrices=(df, matrices))  # búsqueda
    return {"base": base, "df": df, "matrices": matrices, "resumen": resumen, "json": escrito,
            "destino": destino, "cargar": cargar, "rutas": rutas}  # todo lo que se inspecciona


# %% [markdown]
# ## 1–2. Validación antes de cargar datos
#
# Toda falla de la config es `ValueError` y ocurre **antes** de leer los datos: el doble `cargar` no se llama y no
# se crea ni la carpeta de salida ni el YAML de destino. Cada caso es una mutación de la config válida.

# %%
def _quitar(ruta):
    """Mutación: borra la clave ``ruta`` (tupla de claves) de la config."""
    def mutar(cfg):
        nodo = cfg  # baja hasta el padre
        for clave in ruta[:-1]:  # tramos intermedios
            nodo = nodo[clave]  # siguiente nivel
        del nodo[ruta[-1]]  # quita la hoja
    return mutar  # función que muta en el sitio


def _poner(ruta, valor):
    """Mutación: asigna ``valor`` a la clave ``ruta`` (tupla de claves) de la config."""
    def mutar(cfg):
        nodo = cfg  # baja hasta el padre
        for clave in ruta[:-1]:  # tramos intermedios
            nodo = nodo[clave]  # siguiente nivel
        nodo[ruta[-1]] = valor  # asigna la hoja
    return mutar  # función que muta en el sitio


E = ("hpo", "espacio")  # prefijo de las claves del espacio

CASOS_1 = {  # prueba 1 del plan: claves, rangos, opciones y n_ensayos
    "hpo_clave_sobrante": _poner(("hpo", "timeout"), 60),  # timeout no es una clave del bloque
    "hpo_falta_destino": _quitar(("hpo", "destino")),  # sin destino no hay a dónde escribir
    "espacio_clave_sobrante": _poner((*E, "momento"), [0.0, 0.9]),  # momento no se busca
    "espacio_falta_batch": _quitar((*E, "batch")),  # el espacio debe estar completo
    "l2_min_mayor_que_max": _poner((*E, "l2"), [1.0e-2, 1.0e-6]),  # mín > máx
    "lr_min_mayor_que_max": _poner((*E, "lr"), [1.0e-2, 1.0e-4]),  # mín > máx
    "l2_limite_cero": _poner((*E, "l2"), [0.0, 1.0e-2]),  # log de 0
    "lr_limite_negativo": _poner((*E, "lr"), [-1.0e-3, 1.0e-2]),  # log de un negativo
    "activacion_no_admitida": _poner((*E, "activacion"), ["relu", "sigmoid"]),  # no está en ACTIVACIONES
    "optimizador_no_admitido": _poner((*E, "optimizador"), ["adamw"]),  # no es L2 (Parte B §6)
    "n_ensayos_cero": _poner(("hpo", "n_ensayos"), 0),  # sin ensayos no hay mejor
    "n_ensayos_flotante": _poner(("hpo", "n_ensayos"), 2.5),  # no es entero
    "n_ensayos_bool": _poner(("hpo", "n_ensayos"), True),  # bool no es un entero de la config
}


@pytest.mark.parametrize("mutar", CASOS_1.values(), ids=CASOS_1.keys())
def test_01_validacion_antes_de_cargar(mutar, tmp_path):
    """Clave sobrante o faltante, mín > máx, límite ≤ 0 en log, opción no admitida o n_ensayos inválido → ValueError
    sin llamar a cargar ni crear archivos (atrapa validar después de leer los datos o un KeyError/TypeError)."""
    cfg = _cfg(tmp_path)  # config válida
    mutar(cfg)  # la daña
    with pytest.raises(ValueError):  # validación pura
        validar_cfg(cfg)  # sin datos
    cargar = _Cargar(*_datos())  # doble que cuenta llamadas
    with pytest.raises(ValueError):  # la búsqueda valida primero
        buscar(cfg, rutas=_rutas(tmp_path / "entradas"), cargar=cargar, reloj=_reloj(), ahora=_ahora)  # búsqueda
    assert cargar.llamadas == []  # no leyó datos
    assert not (tmp_path / "resultados").exists()  # no creó la carpeta de salida
    assert not (tmp_path / "mlp_final.yaml").exists()  # no escribió el YAML


@pytest.mark.parametrize("clave", ["pos_weight", "device"])
def test_01_espacio_prohibido_con_mensaje(clave, tmp_path):
    """pos_weight o device en el espacio → ValueError que los nombra (atrapa buscar pos_weight, que descalibraría la
    PD, o mezclar dispositivos entre ensayos, con un mensaje genérico que no dice por qué)."""
    cfg = _cfg(tmp_path)  # config válida
    cfg["hpo"]["espacio"][clave] = [1.0, 2.0] if clave == "pos_weight" else ["cpu", "cuda"]  # clave prohibida
    with pytest.raises(ValueError, match=clave):  # mensaje específico
        validar_cfg(cfg)  # sin datos


CASOS_2 = {  # prueba 2 del plan: más reglas de validez
    "dropout_max_uno": _poner((*E, "dropout"), [0.0, 1.0]),  # validar_cfg_mlp exige dropout < 1
    "dropout_negativo": _poner((*E, "dropout"), [-0.1, 0.5]),  # probabilidad negativa
    "activacion_duplicada": _poner((*E, "activacion"), ["relu", "relu"]),  # duplicados sesgan el muestreo
    "batch_duplicado": _poner((*E, "batch"), [32, 32]),  # ídem
    "neuronas_duplicadas": _poner((*E, "neuronas"), [8, 8]),  # ídem
    "neuronas_cero": _poner((*E, "neuronas"), [0, 8]),  # una capa sin neuronas
    "neuronas_bool": _poner((*E, "neuronas"), [True, 8]),  # bool no es una anchura
    "neuronas_flotante": _poner((*E, "neuronas"), [8.0]),  # 8.0 no es un entero
    "batch_cero": _poner((*E, "batch"), [0]),  # minilote vacío
    "batch_bool": _poner((*E, "batch"), [False]),  # bool no es un tamaño
    "lista_vacia": _poner((*E, "activacion"), []),  # nada que elegir
    "lista_no_lista": _poner((*E, "optimizador"), "adam"),  # texto: se iteraría letra por letra
    "rango_longitud_3": _poner((*E, "lr"), [1.0e-4, 1.0e-3, 1.0e-2]),  # rango = [mín, máx]
    "rango_longitud_1": _poner((*E, "dropout"), [0.1]),  # falta el máximo
    "rango_no_lista": _poner((*E, "lr"), "1e-3"),  # texto
    "rango_bool": _poner((*E, "dropout"), [False, 0.5]),  # bool no es un real
    "rango_inf": _poner((*E, "lr"), [1.0e-4, math.inf]),  # no finito
    "rango_nan": _poner((*E, "dropout"), [math.nan, 0.5]),  # no finito
    "n_capas_cero": _poner((*E, "n_capas"), [0, 2]),  # red sin capas ocultas
    "n_capas_flotante": _poner((*E, "n_capas"), [1.5, 3]),  # número de capas no entero
    "n_capas_bool": _poner((*E, "n_capas"), [True, 2]),  # bool no es un entero
    "n_capas_min_mayor": _poner((*E, "n_capas"), [2, 1]),  # mín > máx
    "espacio_no_dict": _poner(("hpo", "espacio"), ["n_capas"]),  # el espacio es un mapeo
    "destino_vacio": _poner(("hpo", "destino"), ""),  # ruta vacía
    "destino_no_texto": _poner(("hpo", "destino"), 3),  # no es una ruta
    "salida_vacia": _poner(("hpo", "salida"), ""),  # ruta vacía
    "base_seed_negativa": _poner(("base_seed",), -1),  # semilla < 0
    "base_seed_texto": _poner(("base_seed",), "0"),  # texto
    "base_seed_bool": _poner(("base_seed",), True),  # bool no es una semilla
    "sin_base_seed": _quitar(("base_seed",)),  # sin semilla no hay reproducibilidad
    "sin_bloque_hpo": _quitar(("hpo",)),  # config de otro subcomando
    "hpo_no_dict": _poner(("hpo",), 40),  # bloque reemplazado por un valor
    "sin_bloque_mlp": _quitar(("mlp",)),  # sin mlp heredado
    "mlp_dropout_uno": _poner(("mlp", "dropout"), 1.0),  # mlp heredado inválido
    "mlp_falta_paciencia": _quitar(("mlp", "paciencia")),  # mlp heredado incompleto
    "mlp_clave_sobrante": _poner(("mlp", "perdida"), "bce"),  # clave de la LGD
}


@pytest.mark.parametrize("mutar", CASOS_2.values(), ids=CASOS_2.keys())
def test_02_validacion_extra_antes_de_cargar(mutar, tmp_path):
    """dropout ≥ 1, duplicados, anchuras 0/bool/flotantes, rangos mal formados o no finitos, rutas vacías, semilla
    inválida o mlp heredado inválido → ValueError antes de cargar (atrapa un ganador inválido que truena a mitad
    de 40 ensayos, o un KeyError/TypeError en lugar del ValueError del contrato)."""
    cfg = _cfg(tmp_path)  # config válida
    mutar(cfg)  # la daña
    with pytest.raises(ValueError):  # validación pura
        validar_cfg(cfg)  # sin datos
    cargar = _Cargar(*_datos())  # doble que cuenta llamadas
    with pytest.raises(ValueError):  # la búsqueda valida primero
        buscar(cfg, rutas=_rutas(tmp_path / "entradas"), cargar=cargar, reloj=_reloj(), ahora=_ahora)  # búsqueda
    assert cargar.llamadas == []  # no leyó datos
    assert not (tmp_path / "resultados").exists()  # no creó la carpeta de salida
    assert not (tmp_path / "mlp_final.yaml").exists()  # no escribió el YAML


def test_02_validar_cfg_devuelve_lo_validado(tmp_path):
    """validar_cfg devuelve (hpo, mlp heredado, base_seed) y acepta listas de un elemento y rangos con mín = máx
    (atrapa un validador más estricto que la decisión de Andrés)."""
    cfg = _cfg(tmp_path, base_seed=7)  # config válida con otra semilla
    cfg["hpo"]["espacio"].update(neuronas=[8], dropout=[0.2, 0.2], lr=[1.0e-3, 1.0e-3], n_capas=[2, 2])  # bordes
    hpo_v, mlp_v, semilla = validar_cfg(cfg)  # validación
    assert hpo_v == cfg["hpo"]  # mismo contenido
    assert mlp_v == cfg["mlp"]  # mlp heredado tal cual
    assert semilla == 7  # base_seed de la config


# %% [markdown]
# ## 3. `sugerir`
#
# Con `optuna.trial.FixedTrial` los valores se fijan por nombre: si el código usa otros nombres que `n_capas`,
# `neuronas_capa{k}` (k desde 1), `activacion`, `dropout`, `l2`, `optimizador`, `lr` y `batch`, FixedTrial falla.

# %%
def test_03_sugerir_con_fixed_trial():
    """Parámetros ya armados: capas = n_capas anchuras elegidas por separado ([32, 16, 32]) y {**mlp, **params}
    pasa validar_cfg_mlp (atrapa una sola anchura para todas las capas, nombres distintos o capas en pirámide)."""
    fijos = {  # un valor por nombre de parámetro de Optuna
        "n_capas": 3, "neuronas_capa1": 32, "neuronas_capa2": 16, "neuronas_capa3": 32, "activacion": "tanh",
        "dropout": 0.1, "l2": 1.0e-4, "optimizador": "sgd", "lr": 1.0e-3, "batch": 512,
    }
    p = sugerir(optuna.trial.FixedTrial(fijos), ESPACIO_B33)  # parámetros armados
    assert p == {"capas": [32, 16, 32], "activacion": "tanh", "dropout": 0.1, "l2": 1.0e-4, "optimizador": "sgd",
                 "lr": 1.0e-3, "batch": 512}  # armado exacto
    assert set(p) == set(CLAVES_BUSCADAS)  # solo las claves buscadas
    validar_cfg_mlp({**MLP_PRUEBA, **p})  # el bloque resultante es válido


def test_03_sugerir_una_capa_no_pide_mas():
    """Con n_capas = 1 solo se pide neuronas_capa1 (atrapa sugerir siempre 4 anchuras y recortar después, que
    ensuciaría el espacio del TPE con parámetros que no se usan)."""
    fijos = {"n_capas": 1, "neuronas_capa1": 64, "activacion": "relu", "dropout": 0.0, "l2": 1.0e-6,
             "optimizador": "adam", "lr": 1.0e-2, "batch": 2048}  # sin neuronas_capa2..4
    p = sugerir(optuna.trial.FixedTrial(fijos), ESPACIO_B33)  # FixedTrial falla si se pide otro nombre
    assert p["capas"] == [64]  # una sola capa


def test_a02_sugerir_declara_las_distribuciones():
    """l2 y lr log-uniformes, dropout lineal, n_capas entero y anchuras categóricas por capa (atrapa log=False en lr o
    l2, que FixedTrial no ve)."""
    estudio = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))  # estudio cualquiera
    trial = estudio.ask()  # ensayo real
    p = sugerir(trial, ESPACIO_B33)  # muestrea
    D = optuna.distributions  # alias
    esperadas = {  # distribución de cada nombre
        "n_capas": D.IntDistribution(1, 4),  # entero con extremos incluidos
        "activacion": D.CategoricalDistribution(ESPACIO_B33["activacion"]),  # categórica
        "dropout": D.FloatDistribution(0.0, 0.5),  # uniforme lineal
        "l2": D.FloatDistribution(1.0e-6, 1.0e-2, log=True),  # log-uniforme
        "optimizador": D.CategoricalDistribution(ESPACIO_B33["optimizador"]),  # categórica
        "lr": D.FloatDistribution(1.0e-4, 1.0e-2, log=True),  # log-uniforme
        "batch": D.CategoricalDistribution(ESPACIO_B33["batch"]),  # categórica
    }
    for k in range(1, len(p["capas"]) + 1):  # una anchura por capa
        esperadas[f"neuronas_capa{k}"] = D.CategoricalDistribution(ESPACIO_B33["neuronas"])  # categórica
    assert trial.distributions == esperadas  # exactamente estas distribuciones


# %% [markdown]
# ## 4. `en_borde`
#
# Un valor está en el borde si queda a $\le f\cdot(\text{máx} - \text{mín})$ de un extremo, con $f = 0.05$; en
# $\log_{10}$ para `lr` y `l2`, lineal para `dropout`. Con mín = máx → `None`.
#
# Con el espacio de B §3.3: lr ∈ [1e-4, 1e-2] (2 décadas: borde a 0.1 décadas), l2 ∈ [1e-6, 1e-2] (4 décadas:
# 0.2) y dropout ∈ [0, 0.5] (0.025).

# %%
def _params(lr=1.0e-3, l2=1.0e-4, dropout=0.25):
    """Parámetros de un ensayo con lr, l2 y dropout dados (los demás cualesquiera)."""
    return {"capas": [32], "activacion": "relu", "dropout": dropout, "l2": l2, "optimizador": "adam", "lr": lr,
            "batch": 512}  # parámetros armados


@pytest.mark.parametrize(("params", "esperado"), [
    (_params(lr=1.05e-4, l2=9.0e-3, dropout=0.02), {"lr": "min", "l2": "max", "dropout": "min"}),  # del plan
    (_params(lr=9.6e-3, l2=1.5e-6, dropout=0.48), {"lr": "max", "l2": "min", "dropout": "max"}),  # al revés
    (_params(), {"lr": None, "l2": None, "dropout": None}),  # al centro
    (_params(dropout=0.025), {"lr": None, "l2": None, "dropout": "min"}),  # justo en el 5 % (≤)
    (_params(dropout=0.026), {"lr": None, "l2": None, "dropout": None}),  # justo fuera
    (_params(lr=10 ** (-4 + 0.09)), {"lr": "min", "l2": None, "dropout": None}),  # 0.09 décadas: dentro
    (_params(lr=10 ** (-4 + 0.11)), {"lr": None, "l2": None, "dropout": None}),  # 0.11 décadas: fuera
], ids=["plan", "al_reves", "centro", "dropout_justo", "dropout_fuera", "lr_dentro_log", "lr_fuera_log"])
def test_04_en_borde(params, esperado):
    """Regla del 5 % en log10 (lr, l2) y lineal (dropout) (atrapa medir lr en escala lineal: 1.05e-4 no estaría en
    el borde de [1e-4, 1e-2] y 10^(-3.91) tampoco; o usar < en lugar de ≤)."""
    assert FRACCION_BORDE == 0.05  # decisión de Andrés
    assert en_borde(params, ESPACIO_B33) == esperado  # exacto


def test_04_en_borde_rango_degenerado_y_fraccion():
    """mín = máx → None; la fracción es inyectable (atrapa dividir entre un ancho 0 o ignorar ``fraccion``)."""
    espacio = {**ESPACIO_B33, "lr": [1.0e-3, 1.0e-3], "dropout": [0.0, 0.0], "l2": [1.0e-4, 1.0e-4]}  # degenerado
    assert en_borde(_params(dropout=0.0), espacio) == {"lr": None, "l2": None, "dropout": None}  # sin borde
    assert en_borde(_params(lr=10 ** -3.6), ESPACIO_B33) == {"lr": None, "l2": None, "dropout": None}  # con 5 %
    assert en_borde(_params(lr=10 ** -3.6), ESPACIO_B33, fraccion=0.3) == {  # 0.4 décadas ≤ 0.3 · 2 décadas
        "lr": "min", "l2": None, "dropout": None}  # la fracción cambia la regla


# %% [markdown]
# ## 5. `elegir_mejor`
#
# Argmin de `logloss_val` entre los ensayos `COMPLETE`; empate → menor `numero` (no el primero de la lista);
# ninguno completo → `RuntimeError`.

# %%
def test_05_elegir_mejor():
    """Argmin, empate → menor número aunque llegue después en la lista, ignora FAIL aunque traiga un valor menor
    (atrapa min() sobre la lista sin filtrar el estado o un desempate por posición)."""
    ensayos = [  # filas mínimas
        {"numero": 0, "estado": "COMPLETE", "logloss_val": 0.5},  # peor
        {"numero": 3, "estado": "COMPLETE", "logloss_val": 0.4},  # empata con el 2, llega antes
        {"numero": 1, "estado": "FAIL", "logloss_val": 0.1},  # menor, pero falló
        {"numero": 2, "estado": "COMPLETE", "logloss_val": 0.4},  # mejor por desempate
    ]
    assert elegir_mejor(ensayos)["numero"] == 2  # menor número entre los empatados


@pytest.mark.parametrize("ensayos", [
    [{"numero": 0, "estado": "FAIL", "logloss_val": None}, {"numero": 1, "estado": "FAIL", "logloss_val": None}],
    [],
], ids=["todos_fail", "vacio"])
def test_05_elegir_mejor_sin_completos(ensayos):
    """Sin ensayos completos → RuntimeError (atrapa un ValueError genérico de min() o elegir un FAIL)."""
    with pytest.raises(RuntimeError):  # no hay mejor
        elegir_mejor(ensayos)  # sin completos


# %% [markdown]
# ## 6. Estudio y muestreador
#
# Optuna 5.0.0 usa TPE **multivariado** por defecto en un objetivo (hallazgo de la lectura del plan). La prueba fija
# ese comportamiento: si cambia la versión, avisa. `n_startup_trials` se compara con el valor por defecto de la
# firma de `TPESampler` (sin escribir 10 a mano).

# %%
@pytest.mark.parametrize("semilla", [0, 5])
def test_06_crear_estudio_y_describir(semilla):
    """TPESampler con la semilla, NopPruner, minimizar, en memoria; describir_muestreador = {nombre, semilla,
    n_startup_trials, multivariate=True} (atrapa reportar un TPE univariado o una semilla escrita a mano)."""
    estudio = crear_estudio(semilla)  # estudio de la búsqueda
    assert isinstance(estudio.sampler, optuna.samplers.TPESampler)  # TPE
    assert isinstance(estudio.pruner, optuna.pruners.NopPruner)  # sin poda
    assert estudio.direction == optuna.study.StudyDirection.MINIMIZE  # log-loss: menor es mejor
    assert isinstance(estudio._storage, optuna.storages.InMemoryStorage)  # en memoria (API privada: solo aquí)
    defecto = inspect.signature(optuna.samplers.TPESampler).parameters["n_startup_trials"].default  # de la firma
    assert describir_muestreador(estudio) == {  # muestreador efectivo
        "nombre": "TPESampler", "semilla": semilla, "n_startup_trials": defecto, "multivariate": True}


def test_06_semilla_llega_al_muestreador():
    """Misma semilla → mismo primer valor; otra semilla → otro (atrapa registrar la semilla sin pasarla a
    TPESampler)."""
    def primero(semilla):  # primer valor muestreado por un estudio nuevo
        return crear_estudio(semilla).ask().suggest_float("x", 0.0, 1.0)  # ensayo 0
    assert primero(3) == primero(3)  # reproducible
    assert primero(3) != primero(4)  # depende de la semilla


# %% [markdown]
# ## 7. `texto_mlp_final` y `escribir_texto_atomico`
#
# El texto empieza con la cabecera "no se edita a mano", `load_config` lo lee sin pérdida (también un flotante con
# 17 cifras) y la escritura es atómica: si `os.replace` falla, el archivo anterior queda intacto y no quedan
# temporales.

# %%
MLP_GANADOR = {**MLP_PRUEBA, "capas": [16, 8], "lr": 0.0012345678901234567, "l2": 3.3e-5}  # flotantes largos
BUSQUEDA_EJEMPLO = {"tipo": "optuna", "ensayo": 2, "n_ensayos": 3, "tiempo_s": 1.5,
                    "evidencia": {"sha256_particion": "a" * 64, "sha256_preprocesador": "b" * 64},
                    "fecha": FECHA_FIJA.isoformat()}  # bloque busqueda de ejemplo


def test_07_texto_mlp_final_se_lee_igual(tmp_path):
    """Cabecera en la 1.ª línea, YAML que load_config lee igual (flotantes exactos), permisos 0o644 y sin temporales
    (atrapa perder cifras al serializar, un YAML sin cabecera o un temporal que queda en configs/)."""
    texto = texto_mlp_final(MLP_GANADOR, BUSQUEDA_EJEMPLO)  # texto del YAML
    assert texto.splitlines()[0] == CABECERA  # cabecera exacta
    assert cargar_yaml(texto) == {"mlp": MLP_GANADOR, "busqueda": BUSQUEDA_EJEMPLO}  # sin pérdida
    ruta = tmp_path / "mlp_final.yaml"  # destino
    assert escribir_texto_atomico(str(ruta), texto) == ruta  # acepta str y devuelve Path
    assert ruta.read_text(encoding="utf-8") == texto  # contenido exacto
    assert load_config(ruta) == {"mlp": MLP_GANADOR, "busqueda": BUSQUEDA_EJEMPLO}  # el cargador del proyecto
    assert stat.S_IMODE(ruta.stat().st_mode) == PERMISOS_LEGIBLES  # legible por todos
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mlp_final.yaml"]  # sin temporales


def test_07_escritura_fallida_conserva_el_anterior(tmp_path, monkeypatch):
    """Si os.replace falla, el archivo anterior queda intacto y se borra el temporal (atrapa escribir directo sobre
    el destino, que dejaría un mlp_final.yaml a medias que B5 heredaría)."""
    ruta = tmp_path / "mlp_final.yaml"  # destino con contenido previo
    ruta.write_text("previo\n", encoding="utf-8")  # versión anterior

    def falla(*args, **kwargs):  # os.replace que no puede mover
        raise OSError("disco lleno (simulado)")  # falla del sistema

    monkeypatch.setattr(os, "replace", falla)  # toda llamada a os.replace falla
    with pytest.raises(OSError):  # la falla se propaga
        escribir_texto_atomico(ruta, texto_mlp_final(MLP_GANADOR, BUSQUEDA_EJEMPLO))  # intento de escritura
    assert ruta.read_text(encoding="utf-8") == "previo\n"  # intacto
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mlp_final.yaml"]  # sin temporales


# %% [markdown]
# ## 8. Listas de un elemento y rangos con mín = máx
#
# Son válidos (decisión de Andrés) y fijan el parámetro en todos los ensayos.

# %%
def test_08_espacio_degenerado_fija_los_parametros(tmp_path):
    """Con listas de un elemento y rangos mín = máx, los 3 ensayos usan los mismos parámetros y en_borde = None
    (atrapa suggest_float(low == high) rechazado o un rango degenerado tomado como borde)."""
    espacio = {"n_capas": [2, 2], "neuronas": [8], "activacion": ["tanh"], "dropout": [0.2, 0.2],
               "l2": [1.0e-4, 1.0e-4], "optimizador": ["adam"], "lr": [1.0e-3, 1.0e-3], "batch": [64]}  # fijo
    _, escrito, _, _, _ = _correr(tmp_path, cfg=_cfg(tmp_path, espacio=espacio))  # búsqueda
    fijos = {"capas": [8, 8], "activacion": "tanh", "dropout": 0.2, "l2": 1.0e-4, "optimizador": "adam",
             "lr": 1.0e-3, "batch": 64}  # los únicos valores posibles
    assert [e["parametros"] for e in escrito["ensayos"]] == [fijos] * N_ENSAYOS  # iguales en todos
    assert escrito["en_borde"] == {"lr": None, "l2": None, "dropout": None}  # ancho 0: sin borde


# %% [markdown]
# ## 9. Forma de `hpo_ensayos.json`
#
# Claves exactas, presupuesto (`tiempo_total_s` = suma de los `tiempo_s` de los fit, con el reloj falso exacto),
# estados, evidencia con dos SHA-256 y `mlp_base` con solo las claves fijas.

# %%
def test_09_forma_del_json(corrida):
    """Claves exactas, tiempos exactos con el reloj falso, estados, evidencia y mlp_base fijo (atrapa un JSON que
    omite el presupuesto, guarda el bloque mlp completo o calcula la huella sobre otros datos)."""
    j, df = corrida["json"], corrida["df"]  # JSON escrito y df de la partición
    assert set(j) == CLAVES_JSON  # claves exactas
    assert j["criterio"] == "logloss_val"  # criterio
    assert j["podador"] == "NopPruner"  # sin poda
    assert j["semilla_ensayos"] == 0  # base_seed
    assert j["muestreador"] == describir_muestreador(crear_estudio(0))  # mismo muestreador que el estudio
    assert j["versiones"] == {"optuna": optuna.__version__, "torch": str(torch.__version__)}  # versiones
    assert j["hilos_torch"] == torch.get_num_threads()  # hilos de torch (determinismo)
    assert j["espacio"] == ESPACIO_PRUEBA  # espacio validado
    assert j["mlp_base"] == {k: MLP_PRUEBA[k] for k in CLAVES_FIJAS}  # solo las claves fijas
    assert j["n_ensayos"] == N_ENSAYOS  # presupuesto en configuraciones
    assert {k: v for k, v in j["estados"].items() if v} == {"COMPLETE": N_ENSAYOS}  # conteo por estado
    assert [e["numero"] for e in j["ensayos"]] == list(range(N_ENSAYOS))  # en orden
    for e in j["ensayos"]:  # cada ensayo
        assert set(e) == CLAVES_ENSAYO  # sin "error" si completó
        assert e["estado"] == "COMPLETE"  # completó
        assert set(e["parametros"]) == set(CLAVES_BUSCADAS)  # parámetros armados
        assert e["tiempo_s"] == PASO_RELOJ  # un paso del reloj por fit
    assert j["tiempo_total_s"] == N_ENSAYOS * PASO_RELOJ  # suma exacta de los fit
    assert j["tiempo_reloj_s"] >= j["tiempo_total_s"]  # el reloj de pared incluye a los fit
    assert set(j["mejor"]) == CLAVES_MEJOR  # bloque mejor
    fila = elegir_mejor(j["ensayos"])  # regla de elección sobre las filas escritas
    assert j["mejor"] == {k: fila[k] for k in CLAVES_MEJOR}  # coherente con las filas
    assert j["en_borde"] == en_borde(j["mejor"]["parametros"], ESPACIO_PRUEBA)  # regla del 5 %
    assert set(j["evidencia"]) == CLAVES_EVIDENCIA  # las dos huellas del núcleo §6
    assert j["evidencia"] == {  # huellas independientes de la implementación
        "sha256_particion": _sha256_particion(df),  # partición del df cargado
        "sha256_preprocesador": hashlib.sha256(corrida["rutas"].preprocesador.read_bytes()).hexdigest(),  # bytes
    }
    assert json.loads(json.dumps(to_jsonable(corrida["resumen"]))) == j  # buscar devuelve lo que escribió
    assert not corrida["rutas"].resultados.exists()  # la salida es hpo.salida, no rutas.resultados
    assert not corrida["rutas"].modelos.exists()  # B4 no guarda modelos


# %% [markdown]
# ## 10. Criterio
#
# La `logloss_val` de un ensayo es la de reentrenar `{**mlp, **parametros}` con la misma semilla y medir en val; y
# coincide con la pérdida de validación de la mejor época que reporta `entrenar` (la misma BCE en float32).

# %%
def test_10_criterio_es_la_logloss_de_val(corrida):
    """Reentrenar el ensayo da la misma logloss_val, auc_val, épocas y parámetros; |logloss_val −
    perdida_val_entrenar| < 1e-5 (atrapa medir en train, sin restaurar los mejores pesos o con otra semilla)."""
    X_tr, y_tr, _ = corrida["matrices"]["train"]  # train
    X_val, y_val, _ = corrida["matrices"]["val"]  # val
    for e in corrida["json"]["ensayos"]:  # cada ensayo
        m = MLPPD().fit(X_tr, y_tr, X_val, y_val, {"mlp": {**MLP_PRUEBA, **e["parametros"]}}, seed=0)  # reentreno
        p = m.predict_proba(X_val)  # PD de val
        assert e["logloss_val"] == pytest.approx(logloss(y_val, p), rel=1e-12)  # mismo criterio
        assert e["auc_val"] == pytest.approx(auc(y_val, p), rel=1e-12)  # misma AUC
        assert e["perdida_val_entrenar"] == m.resultado.mejor_perdida_val  # la del bucle
        assert (e["epocas"], e["mejor_epoca"]) == (m.resultado.epocas_corridas, m.resultado.mejor_epoca)  # épocas
        assert e["n_parametros"] == m.n_parametros()  # parámetros entrenables
        assert abs(e["logloss_val"] - e["perdida_val_entrenar"]) < 1e-5  # mismo split y mismos pesos


def test_a04_medir_ensayo_usa_lo_inyectado():
    """medir_ensayo mide con el criterio y la AUC inyectados sobre (y_val, predict_proba(X_val)) y copia cada campo
    del resultado sin cruzarlos (atrapa epocas ↔ mejor_epoca o medir sobre otro arreglo)."""
    X_val, y_val = np.zeros((4, 2), dtype=np.float32), np.array([0, 1, 0, 1])  # val de juguete
    p_fija = np.array([0.1, 0.8, 0.3, 0.6])  # PD que devuelve el modelo falso
    recibidos = []  # argumentos que reciben las métricas

    class _Resultado:  # campos con valores distintos entre sí
        mejor_perdida_val, epocas_corridas, mejor_epoca = 0.25, 7, 4  # perdida, épocas, mejor época

    class _Modelo:  # modelo falso ya "entrenado"
        resultado = _Resultado()  # resultado del fit

        def predict_proba(self, X):
            assert X is X_val  # mide sobre val
            return p_fija  # PD fija

        def n_parametros(self):
            return 123  # parámetros

    def criterio(y, p):  # criterio falso
        recibidos.append(("criterio", y, p))  # registra
        return 0.5  # valor fijo

    def auc_fn(y, p):  # AUC falsa
        recibidos.append(("auc", y, p))  # registra
        return 0.75  # valor fijo

    r = medir_ensayo(_Modelo(), X_val, y_val, criterio=criterio, auc_fn=auc_fn)  # métricas
    assert r == {"logloss_val": 0.5, "auc_val": 0.75, "perdida_val_entrenar": 0.25, "epocas": 7, "mejor_epoca": 4,
                 "n_parametros": 123}  # cada campo en su sitio
    assert set(r) == CLAVES_METRICAS_ENSAYO  # claves exactas
    assert {n for n, _, _ in recibidos} == {"criterio", "auc"}  # usó ambas
    assert all(y is y_val and p is p_fija for _, y, p in recibidos)  # sobre (y_val, p de val)


# %% [markdown]
# ## 11. Test nunca se usa
#
# Las matrices son un mapeo cuyo acceso a `"test"` lanza `_AccesoATest`. La huella de la partición sí usa el df
# completo (id y split de todos), que no contiene ni X ni y.

# %%
def test_11_test_nunca_se_usa(tmp_path):
    """La búsqueda termina leyendo solo train y val (atrapa una fuga: elegir o medir con test)."""
    df, matrices = _datos()  # datos sintéticos
    sin_test = _MatricesSinTest(matrices)  # acceso a test prohibido
    _correr(tmp_path, df_matrices=(df, sin_test))  # termina sin _AccesoATest
    assert set(sin_test.leidas) <= {"train", "val"}  # nunca leyó test
    assert {"train", "val"} <= set(sin_test.leidas)  # y sí leyó train y val


# %% [markdown]
# ## 12. `FloatingPointError` en un ensayo
#
# Un ensayo que diverge queda `FAIL` con su `error` y su `tiempo_s` (que cuenta en el presupuesto) y la búsqueda
# sigue. Si todos fallan → `RuntimeError` y no se escribe nada (el YAML anterior queda intacto).

# %%
def test_12_ensayo_divergente_queda_fail(tmp_path):
    """El ensayo 1 diverge: FAIL con error, parámetros y tiempo_s exacto; los demás COMPLETE; el presupuesto incluye
    el FAIL (atrapa abortar 40 ensayos por uno o un presupuesto que omite los fallidos)."""
    modelo_cls = _modelo_que_falla(FloatingPointError(MENSAJE_DIVERGE), {1})  # diverge en el ensayo 1
    _, j, _, _, _ = _correr(tmp_path, modelo_cls=modelo_cls)  # búsqueda
    fallido = j["ensayos"][1]  # el ensayo 1
    assert fallido["estado"] == "FAIL"  # marcado como fallido
    assert fallido["error"] == MENSAJE_DIVERGE  # mensaje guardado (Optuna 5 no lo guarda)
    assert set(fallido) == CLAVES_ENSAYO | {"error"}  # mismas claves + error
    assert set(fallido["parametros"]) == set(CLAVES_BUSCADAS)  # parámetros registrados antes del fit
    assert fallido["tiempo_s"] == PASO_RELOJ  # cronometrado aunque falló
    assert all(fallido[k] is None for k in CLAVES_METRICAS_ENSAYO)  # sin métricas
    assert [e["estado"] for e in j["ensayos"]] == ["COMPLETE", "FAIL", "COMPLETE"]  # la búsqueda siguió
    assert {k: v for k, v in j["estados"].items() if v} == {"COMPLETE": 2, "FAIL": 1}  # conteo
    assert j["tiempo_total_s"] == N_ENSAYOS * PASO_RELOJ  # incluye el FAIL
    assert j["mejor"]["numero"] != 1  # el fallido no gana


def test_12_todos_fallan(tmp_path):
    """Todos divergen → RuntimeError, sin JSON y con el YAML anterior intacto (atrapa escribir un mlp_final.yaml
    sin ganador)."""
    cfg = _cfg(tmp_path)  # config válida
    destino = Path(cfg["hpo"]["destino"])  # YAML previo
    destino.write_text("previo\n", encoding="utf-8")  # versión anterior
    modelo_cls = _modelo_que_falla(FloatingPointError(MENSAJE_DIVERGE), set(range(N_ENSAYOS)))  # todos fallan
    with pytest.raises(RuntimeError):  # ningún ensayo completo
        buscar(cfg, rutas=_rutas(tmp_path / "entradas"), cargar=_Cargar(*_datos()), modelo_cls=modelo_cls,
               reloj=_reloj(), ahora=_ahora)  # búsqueda
    assert not (Path(cfg["hpo"]["salida"]) / "hpo_ensayos.json").exists()  # sin JSON
    assert destino.read_text(encoding="utf-8") == "previo\n"  # YAML intacto


# %% [markdown]
# ## 13. Otro error dentro del fit
#
# Solo `FloatingPointError` se captura: un `ValueError` es un bug y sale de la búsqueda sin escribir nada.

# %%
def test_13_value_error_sale_de_la_busqueda(tmp_path):
    """ValueError en el ensayo 1 → se propaga, no queda como FAIL y no se escribe nada (atrapa catch=(Exception,),
    que escondería un bug real como un ensayo fallido)."""
    cfg = _cfg(tmp_path)  # config válida
    modelo_cls = _modelo_que_falla(ValueError("bug simulado"), {1})  # bug en el ensayo 1
    with pytest.raises(ValueError, match="bug simulado"):  # se propaga tal cual
        buscar(cfg, rutas=_rutas(tmp_path / "entradas"), cargar=_Cargar(*_datos()), modelo_cls=modelo_cls,
               reloj=_reloj(), ahora=_ahora)  # búsqueda
    assert not (Path(cfg["hpo"]["salida"]) / "hpo_ensayos.json").exists()  # sin JSON
    assert not Path(cfg["hpo"]["destino"]).exists()  # sin YAML


# %% [markdown]
# ## 14. `--set` y `pos_weight` llegan a los ensayos
#
# Un espía (subclase de `MLPPD` inyectada por `modelo_cls`, sin monkeypatch) registra lo que recibe cada `fit`.

# %%
def test_14_set_llega_a_todos_los_ensayos(tmp_path):
    """mlp.paciencia=5 por --set, pos_weight = 1 y max_epocas heredados, seed = base_seed y solo train/val en cada
    fit (atrapa reconstruir el bloque mlp desde el YAML sin los --set o sembrar con base_seed + número)."""
    recibidos = []  # una entrada por fit

    class _Espia(MLPPD):
        def fit(self, X_tr, y_tr, X_val, y_val, cfg, seed):
            recibidos.append({"X_tr": X_tr, "X_val": X_val, "cfg": copy.deepcopy(cfg), "seed": seed})  # registra
            return super().fit(X_tr, y_tr, X_val, y_val, cfg, seed)  # entrenamiento real

    cfg = apply_overrides(_cfg(tmp_path, base_seed=3), ["mlp.paciencia=5"])  # --set como en el CLI
    df, matrices = _datos()  # datos sintéticos
    _, j, _, _, _ = _correr(tmp_path, cfg=cfg, df_matrices=(df, matrices), modelo_cls=_Espia)  # búsqueda
    assert len(recibidos) == N_ENSAYOS  # un fit por ensayo
    for r, e in zip(recibidos, j["ensayos"], strict=True):  # cada fit con su ensayo
        assert set(r["cfg"]) == {"mlp"}  # solo el bloque mlp
        assert r["cfg"]["mlp"] == {**cfg["mlp"], **e["parametros"]}  # heredado + parámetros del ensayo
        assert r["cfg"]["mlp"]["paciencia"] == 5  # el --set llegó
        assert r["cfg"]["mlp"]["pos_weight"] == 1.0  # pos_weight fijo
        assert r["seed"] == 3  # misma semilla en todos los ensayos
        assert r["X_tr"] is matrices["train"][0]  # entrena con train
        assert r["X_val"] is matrices["val"][0]  # valida con val


# %% [markdown]
# ## 15–16. Reproducibilidad y semilla
#
# Con el reloj real, dos búsquedas con la misma config dan el mismo JSON salvo los tiempos. Otra `base_seed`
# cambia el ensayo 0.

# %%
def test_15_reproducible(tmp_path):
    """Dos búsquedas → mismo JSON salvo tiempo_* y mismo YAML salvo busqueda.tiempo_s (atrapa una semilla que no
    llega al muestreador o al modelo, o un estado global que cambia entre corridas)."""
    a = _cfg(tmp_path / "a")  # primera corrida
    b = _cfg(tmp_path / "b")  # segunda corrida, otras carpetas (mismo preprocesador falso: misma huella)
    _, ja, ya, _, _ = _correr(tmp_path / "a", cfg=a, reloj=time.perf_counter)  # corrida a
    _, jb, yb, _, _ = _correr(tmp_path / "b", cfg=b, reloj=time.perf_counter)  # corrida b
    assert _sin_tiempos(ja) == _sin_tiempos(jb)  # mismo JSON salvo tiempos
    ya_cfg, yb_cfg = load_config(ya), load_config(yb)  # YAML escritos
    assert ya_cfg["mlp"] == yb_cfg["mlp"]  # mismo ganador
    ya_cfg["busqueda"].pop("tiempo_s")  # cronometrado
    yb_cfg["busqueda"].pop("tiempo_s")  # cronometrado
    assert ya_cfg["busqueda"] == yb_cfg["busqueda"]  # misma procedencia salvo el tiempo


def test_16_otra_semilla_cambia_el_ensayo_0(tmp_path):
    """base_seed = 1 cambia los parámetros del ensayo 0 respecto de base_seed = 0 (atrapa un muestreador sin semilla
    de la config, p. ej. una semilla fija en el código)."""
    _, j0, _, _, _ = _correr(tmp_path / "s0", cfg=_cfg(tmp_path / "s0", n_ensayos=1, base_seed=0))  # semilla 0
    _, j1, _, _, _ = _correr(tmp_path / "s1", cfg=_cfg(tmp_path / "s1", n_ensayos=1, base_seed=1))  # semilla 1
    assert j0["ensayos"][0]["parametros"] != j1["ensayos"][0]["parametros"]  # distinto ensayo 0
    assert (j0["semilla_ensayos"], j1["semilla_ensayos"]) == (0, 1)  # semilla registrada


# %% [markdown]
# ## 17. `mlp_final.yaml`
#
# `load_config` lo carga; su `mlp` es `{**mlp, **mejor.parametros}` y pasa `validar_cfg_mlp`; `busqueda` trae la
# procedencia. Un `base.yaml` temporal que lo hereda resuelve el ganador (lo que hará B5).

# %%
def test_17_mlp_final(corrida):
    """mlp = heredado + parámetros del mejor, válido; busqueda = {tipo: optuna, ensayo, n_ensayos, tiempo_s,
    evidencia, fecha}; se hereda (atrapa un YAML que B5 no puede usar o sin procedencia)."""
    destino, j = corrida["destino"], corrida["json"]  # YAML y JSON escritos
    texto = destino.read_text(encoding="utf-8")  # texto del YAML
    assert texto.splitlines()[0] == CABECERA  # cabecera
    cfg = load_config(destino)  # cargador del proyecto
    assert set(cfg) == {"mlp", "busqueda"}  # solo los dos bloques
    assert cfg["mlp"] == {**MLP_PRUEBA, **j["mejor"]["parametros"]}  # ganador sobre lo heredado
    validar_cfg_mlp(cfg["mlp"])  # bloque válido
    assert cfg["busqueda"] == {  # procedencia
        "tipo": "optuna", "ensayo": j["mejor"]["numero"], "n_ensayos": N_ENSAYOS,
        "tiempo_s": j["tiempo_total_s"], "evidencia": j["evidencia"], "fecha": FECHA_FIJA.isoformat()}
    assert set(cfg["busqueda"]) == CLAVES_BUSQUEDA  # claves exactas
    assert stat.S_IMODE(destino.stat().st_mode) == PERMISOS_LEGIBLES  # legible
    base = destino.parent / "base_temporal.yaml"  # base que lo hereda, junto al destino
    base.write_text(f"hereda: {destino.name}\nbase_seed: 0\n", encoding="utf-8")  # como configs/pd/base.yaml
    assert load_config(base)["mlp"] == cfg["mlp"]  # B5 hereda el ganador


# %% [markdown]
# ## 18. Idempotencia con la cadena versionada
#
# La cadena `hpo.yaml → base.yaml → mlp_final.yaml` se copia a `tmp_path` y se corre dos veces sobre el **mismo**
# destino: la segunda corrida hereda el ganador de la primera, pero como `mlp_base` solo guarda las claves fijas y
# los ensayos reemplazan las buscadas, el JSON es el mismo salvo los tiempos.

# %%
def _cadena(tmp_path):
    """Copia hpo.yaml, base.yaml y mlp_final.yaml versionados a tmp_path/configs/pd; devuelve la carpeta."""
    carpeta = tmp_path / "configs" / "pd"  # misma estructura relativa
    carpeta.mkdir(parents=True)  # carpeta nueva
    for nombre in ("hpo.yaml", "base.yaml", "mlp_final.yaml"):  # la cadena de herencia
        shutil.copy(ROOT / "configs" / "pd" / nombre, carpeta / nombre)  # copia (los reales no se tocan)
    return carpeta  # carpeta de la copia


def test_18_idempotente(tmp_path):
    """Dos corridas sobre el mismo destino → mismos ensayos y mismo JSON salvo tiempos (atrapa guardar el bloque mlp
    completo en el JSON, que cambiaría con la herencia circular)."""
    carpeta = _cadena(tmp_path)  # cadena copiada
    destino = carpeta / "mlp_final.yaml"  # destino compartido por las dos corridas
    jsons, cfgs = [], []  # JSON y config de cada corrida
    for k in range(2):  # dos corridas
        cfg = apply_overrides(load_config(carpeta / "hpo.yaml"), [  # se relee: hereda el ganador anterior
            f"hpo.destino={destino}", f"hpo.salida={tmp_path / f'res{k}'}", "hpo.n_ensayos=2",
            *ESPACIO_PEQUENO_SET])  # rutas temporales y espacio pequeño
        _, j, _, _, _ = _correr(tmp_path / f"corrida{k}", cfg=cfg)  # búsqueda
        jsons.append(j)  # guarda el JSON
        cfgs.append(cfg)  # y la config con que corrió
    assert _sin_tiempos(jsons[0]) == _sin_tiempos(jsons[1])  # idempotente
    assert cfgs[1]["mlp"] == {**cfgs[0]["mlp"], **jsons[0]["mejor"]["parametros"]}  # la 2.ª heredó el ganador
    assert load_config(carpeta / "base.yaml")["mlp"] == {  # base.yaml hereda el ganador de la 2.ª corrida
        **cfgs[1]["mlp"], **jsons[1]["mejor"]["parametros"]}  # heredado + parámetros del mejor


# %% [markdown]
# ## 19. Orden de escritura
#
# Se arman en memoria el JSON **y** el texto del YAML; luego se escribe el JSON y al final el YAML. Si la
# serialización del YAML falla, no queda el JSON; si falla la del JSON, el YAML anterior queda intacto.

# %%
def test_19_falla_del_yaml_no_deja_json(tmp_path, monkeypatch):
    """texto_mlp_final falla → no queda hpo_ensayos.json y el YAML anterior sigue (atrapa escribir el JSON antes de
    armar el YAML: quedaría un JSON sin su mlp_final.yaml)."""
    cfg = _cfg(tmp_path)  # config válida
    destino = Path(cfg["hpo"]["destino"])  # YAML previo
    destino.write_text("previo\n", encoding="utf-8")  # versión anterior

    def falla(*args, **kwargs):  # serialización imposible
        raise yaml.representer.RepresenterError("objeto no representable (simulado)")  # error de PyYAML

    monkeypatch.setattr(hpo, "texto_mlp_final", falla)  # buscar compone las funciones del módulo
    with pytest.raises(yaml.representer.RepresenterError):  # la falla se propaga
        buscar(cfg, rutas=_rutas(tmp_path / "entradas"), cargar=_Cargar(*_datos()), reloj=_reloj(), ahora=_ahora)
    assert not (Path(cfg["hpo"]["salida"]) / "hpo_ensayos.json").exists()  # sin JSON
    assert destino.read_text(encoding="utf-8") == "previo\n"  # YAML intacto


def test_19_falla_del_json_no_toca_el_yaml(tmp_path, monkeypatch):
    """save_json falla (p. ej. un NaN) → el YAML anterior queda intacto (atrapa escribir el YAML primero: B5
    heredaría un ganador sin su JSON de búsqueda)."""
    cfg = _cfg(tmp_path)  # config válida
    destino = Path(cfg["hpo"]["destino"])  # YAML previo
    destino.write_text("previo\n", encoding="utf-8")  # versión anterior

    def falla(*args, **kwargs):  # JSON imposible
        raise ValueError("Valor no finito (NaN o inf) (simulado)")  # el error de save_json

    monkeypatch.setattr(hpo, "save_json", falla)  # el JSON no se puede escribir
    with pytest.raises(ValueError, match="simulado"):  # la falla se propaga
        buscar(cfg, rutas=_rutas(tmp_path / "entradas"), cargar=_Cargar(*_datos()), reloj=_reloj(), ahora=_ahora)
    assert destino.read_text(encoding="utf-8") == "previo\n"  # YAML intacto


# %% [markdown]
# ## 20. Configs versionadas
#
# `configs/pd/hpo.yaml` pasa `validar_cfg` con 40 ensayos, el espacio de B §3.3 y `base_seed = 0`; el `mlp` de
# `configs/pd/base.yaml` es válido; `hpo.destino` es el mismo archivo que hereda `base.yaml`.

# %%
def test_20_configs_versionadas():
    """hpo.yaml válido (40, espacio B §3.3, base_seed 0, rutas del proyecto); base.yaml hereda mlp_final.yaml con un
    mlp válido y busqueda.tipo conocido (atrapa un YAML mal escrito antes de la corrida larga o un destino que
    base.yaml no hereda)."""
    cfg = load_config(ROOT / "configs" / "pd" / "hpo.yaml")  # config declarada
    hpo_v, mlp_v, semilla = validar_cfg(cfg)  # validación completa
    assert hpo_v["n_ensayos"] == 40  # presupuesto de B §3.3
    assert hpo_v["espacio"] == ESPACIO_B33  # espacio de B §3.3
    assert (hpo_v["destino"], hpo_v["salida"]) == ("configs/pd/mlp_final.yaml", "results/data/pd")  # rutas
    assert semilla == 0  # base_seed
    validar_cfg_mlp(mlp_v)  # mlp heredado válido
    base = load_config(ROOT / "configs" / "pd" / "base.yaml")  # config de B5
    validar_cfg_mlp(base["mlp"])  # bloque mlp heredado válido
    assert base["busqueda"]["tipo"] in {"sin_busqueda", "optuna"}  # procedencia conocida
    assert "logistica" in base  # B3 sigue en base.yaml
    crudo = cargar_yaml((ROOT / "configs" / "pd" / "base.yaml").read_text(encoding="utf-8"))  # sin resolver
    heredado = (ROOT / "configs" / "pd" / crudo["hereda"]).resolve()  # archivo que hereda base.yaml
    assert heredado == (ROOT / hpo_v["destino"]).resolve()  # es el que reescribe pd-hpo


def test_a03_mlp_fijo_y_buscadas_parten_el_bloque():
    """Claves fijas ∪ buscadas = CLAVES_MLP y disjuntas; mlp_fijo devuelve solo las fijas (atrapa una clave nueva del
    bloque mlp que no quedaría ni en mlp_base ni en los parámetros)."""
    fijas = mlp_fijo(MLP_PRUEBA)  # claves que no se buscan
    assert fijas == {k: MLP_PRUEBA[k] for k in CLAVES_FIJAS}  # exactamente las fijas
    assert set(fijas).isdisjoint(CLAVES_BUSCADAS)  # disjuntas
    assert set(fijas) | set(CLAVES_BUSCADAS) == set(CLAVES_MLP)  # cubren el bloque


def test_a01_validar_cfg_hpo_devuelve_copia(tmp_path):
    """La copia validada no comparte listas con la entrada (atrapa devolver el mismo objeto: mutar el espacio
    validado alteraría la config que imprime el CLI)."""
    cfg = _cfg(tmp_path)  # config válida
    copia = validar_cfg_hpo(cfg["hpo"])  # bloque validado
    assert copia == cfg["hpo"]  # mismo contenido
    copia["espacio"]["neuronas"].append(999)  # muta la copia
    assert cfg["hpo"]["espacio"]["neuronas"] == ESPACIO_PRUEBA["neuronas"]  # la entrada no cambió


def test_a05_buscar_no_muta_la_config(tmp_path):
    """buscar deja la config igual (atrapa mutar cfg["mlp"] con los parámetros del ganador: B5 y el CLI la
    reutilizan)."""
    cfg = _cfg(tmp_path)  # config válida
    antes = copy.deepcopy(cfg)  # copia de referencia
    _correr(tmp_path, cfg=cfg)  # búsqueda
    assert cfg == antes  # sin cambios


# %% [markdown]
# ## 21. `cargar_datos` y `evidencia` públicas
#
# B4 las vuelve públicas (antes `_cargar_datos` y `_evidencia`); `correr_logistica` sigue igual (lo cubren las
# pruebas de `test_logistica.py`). `hpo` las usa por defecto.

# %%
FEATURES_21 = [  # variables del parquet sintético: una numérica y una categórica
    {"nombre": "razon", "tipo": "num", "transformacion": "ninguna"},  # numérica sin transformar
    {"nombre": "clase", "tipo": "cat", "transformacion": "ninguna"},  # categórica
]


def test_21_cargar_datos_y_evidencia_publicas(tmp_path):
    """Están en __all__; cargar_datos = (df, matrices de cargar_matrices); evidencia = las dos SHA-256; hpo usa las de
    experimento por defecto (atrapa un renombre a medias que deja a hpo con una copia propia)."""
    assert {"cargar_datos", "evidencia"} <= set(experimento.__all__)  # públicas
    rng = np.random.default_rng(SEMILLA_DATOS)  # generador
    df = pd.DataFrame({  # parquet mínimo con el esquema del núcleo §5.1 que usa el experimento
        "id": [f"id{i:03d}" for i in range(30)], "split": ["train"] * 15 + ["val"] * 8 + ["test"] * 7,
        "target": rng.binomial(1, 0.3, size=30).astype(np.int8), "razon": rng.normal(size=30),
        "clase": rng.choice(["A", "B"], size=30)})  # 30 créditos
    df.to_parquet(tmp_path / "loans_pd.parquet", index=False)  # parquet
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES_21)  # ajustado SOLO con train
    save_json(tmp_path / "pre.json", pre.to_dict())  # como lo deja A
    rutas = RutasPD(tmp_path / "loans_pd.parquet", tmp_path / "pre.json", tmp_path / "res", tmp_path / "models")
    df_leido, matrices = cargar_datos(rutas)  # una lectura
    pd.testing.assert_frame_equal(df_leido, pd.read_parquet(rutas.parquet))  # el df del parquet
    referencia = cargar_matrices(rutas)  # firma pública de B3
    for split in ("train", "val", "test"):  # cada split
        for a, b in zip(matrices[split], referencia[split], strict=True):  # X, y, ids
            np.testing.assert_array_equal(a, b)  # iguales
    assert evidencia(df_leido, rutas.preprocesador) == {  # huellas independientes
        "sha256_particion": _sha256_particion(df_leido),  # partición
        "sha256_preprocesador": hashlib.sha256((tmp_path / "pre.json").read_bytes()).hexdigest()}  # bytes
    assert hpo.cargar_datos is experimento.cargar_datos  # hpo usa la de experimento
    assert hpo.RUTAS_PD is experimento.RUTAS_PD  # y sus rutas


# %% [markdown]
# ## 22. CLI
#
# `run.py pd-hpo` con la config versionada y `--set` a carpetas temporales. `hpo.cargar_datos` y `hpo.RUTAS_PD` se
# sustituyen con `monkeypatch` (se resuelven al llamarse). Imprime la config resuelta (la imprime el CLI), los
# ensayos completos y fallidos, el mejor con su log-loss y AUC de val, dónde quedaron los archivos y el tiempo.

# %%
def test_22_cli(tmp_path, monkeypatch, capsys):
    """cli.main(["pd-hpo", ...]) escribe en las rutas de --set, no toca configs/ y lo imprime todo (atrapa un
    subcomando sin conectar, rutas fijas en el código o un manejador que no reporta el presupuesto)."""
    real = (ROOT / "configs" / "pd" / "mlp_final.yaml").read_bytes()  # YAML versionado antes
    rutas = _rutas(tmp_path / "entradas")  # preprocesador falso
    monkeypatch.setattr(hpo, "cargar_datos", _Cargar(*_datos()))  # sin datos reales
    monkeypatch.setattr(hpo, "RUTAS_PD", rutas)  # huella del preprocesador falso
    nivel = optuna.logging.get_verbosity()  # nivel de log antes
    try:  # restaura el log global de Optuna al salir
        codigo = cli.main(["pd-hpo", "--config", str(ROOT / "configs" / "pd" / "hpo.yaml"),
                           "--set", f"hpo.destino={tmp_path / 'mlp_final.yaml'}",
                           "--set", f"hpo.salida={tmp_path / 'res'}", "--set", "hpo.n_ensayos=2",
                           *[a for s in ESPACIO_PEQUENO_SET for a in ("--set", s)]])  # CLI completo
        assert optuna.logging.get_verbosity() == optuna.logging.WARNING  # el manejador fija WARNING
    finally:
        optuna.logging.set_verbosity(nivel)  # sin efectos en otras pruebas
    salida = capsys.readouterr().out  # lo impreso
    assert codigo == 0  # éxito
    j = load_json(tmp_path / "res" / "hpo_ensayos.json")  # JSON escrito
    assert j["n_ensayos"] == 2  # el --set llegó
    assert (tmp_path / "mlp_final.yaml").is_file()  # YAML escrito
    assert (ROOT / "configs" / "pd" / "mlp_final.yaml").read_bytes() == real  # el versionado no cambió
    assert cli.CABECERA_CONFIG in salida  # config resuelta
    minusculas = salida.lower()  # para buscar palabras
    for palabra in ("complet", "fallid", "log-loss", "auc"):  # ensayos y métricas del mejor
        assert palabra in minusculas  # se reporta
    assert "hpo_ensayos.json" in salida and "mlp_final.yaml" in salida  # dónde quedaron
    assert "Tiempo de ejecución:" in salida  # tiempo total (núcleo §4.1)


def test_a06_destino_relativo_al_directorio_actual(tmp_path, monkeypatch):
    """destino y salida relativos se resuelven contra el directorio actual (atrapa resolverlos contra la carpeta del
    YAML, que con hpo.yaml en configs/pd/ escribiría configs/pd/configs/pd/mlp_final.yaml)."""
    monkeypatch.chdir(tmp_path)  # directorio actual = tmp_path
    cfg = _cfg(tmp_path)  # config válida
    cfg["hpo"]["destino"], cfg["hpo"]["salida"] = "mlp_final.yaml", "res"  # relativos
    buscar(cfg, rutas=_rutas(tmp_path / "entradas"), cargar=_Cargar(*_datos()), reloj=_reloj(), ahora=_ahora)
    assert (tmp_path / "mlp_final.yaml").is_file()  # YAML en el directorio actual
    assert (tmp_path / "res" / "hpo_ensayos.json").is_file()  # JSON en el directorio actual


# %% [markdown]
# ## 23. Datos reales (lenta)
#
# Con los datos de A, 2 ensayos y pocas épocas; destino y salida en `tmp_path`. Se salta si el clon no trae los
# datos procesados.

# %%
@pytest.mark.lento
def test_23_datos_reales(tmp_path, monkeypatch):
    """Con loans_pd.parquet y el preprocesador de A: escribe los dos archivos y la evidencia es la del preprocesador
    real (atrapa una ruta relativa rota o una incompatibilidad con las 48 columnas reales)."""
    parquet = ROOT / experimento.RUTAS_PD.parquet  # datos reales de A
    pre = ROOT / experimento.RUTAS_PD.preprocesador  # preprocesador real de A
    if not (parquet.is_file() and pre.is_file()):  # el clon no trae los datos procesados
        pytest.skip("faltan data/processed/loans_pd.parquet o models/preprocesador_pd.json")  # se corre a mano
    rutas = RutasPD(parquet, pre, tmp_path / "resultados", tmp_path / "models")  # entradas reales
    cfg = apply_overrides(load_config(ROOT / "configs" / "pd" / "hpo.yaml"), [  # config versionada
        f"hpo.destino={tmp_path / 'mlp_final.yaml'}", f"hpo.salida={tmp_path / 'res'}", "hpo.n_ensayos=2",
        "mlp.max_epocas=2"])  # rápido
    resumen = buscar(cfg, rutas=rutas)  # cargar_datos real
    j = load_json(tmp_path / "res" / "hpo_ensayos.json")  # JSON escrito
    assert j["n_ensayos"] == 2 and len(j["ensayos"]) == 2  # presupuesto
    assert j["evidencia"]["sha256_preprocesador"] == hashlib.sha256(pre.read_bytes()).hexdigest()  # huella real
    assert all(e["n_parametros"] > 0 for e in j["ensayos"] if e["estado"] == "COMPLETE")  # redes entrenadas
    assert load_config(tmp_path / "mlp_final.yaml")["mlp"]["capas"] == resumen["mejor"]["parametros"]["capas"]
