"""Pruebas de la logística de referencia con búsqueda de C y de su experimento (Parte B §3.1, tarea B3).

Solo se usa la interfaz pública de ``src.pd.logistica`` y ``src.pd.experimento``. No se prueba scikit-learn (que lbfgs
converja, que ``log_loss`` o ``LogisticRegression`` estén bien hechos): se prueban nuestras firmas (protocolo
``ModeloPD`` del núcleo §5.3), nuestras decisiones (criterio log-loss de val, empate → menor C, sin refit con val,
avisos como error, validación de la config, determinismo), la condición de optimalidad del objetivo que minimiza
sklearn (prueba 5) y los archivos de resultados del núcleo §5.4 con la evidencia de comparación justa (núcleo §6).

Datos: sintéticos, generados aquí con una logística conocida (núcleo §0 regla 8: solo para desarrollar). La única
excepción es la prueba 25 (marcada ``lento``), que corre el experimento con los datos reales de A.

Pruebas añadidas a las 28 del plan aprobado (cada una atrapa un fallo de nuestra infraestructura que ninguna otra ve):

- ``test_a01_buscar_ajusta_cada_C_solo_con_train``: un espía inyectado por ``crear`` comprueba que cada C se crea con
  ``max_iter`` y ``tol`` de la config, en el orden de la rejilla, y que se ajusta con X_tr (atrapa una fuga: ajustar
  con train + val, o un refit final con val).
- ``test_a02_filas_miden_en_validacion``: ``logloss_val`` y ``auc_val`` del C elegido son las de val (atrapa medir el
  criterio en train).
- ``test_a03_logloss_con_una_sola_clase``: ``logloss`` no falla si y tiene una sola clase (atrapa olvidar
  ``labels=[0, 1]``, que rompería la búsqueda con una validación pequeña).
- ``test_a04_config_base_valida``: ``configs/pd/base.yaml`` tiene solo el bloque ``logistica``, pasa la validación y
  declara 13 valores (atrapa un YAML mal escrito antes de la corrida larga).
- ``test_a05_evaluar_usa_las_metricas_comunes``: ``evaluar`` devuelve las 5 métricas con nombre, calculadas con
  ``src.common.metricas`` y ``logloss`` (atrapa un nombre cambiado o una métrica cruzada).
- ``test_a06_medir_nombra_por_split``: ``medir`` produce ``{metrica}_{split}`` con las PD del modelo en cada split.
- ``test_a07_cargar_matrices_no_reajusta``: las matrices salen del preprocesador guardado, sin ``fit`` (atrapa un
  reajuste con todos los datos, que sería fuga de información, núcleo §5.7).
- ``test_a08_crear_estimador_sin_class_weight``: el estimador no lleva ``class_weight`` ni cambia ``penalty`` (atrapa
  un ``class_weight="balanced"`` que descalibraría la PD).
- ``test_a09_ajustar_sin_avisos_ignora_filtros_globales``: ConvergenceWarning y FutureWarning son error aunque los
  filtros globales los ignoren (atrapa depender de los filtros de pytest).
- ``test_a10_buscar_usa_el_reloj_inyectado``: ``buscar_C`` mide con el reloj que recibe (DIP): cada ``tiempo_s`` es
  exactamente un paso del reloj falso.
- ``test_a11_buscar_falsa_que_falla_no_altera_modelo_entrenado``: si la búsqueda falla en un re-fit, el modelo anterior
  queda intacto (la prueba 16 solo cubre la falla en la validación de datos).
- ``test_a12_pred_test_rechaza_otra_y``: el mismo id con otra y en ``pred_test.parquet`` → ValueError.
- ``test_a13_validacion_acepta_casos_validos``: enteros en la rejilla, rejilla desordenada, un valor y ``max_iter = 1``
  se aceptan (atrapa un validador más estricto que la decisión de Andrés).
- ``test_a14_validacion_acepta_tupla_y_devuelve_lista`` (decisión de Andrés): tupla no vacía válida y copia validada
  como lista.
- ``test_a15_predict_rechaza_clases_distintas``: un estimador con ``classes_`` ≠ [0, 1] no predice (atrapa quitar la
  comprobación de ``classes_``, que ninguna otra prueba ve porque y siempre es 0/1).
- ``test_a16_tiempo_total_del_fit``: ``tiempo_total_s`` es ≥ la suma de los tiempos de las filas y sale del reloj
  inyectado en ``LogisticaPD`` (un paso exacto del reloj falso).
- ``test_a17_cfg_sin_bloque_logistica``: cfg sin la clave ``logistica`` o que no es dict → ValueError (no KeyError ni
  TypeError) en ``bloque_logistica``, ``fit`` y ``correr_logistica``, sin tocar los datos; con el bloque, devuelve
  la copia validada.
- ``test_a18_correr_logistica_valida_la_semilla``: seed = −1 o ``"0"`` → ValueError antes de entrenar y sin archivos.
- ``test_a19_pred_test_rechaza_ids_repetidos``: un id repetido haría ambigua la alineación por id → ValueError.
- ``test_a20_resumir_rechaza_metrica_sin_sentido``: una métrica sin sentido declarado → ValueError (no un KeyError).
- ``test_a21_resumir_sin_corridas``: ``resumir([])`` → ValueError (no un resumen vacío en silencio).
- ``test_a22_tiempo_total_tras_cargar``: un modelo cargado no tiene ``tiempo_total_s`` (no se guarda) → RuntimeError.
- ``test_a23_cargar_rechaza_contenido_no_dict``: un joblib que no es diccionario → ValueError.
- ``test_a24_elegir_indice_sin_filas``: ``elegir_indice([])`` → ValueError (no el ValueError genérico de ``min``).
- ``test_a25_medir_usa_las_metricas_inyectadas``: ``medir`` pasa ``metricas`` a ``evaluar`` (DIP, para B5).
- ``test_a26_escribir_semilla_valida_la_semilla``: ``escribir_semilla`` (pública, la usará B5) rechaza seed = −1 o
  ``"0"`` con ValueError y sin escribir (``correr_logistica`` ya valida antes, así que solo esta prueba lo ve).
- ``test_a27_pred_test_queda_legible``: ``pred_test.parquet`` queda con permisos 0o644 como los JSON de ``save_json``
  (``mkstemp`` crea el temporal en 0o600 y ``os.replace`` los conserva; hallazgo de la corrida real).
"""

import copy  # copias profundas de la config de prueba
import hashlib  # SHA-256 de referencia para las huellas
import itertools  # reloj falso que avanza a paso fijo
import os  # permisos del archivo y sistema operativo
import re  # 64 dígitos hexadecimales de las huellas
import stat  # S_IMODE: bits de permisos
import warnings  # filtros de avisos antes y después del fit
from pathlib import Path  # rutas de los archivos de prueba

import joblib  # archivos guardados a mano para probar cargar
import numpy as np  # datos sintéticos y comparaciones bit a bit
import pandas as pd  # parquet sintético y pred_test
import pytest  # parametrización, fixtures y pytest.raises
import sklearn  # versión instalada (la guarda el joblib)
from scipy.special import expit  # σ de referencia, independiente de la implementación
from sklearn.exceptions import ConvergenceWarning  # aviso de lbfgs sin converger
from sklearn.linear_model import LogisticRegression  # referencia de la prueba de equivalencia
from sklearn.metrics import log_loss  # referencia de la log-loss

from src.common.config import apply_overrides, load_config  # mismo cargador que el CLI (--set)
from src.common.metricas import auc, brier, gini, ks  # métricas comunes (B1)
from src.common.preprocesamiento import Preprocesador  # preprocesador común (núcleo §5.2)
from src.common.results import load_json, save_json  # JSON sin NaN (núcleo §5.4)
from src.pd.experimento import (  # experimento de la PD (resultados del núcleo §5.4)
    METRICAS,  # nombre → función (y, p) -> float
    RUTAS_PD,  # rutas por defecto del proyecto
    SENTIDO,  # nombre → "max" | "min"
    RutasPD,  # rutas inyectables
    actualizar_pred_test,  # agrega la columna de un modelo a pred_test.parquet
    actualizar_summary,  # reemplaza la entrada de un modelo en summary.json
    cargar_matrices,  # parquet + preprocesador guardado → matrices por split
    correr_logistica,  # experimento completo de la logística
    escribir_semilla,  # archivo de una corrida
    evaluar,  # métricas de un vector de PD
    huella_archivo,  # SHA-256 de un archivo
    huella_particion,  # SHA-256 de la partición
    medir,  # métricas por split de un modelo
    resumir,  # media, desv, mejor, peor, n
)
from src.pd.logistica import (  # modelo de referencia y sus piezas
    CLAVES_LOGISTICA,  # claves exactas del bloque logistica
    LogisticaPD,  # el modelo que cumple ModeloPD
    ajustar_sin_avisos,  # ajuste con avisos como error
    bloque_logistica,  # bloque logistica de la config completa
    buscar_C,  # búsqueda de C con validación
    crear_estimador,  # fábrica del estimador de sklearn
    elegir_indice,  # regla de elección (pura)
    esta_en_borde,  # ¿C elegido en el borde de la rejilla?
    logloss,  # criterio de la búsqueda
    validar_cfg_logistica,  # reglas de validez del bloque logistica
)

# Los avisos de convergencia y de deprecación de sklearn son error en todo el módulo (decisión de Andrés).
pytestmark = [  # marcas aplicadas a todas las pruebas del módulo
    pytest.mark.filterwarnings("error::sklearn.exceptions.ConvergenceWarning"),  # lbfgs sin converger = error
    pytest.mark.filterwarnings("error::FutureWarning"),  # parámetros deprecados de sklearn = error
]

# --------------------------------------------------------------------------------------------
# Constantes de prueba (no son del Spec ni del dominio; solo fijan los escenarios sintéticos)
# --------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio (para la prueba con datos reales)
SEMILLA_DATOS = 20261025  # semilla declarada de los datos sintéticos
SEMILLA_MODELO = 0  # semilla por defecto del fit (lbfgs la ignora)
SEMILLA_OTRA = 123  # otra semilla: los coeficientes no deben cambiar
N_TRAIN = 400  # filas de entrenamiento sintéticas
N_VAL = 400  # filas de validación sintéticas
PESOS_GEN = np.array([0.8, -0.6, 0.5, -0.4, 0.3, -0.2])  # coeficientes de la logística generadora
INTERCEPTO_GEN = -0.7  # intercepto de la logística generadora (menos malos que buenos)
N_SENAL = PESOS_GEN.size  # columnas con señal
COL_CEROS = N_SENAL  # índice de la columna toda en ceros (la última)
N_COLUMNAS = N_SENAL + 1  # columnas de X: señal + la columna en ceros
REJILLA_13 = [1.0e-4, 3.1623e-4, 1.0e-3, 3.1623e-3, 1.0e-2, 3.1623e-2, 1.0e-1, 3.1623e-1, 1.0, 3.1623, 10.0,
              31.623, 100.0]  # la rejilla de configs/pd/base.yaml
N_REJILLA_BASE = 13  # valores de la rejilla declarada (decisión de Andrés)
MAX_ITER = 1000  # tope de iteraciones de lbfgs de la config
TOL = 1e-6  # tolerancia de lbfgs de la config
CFG_PRUEBA = {"logistica": {"rejilla_C": list(REJILLA_13), "max_iter": MAX_ITER, "tol": TOL}}  # config completa
C_EQUIVALENCIA = 0.1  # C fijo de la prueba de equivalencia con sklearn
C_UNICO = 0.01  # el valor de la modificación en vivo --set logistica.rejilla_C=[0.01]
OVERRIDE_C_UNICO = ["logistica.rejilla_C=[0.01]"]  # override de la modificación en vivo
TOL_EXPIT = 1e-12  # tolerancia de predict_proba frente a σ(predict_logit) en float64
# lbfgs de sklearn se detiene cuando max|∇f| ≤ gtol = tol (= 1e-6), con f el objetivo de la prueba 5; también puede
# detenerse antes por la reducción relativa de f (ftol = 64·eps). El factor 10 da margen a ese segundo criterio y al
# redondeo al recalcular el gradiente aquí; en estos datos el máximo medido fue ≈ 7.9e-7 (< tol).
TOL_GRADIENTE = 10 * TOL  # cota del gradiente del objetivo en el óptimo
N_RUIDO_FILAS = 60  # pocas filas de train en el escenario de ruido
N_RUIDO_COLUMNAS = 40  # columnas de puro ruido en el escenario de ruido
COEF_SENAL_RUIDO = 0.5  # coeficiente de la única columna con señal en el escenario de ruido
N_VAL_RUIDO = 400  # filas de validación del escenario de ruido
C_MAXIMO_RUIDO = 1e-2  # con tanto ruido la validación debe preferir regularizar fuerte (C pequeño)
MAX_ITER_CORTO = 1  # una sola iteración: lbfgs no converge
LOGLOSS_EMPATE = 0.3  # log-loss repetida en la prueba de empate
REJILLA_CORTA = [1.0e-2, 1.0, 100.0]  # rejilla del experimento sintético (rápido)
N_EXP = {"train": 300, "val": 150, "test": 150}  # filas por split del parquet sintético
ANIO_SPLIT = {"train": 2012, "val": 2013, "test": 2014}  # año de originación de cada split (como la partición real)
CATEGORIAS_EXP = ["A", "B", "C"]  # valores de la categórica sintética
FEATURES_EXP = [  # variables del parquet sintético: 2 numéricas + 1 categórica
    {"nombre": "monto", "tipo": "num", "transformacion": "log1p"},  # numérica con log1p
    {"nombre": "razon", "tipo": "num", "transformacion": "ninguna"},  # numérica sin transformar
    {"nombre": "clase", "tipo": "cat", "transformacion": "ninguna"},  # categórica
]
N_PARAMETROS_REAL = 49  # 48 columnas del preprocesador real (A) + intercepto
HEX64 = re.compile(r"[0-9a-f]{64}")  # SHA-256 en hexadecimal
METRICAS_BASE = {"auc", "gini", "ks", "brier", "logloss"}  # nombres de las métricas (decisión de Andrés)
SPLITS_MEDIDOS = ("val", "test")  # splits que mide el experimento
CLAVES_FILA = {"C", "logloss_val", "auc_val", "n_iter", "tiempo_s"}  # claves exactas de una fila de la búsqueda
CLAVES_EVIDENCIA = {"sha256_particion", "sha256_preprocesador"}  # evidencia de comparación justa (núcleo §6)
CLAVES_SEMILLA = {"modelo", "seed", "config", "metricas", "n_parametros", "tiempo_s", "device", "epocas",
                  "evidencia"}  # núcleo §5.4 + evidencia
CLAVES_BUSQUEDA = {"criterio", "filas", "C_elegido", "en_borde", "tiempo_total_s", "evidencia"}  # busqueda_logistica
CLAVES_SUMMARY = {"n_parametros", "n_configuraciones_busqueda", "tiempo_total_s", "metricas", "evidencia"}  # summary
CLAVES_RESUMEN = {"media", "desv", "mejor", "peor", "n"}  # resumen de una métrica (núcleo §5.4)
PASO_RELOJ = 0.5  # avance del reloj falso por llamada
PERMISOS_LEGIBLES = 0o644  # rw-r--r--: los mismos que deja save_json en los JSON
ENTRADA_MLP = {"n_parametros": 1, "nota": "entrada ajena"}  # entrada de otro modelo que debe sobrevivir


# --------------------------------------------------------------------------------------------
# Fixtures y utilidades
# --------------------------------------------------------------------------------------------
def _logistica(rng, n):
    """n filas float32: N_SENAL columnas N(0, 1) + una columna en ceros; y 0/1 de la logística generadora."""
    senal = rng.normal(size=(n, N_SENAL))  # variables ya "estandarizadas"
    X = np.hstack([senal, np.zeros((n, 1))]).astype(np.float32)  # añade la columna toda en ceros
    y = rng.binomial(1, expit(senal @ PESOS_GEN + INTERCEPTO_GEN)).astype(np.int64)  # etiquetas 0/1
    return X, y  # par (X, y)


@pytest.fixture(scope="module")
def datos():
    """Train y val sintéticos: N_COLUMNAS columnas float32 (la última en ceros), misma logística generadora."""
    rng = np.random.default_rng(SEMILLA_DATOS)  # generador de los datos sintéticos
    X_tr, y_tr = _logistica(rng, N_TRAIN)  # entrenamiento
    X_val, y_val = _logistica(rng, N_VAL)  # validación aparte
    return X_tr, y_tr, X_val, y_val  # tupla de cuatro arreglos


@pytest.fixture(scope="module")
def datos_ruido():
    """Pocas filas de train y muchas columnas de ruido + una de señal: la validación debe pedir C pequeño."""
    rng = np.random.default_rng(SEMILLA_DATOS)  # generador del escenario de ruido

    def generar(n):  # n filas: columna 0 con señal, el resto ruido
        X = rng.normal(size=(n, N_RUIDO_COLUMNAS + 1))  # señal + ruido
        y = rng.binomial(1, expit(COEF_SENAL_RUIDO * X[:, 0])).astype(np.int64)  # solo la columna 0 importa
        return X.astype(np.float32), y  # par (X, y)

    X_tr, y_tr = generar(N_RUIDO_FILAS)  # train pequeño
    X_val, y_val = generar(N_VAL_RUIDO)  # validación aparte
    return X_tr, y_tr, X_val, y_val  # tupla de cuatro arreglos


def _cfg(**cambios):
    """Copia de CFG_PRUEBA con ``cambios`` aplicados al bloque logistica (no muta la constante)."""
    cfg = copy.deepcopy(CFG_PRUEBA)  # copia profunda de la constante
    cfg["logistica"].update(cambios)  # aplica los cambios pedidos
    return cfg  # config completa


def _ajustar(datos, cfg=None, seed=SEMILLA_MODELO, **kwargs):
    """Entrena una LogisticaPD con ``cfg`` (por defecto CFG_PRUEBA); ``kwargs`` van al constructor."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca los datos
    modelo = LogisticaPD(**kwargs)  # modelo sin entrenar (búsqueda opcionalmente inyectada)
    return modelo.fit(X_tr, y_tr, X_val, y_val, cfg if cfg is not None else _cfg(), seed)  # fit devuelve el modelo


@pytest.fixture(scope="module")
def modelo(datos):
    """LogisticaPD entrenada con CFG_PRUEBA (solo para pruebas que no la mutan)."""
    return _ajustar(datos)  # modelo compartido de solo lectura


def _filas(Cs, perdidas):
    """Filas mínimas de la búsqueda para ``elegir_indice`` (C y logloss_val)."""
    return [{"C": c, "logloss_val": p} for c, p in zip(Cs, perdidas)]  # una fila por C


class _Espia:
    """Estimador espía: delega en una LogisticRegression real y registra con qué se ajustó."""

    def __init__(self, registro, C, max_iter, tol):
        self._registro = registro  # lista compartida de llamadas
        self._real = crear_estimador(C, max_iter, tol)  # estimador real al que delega
        registro.append({"C": C, "max_iter": max_iter, "tol": tol, "X": None})  # registra la creación

    def fit(self, X, y):
        self._registro[-1]["X"] = np.array(X, copy=True)  # guarda la X con la que se ajustó
        self._real.fit(X, y)  # ajusta el estimador real
        return self  # como sklearn: fit devuelve el estimador

    def __getattr__(self, nombre):
        return getattr(self._real, nombre)  # predict_proba, n_iter_, coef_, ... del estimador real


class _BuscarFalsa:
    """``buscar`` falsa: registra sus argumentos y devuelve un resultado fijo (o lanza ``error``)."""

    def __init__(self, resultado=None, error=None):
        self.resultado = resultado  # (estimador, filas, indice) que devuelve
        self.error = error  # excepción que lanza en lugar de devolver
        self.llamadas = []  # argumentos recibidos en cada llamada

    def __call__(self, X_tr, y_tr, X_val, y_val, cfg_log, **kwargs):
        self.llamadas.append({"X_tr": X_tr, "X_val": X_val, "cfg_log": cfg_log})  # registra los argumentos
        if self.error is not None:  # modo "falla"
            raise self.error  # simula una búsqueda que se rompe
        return self.resultado  # modo "termina"


class _FallaBusqueda(Exception):
    """Excepción propia de la búsqueda falsa (no se confunde con ValueError ni RuntimeError)."""


def _resultado_falso(datos, C=0.5):
    """(estimador ya ajustado con train, filas, indice) para la búsqueda falsa."""
    X_tr, y_tr, _, _ = datos  # solo train
    estimador = crear_estimador(C, MAX_ITER, TOL).fit(X_tr.astype(np.float64), y_tr)  # estimador real ajustado
    filas = [{"C": C, "logloss_val": 0.6, "auc_val": 0.7, "n_iter": 5, "tiempo_s": 0.01}]  # una fila con las 5 claves
    return estimador, filas, 0  # índice del único C


def _parquet_sintetico(ruta):
    """Escribe un loans_pd.parquet sintético con el esquema del núcleo §5.1 y devuelve el DataFrame."""
    rng = np.random.default_rng(SEMILLA_DATOS)  # generador del parquet sintético
    partes = []  # un DataFrame por split
    for split, n in N_EXP.items():  # train, val, test
        monto = rng.lognormal(mean=9.0, sigma=0.5, size=n)  # numérica positiva (admite log1p)
        razon = rng.normal(size=n)  # numérica centrada
        clase = rng.choice(CATEGORIAS_EXP, size=n)  # categórica
        logit = 0.8 * razon + 0.5 * (clase == "C") - 1.0  # logística generadora
        partes.append(pd.DataFrame({  # filas del split
            "issue_d": pd.Timestamp(f"{ANIO_SPLIT[split]}-06-01"),  # fecha de originación
            "anio_origen": np.int32(ANIO_SPLIT[split]),  # año de originación (int)
            "target": rng.binomial(1, expit(logit)).astype(np.int8),  # objetivo 0/1
            "split": split,  # partición
            "monto": monto, "razon": razon, "clase": clase,  # variables crudas
        }))
    df = pd.concat(partes, ignore_index=True)  # un solo DataFrame en orden train, val, test
    df.insert(0, "id", [f"id{i:05d}" for i in rng.permutation(len(df))])  # id str, no ordenado
    df.to_parquet(ruta, index=False)  # escribe el parquet
    return df  # DataFrame escrito


def _escenario(base):
    """Parquet sintético + preprocesador ajustado con su train en ``base``; devuelve (rutas, df, cfg)."""
    df = _parquet_sintetico(base / "loans_pd.parquet")  # parquet sintético
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES_EXP)  # preprocesador ajustado SOLO con train
    save_json(base / "preprocesador_pd.json", pre.to_dict())  # guardado como lo deja A
    rutas = RutasPD(base / "loans_pd.parquet", base / "preprocesador_pd.json", base / "resultados",
                    base / "models")  # rutas hacia la carpeta temporal
    return rutas, df, _cfg(rejilla_C=list(REJILLA_CORTA))  # rejilla corta: rápido


@pytest.fixture(scope="module")
def experimento(tmp_path_factory):
    """Corre correr_logistica una vez sobre el parquet sintético; devuelve rutas, datos y resultado."""
    rutas, df, cfg = _escenario(tmp_path_factory.mktemp("exp_logistica"))  # escenario sintético
    resultado = correr_logistica(cfg, rutas=rutas)  # experimento completo
    n_columnas = len(Preprocesador.from_dict(load_json(rutas.preprocesador)).nombres_salida())  # d
    return {"df": df, "rutas": rutas, "cfg": cfg, "resultado": resultado,
            "n_columnas": n_columnas}  # todo lo que inspeccionan las pruebas


def _sha256_particion(df):
    """SHA-256 de referencia: texto UTF-8 de las líneas f"{id},{split}\\n" en el orden del parquet."""
    texto = "".join(f"{i},{s}\n" for i, s in zip(df["id"], df["split"]))  # líneas id,split
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()  # hexadecimal


# --------------------------------------------------------------------------------------------
# 1. Validación del bloque logistica
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cambio", [  # cada caso rompe una regla de la config
    {"rejilla_C": [0.0, 1.0]},  # C = 0
    {"rejilla_C": [1.0, -0.1]},  # C < 0
    {"rejilla_C": []},  # lista vacía
    {"extra": 1},  # clave de más
], ids=["C_cero", "C_negativo", "rejilla_vacia", "clave_de_mas"])
def test_01_validacion_rechaza(cambio):
    """C ≤ 0, rejilla vacía o una clave de más → ValueError."""
    cfg_log = _cfg(**cambio)["logistica"]  # bloque inválido
    with pytest.raises(ValueError):  # la validación debe rechazarlo
        validar_cfg_logistica(cfg_log)  # valida el bloque


@pytest.mark.parametrize("clave", CLAVES_LOGISTICA)  # quitar cada una de las tres claves
def test_01_validacion_rechaza_clave_faltante(clave):
    """Falta una de las tres claves → ValueError."""
    cfg_log = _cfg()["logistica"]  # bloque válido
    del cfg_log[clave]  # quita la clave
    with pytest.raises(ValueError):  # la validación debe rechazarlo
        validar_cfg_logistica(cfg_log)  # valida el bloque


def test_01_validacion_devuelve_copia_profunda():
    """``validar_cfg_logistica`` devuelve una copia profunda igual y no muta la recibida."""
    cfg_log = _cfg()["logistica"]  # bloque válido
    original = copy.deepcopy(cfg_log)  # foto para comparar
    validado = validar_cfg_logistica(cfg_log)  # copia validada
    assert validado == original  # mismo contenido
    assert validado is not cfg_log  # otro diccionario
    assert validado["rejilla_C"] is not cfg_log["rejilla_C"]  # otra lista (copia profunda)
    assert cfg_log == original  # la recibida no cambió


def test_01_fit_no_muta_la_config(datos):
    """``fit`` no muta la config completa (B5 la reutiliza entre modelos)."""
    cfg = _cfg()  # config completa
    original = copy.deepcopy(cfg)  # foto para comparar
    _ajustar(datos, cfg)  # entrena
    assert cfg == original  # sin cambios


# --------------------------------------------------------------------------------------------
# 2. Regla de elección (pura)
# --------------------------------------------------------------------------------------------
def test_02_elegir_indice_argmin():
    """Elige el argmin de logloss_val."""
    assert elegir_indice(_filas([0.01, 0.1, 1.0], [0.5, 0.3, 0.4])) == 1  # el mínimo está en medio


def test_02_elegir_indice_empate_gana_menor_C():
    """Empate exacto → gana el menor C aunque la rejilla venga desordenada."""
    assert elegir_indice(_filas([1.0, 0.1], [LOGLOSS_EMPATE, LOGLOSS_EMPATE])) == 1  # el 0.1 está en la posición 1
    assert elegir_indice(_filas([0.1, 1.0], [LOGLOSS_EMPATE, LOGLOSS_EMPATE])) == 0  # y en la posición 0


def test_02_elegir_indice_sin_tolerancia():
    """El argmin es estricto: una diferencia de un ulp decide (no hay empate "aproximado")."""
    peor = np.nextafter(LOGLOSS_EMPATE, 1.0)  # el siguiente flotante después de LOGLOSS_EMPATE
    assert elegir_indice(_filas([1.0, 0.1], [LOGLOSS_EMPATE, peor])) == 0  # gana la menor pérdida, aunque su C sea mayor


def test_02_elegir_indice_no_muta():
    """``elegir_indice`` es pura: no reordena ni modifica las filas."""
    filas = _filas([1.0, 0.1, 0.01], [0.4, 0.3, 0.3])  # filas desordenadas con empate
    original = copy.deepcopy(filas)  # foto para comparar
    assert elegir_indice(filas) == 2  # empate entre 0.1 y 0.01 → 0.01
    assert filas == original  # sin cambios


# --------------------------------------------------------------------------------------------
# 3. Contrato ModeloPD
# --------------------------------------------------------------------------------------------
def test_03_contrato_predicciones(datos):
    """predict_proba y predict_logit en float64 (n,); proba = σ(logit); PD en (0, 1); fit devuelve el modelo."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    m = LogisticaPD()  # modelo nuevo
    assert m.fit(X_tr, y_tr, X_val, y_val, _cfg(), SEMILLA_MODELO) is m  # fit devuelve el mismo objeto
    p = m.predict_proba(X_val)  # PD de validación
    z = m.predict_logit(X_val)  # logit de validación
    assert p.dtype == np.float64 and p.shape == (N_VAL,)  # forma y tipo de la PD
    assert z.dtype == np.float64 and z.shape == (N_VAL,)  # forma y tipo del logit
    np.testing.assert_allclose(p, expit(z), rtol=0, atol=TOL_EXPIT)  # proba = σ(logit)
    assert np.all((p > 0) & (p < 1))  # PD estrictamente en (0, 1)


# --------------------------------------------------------------------------------------------
# 4. Equivalencia con sklearn
# --------------------------------------------------------------------------------------------
def test_04_equivalencia_con_sklearn(datos):
    """Con un solo C, los coeficientes son bit a bit los de LogisticRegression(lbfgs) ajustada con train."""
    X_tr, y_tr, _, _ = datos  # desempaca
    m = _ajustar(datos, _cfg(rejilla_C=[C_EQUIVALENCIA]))  # rejilla de un valor
    ref = LogisticRegression(C=C_EQUIVALENCIA, solver="lbfgs", max_iter=MAX_ITER, tol=TOL)  # referencia
    ref.fit(X_tr.astype(np.float64), y_tr)  # ajustada solo con train
    w, b = m.coeficientes()  # coeficientes del modelo
    assert w.dtype == np.float64 and w.shape == (N_COLUMNAS,)  # forma y tipo de w
    assert isinstance(b, float)  # b es un float nativo
    assert np.array_equal(w, ref.coef_[0])  # mismos pesos
    assert b == ref.intercept_[0]  # mismo intercepto


# --------------------------------------------------------------------------------------------
# 5. Optimalidad
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("C", [1e-3, 1.0, 100.0])  # regularización fuerte, media y débil
def test_05_optimalidad(datos, C):
    """En el óptimo de (1/n)Σ logloss + ‖w‖²/(2Cn), el gradiente (1/n)Xᵀ(p − y) + w/(Cn) es ≈ 0 y mean(p − y) ≈ 0."""
    X_tr, y_tr, _, _ = datos  # desempaca
    m = _ajustar(datos, _cfg(rejilla_C=[C]))  # C fijo
    X = X_tr.astype(np.float64)  # misma precisión que el ajuste
    n = X.shape[0]  # filas de train
    w, _ = m.coeficientes()  # pesos (el intercepto no se penaliza)
    residuo = m.predict_proba(X_tr) - y_tr  # p − y
    gradiente_w = X.T @ residuo / n + w / (C * n)  # ∂f/∂w
    assert np.max(np.abs(gradiente_w)) < TOL_GRADIENTE  # ≈ 0 en cada peso
    assert abs(np.mean(residuo)) < TOL_GRADIENTE  # ∂f/∂b ≈ 0


# --------------------------------------------------------------------------------------------
# 6. Búsqueda
# --------------------------------------------------------------------------------------------
def test_06_busqueda_registra_cada_C(modelo):
    """Una fila por C, en el orden de la rejilla, con exactamente 5 claves; el elegido es el argmin de logloss_val."""
    filas = modelo.busqueda  # filas de la búsqueda
    assert len(filas) == len(REJILLA_13)  # presupuesto = tamaño de la rejilla
    assert [f["C"] for f in filas] == REJILLA_13  # orden de la rejilla
    assert all(set(f) == CLAVES_FILA for f in filas)  # exactamente las 5 claves
    assert all(isinstance(f["n_iter"], int) and f["n_iter"] >= 1 for f in filas)  # iteraciones de lbfgs
    assert all(f["tiempo_s"] >= 0 for f in filas)  # tiempos no negativos
    perdidas = [f["logloss_val"] for f in filas]  # criterio por C
    assert modelo.C_elegido == REJILLA_13[int(np.argmin(perdidas))]  # el elegido es el argmin
    assert isinstance(modelo.C_elegido, float)  # C_elegido es float


def test_06_busqueda_regulariza_con_ruido(datos_ruido):
    """Con pocas filas y mucho ruido, la validación elige un C pequeño (≤ 1e-2)."""
    m = _ajustar(datos_ruido)  # rejilla completa
    assert m.C_elegido <= C_MAXIMO_RUIDO  # regularización fuerte


def test_06_buscar_C_devuelve_estimador_filas_indice(datos):
    """``buscar_C`` devuelve (estimador del C elegido, filas, índice del argmin)."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    X_tr64, X_val64 = X_tr.astype(np.float64), X_val.astype(np.float64)  # la búsqueda trabaja en float64
    estimador, filas, indice = buscar_C(X_tr64, y_tr, X_val64, y_val, _cfg()["logistica"])  # búsqueda directa
    assert len(filas) == len(REJILLA_13)  # una fila por C
    assert indice == int(np.argmin([f["logloss_val"] for f in filas]))  # argmin de la log-loss de val
    assert estimador.C == filas[indice]["C"]  # el estimador es el del C elegido


# --------------------------------------------------------------------------------------------
# 7. Encogimiento
# --------------------------------------------------------------------------------------------
def test_07_encogimiento(datos):
    """‖w‖₂ es estrictamente decreciente al recorrer la rejilla de mayor a menor C."""
    normas = [np.linalg.norm(_ajustar(datos, _cfg(rejilla_C=[C])).coeficientes()[0])  # ‖w‖₂ con C fijo
              for C in sorted(REJILLA_13, reverse=True)]  # de mayor a menor C
    assert np.all(np.diff(normas) < 0)  # cada norma es menor que la anterior


# --------------------------------------------------------------------------------------------
# 8. Determinismo
# --------------------------------------------------------------------------------------------
def test_08_determinismo_entre_semillas(datos):
    """lbfgs es determinista: con seed = 0 y seed = 123 los coeficientes son idénticos."""
    m0 = _ajustar(datos, seed=SEMILLA_MODELO)  # semilla 0
    m1 = _ajustar(datos, seed=SEMILLA_OTRA)  # semilla 123
    w0, b0 = m0.coeficientes()  # coeficientes con 0
    w1, b1 = m1.coeficientes()  # coeficientes con 123
    assert np.array_equal(w0, w1)  # mismos pesos
    assert b0 == b1  # mismo intercepto
    assert m0.C_elegido == m1.C_elegido  # mismo C elegido
    assert [f["logloss_val"] for f in m0.busqueda] == [f["logloss_val"] for f in m1.busqueda]  # misma búsqueda


# --------------------------------------------------------------------------------------------
# 9. Parámetros
# --------------------------------------------------------------------------------------------
def test_09_n_parametros(modelo):
    """n_parametros() = d pesos + 1 intercepto, como int."""
    assert modelo.n_parametros() == N_COLUMNAS + 1  # d + 1
    assert type(modelo.n_parametros()) is int  # entero nativo (va al JSON)


# --------------------------------------------------------------------------------------------
# 10. Guardar y cargar
# --------------------------------------------------------------------------------------------
def test_10_guardar_cargar(modelo, datos, tmp_path):
    """guardar → cargar conserva predicciones (bit a bit), C_elegido, búsqueda y parámetros."""
    _, _, X_val, _ = datos  # validación
    ruta = tmp_path / "pd_logistica.joblib"  # archivo del modelo
    modelo.guardar(ruta)  # guarda
    cargado = LogisticaPD.cargar(ruta)  # carga
    assert isinstance(cargado, LogisticaPD)  # devuelve el tipo correcto
    assert np.array_equal(cargado.predict_proba(X_val), modelo.predict_proba(X_val))  # mismas PD
    assert np.array_equal(cargado.predict_logit(X_val), modelo.predict_logit(X_val))  # mismos logits
    assert cargado.C_elegido == modelo.C_elegido  # mismo C
    assert cargado.busqueda == modelo.busqueda  # misma búsqueda
    assert cargado.n_parametros() == modelo.n_parametros()  # mismos parámetros


# --------------------------------------------------------------------------------------------
# 11. Avisos
# --------------------------------------------------------------------------------------------
def test_11_convergencia_es_error(datos):
    """Con max_iter = 1, lbfgs no converge y el fit lanza ConvergenceWarning como excepción."""
    with pytest.raises(ConvergenceWarning):  # el aviso se vuelve error
        _ajustar(datos, _cfg(max_iter=MAX_ITER_CORTO))  # una sola iteración


def test_11_ajuste_normal_sin_avisos_de_deprecacion(datos):
    """Un ajuste normal no emite FutureWarning ni DeprecationWarning (sin penalty, multi_class, ...)."""
    with warnings.catch_warnings(record=True) as avisos:  # captura todos los avisos
        warnings.simplefilter("always")  # registra todo lo que se emita
        _ajustar(datos)  # ajuste normal
    tipos = [a.category for a in avisos]  # categorías emitidas
    assert not any(issubclass(t, (FutureWarning, DeprecationWarning)) for t in tipos)  # ninguno de deprecación


def test_11_fit_no_altera_los_filtros_globales(datos):
    """``warnings.filters`` es igual antes y después de un fit (bueno o fallido).

    Se parte de un entorno que ignora todo: con los filtros "error" del ``pytestmark`` ya instalados, un
    ``simplefilter("error", ...)`` global (sin ``catch_warnings`` local) no cambiaría la lista y la prueba no lo vería.
    """
    with warnings.catch_warnings():  # filtros temporales de la prueba
        warnings.simplefilter("ignore")  # entorno distinto de los filtros que pone ajustar_sin_avisos
        antes = list(warnings.filters)  # filtros antes
        _ajustar(datos)  # fit normal
        assert list(warnings.filters) == antes  # sin cambios
        with pytest.raises(ConvergenceWarning):  # fit que falla por convergencia (error aunque el entorno ignore)
            _ajustar(datos, _cfg(max_iter=MAX_ITER_CORTO))  # una sola iteración
        assert list(warnings.filters) == antes  # tampoco cambian tras la falla


# --------------------------------------------------------------------------------------------
# 12. Columna sin información
# --------------------------------------------------------------------------------------------
def test_12_columna_en_ceros(modelo):
    """La columna toda en ceros tiene coeficiente exactamente 0.0 (no recibe gradiente)."""
    w, _ = modelo.coeficientes()  # pesos
    assert w[COL_CEROS] == 0.0  # exactamente cero


# --------------------------------------------------------------------------------------------
# 13. Validación extra y modificación en vivo
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cambio", [  # cada caso rompe una regla de tipo o de rango
    {"rejilla_C": [True]},  # bool no cuenta como real
    {"rejilla_C": ["0.1"]},  # texto no cuenta como real
    {"rejilla_C": [float("nan")]},  # NaN
    {"rejilla_C": [float("inf")]},  # inf
    {"rejilla_C": 0.01},  # escalar en lugar de lista
    {"rejilla_C": ()},  # tupla vacía
    {"rejilla_C": [0.1, 0.1]},  # duplicados
    {"max_iter": 1.0},  # max_iter flotante
    {"max_iter": True},  # max_iter bool
    {"max_iter": 0},  # max_iter < 1
    {"tol": 0},  # tol = 0
    {"tol": float("inf")},  # tol infinita
    {"tol": float("nan")},  # tol NaN
    {"tol": True},  # tol bool
], ids=["C_bool", "C_texto", "C_nan", "C_inf", "rejilla_escalar", "tupla_vacia", "duplicados", "max_iter_float",
        "max_iter_bool", "max_iter_cero", "tol_cero", "tol_inf", "tol_nan", "tol_bool"])
def test_13_validacion_extra(cambio):
    """Tipos y rangos inválidos → ValueError (nunca TypeError)."""
    cfg_log = _cfg(**cambio)["logistica"]  # bloque inválido
    with pytest.raises(ValueError):  # toda falla es ValueError
        validar_cfg_logistica(cfg_log)  # valida el bloque


def test_13_validacion_rechaza_bloque_no_dict():
    """Un bloque que no es diccionario → ValueError."""
    with pytest.raises(ValueError):  # toda falla es ValueError
        validar_cfg_logistica(list(REJILLA_13))  # una lista en lugar del bloque


def test_13_fit_valida_la_config(datos):
    """``fit`` valida la config antes de ajustar (una rejilla con C = 0 no llega a sklearn)."""
    with pytest.raises(ValueError):  # la validación lo detiene
        _ajustar(datos, _cfg(rejilla_C=[0.0]))  # C inválido


def test_13_override_en_vivo(datos):
    """``--set logistica.rejilla_C=[0.01]`` funciona: una sola fila y C_elegido = 0.01."""
    cfg = apply_overrides(_cfg(), OVERRIDE_C_UNICO)  # mismo camino que el CLI
    m = _ajustar(datos, cfg)  # entrena con C fijo
    assert len(m.busqueda) == 1  # una sola configuración
    assert m.C_elegido == C_UNICO  # el valor del override


# --------------------------------------------------------------------------------------------
# 14. Antes de entrenar
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("uso", [  # cada uso requiere un modelo entrenado
    lambda m, X, r: m.predict_proba(X),  # PD
    lambda m, X, r: m.predict_logit(X),  # logit
    lambda m, X, r: m.n_parametros(),  # parámetros
    lambda m, X, r: m.coeficientes(),  # coeficientes
    lambda m, X, r: m.busqueda,  # filas de la búsqueda
    lambda m, X, r: m.C_elegido,  # C elegido
    lambda m, X, r: m.guardar(r),  # guardar
], ids=["predict_proba", "predict_logit", "n_parametros", "coeficientes", "busqueda", "C_elegido", "guardar"])
def test_14_sin_entrenar(uso, datos, tmp_path):
    """Antes de fit o cargar, cualquier uso → RuntimeError (y guardar no deja archivo)."""
    _, _, X_val, _ = datos  # validación
    ruta = tmp_path / "modelo.joblib"  # ruta del intento de guardar
    with pytest.raises(RuntimeError):  # el modelo no está entrenado
        uso(LogisticaPD(), X_val, ruta)  # uso sobre un modelo nuevo
    assert not ruta.exists()  # no se escribió nada


# --------------------------------------------------------------------------------------------
# 15. Datos inválidos
# --------------------------------------------------------------------------------------------
def _nan_en(a):
    """Copia flotante de ``a`` con un NaN en la primera posición."""
    b = np.array(a, dtype=np.float64, copy=True)  # copia flotante
    b.flat[0] = np.nan  # un NaN
    return b  # arreglo con NaN


CASOS_DATOS = {  # nombre → transformación de (X_tr, y_tr, X_val, y_val) que lo vuelve inválido
    "X_tr_1d": lambda Xt, yt, Xv, yv: (Xt[:, 0], yt, Xv, yv),  # X de una dimensión
    "X_val_otras_columnas": lambda Xt, yt, Xv, yv: (Xt, yt, Xv[:, :-1], yv),  # val con una columna menos
    "y_tr_con_2": lambda Xt, yt, Xv, yv: (Xt, np.where(np.arange(yt.size) == 0, 2, yt), Xv, yv),  # etiqueta 2
    "y_val_con_2": lambda Xt, yt, Xv, yv: (Xt, yt, Xv, np.where(np.arange(yv.size) == 0, 2, yv)),  # etiqueta 2 en val
    "y_tr_con_nan": lambda Xt, yt, Xv, yv: (Xt, _nan_en(yt), Xv, yv),  # y con NaN
    "y_tr_columna": lambda Xt, yt, Xv, yv: (Xt, yt.reshape(-1, 1), Xv, yv),  # y de forma (n, 1)
    "X_tr_con_nan": lambda Xt, yt, Xv, yv: (_nan_en(Xt), yt, Xv, yv),  # X con NaN
    "X_val_con_nan": lambda Xt, yt, Xv, yv: (Xt, yt, _nan_en(Xv), yv),  # X de val con NaN
    "y_tr_una_clase": lambda Xt, yt, Xv, yv: (Xt, np.zeros_like(yt), Xv, yv),  # una sola clase en train
    "filas_distintas": lambda Xt, yt, Xv, yv: (Xt, yt[:-1], Xv, yv),  # X e y con distinto número de filas
}


@pytest.mark.parametrize("caso", list(CASOS_DATOS))  # cada caso de datos inválidos
def test_15_datos_invalidos(caso, datos):
    """Datos mal formados → ValueError antes de ajustar."""
    X_tr, y_tr, X_val, y_val = CASOS_DATOS[caso](*datos)  # datos inválidos
    with pytest.raises(ValueError):  # se rechazan
        LogisticaPD().fit(X_tr, y_tr, X_val, y_val, _cfg(), SEMILLA_MODELO)  # fit


@pytest.mark.parametrize("metodo", ["predict_proba", "predict_logit"])  # ambos predictores
def test_15_predict_con_otras_columnas(metodo, modelo, datos):
    """Predecir con otra cantidad de columnas → ValueError."""
    _, _, X_val, _ = datos  # validación
    with pytest.raises(ValueError):  # se rechaza
        getattr(modelo, metodo)(X_val[:, :-1])  # una columna menos


# --------------------------------------------------------------------------------------------
# 16. Atomicidad
# --------------------------------------------------------------------------------------------
def test_16_refit_fallido_no_altera(datos):
    """Un fit con datos inválidos sobre un modelo entrenado lo deja intacto."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    m = _ajustar(datos)  # modelo entrenado propio (se intenta mutar)
    p_antes, C_antes, filas_antes = m.predict_proba(X_val), m.C_elegido, m.busqueda  # foto del estado
    with pytest.raises(ValueError):  # re-fit inválido
        m.fit(_nan_en(X_tr), y_tr, X_val, y_val, _cfg(rejilla_C=[C_UNICO]), SEMILLA_MODELO)  # X con NaN
    assert np.array_equal(m.predict_proba(X_val), p_antes)  # mismas PD
    assert m.C_elegido == C_antes  # mismo C
    assert m.busqueda == filas_antes  # misma búsqueda


# --------------------------------------------------------------------------------------------
# 17. float32 vs float64
# --------------------------------------------------------------------------------------------
def test_17_float32_y_float64(datos):
    """Los mismos valores en float32 o float64 dan coeficientes y predicciones idénticos."""
    X_tr, y_tr, X_val, y_val = datos  # float32
    m32 = _ajustar(datos)  # entrada float32
    m64 = _ajustar((X_tr.astype(np.float64), y_tr, X_val.astype(np.float64), y_val))  # entrada float64
    assert np.array_equal(m32.coeficientes()[0], m64.coeficientes()[0])  # mismos pesos
    assert m32.coeficientes()[1] == m64.coeficientes()[1]  # mismo intercepto
    assert np.array_equal(m32.predict_proba(X_val), m64.predict_proba(X_val.astype(np.float64)))  # mismas PD


# --------------------------------------------------------------------------------------------
# 18. Copias
# --------------------------------------------------------------------------------------------
def test_18_devuelve_copias(datos):
    """Mutar lo que devuelven coeficientes() y busqueda no altera el modelo."""
    _, _, X_val, _ = datos  # validación
    m = _ajustar(datos)  # modelo propio (se intenta mutar)
    p_antes = m.predict_proba(X_val)  # PD antes
    w, _ = m.coeficientes()  # pesos
    w[:] = 99.0  # muta la copia
    filas = m.busqueda  # filas
    filas[0]["logloss_val"] = -1.0  # muta una fila
    filas.append({"C": -1.0})  # agrega una fila
    assert np.array_equal(m.predict_proba(X_val), p_antes)  # las PD no cambian
    assert not np.array_equal(m.coeficientes()[0], w)  # los pesos internos no cambian
    assert len(m.busqueda) == len(REJILLA_13)  # la búsqueda conserva su tamaño
    assert m.busqueda[0]["logloss_val"] != -1.0  # ni sus valores


# --------------------------------------------------------------------------------------------
# 19. cargar rechaza archivos inconsistentes
# --------------------------------------------------------------------------------------------
def _contenido_valido(datos):
    """Diccionario joblib válido armado a mano (sin pasar por guardar)."""
    estimador, filas, _ = _resultado_falso(datos)  # estimador ajustado y sus filas
    return {"estimador": estimador, "C_elegido": filas[0]["C"], "busqueda": filas,
            "n_entradas": N_COLUMNAS, "version_sklearn": sklearn.__version__}  # las 5 claves


def test_19_cargar_contenido_valido(datos, tmp_path):
    """Control positivo: un archivo armado a mano con las 5 claves carga y predice."""
    _, _, X_val, _ = datos  # validación
    contenido = _contenido_valido(datos)  # diccionario válido
    joblib.dump(contenido, tmp_path / "ok.joblib")  # archivo a mano
    m = LogisticaPD.cargar(tmp_path / "ok.joblib")  # carga
    assert np.array_equal(m.predict_proba(X_val),
                          contenido["estimador"].predict_proba(X_val.astype(np.float64))[:, 1])  # mismas PD
    assert m.C_elegido == contenido["C_elegido"]  # mismo C


@pytest.mark.parametrize("dano", [  # cada daño vuelve inconsistente el archivo
    lambda c: c.pop("version_sklearn"),  # clave faltante
    lambda c: c.pop("estimador"),  # clave faltante
    lambda c: c.update(version_sklearn="0.0.0"),  # otra versión de sklearn
    lambda c: c.update(n_entradas=N_COLUMNAS + 1),  # n_entradas no coincide con coef_
], ids=["sin_version", "sin_estimador", "otra_version", "n_entradas_distinto"])
def test_19_cargar_rechaza(dano, datos, tmp_path):
    """Clave faltante, versión de sklearn distinta o n_entradas ≠ coef_.shape[1] → ValueError."""
    contenido = _contenido_valido(datos)  # diccionario válido
    dano(contenido)  # lo daña
    joblib.dump(contenido, tmp_path / "malo.joblib")  # archivo a mano
    with pytest.raises(ValueError):  # se rechaza
        LogisticaPD.cargar(tmp_path / "malo.joblib")  # carga


# --------------------------------------------------------------------------------------------
# 20. Inversión de dependencias (buscar inyectada)
# --------------------------------------------------------------------------------------------
def test_20_buscar_inyectada_recibe_copia_y_float64(datos):
    """fit pasa a buscar la COPIA validada del bloque (no el objeto de cfg) y X en float64; luego usa su resultado."""
    X_tr, y_tr, X_val, y_val = datos  # float32
    falsa = _BuscarFalsa(resultado=_resultado_falso(datos))  # búsqueda falsa que termina
    cfg = _cfg()  # config completa
    m = LogisticaPD(buscar=falsa).fit(X_tr, y_tr, X_val, y_val, cfg, SEMILLA_MODELO)  # fit con la falsa
    assert len(falsa.llamadas) == 1  # una sola búsqueda
    llamada = falsa.llamadas[0]  # argumentos recibidos
    assert llamada["cfg_log"] == cfg["logistica"]  # mismo contenido
    assert llamada["cfg_log"] is not cfg["logistica"]  # pero es la copia validada
    assert llamada["X_tr"].dtype == np.float64 and llamada["X_val"].dtype == np.float64  # X en float64
    estimador = falsa.resultado[0]  # estimador que devolvió la falsa
    assert m.C_elegido == falsa.resultado[1][0]["C"]  # el C de la fila elegida
    assert m.busqueda == falsa.resultado[1]  # las filas de la falsa
    assert np.array_equal(m.predict_proba(X_val), estimador.predict_proba(X_val.astype(np.float64))[:, 1])  # su estimador


def test_20_buscar_inyectada_que_falla_deja_sin_entrenar(datos):
    """Si buscar lanza, el modelo sigue sin entrenar (el estado se asigna solo al terminar)."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    m = LogisticaPD(buscar=_BuscarFalsa(error=_FallaBusqueda("falla simulada")))  # búsqueda que se rompe
    with pytest.raises(_FallaBusqueda):  # la excepción se propaga
        m.fit(X_tr, y_tr, X_val, y_val, _cfg(), SEMILLA_MODELO)  # fit
    with pytest.raises(RuntimeError):  # sigue sin entrenar
        m.predict_proba(X_val)  # predicción


# --------------------------------------------------------------------------------------------
# 21. resumir
# --------------------------------------------------------------------------------------------
def test_21_resumir_una_corrida():
    """n = 1: desv = 0 por convención y mejor = peor = media."""
    corrida = {"auc_test": 0.7, "brier_test": 0.2}  # una sola corrida
    resumen = resumir([corrida])  # resumen
    for nombre, valor in corrida.items():  # cada métrica
        r = resumen[nombre]  # su resumen
        assert set(r) == CLAVES_RESUMEN  # exactamente las 5 claves
        assert r["n"] == 1 and r["desv"] == 0  # n = 1 → desv = 0
        assert r["media"] == r["mejor"] == r["peor"] == valor  # todo igual al valor


def test_21_resumir_tres_corridas():
    """n = 3: desv muestral (ddof = 1); mejor = máx en auc/gini/ks y mín en brier/logloss; sin NaN."""
    valores = {  # tres corridas por métrica
        "auc_test": [0.70, 0.75, 0.72], "gini_test": [0.40, 0.50, 0.44], "ks_test": [0.30, 0.36, 0.33],
        "brier_test": [0.20, 0.18, 0.22], "logloss_test": [0.60, 0.55, 0.65],
    }
    corridas = [{k: v[i] for k, v in valores.items()} for i in range(3)]  # lista de diccionarios
    resumen = resumir(corridas)  # resumen
    for nombre, x in valores.items():  # cada métrica
        r = resumen[nombre]  # su resumen
        mayor_es_mejor = nombre.split("_")[0] in {"auc", "gini", "ks"}  # sentido esperado
        assert r["n"] == 3  # tres corridas
        assert r["media"] == pytest.approx(np.mean(x), rel=1e-12)  # media
        assert r["desv"] == pytest.approx(np.std(x, ddof=1), rel=1e-12)  # desviación muestral
        assert r["mejor"] == (max(x) if mayor_es_mejor else min(x))  # mejor según el sentido
        assert r["peor"] == (min(x) if mayor_es_mejor else max(x))  # peor según el sentido
        assert not any(np.isnan(v) for v in (r["media"], r["desv"], r["mejor"], r["peor"]))  # nunca NaN


def test_21_sentido_declarado():
    """SENTIDO y METRICAS cubren las mismas 5 métricas; auc, gini, ks → max; brier, logloss → min."""
    assert set(METRICAS) == METRICAS_BASE  # nombres de las métricas
    assert SENTIDO == {"auc": "max", "gini": "max", "ks": "max", "brier": "min", "logloss": "min"}  # sentidos


# --------------------------------------------------------------------------------------------
# 22. actualizar_summary
# --------------------------------------------------------------------------------------------
def test_22_actualizar_summary_conserva_ajenas(tmp_path):
    """Reemplaza SOLO la entrada del modelo y conserva la de otro modelo ("mlp")."""
    ruta = tmp_path / "summary.json"  # summary de prueba
    actualizar_summary(ruta, "logistica", {"n_parametros": 0})  # crea el archivo
    actualizar_summary(ruta, "mlp", ENTRADA_MLP)  # agrega otra entrada
    actualizar_summary(ruta, "logistica", {"n_parametros": 7})  # reemplaza la de la logística
    assert load_json(ruta) == {"logistica": {"n_parametros": 7}, "mlp": ENTRADA_MLP}  # ambas, la nueva gana


def test_22_actualizar_summary_raiz_no_dict(tmp_path):
    """Un summary cuya raíz no es un diccionario → ValueError, sin tocar el archivo."""
    ruta = tmp_path / "summary.json"  # summary de prueba
    save_json(ruta, [1, 2, 3])  # raíz lista
    with pytest.raises(ValueError):  # se rechaza
        actualizar_summary(ruta, "logistica", {"n_parametros": 7})  # intento de actualizar
    assert load_json(ruta) == [1, 2, 3]  # el archivo no cambió


# --------------------------------------------------------------------------------------------
# 23. actualizar_pred_test
# --------------------------------------------------------------------------------------------
IDS_PRED = ["a", "b", "c", "d"]  # ids de prueba de pred_test
Y_PRED = np.array([0, 1, 0, 1])  # y de prueba de pred_test
P_MLP = np.array([0.1, 0.9, 0.2, 0.8])  # predicción ajena (mlp)
P_LOG = np.array([0.15, 0.85, 0.25, 0.75])  # predicción de la logística


def _por_id(df, columna):
    """Diccionario id → valor de ``columna``."""
    return dict(zip(df["id"], df[columna]))  # alineado por id


def test_23_pred_test_conserva_ajenas(tmp_path):
    """Agregar pred_logistica conserva pred_mlp y alinea por id."""
    ruta = tmp_path / "pred_test.parquet"  # archivo de prueba
    actualizar_pred_test(ruta, IDS_PRED, Y_PRED, "pred_mlp", P_MLP)  # crea con la columna ajena
    actualizar_pred_test(ruta, IDS_PRED, Y_PRED, "pred_logistica", P_LOG)  # agrega la logística
    df = pd.read_parquet(ruta)  # lee
    assert set(df.columns) == {"id", "y", "pred_mlp", "pred_logistica"}  # ambas columnas
    assert _por_id(df, "pred_mlp") == dict(zip(IDS_PRED, P_MLP))  # la ajena sigue igual
    assert _por_id(df, "pred_logistica") == dict(zip(IDS_PRED, P_LOG))  # la nueva, por id


def test_23_pred_test_alinea_por_id(tmp_path):
    """Los mismos ids en otro orden se alinean por id (no por posición)."""
    ruta = tmp_path / "pred_test.parquet"  # archivo de prueba
    actualizar_pred_test(ruta, IDS_PRED, Y_PRED, "pred_mlp", P_MLP)  # crea
    orden = [2, 0, 3, 1]  # permutación
    actualizar_pred_test(ruta, [IDS_PRED[i] for i in orden], Y_PRED[orden], "pred_logistica", P_LOG[orden])  # otro orden
    df = pd.read_parquet(ruta)  # lee
    assert _por_id(df, "pred_logistica") == dict(zip(IDS_PRED, P_LOG))  # cada id con su predicción
    assert _por_id(df, "pred_mlp") == dict(zip(IDS_PRED, P_MLP))  # la ajena no se desalineó
    assert _por_id(df, "y") == dict(zip(IDS_PRED, Y_PRED))  # y tampoco


def test_23_pred_test_otros_ids(tmp_path):
    """Otro conjunto de ids → ValueError, sin tocar el archivo."""
    ruta = tmp_path / "pred_test.parquet"  # archivo de prueba
    actualizar_pred_test(ruta, IDS_PRED, Y_PRED, "pred_mlp", P_MLP)  # crea
    antes = pd.read_parquet(ruta)  # foto
    with pytest.raises(ValueError):  # otros ids
        actualizar_pred_test(ruta, ["a", "b", "c", "z"], Y_PRED, "pred_logistica", P_LOG)  # "z" en lugar de "d"
    pd.testing.assert_frame_equal(pd.read_parquet(ruta), antes)  # el archivo no cambió


# --------------------------------------------------------------------------------------------
# 24. correr_logistica (rápida, parquet sintético)
# --------------------------------------------------------------------------------------------
def test_24_correr_logistica_escribe_los_archivos(experimento):
    """Escribe semilla, búsqueda, summary, pred_test y joblib con exactamente las claves acordadas."""
    rutas, cfg, n_cols = experimento["rutas"], experimento["cfg"], experimento["n_columnas"]  # escenario
    semilla = load_json(rutas.resultados / "semillas" / "logistica_seed0.json")  # load_json falla si hay NaN
    busqueda = load_json(rutas.resultados / "busqueda_logistica.json")  # búsqueda
    summary = load_json(rutas.resultados / "summary.json")  # summary
    assert (rutas.modelos / "pd_logistica.joblib").is_file()  # modelo guardado
    assert set(semilla) == CLAVES_SEMILLA  # claves de la semilla
    assert semilla["modelo"] == "logistica" and semilla["seed"] == 0 and semilla["device"] == "cpu"  # identificación
    assert semilla["config"] == {**cfg["logistica"], "C_elegido": busqueda["C_elegido"]}  # bloque + C elegido
    assert set(semilla["metricas"]) == {f"{m}_{s}" for m in METRICAS_BASE for s in SPLITS_MEDIDOS}  # 10 métricas
    assert semilla["n_parametros"] == n_cols + 1  # d + 1
    elegida = [f for f in busqueda["filas"] if f["C"] == busqueda["C_elegido"]][0]  # fila del C elegido
    assert semilla["tiempo_s"] == elegida["tiempo_s"]  # tiempo del fit del C elegido
    assert semilla["epocas"] == elegida["n_iter"]  # iteraciones de lbfgs del C elegido
    assert set(busqueda) == CLAVES_BUSQUEDA  # claves de la búsqueda
    assert busqueda["criterio"] == "logloss_val"  # criterio declarado
    assert [f["C"] for f in busqueda["filas"]] == REJILLA_CORTA  # una fila por C, en orden
    assert busqueda["en_borde"] == esta_en_borde(REJILLA_CORTA, busqueda["C_elegido"])  # borde coherente
    assert busqueda["tiempo_total_s"] >= sum(f["tiempo_s"] for f in busqueda["filas"]) > 0  # presupuesto de tiempo
    entrada = summary["logistica"]  # entrada de la logística
    assert set(entrada) == CLAVES_SUMMARY  # claves del summary
    assert entrada["n_configuraciones_busqueda"] == len(REJILLA_CORTA)  # presupuesto = tamaño de la rejilla
    assert entrada["n_parametros"] == n_cols + 1  # d + 1
    assert entrada["tiempo_total_s"] == busqueda["tiempo_total_s"] > 0  # el mismo reloj de todo el fit
    assert set(entrada["metricas"]) == set(semilla["metricas"])  # las mismas métricas
    for nombre, r in entrada["metricas"].items():  # cada resumen
        assert r == {"media": semilla["metricas"][nombre], "desv": 0, "mejor": semilla["metricas"][nombre],
                     "peor": semilla["metricas"][nombre], "n": 1}  # una corrida: desv = 0
    assert experimento["resultado"] == entrada  # devuelve la entrada del summary


def test_24_pred_test_del_experimento(experimento):
    """pred_test.parquet tiene id, y, pred_logistica para los ids de test y las PD del modelo guardado."""
    rutas, df = experimento["rutas"], experimento["df"]  # escenario
    pred = pd.read_parquet(rutas.resultados / "pred_test.parquet")  # predicciones de test
    assert list(pred.columns) == ["id", "y", "pred_logistica"]  # columnas exactas
    test = df[df["split"] == "test"]  # filas de test del parquet
    assert set(pred["id"]) == set(test["id"]) and len(pred) == len(test)  # solo test, sin duplicados
    assert _por_id(pred, "y") == dict(zip(test["id"], test["target"]))  # y = target, por id
    matrices = cargar_matrices(rutas)  # mismas matrices que el experimento
    X_te, _, ids_te = matrices["test"]  # test
    p = LogisticaPD.cargar(rutas.modelos / "pd_logistica.joblib").predict_proba(X_te)  # PD del modelo guardado
    assert _por_id(pred, "pred_logistica") == dict(zip(ids_te, p))  # mismas PD, por id


def test_24_metricas_de_la_semilla(experimento):
    """Las métricas de la semilla son las del modelo guardado en val y test (test se mide con el modelo final)."""
    rutas = experimento["rutas"]  # escenario
    semilla = load_json(rutas.resultados / "semillas" / "logistica_seed0.json")  # semilla
    m = LogisticaPD.cargar(rutas.modelos / "pd_logistica.joblib")  # modelo guardado
    esperado = medir(m, cargar_matrices(rutas))  # métricas recalculadas
    assert semilla["metricas"] == pytest.approx(esperado, rel=1e-12)  # coinciden


# --------------------------------------------------------------------------------------------
# 25. correr_logistica con los datos reales (lenta)
# --------------------------------------------------------------------------------------------
@pytest.mark.lento
def test_25_correr_logistica_datos_reales(tmp_path):
    """Con los datos de A y configs/pd/base.yaml: 5 archivos, 13 configuraciones, 49 parámetros; conserva lo ajeno."""
    parquet, pre = ROOT / RUTAS_PD.parquet, ROOT / RUTAS_PD.preprocesador  # datos reales de A
    if not (parquet.is_file() and pre.is_file()):  # el clon no trae los datos procesados
        pytest.skip("faltan data/processed/loans_pd.parquet o models/preprocesador_pd.json")  # se corre a mano
    rutas = RutasPD(parquet, pre, tmp_path / "resultados", tmp_path / "models")  # resultados en tmp_path
    cfg = load_config(ROOT / "configs" / "pd" / "base.yaml")  # config declarada
    entrada = correr_logistica(cfg, rutas=rutas)  # experimento completo
    archivos = [rutas.resultados / "semillas" / "logistica_seed0.json", rutas.resultados / "busqueda_logistica.json",
                rutas.resultados / "summary.json", rutas.resultados / "pred_test.parquet",
                rutas.modelos / "pd_logistica.joblib"]  # los 5 archivos
    assert all(a.is_file() for a in archivos)  # todos existen
    assert entrada["n_configuraciones_busqueda"] == N_REJILLA_BASE  # 13 configuraciones
    assert entrada["tiempo_total_s"] > 0  # tiempo registrado
    assert entrada["n_parametros"] == N_PARAMETROS_REAL  # 48 + 1
    actualizar_summary(rutas.resultados / "summary.json", "mlp", ENTRADA_MLP)  # entrada ajena
    pred = pd.read_parquet(rutas.resultados / "pred_test.parquet")  # predicciones de test
    actualizar_pred_test(rutas.resultados / "pred_test.parquet", pred["id"].tolist(), pred["y"].to_numpy(),
                         "pred_mlp", pred["pred_logistica"].to_numpy())  # columna ajena
    correr_logistica(cfg, rutas=rutas)  # segunda corrida
    assert load_json(rutas.resultados / "summary.json")["mlp"] == ENTRADA_MLP  # la entrada ajena sigue
    assert "pred_mlp" in pd.read_parquet(rutas.resultados / "pred_test.parquet").columns  # la columna ajena sigue


# --------------------------------------------------------------------------------------------
# 26. Huellas
# --------------------------------------------------------------------------------------------
def test_26_huella_particion():
    """Estable, sensible al split y igual al SHA-256 del texto f"{id},{split}\\n" en orden."""
    ids, splits = ["1", "2", "3"], ["train", "val", "test"]  # partición mínima
    texto = "1,train\n2,val\n3,test\n"  # texto definido
    h = huella_particion(ids, splits)  # huella
    assert h == huella_particion(list(ids), list(splits))  # estable
    assert h == hashlib.sha256(texto.encode("utf-8")).hexdigest()  # definición exacta
    assert h != huella_particion(ids, ["train", "val", "val"])  # cambia si cambia un split
    assert h != huella_particion(["2", "1", "3"], ["val", "train", "test"])  # depende del orden


def test_26_huella_archivo(tmp_path):
    """SHA-256 de los bytes del archivo; cambia si cambia un byte."""
    ruta = tmp_path / "pre.json"  # archivo de prueba
    ruta.write_bytes(b'{"a": 1}\n')  # contenido
    h = huella_archivo(ruta)  # huella
    assert h == hashlib.sha256(b'{"a": 1}\n').hexdigest()  # definición exacta
    ruta.write_bytes(b'{"a": 2}\n')  # cambia un byte
    assert huella_archivo(ruta) != h  # cambia la huella


def test_26_evidencias_iguales(experimento):
    """Las evidencias de semilla, búsqueda y summary son iguales entre sí y a las huellas recalculadas."""
    rutas, df = experimento["rutas"], experimento["df"]  # escenario
    evidencias = [load_json(rutas.resultados / "semillas" / "logistica_seed0.json")["evidencia"],
                  load_json(rutas.resultados / "busqueda_logistica.json")["evidencia"],
                  load_json(rutas.resultados / "summary.json")["logistica"]["evidencia"]]  # las tres
    esperado = {"sha256_particion": _sha256_particion(df),  # partición en el orden del parquet
                "sha256_preprocesador": hashlib.sha256(rutas.preprocesador.read_bytes()).hexdigest()}  # bytes del JSON
    assert all(e == esperado for e in evidencias)  # iguales entre sí y a la referencia


# --------------------------------------------------------------------------------------------
# 27. Formato de la evidencia
# --------------------------------------------------------------------------------------------
def test_27_evidencia_hex64(experimento):
    """Los tres JSON llevan evidencia con exactamente sha256_particion y sha256_preprocesador (64 hex)."""
    rutas = experimento["rutas"]  # escenario
    evidencias = [load_json(rutas.resultados / "semillas" / "logistica_seed0.json")["evidencia"],
                  load_json(rutas.resultados / "busqueda_logistica.json")["evidencia"],
                  load_json(rutas.resultados / "summary.json")["logistica"]["evidencia"]]  # las tres
    for e in evidencias:  # cada evidencia
        assert set(e) == CLAVES_EVIDENCIA  # exactamente las dos claves
        assert all(HEX64.fullmatch(e[k]) for k in CLAVES_EVIDENCIA)  # 64 dígitos hexadecimales


# --------------------------------------------------------------------------------------------
# 28. Borde de la rejilla
# --------------------------------------------------------------------------------------------
def test_28_esta_en_borde():
    """C en el mínimo o el máximo → True; en el interior → False; rejilla de un valor → False (sin desorden)."""
    rejilla = [1.0, 0.01, 0.1]  # desordenada a propósito: el borde es por valor, no por posición
    assert esta_en_borde(rejilla, 0.01) is True  # mínimo (en la posición 1)
    assert esta_en_borde(rejilla, 1.0) is True  # máximo (en la posición 0)
    assert esta_en_borde(rejilla, 0.1) is False  # interior (en la última posición)
    assert esta_en_borde([C_UNICO], C_UNICO) is False  # un solo valor: no hay búsqueda ni borde


# --------------------------------------------------------------------------------------------
# Pruebas añadidas (ver el docstring del módulo)
# --------------------------------------------------------------------------------------------
def test_a01_buscar_ajusta_cada_C_solo_con_train(datos):
    """Cada C se crea con max_iter y tol de la config, en el orden de la rejilla, y se ajusta solo con X_tr."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    X_tr64, X_val64 = X_tr.astype(np.float64), X_val.astype(np.float64)  # float64
    registro = []  # llamadas del espía
    crear = lambda C, max_iter, tol: _Espia(registro, C, max_iter, tol)  # noqa: E731 — fábrica espía
    buscar_C(X_tr64, y_tr, X_val64, y_val, _cfg(rejilla_C=list(REJILLA_CORTA))["logistica"], crear=crear)  # búsqueda
    assert [r["C"] for r in registro] == REJILLA_CORTA  # un estimador por C, en orden (sin refit extra)
    assert all(r["max_iter"] == MAX_ITER and r["tol"] == TOL for r in registro)  # parámetros de la config
    assert all(np.array_equal(r["X"], X_tr64) for r in registro)  # cada ajuste usa solo train


def test_a02_filas_miden_en_validacion(modelo, datos):
    """logloss_val y auc_val del C elegido son las del modelo en validación."""
    _, _, X_val, y_val = datos  # validación
    fila = [f for f in modelo.busqueda if f["C"] == modelo.C_elegido][0]  # fila del C elegido
    p = modelo.predict_proba(X_val)  # PD de validación
    assert fila["logloss_val"] == pytest.approx(log_loss(y_val, p, labels=[0, 1]), rel=1e-12)  # criterio en val
    assert fila["auc_val"] == pytest.approx(auc(y_val, p), rel=1e-12)  # AUC en val


def test_a03_logloss_con_una_sola_clase():
    """logloss usa labels=[0, 1]: no falla con una sola clase y coincide con sklearn."""
    y = np.zeros(4, dtype=np.int64)  # una sola clase
    p = np.array([0.1, 0.2, 0.3, 0.4])  # PD cualquiera
    assert logloss(y, p) == pytest.approx(log_loss(y, p, labels=[0, 1]), rel=1e-12)  # igual a sklearn
    y2, p2 = np.array([0, 1, 1, 0]), np.array([0.2, 0.7, 0.6, 0.1])  # dos clases
    assert logloss(y2, p2) == pytest.approx(log_loss(y2, p2), rel=1e-12)  # igual a sklearn


def test_a04_config_base_valida():
    """configs/pd/base.yaml tiene solo el bloque logistica, válido, con 13 valores de C."""
    cfg = load_config(ROOT / "configs" / "pd" / "base.yaml")  # config declarada
    assert set(cfg) == {"logistica"}  # B5 añade el bloque mlp
    validado = validar_cfg_logistica(cfg["logistica"])  # pasa la validación
    assert validado == CFG_PRUEBA["logistica"]  # los valores declarados en el prompt
    assert len(validado["rejilla_C"]) == N_REJILLA_BASE  # 13 configuraciones


def test_a05_evaluar_usa_las_metricas_comunes():
    """evaluar devuelve auc, gini, ks, brier y logloss calculadas con src.common.metricas y logloss."""
    rng = np.random.default_rng(SEMILLA_DATOS)  # generador
    p = rng.uniform(0.01, 0.99, size=200)  # PD cualquiera
    y = rng.binomial(1, p)  # etiquetas coherentes con p
    r = evaluar(y, p)  # métricas
    assert set(r) == METRICAS_BASE  # los 5 nombres
    esperado = {"auc": auc(y, p), "gini": gini(y, p), "ks": ks(y, p), "brier": brier(y, p),
                "logloss": logloss(y, p)}  # referencia por nombre
    assert r == pytest.approx(esperado, rel=1e-12)  # cada métrica con su función


def test_a06_medir_nombra_por_split(modelo, datos):
    """medir produce {metrica}_{split} con las PD del modelo en cada split."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    ids_tr, ids_val = np.arange(N_TRAIN).astype(str), np.arange(N_VAL).astype(str)  # ids cualesquiera
    m_datos = {"val": (X_val, y_val, ids_val), "test": (X_tr, y_tr, ids_tr)}  # train hace de "test" aquí
    r = medir(modelo, m_datos)  # métricas por split
    assert set(r) == {f"{m}_{s}" for m in METRICAS_BASE for s in SPLITS_MEDIDOS}  # 10 nombres
    for s, (X, y, _) in m_datos.items():  # cada split
        esperado = evaluar(y, modelo.predict_proba(X))  # métricas del split
        assert all(r[f"{m}_{s}"] == pytest.approx(v, rel=1e-12) for m, v in esperado.items())  # coinciden


def test_a07_cargar_matrices_no_reajusta(experimento):
    """Las matrices salen del preprocesador guardado (sin fit), con y = target e ids en el orden del parquet."""
    rutas, df = experimento["rutas"], experimento["df"]  # escenario
    pre = Preprocesador.from_dict(load_json(rutas.preprocesador))  # el preprocesador guardado
    matrices = cargar_matrices(rutas)  # matrices por split
    assert set(matrices) == {"train", "val", "test"}  # tres splits
    for split, (X, y, ids) in matrices.items():  # cada split
        parte = df[df["split"] == split]  # filas del split en el orden del parquet
        assert np.array_equal(X, pre.transform(parte))  # misma transformación, sin reajuste
        assert np.array_equal(np.asarray(y), parte["target"].to_numpy())  # y = target
        assert list(ids) == parte["id"].tolist()  # ids en orden


def test_a08_crear_estimador_sin_class_weight():
    """crear_estimador: lbfgs con C, max_iter y tol; sin class_weight y con penalty por defecto."""
    est = crear_estimador(C_EQUIVALENCIA, MAX_ITER, TOL)  # estimador
    por_defecto = LogisticRegression().get_params()  # parámetros por defecto de sklearn
    params = est.get_params()  # parámetros del nuestro
    assert isinstance(est, LogisticRegression)  # tipo
    assert (params["C"], params["solver"], params["max_iter"], params["tol"]) == (C_EQUIVALENCIA, "lbfgs",
                                                                                  MAX_ITER, TOL)  # los pedidos
    otros = {k: v for k, v in params.items() if k not in {"C", "solver", "max_iter", "tol"}}  # el resto
    assert otros == {k: por_defecto[k] for k in otros}  # todo lo demás por defecto (class_weight, penalty, ...)


class _EstimadorQueAvisa:
    """Estimador falso cuyo fit emite el aviso indicado."""

    def __init__(self, categoria):
        self.categoria = categoria  # clase del aviso que emite

    def fit(self, X, y):
        warnings.warn("aviso simulado", self.categoria)  # emite el aviso
        return self  # como sklearn


@pytest.mark.parametrize("categoria", [ConvergenceWarning, FutureWarning])  # los dos avisos que son error
def test_a09_ajustar_sin_avisos_ignora_filtros_globales(categoria, datos):
    """Los avisos son error dentro de ajustar_sin_avisos aunque los filtros globales los ignoren."""
    X_tr, y_tr, _, _ = datos  # train
    with warnings.catch_warnings():  # filtros temporales de la prueba
        warnings.simplefilter("ignore")  # el entorno ignora todo
        with pytest.raises(categoria):  # aun así es error
            ajustar_sin_avisos(_EstimadorQueAvisa(categoria), X_tr, y_tr)  # ajuste que avisa


def test_a10_buscar_usa_el_reloj_inyectado(datos):
    """buscar_C mide con el reloj que recibe (DIP): cada tiempo_s es exactamente un paso del reloj falso."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    contador = itertools.count()  # llamadas al reloj
    reloj = lambda: next(contador) * PASO_RELOJ  # noqa: E731 — reloj falso que avanza PASO_RELOJ por llamada
    _, filas, _ = buscar_C(X_tr.astype(np.float64), y_tr, X_val.astype(np.float64), y_val,
                           _cfg(rejilla_C=list(REJILLA_CORTA))["logistica"], reloj=reloj)  # búsqueda
    assert all(f["tiempo_s"] == PASO_RELOJ for f in filas)  # dos lecturas consecutivas por ajuste: un paso exacto


def test_a11_buscar_falsa_que_falla_no_altera_modelo_entrenado(datos):
    """Un re-fit cuya búsqueda falla deja intacto el modelo entrenado antes."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    falsa = _BuscarFalsa(resultado=_resultado_falso(datos))  # primero termina
    m = LogisticaPD(buscar=falsa).fit(X_tr, y_tr, X_val, y_val, _cfg(), SEMILLA_MODELO)  # modelo entrenado
    p_antes, C_antes = m.predict_proba(X_val), m.C_elegido  # foto del estado
    falsa.error = _FallaBusqueda("falla simulada")  # ahora la búsqueda se rompe
    with pytest.raises(_FallaBusqueda):  # la excepción se propaga
        m.fit(X_tr, y_tr, X_val, y_val, _cfg(), SEMILLA_MODELO)  # re-fit
    assert np.array_equal(m.predict_proba(X_val), p_antes)  # mismas PD
    assert m.C_elegido == C_antes  # mismo C


def test_a12_pred_test_rechaza_otra_y(tmp_path):
    """El mismo id con otra y → ValueError, sin tocar el archivo."""
    ruta = tmp_path / "pred_test.parquet"  # archivo de prueba
    actualizar_pred_test(ruta, IDS_PRED, Y_PRED, "pred_mlp", P_MLP)  # crea
    antes = pd.read_parquet(ruta)  # foto
    with pytest.raises(ValueError):  # y distinta para los mismos ids
        actualizar_pred_test(ruta, IDS_PRED, 1 - Y_PRED, "pred_logistica", P_LOG)  # y invertida
    pd.testing.assert_frame_equal(pd.read_parquet(ruta), antes)  # el archivo no cambió


@pytest.mark.parametrize("cambio", [  # casos válidos que no deben rechazarse
    {"rejilla_C": [1, 10]},  # enteros: son reales finitos > 0
    {"rejilla_C": [1.0, 0.01, 0.1]},  # desordenada: no se exige orden
    {"rejilla_C": [C_UNICO]},  # un solo valor (C fijo)
    {"max_iter": 1},  # el mínimo permitido
], ids=["enteros", "desordenada", "un_valor", "max_iter_uno"])
def test_a13_validacion_acepta_casos_validos(cambio):
    """Enteros, rejilla desordenada, un solo valor y max_iter = 1 son válidos."""
    cfg_log = _cfg(**cambio)["logistica"]  # bloque válido
    assert validar_cfg_logistica(cfg_log) == cfg_log  # pasa sin cambios


def test_a14_validacion_acepta_tupla_y_devuelve_lista():
    """Una tupla no vacía es una rejilla válida (como mlp.capas); la copia validada la devuelve como lista."""
    tupla = tuple(REJILLA_CORTA)  # rejilla como tupla
    validado = validar_cfg_logistica(_cfg(rejilla_C=tupla)["logistica"])  # valida
    assert isinstance(validado["rejilla_C"], list)  # lista en la copia
    assert validado["rejilla_C"] == list(tupla)  # mismos valores, mismo orden


def test_a15_predict_rechaza_clases_distintas(datos, tmp_path):
    """Un estimador cuyas clases no son [0, 1] (p. ej. ajustado con y ∈ {1, 2}) → ValueError al predecir."""
    X_tr, y_tr, X_val, _ = datos  # desempaca
    contenido = _contenido_valido(datos)  # diccionario válido
    contenido["estimador"] = crear_estimador(C_UNICO, MAX_ITER, TOL).fit(X_tr.astype(np.float64), y_tr + 1)  # clases 1, 2
    joblib.dump(contenido, tmp_path / "clases.joblib")  # archivo a mano
    with pytest.raises(ValueError):  # la columna 1 de predict_proba no sería P(malo)
        LogisticaPD.cargar(tmp_path / "clases.joblib").predict_proba(X_val)  # cargar o predecir lo rechazan


def test_a16_tiempo_total_del_fit(modelo, datos):
    """tiempo_total_s es ≥ la suma de los tiempos de cada C y sale del reloj inyectado en LogisticaPD."""
    assert modelo.tiempo_total_s >= sum(f["tiempo_s"] for f in modelo.busqueda) > 0  # presupuesto de tiempo real
    contador = itertools.count()  # llamadas al reloj del fit
    reloj = lambda: next(contador) * PASO_RELOJ  # noqa: E731 — reloj falso que avanza PASO_RELOJ por llamada
    m = _ajustar(datos, _cfg(rejilla_C=list(REJILLA_CORTA)), reloj=reloj)  # fit con el reloj falso
    assert m.tiempo_total_s == PASO_RELOJ  # inicio y fin del fit: un paso exacto


def test_a17_cfg_sin_bloque_logistica(datos, tmp_path):
    """cfg sin "logistica" o que no es dict → ValueError en fit y en correr_logistica (antes de leer datos)."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    rutas = RutasPD(tmp_path / "no_existe.parquet", tmp_path / "no_existe.json", tmp_path / "res",
                    tmp_path / "mod")  # rutas inexistentes: si se leyeran, el error sería otro
    cfg_ok = _cfg()  # config válida
    assert bloque_logistica(cfg_ok) == cfg_ok["logistica"]  # mismo contenido
    assert bloque_logistica(cfg_ok) is not cfg_ok["logistica"]  # pero es la copia validada
    for cfg in ({"mlp": {}}, [CFG_PRUEBA["logistica"]], None):  # sin bloque, lista, None
        with pytest.raises(ValueError):  # la función pública
            bloque_logistica(cfg)  # config inválida
        with pytest.raises(ValueError):  # fit
            LogisticaPD().fit(X_tr, y_tr, X_val, y_val, cfg, SEMILLA_MODELO)  # config inválida
        with pytest.raises(ValueError):  # experimento
            correr_logistica(cfg, rutas=rutas)  # config inválida
    assert not any(tmp_path.iterdir())  # no se creó nada


class _ModeloQueNoDebeUsarse:
    """``modelo_cls`` falso: registra si alguien lo instancia (correr_logistica no debe llegar a entrenar)."""

    instancias = 0  # contador de instancias

    def __init__(self):
        type(self).instancias += 1  # cuenta la instancia
        raise AssertionError("correr_logistica entrenó con una semilla inválida")  # nunca debe ocurrir


@pytest.mark.parametrize("seed", [-1, "0"], ids=["negativa", "texto"])  # semillas inválidas
def test_a18_correr_logistica_valida_la_semilla(seed, tmp_path):
    """seed = −1 o "0" → ValueError antes de cargar datos o entrenar, sin crear ningún archivo."""
    rutas, _, cfg = _escenario(tmp_path)  # parquet y preprocesador sintéticos
    _ModeloQueNoDebeUsarse.instancias = 0  # reinicia el contador
    with pytest.raises(ValueError):  # semilla inválida
        correr_logistica(cfg, rutas=rutas, modelo_cls=_ModeloQueNoDebeUsarse, seed=seed)  # experimento
    assert _ModeloQueNoDebeUsarse.instancias == 0  # no llegó a entrenar
    assert not rutas.resultados.exists() and not rutas.modelos.exists()  # ningún archivo de salida


def test_a19_pred_test_rechaza_ids_repetidos(tmp_path):
    """Un id repetido → ValueError, sin crear el archivo."""
    ruta = tmp_path / "pred_test.parquet"  # archivo de prueba
    with pytest.raises(ValueError):  # ids repetidos
        actualizar_pred_test(ruta, ["a", "a", "b", "c"], Y_PRED, "pred_logistica", P_LOG)  # "a" dos veces
    assert not ruta.exists()  # no se escribió nada


def test_a20_resumir_rechaza_metrica_sin_sentido():
    """Una métrica cuya base no está en SENTIDO → ValueError."""
    with pytest.raises(ValueError):  # sin sentido declarado
        resumir([{"exactitud_test": 0.9}])  # "exactitud" no está en SENTIDO


def test_a21_resumir_sin_corridas():
    """resumir([]) → ValueError."""
    with pytest.raises(ValueError):  # sin corridas
        resumir([])  # lista vacía


def test_a22_tiempo_total_tras_cargar(modelo, tmp_path):
    """Un modelo cargado no tiene tiempo_total_s (no se guarda en disco) → RuntimeError."""
    modelo.guardar(tmp_path / "m.joblib")  # guarda
    with pytest.raises(RuntimeError):  # sin fit en esta sesión
        LogisticaPD.cargar(tmp_path / "m.joblib").tiempo_total_s  # noqa: B018 — se accede para provocar el error


def test_a23_cargar_rechaza_contenido_no_dict(datos, tmp_path):
    """Un joblib cuyo contenido no es un diccionario → ValueError."""
    joblib.dump(_resultado_falso(datos)[0], tmp_path / "estimador.joblib")  # el estimador solo, sin diccionario
    with pytest.raises(ValueError):  # contenido inesperado
        LogisticaPD.cargar(tmp_path / "estimador.joblib")  # carga


def test_a24_elegir_indice_sin_filas():
    """elegir_indice([]) → ValueError con mensaje propio."""
    with pytest.raises(ValueError, match="filas"):  # mensaje de nuestra guarda, no el de min()
        elegir_indice([])  # sin filas


def test_a25_medir_usa_las_metricas_inyectadas(modelo, datos):
    """medir pasa ``metricas`` a evaluar: con una métrica falsa, solo aparece ella por split."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca
    m_datos = {"val": (X_val, y_val, None), "test": (X_tr, y_tr, None)}  # ids no se usan
    r = medir(modelo, m_datos, metricas={"uno": lambda y, p: 1.0})  # métrica falsa
    assert r == {"uno_val": 1.0, "uno_test": 1.0}  # solo la inyectada


@pytest.mark.parametrize("seed", [-1, "0"], ids=["negativa", "texto"])  # semillas inválidas
def test_a26_escribir_semilla_valida_la_semilla(seed, tmp_path):
    """escribir_semilla con seed = −1 o "0" → ValueError, sin crear el archivo."""
    rutas = RutasPD(tmp_path / "p.parquet", tmp_path / "pre.json", tmp_path / "res", tmp_path / "mod")  # salida temporal
    with pytest.raises(ValueError):  # semilla inválida
        escribir_semilla(rutas, "logistica", seed, {"modelo": "logistica"})  # registro mínimo
    assert not rutas.resultados.exists()  # no se escribió nada


@pytest.mark.skipif(os.name == "nt", reason="Windows no usa los bits de permisos POSIX")  # solo POSIX
def test_a27_pred_test_queda_legible(tmp_path):
    """pred_test.parquet queda con permisos 0o644 al crearlo y al reemplazarlo (no 0o600 del temporal)."""
    ruta = tmp_path / "pred_test.parquet"  # archivo de prueba
    actualizar_pred_test(ruta, IDS_PRED, Y_PRED, "pred_mlp", P_MLP)  # crea
    assert stat.S_IMODE(os.stat(ruta).st_mode) == PERMISOS_LEGIBLES  # legible al crearlo
    actualizar_pred_test(ruta, IDS_PRED, Y_PRED, "pred_logistica", P_LOG)  # reemplaza
    assert stat.S_IMODE(os.stat(ruta).st_mode) == PERMISOS_LEGIBLES  # y al reemplazarlo
