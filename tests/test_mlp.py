"""Pruebas del MLP de PD y del bucle de entrenamiento genérico (Parte B §3.2, tarea B2).

Solo se usa la interfaz pública de ``src.common.torch_utils``, ``src.pd.entrenar`` y ``src.pd.mlp``. No se prueba
PyTorch (que Adam, ``weight_decay``, ``Dropout``, ``eval()`` o ``manual_seed`` hagan lo que dicen): se prueban
nuestras firmas, nuestras decisiones (L2 sin sesgos, parada temprana con mejora estricta, pérdida no finita →
``FloatingPointError``, logit sin sigmoide) y el determinismo esperado en CPU (núcleo §7.1).

Datos: sintéticos, generados aquí con una logística conocida (núcleo §0 regla 8: solo para desarrollar).
"""

import copy
import re

import numpy as np
import pytest
import torch
from scipy.special import expit
from torch import nn

from src.common.config import apply_overrides
from src.common.seeds import fijar_semillas
from src.common.torch_utils import a_numpy, a_tensor, resolver_device
from src.pd.entrenar import (
    ParadaTemprana,
    ResultadoEntrenamiento,
    construir_optimizador,
    entrenar,
    epoca_entrenamiento,
    evaluar_perdida,
    grupos_de_parametros,
    iterar_minilotes,
)
from src.pd.mlp import (
    ACTIVACIONES,
    MLPPD,
    construir_perdida,
    construir_red,
    contar_parametros,
    validar_cfg_mlp,
)

# --------------------------------------------------------------------------------------------
# Constantes de prueba (no son del Spec ni del dominio; solo fijan los escenarios sintéticos)
# --------------------------------------------------------------------------------------------
SEMILLA = 20261024  # semilla declarada de los datos sintéticos
SEMILLA_MODELO = 7  # semilla con la que se entrena el MLP en las pruebas
SEMILLA_OTRA = 8  # una semilla distinta para comprobar que la semilla importa
N_COLUMNAS = 48  # número de columnas del preprocesador real (A); aquí solo fija la forma
N_TRAIN = 600  # filas de entrenamiento sintéticas
N_VAL = 200  # filas de validación sintéticas
ESCALA_PESOS = 0.3  # desviación de los coeficientes de la logística generadora
INTERCEPTO = -1.0  # intercepto de la logística generadora (deja menos malos que buenos)
TOL = 1e-12  # tolerancia numérica para comparar flotantes en float64

# Configuración de prueba del bloque mlp (núcleo §5.3 + pos_weight de PD). Constante de prueba, no del Spec.
CFG_PRUEBA = {
    "mlp": {
        "capas": [48, 24],  # dos capas ocultas
        "activacion": "relu",  # activación de las capas ocultas
        "dropout": 0.2,  # probabilidad de apagar una neurona al entrenar
        "l2": 1e-4,  # coeficiente λ de la penalización L2 (weight_decay)
        "optimizador": "adam",  # Adam (nunca AdamW)
        "lr": 1e-3,  # tasa de aprendizaje
        "momento": 0.9,  # momento de SGD (Adam lo ignora)
        "batch": 512,  # tamaño de minilote
        "max_epocas": 60,  # tope de épocas
        "paciencia": 5,  # épocas sin mejora antes de detener
        "device": "cpu",  # las pruebas corren en CPU
        "pos_weight": 1.0,  # peso de la clase positiva en la BCE (decisión de Andrés: 1)
    }
}

# Conteos de parámetros hechos a mano con 48 entradas (pesos + sesgos de cada Linear).
N_PARAMS_48_24 = (48 * 48 + 48) + (48 * 24 + 24) + (24 * 1 + 1)  # 2352 + 1176 + 25 = 3553
N_PARAMS_64_32 = (48 * 64 + 64) + (64 * 32 + 32) + (32 * 1 + 1)  # 3136 + 2080 + 33 = 5249
N_PARAMS_SALIDA_24 = 24 * 1 + 1  # parámetros de la capa de salida Linear(24, 1)

DROPOUT_ALTO = 0.5  # dropout grande: si se aplicara al predecir, dos predicciones diferirían
L2_FUERTE = 1e-1  # λ grande para que el encogimiento de los pesos sea claro
L2_PRUEBA = 1e-3  # λ cualquiera > 0 para revisar los grupos de parámetros
EPOCAS_CORTAS = 2  # épocas de las pruebas de combinaciones
OVERRIDES_EN_VIVO = ["mlp.capas=[64,32]", "mlp.dropout=0.3", "mlp.lr=1e-2", "mlp.max_epocas=5"]  # núcleo §4.1
MAX_EPOCAS_OVERRIDE = 5  # el valor de mlp.max_epocas en OVERRIDES_EN_VIVO
DROPOUT_OVERRIDE = 0.3  # el valor de mlp.dropout en OVERRIDES_EN_VIVO
CAPAS_OVERRIDE = [64, 32]  # el valor de mlp.capas en OVERRIDES_EN_VIVO

# Escenarios del bucle genérico (regresión lineal con nn.Linear).
N_REG = 256  # filas de la regresión sintética
D_REG = 3  # entradas de la regresión sintética
BATCH_REG = 64  # minilote de la regresión
LR_REG = 1e-2  # tasa de aprendizaje de la regresión
EPOCAS_REG = 20  # épocas de la regresión que debe bajar la pérdida
PACIENCIA_SOBREAJUSTE = 3  # paciencia del escenario que sobreajusta
MAX_EPOCAS_SOBREAJUSTE = 50  # tope holgado: la parada debe llegar antes
MAX_EPOCAS_SIN_PARADA = 8  # tope corto con paciencia mayor: termina por max_epocas
PACIENCIA_SIN_PARADA = 100  # paciencia que nunca se agota en MAX_EPOCAS_SIN_PARADA
EPOCA_NO_FINITA = 2  # época en la que la pérdida guionizada deja de ser finita
CAPAS_EVAL = [16]  # una capa oculta para la prueba de evaluar_perdida con dropout
N_EPOCA = 10  # filas de la prueba de la media ponderada de la época
BATCH_EPOCA = 3  # minilotes 3 + 3 + 3 + 1: el último es incompleto
PERDIDA_CONSTANTE = 0.7  # pérdida guionizada constante de cada minilote
TOL_F32 = 1e-6  # tolerancia relativa para cantidades acumuladas en float32
POS_WEIGHT_PRUEBA = 2.0  # w₊ de la prueba de la BCE ponderada
N_COLUMNAS_OTRAS = 5  # columnas de un re-fit inválido


# --------------------------------------------------------------------------------------------
# Fixtures y utilidades
# --------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restaurar_determinismo():
    """Guarda y restaura el modo determinista global de torch para no contaminar otras pruebas."""
    previo = torch.are_deterministic_algorithms_enabled()  # estado antes de la prueba
    yield  # corre la prueba
    torch.use_deterministic_algorithms(previo)  # deja el estado como estaba


def _logistica(rng, n, pesos):
    """n filas float32 de N(0, 1) y etiquetas 0/1 de una logística con ``pesos`` e ``INTERCEPTO``."""
    X = rng.normal(size=(n, N_COLUMNAS)).astype(np.float32)  # variables ya "estandarizadas"
    p = expit(X.astype(np.float64) @ pesos + INTERCEPTO)  # P(malo) verdadera
    y = rng.binomial(1, p).astype(np.int64)  # etiquetas 0/1 de forma (n,)
    return X, y  # par (X, y)


@pytest.fixture(scope="module")
def datos_pd():
    """Train y val sintéticos de PD: 48 columnas float32, misma logística generadora."""
    rng = np.random.default_rng(SEMILLA)  # generador de los datos sintéticos
    pesos = rng.normal(scale=ESCALA_PESOS, size=N_COLUMNAS)  # coeficientes verdaderos
    X_tr, y_tr = _logistica(rng, N_TRAIN, pesos)  # entrenamiento
    X_val, y_val = _logistica(rng, N_VAL, pesos)  # validación aparte
    return X_tr, y_tr, X_val, y_val  # tupla de cuatro arreglos


def _cfg(**cambios):
    """Copia de CFG_PRUEBA con ``cambios`` aplicados al bloque mlp (no muta la constante)."""
    cfg = copy.deepcopy(CFG_PRUEBA)  # copia profunda de la constante
    cfg["mlp"].update(cambios)  # aplica los cambios pedidos
    return cfg  # config completa


def _ajustar(datos, cfg=None, seed=SEMILLA_MODELO, **kwargs):
    """Entrena un MLPPD con ``cfg`` (por defecto CFG_PRUEBA) y lo devuelve; ``kwargs`` van al constructor."""
    X_tr, y_tr, X_val, y_val = datos  # desempaca los datos
    modelo = MLPPD(**kwargs)  # modelo sin entrenar (fábricas opcionalmente inyectadas)
    return modelo.fit(X_tr, y_tr, X_val, y_val, cfg if cfg is not None else _cfg(), seed)  # fit devuelve el modelo


class _Captura:
    """Envoltorio de ``construir_red`` que guarda la red construida y sus argumentos (para inspeccionarlos)."""

    def __init__(self):
        self.red = None  # red construida por el modelo
        self.args = None  # argumentos posicionales recibidos
        self.kwargs = None  # argumentos con nombre recibidos

    def __call__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs  # recuerda cómo la llamaron
        self.red = construir_red(*args, **kwargs)  # delega en la fábrica real
        return self.red  # devuelve la misma red que usará el modelo


def _norma_pesos(red):
    """Norma euclídea de todos los pesos (parámetros con ndim > 1), sin los sesgos."""
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in red.parameters() if p.ndim > 1)))  # ‖W‖


def _regresion(signo_val):
    """Regresión lineal sintética: train y = X·w; val y = signo_val·X·w (signo −1 fuerza el sobreajuste)."""
    rng = np.random.default_rng(SEMILLA)  # generador de la regresión
    w = rng.normal(size=(D_REG, 1))  # coeficientes verdaderos
    X_tr = rng.normal(size=(N_REG, D_REG))  # entradas de train
    X_val = rng.normal(size=(N_REG, D_REG))  # entradas de val
    a_t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731 — convierte a tensor float32
    return a_t(X_tr), a_t(X_tr @ w), a_t(X_val), a_t(signo_val * (X_val @ w))  # tensores (n, d) y (n, 1)


def _entrenar_regresion(signo_val, max_epocas, paciencia):
    """Ajusta nn.Linear(D_REG, 1) con MSE y Adam sobre la regresión sintética; devuelve (red, datos, resultado)."""
    rng = fijar_semillas(SEMILLA_MODELO)  # fija torch/random y da el generador de minilotes
    red = nn.Linear(D_REG, 1)  # modelo lineal (sirve para probar que entrenar es genérico)
    perdida = nn.MSELoss()  # pérdida de regresión, no de PD
    optimizador = torch.optim.Adam(red.parameters(), lr=LR_REG)  # optimizador cualquiera
    datos = _regresion(signo_val)  # (X_tr, y_tr, X_val, y_val)
    resultado = entrenar(
        red, perdida, optimizador, *datos, batch=BATCH_REG, max_epocas=max_epocas, paciencia=paciencia, rng=rng
    )  # bucle genérico
    return red, perdida, datos, resultado  # todo lo necesario para revisar


# ============================================================================================
# Bloque 1 — torch_utils
# ============================================================================================
def test_resolver_device_cpu_e_invalido():
    """1a. 'cpu' → cpu; un nombre que no es auto|cpu|cuda → ValueError."""
    assert resolver_device("cpu") == torch.device("cpu")  # 'cpu' tal cual
    with pytest.raises(ValueError):  # nombre desconocido
        resolver_device("tpu")  # no está en auto|cpu|cuda


@pytest.mark.skipif(torch.cuda.is_available(), reason="con GPU 'auto' elige cuda")
def test_resolver_device_auto_sin_gpu():
    """1b. 'auto' sin GPU → cpu (si devolviera cuda a ciegas, fallaría al mover tensores)."""
    assert resolver_device("auto") == torch.device("cpu")  # cae a cpu


@pytest.mark.skipif(torch.cuda.is_available(), reason="solo tiene sentido sin GPU")
def test_resolver_device_cuda_sin_gpu():
    """1c (añadida). 'cuda' pedido explícitamente sin GPU → RuntimeError (no caer a cpu en silencio)."""
    with pytest.raises(RuntimeError):  # el usuario pidió GPU y no hay
        resolver_device("cuda")  # no debe degradarse a cpu


def test_a_tensor_y_a_numpy():
    """2. a_tensor da float32 con forma (n,) o (n, 1) si columna=True; a_numpy da float64 (n,)."""
    cpu = torch.device("cpu")  # dispositivo de prueba
    v = np.arange(5, dtype=np.float64)  # vector float64 (debe pasar a float32)
    t = a_tensor(v, cpu)  # sin columna
    assert t.dtype == torch.float32 and tuple(t.shape) == (5,)  # float32 y (n,)
    c = a_tensor(v, cpu, columna=True)  # como columna (objetivo de la BCE)
    assert c.dtype == torch.float32 and tuple(c.shape) == (5, 1)  # (n, 1)
    X = a_tensor(np.ones((4, 3), dtype=np.float32), cpu)  # matriz de diseño
    assert tuple(X.shape) == (4, 3)  # se conserva (n, d)
    r = a_numpy(c)  # de vuelta a numpy desde una columna
    assert r.dtype == np.float64 and r.shape == (5,)  # float64 y aplanado a (n,)
    np.testing.assert_array_equal(r, v)  # mismos valores


@pytest.mark.parametrize("malo", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_a_tensor_rechaza_no_finitos(malo):
    """2b. NaN o ±inf en la entrada → ValueError (no entrenar en silencio con basura)."""
    x = np.array([0.0, malo, 1.0])  # vector con un valor no finito
    with pytest.raises(ValueError):  # debe rechazarlo
        a_tensor(x, torch.device("cpu"))  # conversión


@pytest.mark.parametrize(
    "forma, columna", [((), False), ((2, 2, 2), False), ((3, 2), True)], ids=["escalar", "3d", "2d_como_columna"]
)
def test_a_tensor_rechaza_ndim_invalido(forma, columna):
    """2c (añadida). Escalar o 3-D → ValueError; con columna=True la entrada debe ser 1-D (una 2-D → ValueError)."""
    with pytest.raises(ValueError):  # ndim fuera de {1, 2}, o 2-D pedida como columna
        a_tensor(np.zeros(forma), torch.device("cpu"), columna=columna)  # conversión


# ============================================================================================
# Bloque 2 — red y config
# ============================================================================================
@pytest.mark.parametrize("capas, esperado", [([48, 24], N_PARAMS_48_24), ([64, 32], N_PARAMS_64_32)])
def test_contar_parametros(capas, esperado):
    """3. Con 48 entradas: [48, 24] → 3553 y [64, 32] → 5249 (cuentas a mano arriba)."""
    red = construir_red(N_COLUMNAS, capas, "relu", CFG_PRUEBA["mlp"]["dropout"])  # red de PD
    assert contar_parametros(red) == esperado  # pesos + sesgos


def test_contar_parametros_solo_entrenables():
    """3b (añadida). n_parametros cuenta solo los entrenables (núcleo §5.3): congelar la salida los descuenta."""
    red = construir_red(N_COLUMNAS, [48, 24], "relu", 0.0)  # red de PD
    for p in red[-1].parameters():  # parámetros de la capa de salida
        p.requires_grad_(False)  # congelados
    assert contar_parametros(red) == N_PARAMS_48_24 - N_PARAMS_SALIDA_24  # se descuentan


@pytest.mark.parametrize("activacion", sorted(ACTIVACIONES))
def test_estructura_de_la_red(activacion):
    """4. [Linear → activación → Dropout] por capa oculta + Linear(·, 1) final; sin Sigmoid (salida = logit)."""
    capas = CFG_PRUEBA["mlp"]["capas"]  # dos capas ocultas
    red = construir_red(N_COLUMNAS, capas, activacion, CFG_PRUEBA["mlp"]["dropout"])  # red de PD
    modulos = list(red)  # módulos en orden
    assert isinstance(red, nn.Sequential)  # contrato: nn.Sequential
    assert len(modulos) == 3 * len(capas) + 1  # tres módulos por capa oculta + salida
    entradas = N_COLUMNAS  # anchura que entra a cada capa
    for i, ancho in enumerate(capas):  # recorre las capas ocultas
        lineal, act, drop = modulos[3 * i : 3 * i + 3]  # trío de la capa i
        assert isinstance(lineal, nn.Linear)  # primero el Linear
        assert (lineal.in_features, lineal.out_features) == (entradas, ancho)  # con las anchuras de la config
        assert type(act) is ACTIVACIONES[activacion]  # luego la activación pedida
        assert isinstance(drop, nn.Dropout) and drop.p == CFG_PRUEBA["mlp"]["dropout"]  # luego el Dropout
        entradas = ancho  # la siguiente capa recibe este ancho
    assert isinstance(modulos[-1], nn.Linear) and modulos[-1].out_features == 1  # salida Linear(·, 1)
    assert modulos[-1].in_features == capas[-1]  # conectada a la última oculta
    assert not any(isinstance(m, nn.Sigmoid) for m in red.modules())  # nada de Sigmoid: devuelve el logit


def test_validar_cfg_mlp_acepta_la_valida():
    """5b. La config de prueba es válida y validar_cfg_mlp la devuelve como dict con las mismas claves."""
    validada = validar_cfg_mlp(copy.deepcopy(CFG_PRUEBA["mlp"]))  # valida una copia
    assert isinstance(validada, dict) and set(validada) == set(CFG_PRUEBA["mlp"])  # mismo bloque


CASOS_INVALIDOS = {
    "falta_clave": ("__quitar__", "paciencia"),  # se quita una clave obligatoria
    "sobra_clave": ("__sobra__", "perdida"),  # clave de LGD que no va en PD
    "capas_vacia": ("capas", []),  # sin capas ocultas
    "capas_cero": ("capas", [32, 0]),  # una capa sin neuronas
    "capas_negativa": ("capas", [-4]),  # anchura negativa
    "dropout_negativo": ("dropout", -0.1),  # fuera de [0, 1)
    "dropout_uno": ("dropout", 1.0),  # 1 apaga todo: fuera de [0, 1)
    "l2_negativo": ("l2", -1e-4),  # λ < 0
    "lr_cero": ("lr", 0.0),  # lr ≤ 0
    "lr_negativo": ("lr", -1e-3),  # lr ≤ 0
    "batch_cero": ("batch", 0),  # < 1
    "max_epocas_cero": ("max_epocas", 0),  # < 1
    "paciencia_cero": ("paciencia", 0),  # < 1
    "activacion_desconocida": ("activacion", "sigmoid"),  # no está en relu|leaky_relu|tanh
    "optimizador_desconocido": ("optimizador", "adamw"),  # AdamW no es L2: no se admite
    "momento_negativo": ("momento", -0.1),  # momento < 0
    "pos_weight_cero": ("pos_weight", 0.0),  # pos_weight ≤ 0
    "pos_weight_negativo": ("pos_weight", -1.0),  # pos_weight ≤ 0
    "capas_flotante": ("capas", [32.0, 16]),  # anchura no entera (aunque valga lo mismo)
    "capas_bool": ("capas", [True]),  # bool no cuenta como entero
    "device_desconocido": ("device", "tpu"),  # fuera de auto|cpu|cuda
    "lr_inf": ("lr", float("inf")),  # lr no finito
    "pos_weight_inf": ("pos_weight", float("inf")),  # pos_weight no finito
}


@pytest.mark.parametrize("clave, valor", list(CASOS_INVALIDOS.values()), ids=list(CASOS_INVALIDOS))
def test_validar_cfg_mlp_rechaza(clave, valor):
    """5. Cada caso inválido del bloque mlp → ValueError."""
    cfg = copy.deepcopy(CFG_PRUEBA["mlp"])  # parte de la válida
    if clave == "__quitar__":  # caso: falta una clave
        del cfg[valor]  # la quita
    elif clave == "__sobra__":  # caso: sobra una clave
        cfg[valor] = "mse"  # la añade
    else:  # caso: valor fuera de rango
        cfg[clave] = valor  # lo cambia
    with pytest.raises(ValueError):  # debe rechazarse
        validar_cfg_mlp(cfg)  # validación


def test_fit_valida_la_config(datos_pd):
    """5c (añadida). fit también valida: una config inválida falla en fit con ValueError (no entrena con ella)."""
    with pytest.raises(ValueError):  # dropout fuera de [0, 1)
        _ajustar(datos_pd, _cfg(dropout=1.0))  # fit con config inválida


# ============================================================================================
# Bloque 3 — optimizador
# ============================================================================================
def test_grupos_de_parametros():
    """6. Dos grupos: pesos (ndim > 1) con weight_decay = l2 y sesgos (ndim == 1) con 0; sin repetir ni omitir."""
    red = construir_red(N_COLUMNAS, CFG_PRUEBA["mlp"]["capas"], "relu", 0.0)  # red de PD
    grupos = grupos_de_parametros(red, L2_PRUEBA)  # grupos para el optimizador
    assert len(grupos) == 2  # pesos y sesgos
    por_decay = {g["weight_decay"]: list(g["params"]) for g in grupos}  # grupo según su decaimiento
    assert set(por_decay) == {L2_PRUEBA, 0}  # uno con λ y otro con 0
    assert all(p.ndim > 1 for p in por_decay[L2_PRUEBA])  # L2 solo en pesos
    assert all(p.ndim == 1 for p in por_decay[0])  # sesgos sin L2
    ids_grupos = [id(p) for g in grupos for p in g["params"]]  # identidad de cada parámetro agrupado
    assert len(ids_grupos) == len(set(ids_grupos))  # ninguno repetido
    assert set(ids_grupos) == {id(p) for p in red.parameters()}  # ninguno omitido


def test_construir_optimizador():
    """7. 'adam' → exactamente torch.optim.Adam (no AdamW) con lr; 'sgd' → SGD con lr y momentum = momento."""
    red = construir_red(N_COLUMNAS, CFG_PRUEBA["mlp"]["capas"], "relu", 0.0)  # red de PD
    adam = construir_optimizador(red, _cfg()["mlp"])  # optimizador por defecto
    assert type(adam) is torch.optim.Adam  # Adam exacto (AdamW es otra clase)
    assert all(g["lr"] == CFG_PRUEBA["mlp"]["lr"] for g in adam.param_groups)  # lr de la config
    sgd = construir_optimizador(red, _cfg(optimizador="sgd")["mlp"])  # SGD con momento
    assert type(sgd) is torch.optim.SGD  # SGD exacto
    assert all(g["momentum"] == CFG_PRUEBA["mlp"]["momento"] for g in sgd.param_groups)  # momento de la config
    assert all(g["lr"] == CFG_PRUEBA["mlp"]["lr"] for g in sgd.param_groups)  # lr de la config
    decays = sorted(g["weight_decay"] for g in sgd.param_groups)  # decaimiento por grupo
    assert decays == [0, CFG_PRUEBA["mlp"]["l2"]]  # usa los dos grupos (sesgos sin L2)


# ============================================================================================
# Bloque 4 — bucle
# ============================================================================================
@pytest.mark.parametrize("n, batch", [(10, 3), (10, 5), (4, 16)], ids=["resto", "exacto", "batch_mayor"])
def test_iterar_minilotes_cubre_todo(n, batch):
    """8a (añadida). Los minilotes recorren cada índice exactamente una vez (incluido el último lote incompleto)."""
    lotes = list(iterar_minilotes(n, batch, np.random.default_rng(SEMILLA)))  # todos los minilotes
    assert all(len(l) <= batch for l in lotes)  # ninguno excede el tamaño
    np.testing.assert_array_equal(np.sort(np.concatenate(lotes)), np.arange(n))  # cobertura exacta, sin repetir


def test_iterar_minilotes_depende_del_rng():
    """8b (añadida). El orden sale del rng recibido: misma semilla → mismo orden, y sí está barajado."""
    batch = CFG_PRUEBA["mlp"]["batch"]  # tamaño de minilote de la config de prueba
    a = np.concatenate(list(iterar_minilotes(N_TRAIN, batch, np.random.default_rng(SEMILLA))))  # primer recorrido
    b = np.concatenate(list(iterar_minilotes(N_TRAIN, batch, np.random.default_rng(SEMILLA))))  # misma semilla
    c = np.concatenate(list(iterar_minilotes(N_TRAIN, batch, np.random.default_rng(SEMILLA_OTRA))))  # otra semilla
    np.testing.assert_array_equal(a, b)  # misma semilla, mismo orden
    assert not np.array_equal(a, c)  # otra semilla, otro orden (no hay un rng fijo por dentro)
    assert not np.array_equal(a, np.arange(N_TRAIN))  # barajado (no es el orden original)


def test_parada_temprana_guionizada():
    """8. Pérdidas [1.0, 0.9, 0.9, 0.95, 0.96] con paciencia 3: se detiene en la época 5 y la mejor es la 2.

    La igualdad (0.9 en la época 3) no cuenta como mejora (min_delta = 0, mejora estricta).
    """
    perdidas = [1.0, 0.9, 0.9, 0.95, 0.96]  # pérdidas de validación guionizadas
    paciencia = 3  # épocas sin mejora toleradas
    parada = ParadaTemprana(paciencia)  # objeto bajo prueba
    red = nn.Linear(1, 1)  # cualquier red: solo se guarda su state_dict
    decisiones = [parada.actualizar(p, epoca, red) for epoca, p in enumerate(perdidas, start=1)]  # épocas desde 1
    assert decisiones == [False, False, False, False, True]  # True por primera vez en la época 5
    assert parada.mejor_epoca == 2  # la 3 empata y no mejora
    assert parada.mejor_perdida == pytest.approx(0.9, abs=TOL)  # la mejor pérdida


def test_parada_temprana_reinicia_la_cuenta():
    """8c (añadida). Una mejora reinicia la cuenta de paciencia: [1.0, 1.1, 1.2, 0.5, 0.6, 0.7, 0.8] para en la 7."""
    perdidas = [1.0, 1.1, 1.2, 0.5, 0.6, 0.7, 0.8]  # mejora en la 4 tras dos épocas malas
    parada = ParadaTemprana(3)  # paciencia 3
    red = nn.Linear(1, 1)  # cualquier red
    decisiones = [parada.actualizar(p, epoca, red) for epoca, p in enumerate(perdidas, start=1)]  # épocas desde 1
    assert decisiones == [False] * 6 + [True]  # sin reinicio pararía en la 4
    assert parada.mejor_epoca == 4  # la mejor es la 4


@pytest.mark.parametrize("conjunto", ["train", "val"])
def test_entrenar_rechaza_formas_distintas(conjunto):
    """9. Salida (n, 1) y objetivo (n,) en train o en val → ValueError (MSELoss difundiría a (n, n) en silencio)."""
    X_tr, y_tr, X_val, y_val = _regresion(1.0)  # datos con objetivo (n, 1)
    if conjunto == "train":  # se aplana solo un objetivo
        y_tr = y_tr.squeeze(1)  # train (n,)
    else:  # o el otro
        y_val = y_val.squeeze(1)  # val (n,)
    red = nn.Linear(D_REG, 1)  # salida (n, 1)
    optimizador = torch.optim.Adam(red.parameters(), lr=LR_REG)  # optimizador cualquiera
    with pytest.raises(ValueError):  # formas incompatibles
        entrenar(
            red, nn.MSELoss(), optimizador, X_tr, y_tr, X_val, y_val,
            batch=BATCH_REG, max_epocas=EPOCAS_CORTAS, paciencia=1, rng=np.random.default_rng(SEMILLA),
        )  # un objetivo aplanado a (n,)


@pytest.mark.parametrize("conjunto", ["train", "val"])
def test_entrenar_rechaza_datos_vacios(conjunto):
    """9b (añadida, R2). Un conjunto sin filas → ValueError que nombra el conjunto (no una media 0/0)."""
    X_tr, y_tr, X_val, y_val = _regresion(1.0)  # datos válidos
    if conjunto == "train":  # se vacía solo un conjunto
        X_tr, y_tr = X_tr[:0], y_tr[:0]  # train sin filas
    else:  # o el otro
        X_val, y_val = X_val[:0], y_val[:0]  # val sin filas
    red = nn.Linear(D_REG, 1)  # red cualquiera
    optimizador = torch.optim.SGD(red.parameters(), lr=LR_REG)  # optimizador cualquiera
    with pytest.raises(ValueError, match=conjunto):  # el mensaje dice qué conjunto
        entrenar(
            red, nn.MSELoss(), optimizador, X_tr, y_tr, X_val, y_val,
            batch=BATCH_REG, max_epocas=EPOCAS_CORTAS, paciencia=1, rng=np.random.default_rng(SEMILLA),
        )  # conjunto vacío


class _PerdidaConstante(nn.Module):
    """Pérdida guionizada: vale PERDIDA_CONSTANTE en todo minilote (conserva el grafo para backward)."""

    def forward(self, salida, objetivo):
        return salida.sum() * 0.0 + PERDIDA_CONSTANTE  # constante, pero derivable respecto de la salida


def test_epoca_entrenamiento_media_ponderada():
    """9c (añadida). Con pérdida constante c en minilotes 3 + 3 + 3 + 1, la época devuelve c (media ponderada).

    Una implementación que divide la suma por el tamaño del último lote (o la deja sin dividir) da 10·c.
    """
    fijar_semillas(SEMILLA_MODELO)  # pesos iniciales reproducibles
    red = nn.Linear(D_REG, 1)  # red cualquiera
    X = torch.zeros(N_EPOCA, D_REG)  # entradas cualesquiera
    y = torch.zeros(N_EPOCA, 1)  # objetivo cualquiera
    optimizador = torch.optim.SGD(red.parameters(), lr=LR_REG)  # optimizador cualquiera
    media = epoca_entrenamiento(red, _PerdidaConstante(), optimizador, X, y, BATCH_EPOCA, np.random.default_rng(SEMILLA))
    assert media == pytest.approx(PERDIDA_CONSTANTE, rel=TOL_F32)  # (3c + 3c + 3c + 1c) / 10 = c


class _PerdidaNoFinita(nn.Module):
    """MSE que devuelve ``valor`` (NaN o inf) en la fase pedida a partir de la época EPOCA_NO_FINITA.

    Distingue la fase por ``torch.is_grad_enabled()`` (evaluar_perdida corre en no_grad). Supone batch ≥ n, de modo que
    hay una llamada de train y una de val por época.
    """

    def __init__(self, valor, fase):
        super().__init__()  # inicializa nn.Module
        self.valor, self.fase = valor, fase  # qué devolver y en qué fase
        self.llamadas = {"train": 0, "val": 0}  # llamadas por fase (= épocas con batch ≥ n)

    def forward(self, salida, objetivo):
        fase = "train" if torch.is_grad_enabled() else "val"  # fase de esta llamada
        self.llamadas[fase] += 1  # cuenta la llamada
        base = ((salida - objetivo) ** 2).mean()  # MSE normal
        if fase == self.fase and self.llamadas[fase] >= EPOCA_NO_FINITA:  # época y fase guionizadas
            return base * self.valor  # NaN o inf (nan·x = nan; inf·x = inf si x > 0)
        return base  # pérdida finita


@pytest.mark.parametrize("valor", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("fase", ["train", "val"])
def test_entrenar_perdida_no_finita(valor, fase):
    """10. Pérdida NaN/inf en train o en val → FloatingPointError que nombra la época (aquí la 2)."""
    X_tr, y_tr, X_val, y_val = _regresion(1.0)  # datos válidos
    fijar_semillas(SEMILLA_MODELO)  # pesos iniciales reproducibles
    red = nn.Linear(D_REG, 1)  # red cualquiera
    optimizador = torch.optim.SGD(red.parameters(), lr=LR_REG)  # optimizador cualquiera
    with pytest.raises(FloatingPointError) as exc:  # debe abortar
        entrenar(
            red, _PerdidaNoFinita(valor, fase), optimizador, X_tr, y_tr, X_val, y_val,
            batch=N_REG, max_epocas=MAX_EPOCAS_SIN_PARADA, paciencia=PACIENCIA_SIN_PARADA,
            rng=np.random.default_rng(SEMILLA),
        )  # un minilote por época
    assert re.search(rf"\b{EPOCA_NO_FINITA}\b", str(exc.value))  # el mensaje dice la época
    assert fase in str(exc.value)  # y la fase (train o val) en la que ocurrió


def test_entrenar_es_generico_regresion():
    """11. entrenar ajusta nn.Linear(3, 1) con MSELoss (no sabe que es PD): la pérdida de val baja."""
    _, _, _, resultado = _entrenar_regresion(1.0, EPOCAS_REG, PACIENCIA_SIN_PARADA)  # val = misma relación
    assert isinstance(resultado, ResultadoEntrenamiento)  # tipo de retorno del contrato
    hist = resultado.historial  # una entrada por época
    assert [h["epoca"] for h in hist] == list(range(1, EPOCAS_REG + 1))  # épocas numeradas desde 1
    assert all(set(h) == {"epoca", "perdida_train", "perdida_val"} for h in hist)  # claves del historial
    assert hist[-1]["perdida_val"] < hist[0]["perdida_val"]  # aprendió
    assert hist[-1]["perdida_train"] < hist[0]["perdida_train"]  # también en train
    assert resultado.mejor_perdida_val == min(h["perdida_val"] for h in hist)  # la mejor es el mínimo


def test_entrenar_restaura_mejores_pesos_por_paciencia():
    """12. Escenario que sobreajusta (val con el signo opuesto): para por paciencia y deja los mejores pesos."""
    red, perdida, (_, _, X_val, y_val), r = _entrenar_regresion(-1.0, MAX_EPOCAS_SOBREAJUSTE, PACIENCIA_SOBREAJUSTE)
    assert r.mejor_epoca < r.epocas_corridas < MAX_EPOCAS_SOBREAJUSTE  # paró antes del tope
    assert r.epocas_corridas == r.mejor_epoca + PACIENCIA_SOBREAJUSTE  # justo al agotar la paciencia
    assert len(r.historial) == r.epocas_corridas  # una entrada por época corrida
    mejor = r.historial[r.mejor_epoca - 1]["perdida_val"]  # pérdida de val en la mejor época
    assert r.mejor_perdida_val == mejor  # el resultado la reporta
    assert r.historial[-1]["perdida_val"] > mejor  # los últimos pesos eran peores (la restauración importa)
    assert evaluar_perdida(red, perdida, X_val, y_val) == mejor  # la red quedó con los mejores pesos


def test_entrenar_restaura_mejores_pesos_por_max_epocas():
    """12b (añadida). Si termina por max_epocas sin agotar la paciencia, también restaura los mejores pesos."""
    red, perdida, (_, _, X_val, y_val), r = _entrenar_regresion(-1.0, MAX_EPOCAS_SIN_PARADA, PACIENCIA_SIN_PARADA)
    assert r.epocas_corridas == MAX_EPOCAS_SIN_PARADA  # corrió todas las épocas
    assert r.mejor_epoca < MAX_EPOCAS_SIN_PARADA  # la mejor no fue la última
    mejor = r.historial[r.mejor_epoca - 1]["perdida_val"]  # pérdida en la mejor época
    assert evaluar_perdida(red, perdida, X_val, y_val) == mejor  # quedó con los mejores pesos


def test_evaluar_perdida_sin_dropout():
    """12c (añadida). evaluar_perdida mide en eval(): con dropout 0.5 y la red en train(), dos medidas son iguales."""
    red = construir_red(D_REG, CAPAS_EVAL, "relu", DROPOUT_ALTO)  # red pequeña con mucho dropout
    X_tr, y_tr, _, _ = _regresion(1.0)  # datos (n, 3) y (n, 1)
    red.train()  # parte en modo entrenamiento (dropout activo)
    a = evaluar_perdida(red, nn.MSELoss(), X_tr, y_tr)  # primera medición
    red.train()  # vuelve a modo entrenamiento antes de la segunda
    b = evaluar_perdida(red, nn.MSELoss(), X_tr, y_tr)  # segunda medición
    assert a == b  # sin dropout al medir: idénticas


# ============================================================================================
# Bloque 5 — contrato ModeloPD (MLPPD)
# ============================================================================================
@pytest.fixture(scope="module")
def modelo_base(datos_pd):
    """MLPPD entrenado una vez con CFG_PRUEBA y SEMILLA_MODELO (compartido por las pruebas de solo lectura)."""
    return _ajustar(datos_pd)  # modelo entrenado


def test_predict_proba_es_sigmoide_del_logit(modelo_base, datos_pd):
    """13. predict_proba = sigmoid(predict_logit); forma (n,), float64, en [0, 1]."""
    X_val = datos_pd[2]  # validación
    logit = modelo_base.predict_logit(X_val)  # logit
    p = modelo_base.predict_proba(X_val)  # probabilidad de malo
    assert p.shape == (N_VAL,) and logit.shape == (N_VAL,)  # forma (n,)
    assert p.dtype == np.float64 and logit.dtype == np.float64  # float64
    assert np.all((p >= 0) & (p <= 1))  # es una probabilidad
    np.testing.assert_allclose(p, expit(logit), rtol=0, atol=TOL)  # p = σ(logit)
    assert np.unique(np.round(p, 6)).size > 1  # no es constante (la red aprendió algo)


def test_resultado_del_fit(modelo_base):
    """13b (añadida). El modelo expone el ResultadoEntrenamiento de su fit, con épocas dentro del tope."""
    r = modelo_base.resultado  # resultado del bucle
    assert isinstance(r, ResultadoEntrenamiento)  # tipo del contrato
    assert 1 <= r.mejor_epoca <= r.epocas_corridas <= CFG_PRUEBA["mlp"]["max_epocas"]  # coherencia de épocas
    assert len(r.historial) == r.epocas_corridas  # una entrada por época


def test_dropout_no_se_aplica_al_predecir(datos_pd):
    """14. Con dropout 0.5, dos llamadas a predict_proba sobre los mismos datos son idénticas."""
    captura = _Captura()  # para poner la red en train() desde fuera
    modelo = _ajustar(datos_pd, _cfg(dropout=DROPOUT_ALTO, max_epocas=EPOCAS_CORTAS), construir_red=captura)
    X_val = datos_pd[2]  # validación
    captura.red.train()  # si predict no llamara a eval(), el dropout quedaría activo
    primera = modelo.predict_proba(X_val)  # primera predicción
    captura.red.train()  # de nuevo en train() antes de la segunda
    np.testing.assert_array_equal(primera, modelo.predict_proba(X_val))  # idénticas


def test_determinismo_por_semilla(modelo_base, datos_pd):
    """15. Misma semilla en CPU → predicciones idénticas bit a bit; semilla distinta → distintas."""
    X_val = datos_pd[2]  # validación
    igual = _ajustar(datos_pd, seed=SEMILLA_MODELO)  # mismo fit que modelo_base
    otro = _ajustar(datos_pd, seed=SEMILLA_OTRA)  # otra semilla
    np.testing.assert_array_equal(igual.predict_proba(X_val), modelo_base.predict_proba(X_val))  # bit a bit
    assert not np.array_equal(otro.predict_proba(X_val), modelo_base.predict_proba(X_val))  # la semilla importa


def test_fit_en_cpu_activa_el_determinismo(datos_pd):
    """15c (añadida). fit con device cpu enciende torch.use_deterministic_algorithms (núcleo §7.1).

    La prueba 15 no lo atrapa: con estas operaciones la CPU ya da lo mismo sin el modo determinista.
    """
    torch.use_deterministic_algorithms(False)  # parte de apagado para que la prueba pueda fallar
    _ajustar(datos_pd, _cfg(max_epocas=EPOCAS_CORTAS))  # fit corto en CPU
    assert torch.are_deterministic_algorithms_enabled()  # fit lo encendió


def test_fit_no_muta_la_config(datos_pd):
    """15b (añadida). fit no modifica la config recibida (B4/B5 la reutilizan entre ensayos y semillas)."""
    cfg = _cfg(max_epocas=EPOCAS_CORTAS)  # config corta
    original = copy.deepcopy(cfg)  # copia para comparar
    _ajustar(datos_pd, cfg)  # entrena
    assert cfg == original  # intacta


def test_l2_encoge_los_pesos(datos_pd):
    """16. Misma semilla: con L2 fuerte la norma de los pesos (sin sesgos) es menor que con l2 = 0."""
    sin_l2, con_l2 = _Captura(), _Captura()  # capturan la red de cada fit
    _ajustar(datos_pd, _cfg(l2=0.0), construir_red=sin_l2)  # sin penalización
    _ajustar(datos_pd, _cfg(l2=L2_FUERTE), construir_red=con_l2)  # con penalización fuerte
    assert _norma_pesos(con_l2.red) < _norma_pesos(sin_l2.red)  # L2 encoge ‖W‖


@pytest.mark.parametrize("activacion", ["relu", "leaky_relu", "tanh"])
@pytest.mark.parametrize("optimizador", ["adam", "sgd"])
def test_combinaciones_optimizador_activacion(datos_pd, optimizador, activacion):
    """17. Las 2 × 3 combinaciones optimizador × activación entrenan desde la config, sin tocar código."""
    cfg = _cfg(optimizador=optimizador, activacion=activacion, max_epocas=EPOCAS_CORTAS)  # combinación
    captura = _Captura()  # para ver la activación usada
    modelo = _ajustar(datos_pd, cfg, construir_red=captura)  # entrena
    assert modelo.resultado.epocas_corridas == EPOCAS_CORTAS  # corrió las épocas pedidas
    assert any(type(m) is ACTIVACIONES[activacion] for m in captura.red.modules())  # usó la activación pedida
    assert np.all(np.isfinite(modelo.predict_logit(datos_pd[2])))  # predice valores finitos


def test_overrides_en_vivo(datos_pd):
    """18. --set mlp.capas=[64,32] mlp.dropout=0.3 mlp.lr=1e-2 mlp.max_epocas=5 funciona sin tocar código."""
    cfg = apply_overrides(CFG_PRUEBA, OVERRIDES_EN_VIVO)  # mismo cargador que el CLI
    captura = _Captura()  # para ver la red construida
    modelo = _ajustar(datos_pd, cfg, construir_red=captura)  # entrena con los overrides
    assert modelo.resultado.epocas_corridas <= MAX_EPOCAS_OVERRIDE  # respeta max_epocas
    assert modelo.n_parametros() == N_PARAMS_64_32  # respeta capas
    assert [m.out_features for m in captura.red if isinstance(m, nn.Linear)] == CAPAS_OVERRIDE + [1]  # anchuras
    assert all(m.p == DROPOUT_OVERRIDE for m in captura.red if isinstance(m, nn.Dropout))  # respeta dropout


def test_construir_perdida_pondera_la_clase_positiva():
    """18b (añadida). BCE con w₊ = 2 en z = 0: y = 1 → w₊·log 2 = 2·log 2; y = 0 → log 2 (celda de la pérdida)."""
    perdida = construir_perdida(POS_WEIGHT_PRUEBA, torch.device("cpu"))  # BCEWithLogitsLoss(pos_weight=2)
    z = torch.zeros(1, 1)  # logit 0 → σ(z) = 1/2
    malo = float(perdida(z, torch.ones(1, 1)))  # −w₊·log σ(0) = w₊·log 2
    bueno = float(perdida(z, torch.zeros(1, 1)))  # −log(1 − σ(0)) = log 2
    assert malo == pytest.approx(POS_WEIGHT_PRUEBA * np.log(2), rel=TOL_F32)  # pondera a los malos
    assert bueno == pytest.approx(np.log(2), rel=TOL_F32)  # no pondera a los buenos


CASOS_Y_INVALIDOS = {
    "con_un_2": lambda y: np.where(np.arange(y.size) == 0, 2, y),  # etiqueta fuera de {0, 1}
    "con_nan": lambda y: np.where(np.arange(y.size) == 0, np.nan, y.astype(float)),  # etiqueta faltante
    "longitud_mala": lambda y: y[:-1],  # una etiqueta menos que filas
}


@pytest.mark.parametrize("dañar", list(CASOS_Y_INVALIDOS.values()), ids=list(CASOS_Y_INVALIDOS))
@pytest.mark.parametrize("cual", ["y_tr", "y_val"])
def test_fit_rechaza_objetivo_invalido(datos_pd, cual, dañar):
    """18c (añadida). y_tr o y_val con un 2, un NaN o la longitud equivocada → ValueError en fit."""
    X_tr, y_tr, X_val, y_val = datos_pd  # datos válidos
    if cual == "y_tr":  # se daña solo un objetivo
        y_tr = dañar(y_tr)  # train dañado
    else:  # o el otro
        y_val = dañar(y_val)  # val dañado
    with pytest.raises(ValueError):  # objetivo inválido
        MLPPD().fit(X_tr, y_tr, X_val, y_val, _cfg(max_epocas=EPOCAS_CORTAS), SEMILLA_MODELO)  # fit


def test_fit_rechaza_x_val_con_otras_columnas(datos_pd):
    """18d (añadida). X_val con una columna menos que X_tr → ValueError (no un error críptico de matmul)."""
    X_tr, y_tr, X_val, y_val = datos_pd  # datos válidos
    with pytest.raises(ValueError):  # columnas incompatibles
        MLPPD().fit(X_tr, y_tr, X_val[:, :-1], y_val, _cfg(max_epocas=EPOCAS_CORTAS), SEMILLA_MODELO)  # fit


def test_refit_fallido_no_altera_el_modelo(datos_pd):
    """18e (añadida, B2). Un re-fit que falla por validación deja el modelo entrenado como estaba."""
    X_tr, y_tr, X_val, y_val = datos_pd  # datos válidos (48 columnas)
    modelo = _ajustar(datos_pd, _cfg(max_epocas=EPOCAS_CORTAS))  # modelo entrenado
    antes = modelo.predict_proba(X_val)  # predicción de referencia
    with pytest.raises(ValueError):  # X_tr con 5 columnas y X_val con 48: incompatibles
        modelo.fit(X_tr[:, :N_COLUMNAS_OTRAS], y_tr, X_val, y_val, _cfg(max_epocas=EPOCAS_CORTAS), SEMILLA_MODELO)
    np.testing.assert_array_equal(modelo.predict_proba(X_val), antes)  # sigue funcionando e igual


def test_fit_con_val_de_una_sola_clase(datos_pd):
    """18f (añadida, humo). y_val sin malos (una sola clase) es válido: fit entrena y predice valores finitos."""
    X_tr, y_tr, X_val, _ = datos_pd  # datos válidos
    y_val = np.zeros(X_val.shape[0], dtype=np.int64)  # todos buenos
    modelo = MLPPD().fit(X_tr, y_tr, X_val, y_val, _cfg(max_epocas=EPOCAS_CORTAS), SEMILLA_MODELO)  # entrena
    assert np.all(np.isfinite(modelo.predict_proba(X_val)))  # predice sin problemas


def test_guardar_y_cargar(modelo_base, datos_pd, tmp_path):
    """19. guardar → cargar da predicciones idénticas y el mismo n_parametros, sin pasar la config."""
    ruta = tmp_path / "pd_mlp.pt"  # archivo temporal
    modelo_base.guardar(ruta)  # serializa
    cargado = MLPPD.cargar(ruta)  # reconstruye solo desde el archivo
    X_val = datos_pd[2]  # validación
    assert isinstance(cargado, MLPPD)  # classmethod devuelve la clase
    np.testing.assert_array_equal(cargado.predict_proba(X_val), modelo_base.predict_proba(X_val))  # idénticas
    np.testing.assert_array_equal(cargado.predict_logit(X_val), modelo_base.predict_logit(X_val))  # idénticos
    assert cargado.n_parametros() == modelo_base.n_parametros()  # misma arquitectura


@pytest.mark.parametrize("metodo", ["predict_proba", "predict_logit"])
def test_predecir_antes_de_fit(datos_pd, metodo):
    """20a. predict_* antes de fit → RuntimeError."""
    with pytest.raises(RuntimeError):  # no hay red entrenada
        getattr(MLPPD(), metodo)(datos_pd[2])  # predicción sin fit


@pytest.mark.parametrize("metodo", ["predict_proba", "predict_logit"])
def test_predecir_con_otras_columnas(modelo_base, datos_pd, metodo):
    """20b. X con otro número de columnas que el de fit → ValueError (no un error críptico de matmul)."""
    X_mal = datos_pd[2][:, :-1]  # una columna menos
    with pytest.raises(ValueError):  # columnas incompatibles
        getattr(modelo_base, metodo)(X_mal)  # predicción
