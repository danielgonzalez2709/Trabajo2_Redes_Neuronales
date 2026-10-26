"""Pruebas de caja negra de la transformación probabilidad → puntaje de la scorecard (Parte C §3.1, tarea C1).

Solo se usa la interfaz pública de ``src.scorecard.scorecard`` y el cargador de configuración común
(``src.common.config``: ``load_config`` y ``apply_overrides``, núcleo §4.1).

Convenciones que se prueban (C §3.1):

- p = PD = P(malo); odds de **buen** pagador o = (1 − p)/p (bueno:malo).
- factor = pdo / ln 2 (logaritmo natural); offset = base − factor · ln(odds_base).
- score = offset + factor · ln(o) = offset − factor · logit(p): mayor puntaje = menor riesgo.
- Inversa: logit = (offset − score)/factor; p = σ(logit).
- Valores por defecto (``configs/scorecard/base.yaml``): base = 600, odds_base = 50, pdo = 20.

Qué se prueba: la aceptación de C1 (score(odds_base) = base; duplicar las odds suma PDO), casos hechos a mano cuya
cuenta va en un comentario, monotonía estricta, equivalencia PD ↔ logit, ida y vuelta, la config y los overrides
``--set scorecard.*``, forma y tipo de la salida, y los errores de validación (ValueError).

Los puntos calientes de C §6 que atrapan estas pruebas: odds malo:bueno en vez de bueno:malo (signo de la monotonía y
casos a mano), log10 en vez de ln (factor y PDO), y el signo del factor.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.common.config import apply_overrides, load_config
from src.scorecard.scorecard import (
    CLAVES_PARAMETROS,
    CLAVES_SCORECARD,
    bloque_scorecard,
    logit_desde_score,
    parametros_scorecard,
    pd_desde_score,
    score_desde_logit,
    score_desde_pd,
    validar_cfg_scorecard,
)

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RAIZ / "configs" / "scorecard" / "base.yaml"

# Tolerancias: TOL_PUNTOS para puntajes (factor · ln 2 no es exacto en coma flotante);
# TOL_IDA_VUELTA para las idas y vueltas en rango moderado.
TOL_PUNTOS = 1e-9
TOL_IDA_VUELTA = 1e-12

# Valores por defecto de C §3.1 (decisión de Juan José, 25-oct).
CFG_DEFECTO = {"base": 600, "odds_base": 50, "pdo": 20}
# Otro juego de parámetros para no depender de los valores por defecto.
CFG_OTRA = {"base": 500, "odds_base": 20, "pdo": 40}

ODDS_PRUEBA = [0.5, 1.0, 3.0, 20.0, 50.0, 137.0, 1000.0]


def pd_de_odds(o: float) -> float:
    """PD con odds de buen pagador o = (1 − p)/p ⇒ p = 1/(1 + o)."""
    return 1.0 / (1.0 + o)


def params_validos() -> dict:
    """Parámetros completos con los valores por defecto (factor y offset calculados aquí, a mano)."""
    factor = 20 / math.log(2)
    return {
        "base": 600.0,
        "odds_base": 50.0,
        "pdo": 20.0,
        "factor": factor,
        "offset": 600 - factor * math.log(50),
    }


# --------------------------------------------------------------------------------------------
# 1. Aceptación: score(odds_base) = base
# --------------------------------------------------------------------------------------------


def test_01_score_en_odds_base_es_base_por_defecto():
    """Con 600/50/20, p = 1/51 (odds 50:1) da 600 puntos."""
    params = parametros_scorecard(CFG_DEFECTO)
    # p = 1/51 ⇒ o = (50/51)/(1/51) = 50 = odds_base ⇒ score = base = 600.
    assert score_desde_pd(1 / 51, params) == pytest.approx(600.0, abs=TOL_PUNTOS)


@pytest.mark.parametrize("cfg", [CFG_DEFECTO, CFG_OTRA, {"base": 0, "odds_base": 1, "pdo": 10}])
def test_01b_score_en_odds_base_es_base_con_otros_parametros(cfg):
    """score(odds_base) = base para cualquier juego de parámetros."""
    params = parametros_scorecard(cfg)
    p = pd_de_odds(cfg["odds_base"])
    assert score_desde_pd(p, params) == pytest.approx(cfg["base"], abs=TOL_PUNTOS)


# --------------------------------------------------------------------------------------------
# 2. Aceptación: duplicar las odds suma PDO
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [CFG_DEFECTO, CFG_OTRA])
@pytest.mark.parametrize("o", ODDS_PRUEBA)
def test_02_duplicar_odds_suma_pdo(cfg, o):
    """score(2o) − score(o) = pdo y score(o/2) − score(o) = −pdo."""
    params = parametros_scorecard(cfg)
    s = score_desde_pd(pd_de_odds(o), params)
    s_doble = score_desde_pd(pd_de_odds(2 * o), params)
    s_mitad = score_desde_pd(pd_de_odds(o / 2), params)
    assert s_doble - s == pytest.approx(cfg["pdo"], abs=TOL_PUNTOS)
    assert s_mitad - s == pytest.approx(-cfg["pdo"], abs=TOL_PUNTOS)


# --------------------------------------------------------------------------------------------
# 3. Casos a mano (600/50/20)
# --------------------------------------------------------------------------------------------


def test_03_factor_y_offset_a_mano():
    """factor = 20/ln 2 y offset = 600 − factor · ln 50."""
    params = parametros_scorecard(CFG_DEFECTO)
    # factor = 20 / ln 2 = 20 / 0.6931471805599453 = 28.85390081777927
    assert params["factor"] == pytest.approx(28.85390081777927, abs=TOL_PUNTOS)
    # offset = 600 − factor · ln 50 = 600 − 20 · log2(50) = 600 − 20 · 5.643856189774724
    #        = 600 − 112.87712379549448 = 487.1228762045055
    assert params["offset"] == pytest.approx(487.1228762045055, abs=TOL_PUNTOS)
    assert params["base"] == 600.0
    assert params["odds_base"] == 50.0
    assert params["pdo"] == 20.0


@pytest.mark.parametrize(
    ("o", "esperado"),
    [
        # odds 100 = 2 · 50 ⇒ 600 + 20 · log2(2) = 600 + 20 = 620
        (100.0, 620.0),
        # odds 25 = 50 / 2 ⇒ 600 + 20 · log2(1/2) = 600 − 20 = 580
        (25.0, 580.0),
        # odds 50 = odds_base ⇒ 600
        (50.0, 600.0),
        # odds 200 = 4 · 50 ⇒ 600 + 20 · log2(4) = 600 + 40 = 640
        (200.0, 640.0),
        # odds 1 (p = 0.5) ⇒ 600 + 20 · log2(1/50) = 600 − 20 · 5.643856189774724 = 487.1228762045055 (= offset)
        (1.0, 487.1228762045055),
    ],
)
def test_03b_puntajes_a_mano(o, esperado):
    """Puntajes calculados a mano para varias odds de buen pagador."""
    params = parametros_scorecard(CFG_DEFECTO)
    assert score_desde_pd(pd_de_odds(o), params) == pytest.approx(esperado, abs=TOL_PUNTOS)


def test_03c_p_un_medio_es_offset():
    """p = 0.5 ⇒ logit = 0 ⇒ score = offset."""
    params = parametros_scorecard(CFG_DEFECTO)
    # logit(0.5) = ln(0.5/0.5) = 0 ⇒ score = offset − factor · 0 = offset = 487.1228762045055
    assert score_desde_pd(0.5, params) == pytest.approx(487.1228762045055, abs=TOL_PUNTOS)
    assert score_desde_logit(0.0, params) == pytest.approx(params["offset"], abs=TOL_PUNTOS)


def test_03d_inversa_a_mano():
    """600 puntos ⇒ logit = −ln 50 y p = 1/51; 620 puntos ⇒ p = 1/101."""
    params = parametros_scorecard(CFG_DEFECTO)
    # logit = (offset − 600)/factor = −factor · ln 50 / factor = −ln 50 = −3.912023005428146
    assert logit_desde_score(600.0, params) == pytest.approx(-3.912023005428146, abs=TOL_IDA_VUELTA)
    # p = σ(−ln 50) = 1/(1 + 50) = 1/51
    assert pd_desde_score(600.0, params) == pytest.approx(1 / 51, abs=TOL_IDA_VUELTA)
    # 620 ⇒ odds 100 ⇒ p = 1/101
    assert pd_desde_score(620.0, params) == pytest.approx(1 / 101, abs=TOL_IDA_VUELTA)


def test_03e_usa_factor_y_offset_recibidos():
    """score_desde_logit y logit_desde_score usan factor y offset tal como vienen (no los recalculan)."""
    params = {"base": 600.0, "odds_base": 50.0, "pdo": 20.0, "factor": 10.0, "offset": 100.0}
    # score = 100 − 10 · 2 = 80
    assert score_desde_logit(2.0, params) == pytest.approx(80.0, abs=TOL_PUNTOS)
    # logit = (100 − 80)/10 = 2
    assert logit_desde_score(80.0, params) == pytest.approx(2.0, abs=TOL_IDA_VUELTA)


def test_03f_funciones_de_pd_usan_factor_y_offset_recibidos():
    """score_desde_pd y pd_desde_score usan factor y offset tal como vienen (no los recalculan)."""
    params = {"base": 600.0, "odds_base": 50.0, "pdo": 20.0, "factor": 10.0, "offset": 100.0}
    # p = σ(2) = 1/(1 + e^−2) = 1/(1 + 0.1353352832366127) = 0.8807970779778823 ⇒ logit(p) = 2
    p = 0.8807970779778823
    # score = offset − factor · logit(p) = 100 − 10 · 2 = 80
    assert score_desde_pd(p, params) == pytest.approx(80.0, abs=TOL_PUNTOS)
    # logit = (100 − 80)/10 = 2 ⇒ p = σ(2) = 0.8807970779778823
    assert pd_desde_score(80.0, params) == pytest.approx(0.8807970779778823, abs=TOL_IDA_VUELTA)


# --------------------------------------------------------------------------------------------
# 4. Monotonía estricta: mayor riesgo ⇒ menor puntaje
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [CFG_DEFECTO, CFG_OTRA])
def test_04_score_decrece_estrictamente_con_p(cfg):
    params = parametros_scorecard(cfg)
    p = np.linspace(1e-6, 1 - 1e-6, 1001)
    s = score_desde_pd(p, params)
    assert np.all(np.diff(s) < 0)


@pytest.mark.parametrize("cfg", [CFG_DEFECTO, CFG_OTRA])
def test_04b_score_decrece_estrictamente_con_logit(cfg):
    params = parametros_scorecard(cfg)
    z = np.linspace(-30.0, 30.0, 1001)
    s = score_desde_logit(z, params)
    assert np.all(np.diff(s) < 0)


def test_04c_mas_riesgo_menos_puntos():
    """Un malo más probable (p = 0.3) puntúa menos que uno menos probable (p = 0.01)."""
    params = parametros_scorecard(CFG_DEFECTO)
    assert score_desde_pd(0.3, params) < score_desde_pd(0.01, params)


# --------------------------------------------------------------------------------------------
# 5. score_desde_pd ≡ score_desde_logit(logit(p))
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [CFG_DEFECTO, CFG_OTRA])
def test_05_pd_y_logit_coinciden(cfg):
    params = parametros_scorecard(cfg)
    p = np.array([1e-4, 0.01, 1 / 51, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-4])
    logit_ref = np.log(p / (1 - p))
    np.testing.assert_allclose(
        score_desde_pd(p, params), score_desde_logit(logit_ref, params), rtol=1e-12, atol=TOL_PUNTOS
    )


# --------------------------------------------------------------------------------------------
# 6. Ida y vuelta
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [CFG_DEFECTO, CFG_OTRA])
def test_06_ida_y_vuelta_pd(cfg):
    params = parametros_scorecard(cfg)
    p = np.linspace(1e-4, 1 - 1e-4, 501)
    np.testing.assert_allclose(
        pd_desde_score(score_desde_pd(p, params), params), p, rtol=TOL_IDA_VUELTA, atol=TOL_IDA_VUELTA
    )


@pytest.mark.parametrize("cfg", [CFG_DEFECTO, CFG_OTRA])
def test_06b_ida_y_vuelta_logit(cfg):
    params = parametros_scorecard(cfg)
    z = np.linspace(-9.0, 9.0, 501)
    np.testing.assert_allclose(
        logit_desde_score(score_desde_logit(z, params), params), z, rtol=TOL_IDA_VUELTA, atol=TOL_IDA_VUELTA
    )


def test_06c_pd_desde_score_satura_sin_error():
    """Puntajes extremos pueden saturar la PD a 0 o 1 sin error ni avisos (la sigmoide no debe desbordar)."""
    params = parametros_scorecard(CFG_DEFECTO)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # un RuntimeWarning de desbordamiento haría fallar la prueba
        p = pd_desde_score(np.array([-1e6, 1e6]), params)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert p[0] > p[1]


# --------------------------------------------------------------------------------------------
# 7. Config y overrides --set scorecard.*
# --------------------------------------------------------------------------------------------


def test_07_config_base():
    """configs/scorecard/base.yaml: el bloque scorecard es EXACTAMENTE {base: 600, odds_base: 50, pdo: 20} y
    bloque_scorecard lo valida.

    No se exige que ``scorecard`` sea la única clave de raíz: la tarea C3 añade a la misma config los bloques
    ``modelo``, ``rutas``, ``distribucion`` y ``salida`` (los comprueba ``tests/test_puntos.py``). Lo que C1 fija es
    el bloque ``scorecard``, y ese sigue siendo exacto.
    """
    cfg = load_config(RUTA_CONFIG)
    assert "scorecard" in cfg
    assert cfg["scorecard"] == {"base": 600, "odds_base": 50, "pdo": 20}
    assert set(cfg["scorecard"]) == {"base", "odds_base", "pdo"}
    assert bloque_scorecard(cfg) == {"base": 600.0, "odds_base": 50.0, "pdo": 20.0}


def test_07b_overrides_cambian_los_puntajes():
    """--set scorecard.pdo=40 --set scorecard.base=500 cambia factor, offset y puntajes como se predice."""
    cfg = apply_overrides(load_config(RUTA_CONFIG), ["scorecard.pdo=40", "scorecard.base=500"])
    params = parametros_scorecard(bloque_scorecard(cfg))
    # factor = 40 / ln 2 = 57.70780163555854
    assert params["factor"] == pytest.approx(40 / math.log(2), abs=TOL_PUNTOS)
    assert params["factor"] == pytest.approx(57.70780163555854, abs=TOL_PUNTOS)
    # offset = 500 − factor · ln 50 = 500 − 40 · log2(50) = 274.245752409011
    assert params["offset"] == pytest.approx(500 - (40 / math.log(2)) * math.log(50), abs=TOL_PUNTOS)
    assert params["offset"] == pytest.approx(274.245752409011, abs=TOL_PUNTOS)
    # odds_base sigue en 50 ⇒ p = 1/51 da base = 500
    assert score_desde_pd(1 / 51, params) == pytest.approx(500.0, abs=TOL_PUNTOS)
    # duplicar las odds (50 → 100) suma 40
    diferencia = score_desde_pd(1 / 101, params) - score_desde_pd(1 / 51, params)
    assert diferencia == pytest.approx(40.0, abs=TOL_PUNTOS)


def test_07c_clave_inexistente_falla():
    """Una errata en --set (scorecard.pod) es un error del cargador."""
    cfg = load_config(RUTA_CONFIG)
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["scorecard.pod=40"])


# --------------------------------------------------------------------------------------------
# 8. Forma y tipo de la salida
# --------------------------------------------------------------------------------------------

FUNCIONES = [score_desde_logit, score_desde_pd, logit_desde_score, pd_desde_score]
# Entrada válida de cada función (score_desde_pd necesita p en (0, 1); las demás, cualquier real finito).
ENTRADA_VALIDA = {
    score_desde_logit: [-2.0, 0.0, 1.5],
    score_desde_pd: [0.05, 0.3, 0.8],
    logit_desde_score: [450.0, 600.0, 700.0],
    pd_desde_score: [450.0, 600.0, 700.0],
}


@pytest.mark.parametrize("f", FUNCIONES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("convertir", [float, np.float64, np.float32], ids=["float", "np.float64", "np.float32"])
def test_08_escalar_devuelve_np_float64(f, convertir):
    r = f(convertir(ENTRADA_VALIDA[f][1]), params_validos())
    assert isinstance(r, np.float64)
    assert np.ndim(r) == 0


@pytest.mark.parametrize("f", FUNCIONES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_08b_array_1d_conserva_forma(f, dtype):
    x = np.asarray(ENTRADA_VALIDA[f], dtype=dtype)
    r = f(x, params_validos())
    assert isinstance(r, np.ndarray)
    assert r.shape == (3,)
    assert r.dtype == np.float64


@pytest.mark.parametrize("f", FUNCIONES, ids=lambda f: f.__name__)
def test_08c_array_2d_conserva_forma(f):
    x = np.tile(np.asarray(ENTRADA_VALIDA[f], dtype=np.float64), (2, 1))
    r = f(x, params_validos())
    assert isinstance(r, np.ndarray)
    assert r.shape == (2, 3)
    assert r.dtype == np.float64
    # cada fila coincide con la versión 1-D
    np.testing.assert_array_equal(r[0], f(np.asarray(ENTRADA_VALIDA[f]), params_validos()))


@pytest.mark.parametrize("f", FUNCIONES, ids=lambda f: f.__name__)
def test_08d_lista_devuelve_array(f):
    r = f(list(ENTRADA_VALIDA[f]), params_validos())
    assert isinstance(r, np.ndarray)
    assert r.shape == (3,)
    assert r.dtype == np.float64


@pytest.mark.parametrize("vacio", [[], np.array([])], ids=["lista", "np.array"])
@pytest.mark.parametrize("f", FUNCIONES, ids=lambda f: f.__name__)
def test_08e_entrada_vacia_devuelve_array_vacio(f, vacio):
    """Una entrada vacía da un array float64 vacío de forma (0,), sin error."""
    r = f(vacio, params_validos())
    assert isinstance(r, np.ndarray)
    assert r.shape == (0,)
    assert r.dtype == np.float64


# --------------------------------------------------------------------------------------------
# 9. Errores
# --------------------------------------------------------------------------------------------

VALORES_NO_REALES = [
    pytest.param(True, id="bool"),
    pytest.param(np.bool_(True), id="np.bool_"),
    pytest.param("600", id="texto"),
    pytest.param(None, id="None"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="-inf"),
]


@pytest.mark.parametrize("cfg_sc", [None, [600, 50, 20], "scorecard", 600], ids=["None", "lista", "texto", "int"])
def test_09_validar_rechaza_no_dict(cfg_sc):
    with pytest.raises(ValueError):
        validar_cfg_scorecard(cfg_sc)


@pytest.mark.parametrize("clave", CLAVES_SCORECARD)
def test_09b_validar_rechaza_clave_faltante(clave):
    cfg_sc = {k: v for k, v in CFG_DEFECTO.items() if k != clave}
    with pytest.raises(ValueError):
        validar_cfg_scorecard(cfg_sc)


def test_09c_validar_rechaza_clave_sobrante():
    with pytest.raises(ValueError):
        validar_cfg_scorecard({**CFG_DEFECTO, "pod": 20})


@pytest.mark.parametrize("valor", VALORES_NO_REALES)
@pytest.mark.parametrize("clave", CLAVES_SCORECARD)
def test_09d_validar_rechaza_valor_no_real_finito(clave, valor):
    with pytest.raises(ValueError):
        validar_cfg_scorecard({**CFG_DEFECTO, clave: valor})


@pytest.mark.parametrize("clave", ["pdo", "odds_base"])
@pytest.mark.parametrize("valor", [0, 0.0, -1, -20.5])
def test_09e_validar_rechaza_pdo_u_odds_no_positivos(clave, valor):
    with pytest.raises(ValueError):
        validar_cfg_scorecard({**CFG_DEFECTO, clave: valor})


@pytest.mark.parametrize("base", [0, -100, 1e-3, 600.5])
def test_09f_validar_acepta_cualquier_base_finita(base):
    r = validar_cfg_scorecard({**CFG_DEFECTO, "base": base})
    assert r["base"] == float(base)


def test_09g_validar_devuelve_copia_float_sin_mutar():
    entrada = dict(CFG_DEFECTO)
    copia = dict(entrada)
    r = validar_cfg_scorecard(entrada)
    assert entrada == copia
    assert all(type(entrada[k]) is int for k in CLAVES_SCORECARD)
    assert r is not entrada
    assert set(r) == set(CLAVES_SCORECARD)
    assert all(type(r[k]) is float for k in CLAVES_SCORECARD)
    assert r == {"base": 600.0, "odds_base": 50.0, "pdo": 20.0}


@pytest.mark.parametrize(
    "cfg", [None, [], "scorecard", {}, {"pd": CFG_DEFECTO}], ids=["None", "lista", "texto", "vacio", "sin_bloque"]
)
def test_09h_bloque_rechaza_cfg_sin_bloque(cfg):
    with pytest.raises(ValueError):
        bloque_scorecard(cfg)


@pytest.mark.parametrize("bloque", [None, [600, 50, 20], {"base": 600, "pdo": 20}, {**CFG_DEFECTO, "pdo": 0}])
def test_09i_bloque_valida_el_contenido(bloque):
    with pytest.raises(ValueError):
        bloque_scorecard({"scorecard": bloque})


def test_09j_bloque_devuelve_la_copia_validada():
    assert bloque_scorecard({"scorecard": dict(CFG_OTRA)}) == {"base": 500.0, "odds_base": 20.0, "pdo": 40.0}


def _params_con(**cambios) -> dict:
    return {**params_validos(), **cambios}


PARAMS_INVALIDOS = [
    pytest.param(dict(CFG_DEFECTO), id="bloque_de_3_claves"),
    pytest.param({k: v for k, v in params_validos().items() if k != "factor"}, id="sin_factor"),
    pytest.param({k: v for k, v in params_validos().items() if k != "offset"}, id="sin_offset"),
    pytest.param({k: v for k, v in params_validos().items() if k != "base"}, id="sin_base"),
    pytest.param(_params_con(extra=1.0), id="clave_extra"),
    pytest.param(_params_con(factor=0.0), id="factor_0"),
    pytest.param(_params_con(factor=-28.85390081777927), id="factor_negativo"),
    pytest.param(_params_con(factor=float("nan")), id="factor_nan"),
    pytest.param(_params_con(factor=float("inf")), id="factor_inf"),
    pytest.param(_params_con(offset=float("inf")), id="offset_inf"),
    pytest.param(_params_con(offset=float("-inf")), id="offset_-inf"),
    pytest.param(_params_con(offset=float("nan")), id="offset_nan"),
    pytest.param(_params_con(factor="28.85"), id="factor_texto"),
    pytest.param(_params_con(offset=None), id="offset_None"),
    pytest.param(None, id="None"),
]


@pytest.mark.parametrize("params", PARAMS_INVALIDOS)
@pytest.mark.parametrize("f", FUNCIONES, ids=lambda f: f.__name__)
def test_09k_params_invalidos(f, params):
    with pytest.raises(ValueError):
        f(ENTRADA_VALIDA[f][1], params)


@pytest.mark.parametrize(
    "valor",
    [True, False, np.bool_(True), np.bool_(False)],
    ids=["True", "False", "np.bool_True", "np.bool_False"],
)
@pytest.mark.parametrize("clave", ["factor", "offset"])
@pytest.mark.parametrize("f", FUNCIONES, ids=lambda f: f.__name__)
def test_09l_params_rechaza_bool_en_factor_u_offset(f, clave, valor):
    """Un bool no es un real de los parámetros (float(True) = 1.0 lo colaría como factor u offset)."""
    with pytest.raises(ValueError):
        f(ENTRADA_VALIDA[f][1], _params_con(**{clave: valor}))


P_FUERA = [
    pytest.param(0.0, id="0"),
    pytest.param(1.0, id="1"),
    pytest.param(-0.1, id="-0.1"),
    pytest.param(1.1, id="1.1"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="-inf"),
]


@pytest.mark.parametrize("p", P_FUERA)
def test_09m_score_desde_pd_rechaza_p_fuera_de_0_1_escalar(p):
    with pytest.raises(ValueError):
        score_desde_pd(p, params_validos())


@pytest.mark.parametrize("p", P_FUERA)
def test_09n_score_desde_pd_rechaza_p_fuera_de_0_1_en_array(p):
    with pytest.raises(ValueError):
        score_desde_pd(np.array([0.2, p, 0.5]), params_validos())


NO_FINITOS = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="-inf"),
]


@pytest.mark.parametrize("valor", NO_FINITOS)
@pytest.mark.parametrize("f", [score_desde_logit, logit_desde_score, pd_desde_score], ids=lambda f: f.__name__)
def test_09o_logit_o_score_no_finito(f, valor):
    with pytest.raises(ValueError):
        f(valor, params_validos())
    with pytest.raises(ValueError):
        f(np.array([ENTRADA_VALIDA[f][0], valor]), params_validos())


ENTRADAS_NO_NUMERICAS = [
    pytest.param("0.3", id="texto_numerico"),
    pytest.param(["0.3"], id="lista_de_texto"),
    pytest.param(True, id="bool"),
    pytest.param(np.array([True]), id="array_bool"),
    pytest.param("abc", id="texto"),
    pytest.param(None, id="None"),
    pytest.param(1 + 2j, id="complejo"),
    pytest.param([[0.1], [0.2, 0.3]], id="lista_irregular"),
]


@pytest.mark.parametrize("x", ENTRADAS_NO_NUMERICAS)
@pytest.mark.parametrize("f", FUNCIONES, ids=lambda f: f.__name__)
def test_09p_entrada_no_numerica(f, x):
    """Texto (aunque sea numérico), bool, None y complejos no son entradas: np.asarray los convertiría en silencio."""
    with pytest.raises(ValueError):
        f(x, params_validos())


@pytest.mark.parametrize(
    ("f", "x", "cambios"),
    [
        # −factor · 1e308 = −28.85 · 1e308 desborda a −inf en float64 (máximo ≈ 1.8e308)
        pytest.param(score_desde_logit, 1e308, {}, id="score_desde_logit"),
        # logit(0.01) = ln(0.01/0.99) ≈ −4.6 ⇒ −1e308 · (−4.6) desborda a +inf
        pytest.param(score_desde_pd, 0.01, {"factor": 1e308}, id="score_desde_pd"),
        # (offset − 1e308)/1e-300 ≈ −1e608 desborda a −inf
        pytest.param(logit_desde_score, 1e308, {"factor": 1e-300}, id="logit_desde_score"),
        # pdo = 0.1 ⇒ factor = 0.1/ln 2 = 0.14426950408889636 < 1; logit = (offset + 1e308)/0.1443 ≈ 6.9e308 > 1.8e308
        # desborda a +inf antes de la sigmoide ⇒ ValueError (pd_desde_score solo satura si el logit es finito)
        pytest.param(pd_desde_score, -1e308, {"factor": 0.1 / math.log(2)}, id="pd_desde_score"),
    ],
)
def test_09q_resultado_no_representable(f, x, cambios):
    """Entrada finita con salida infinita → ValueError, sin RuntimeWarning de desbordamiento."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # un RuntimeWarning de overflow haría fallar la prueba con otro tipo de error
        with pytest.raises(ValueError):
            f(x, _params_con(**cambios))


CLAVES_MEZCLADAS = {1: 2, "x": 3}  # claves de tipos distintos: sorted() sin key lanzaría TypeError


def test_09r_claves_de_tipos_mezclados():
    """Claves sobrantes o faltantes de tipos mezclados dan ValueError (no un TypeError al armar el mensaje)."""
    with pytest.raises(ValueError):
        validar_cfg_scorecard({**CFG_DEFECTO, **CLAVES_MEZCLADAS})
    with pytest.raises(ValueError):
        validar_cfg_scorecard(dict(CLAVES_MEZCLADAS))  # faltan las tres y sobran las mezcladas
    with pytest.raises(ValueError):
        bloque_scorecard({"scorecard": {**CFG_DEFECTO, **CLAVES_MEZCLADAS}})
    with pytest.raises(ValueError):
        bloque_scorecard({1: CFG_DEFECTO, "x": 3})  # cfg con claves no texto y sin el bloque
    with pytest.raises(ValueError):
        parametros_scorecard({**CFG_DEFECTO, **CLAVES_MEZCLADAS})
    for f in FUNCIONES:  # los parámetros de las cuatro transformaciones
        with pytest.raises(ValueError):
            f(ENTRADA_VALIDA[f][1], {**params_validos(), **CLAVES_MEZCLADAS})
        with pytest.raises(ValueError):
            f(ENTRADA_VALIDA[f][1], dict(CLAVES_MEZCLADAS))


# --------------------------------------------------------------------------------------------
# 10. parametros_scorecard
# --------------------------------------------------------------------------------------------


def test_10_claves_publicas():
    assert CLAVES_SCORECARD == ("base", "odds_base", "pdo")
    assert CLAVES_PARAMETROS == ("base", "odds_base", "pdo", "factor", "offset")


@pytest.mark.parametrize("cfg", [CFG_DEFECTO, CFG_OTRA])
def test_10b_parametros_claves_exactas_float_sin_mutar(cfg):
    entrada = dict(cfg)
    copia = dict(entrada)
    params = parametros_scorecard(entrada)
    assert entrada == copia
    assert set(params) == set(CLAVES_PARAMETROS)
    assert len(params) == len(CLAVES_PARAMETROS)
    assert all(type(params[k]) is float for k in CLAVES_PARAMETROS)
    factor = cfg["pdo"] / math.log(2)
    assert params["factor"] == pytest.approx(factor, abs=TOL_PUNTOS)
    assert params["offset"] == pytest.approx(cfg["base"] - factor * math.log(cfg["odds_base"]), abs=TOL_PUNTOS)


@pytest.mark.parametrize("cfg_sc", [None, {"base": 600, "pdo": 20}, {**CFG_DEFECTO, "pdo": -20}])
def test_10c_parametros_valida_el_bloque(cfg_sc):
    with pytest.raises(ValueError):
        parametros_scorecard(cfg_sc)
