"""Pruebas de la scorecard de la logística: puntos por variable (Parte C §3.2, tarea C3).

Aceptación de C3 (C §4): **la suma de puntos = score**. Con la logística, logit(p) = β₀ + Σ βⱼ xⱼ (xⱼ = columnas de
``Preprocesador.transform``) y, como score = offset − factor · logit(p) (C §3.1, tarea C1):

- puntos base = offset − factor · β₀;
- puntosⱼ = −factor · βⱼ · xⱼ (β > 0 ⇒ MENOS puntos: más riesgo, menor puntaje);
- score = puntos base + Σⱼ puntosⱼ.

Solo se usa la interfaz pública de ``src.scorecard.puntos`` (más ``src.scorecard.scorecard`` de C1, el cargador de
configuración, el preprocesador común, la logística de B3 y las huellas de ``src/pd/experimento.py``). Toda escritura
va a ``tmp_path``; nunca a ``results/``.

Bloques:
1. modelo de juguete con cuentas a mano (cada número esperado lleva su cuenta en un comentario, calculada con Python);
2–5. modelo real: aditividad, signo, cero en la media y fracciones de train;
6. ``--set scorecard.pdo=40`` duplica todos los puntos;
7. bandas e histograma (incluye casos a mano);
8. formato de ``scorecard.json``;
9. fuga: lo que se estima con train no depende de val ni de test;
10. errores (ValueError);
11. ``manejador`` escribe ``scorecard.json``.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.config import apply_overrides, load_config
from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json, to_jsonable
from src.common.seeds import make_rng
from src.pd.experimento import huella_archivo, huella_particion
from src.pd.logistica import LogisticaPD
from src.scorecard import puntos
from src.scorecard.puntos import (
    ARCHIVO_SCORECARD,
    CLAVES_DISTRIBUCION,
    CLAVES_RAIZ,
    CLAVES_RUTAS,
    MODELOS,
    SPLITS,
    TOL_ADITIVIDAD,
    construir_scorecard,
    cortes_bandas,
    distribucion_split,
    entrada_categorica,
    entrada_numerica,
    error_aditividad,
    grupos_columnas,
    histograma,
    manejador,
    puntos_base,
    puntos_columnas,
    puntos_por_variable,
    puntos_valor_crudo,
    tabla_bandas,
    preparar_splits,
    validar_cfg_puntos,
)
from src.scorecard.scorecard import (
    CLAVES_PARAMETROS,
    bloque_scorecard,
    parametros_scorecard,
    score_desde_logit,
    score_desde_pd,
)

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RAIZ / "configs" / "scorecard" / "base.yaml"

TOL = 1e-9  # cuentas a mano en float64 (factor = 20/ln 2 no es exacto en binario)
TOL_FRACCION = 1e-12  # fracciones de train: medias de columnas 0/1
TOL_REL_PDO = 1e-12  # relación ×2 al pasar de pdo 20 a pdo 40
TOL_BANDA = 0.01  # ±1 punto porcentual en la proporción de train por banda
SEMILLA_PRUEBA = 20261025  # semilla propia de la prueba de fuga (no la de la config)

# Config que debe tener configs/scorecard/base.yaml tras C3 (valores fijados en el plan de la tarea por Juan José).
CFG_ESPERADA = {
    "scorecard": {"base": 600, "odds_base": 50, "pdo": 20},
    "modelo": "logistica",
    "rutas": {
        "modelo": "models/pd_logistica.joblib",
        "preprocesador": "models/preprocesador_pd.json",
        "datos": "data/processed/loans_pd.parquet",
        "features": "configs/features.yaml",
    },
    "distribucion": {
        "percentiles_tabla": [1, 5, 25, 50, 75, 95, 99],
        "percentiles_rango": [1, 99],
        "fraccion_min_categoria": 0.01,
        "ancho_bin": 10,
        "n_bandas": 10,
    },
    "salida": "results/data/scorecard",
}

CLAVES_SCORECARD_JSON = {"base", "odds_base", "pdo", "factor", "offset", "modelo", "puntos_base", "aditividad",
                         "puntos_por_variable", "distribucion_scores", "evidencia"}
CLAVES_NUM = {"nombre", "tipo", "etiqueta_es", "beta", "puntos_por_unidad_estandarizada", "transformacion", "tabla",
              "rango_puntos"}
CLAVES_CAT = {"nombre", "tipo", "etiqueta_es", "categorias", "otra", "rango_puntos"}
CLAVES_TRANSFORMACION = {"mediana", "recorte", "log1p", "media", "desv"}
CLAVES_FILA_TABLA = {"percentil", "valor", "puntos"}
CLAVES_CATEGORIA = {"categoria", "etiqueta", "beta", "puntos", "fraccion_train"}
CLAVES_OTRA = {"categoria", "interna", "beta", "puntos"}
CLAVES_DIST_SCORES = {"cortes_bandas", "n_bandas", "ancho_bin", "train", "val", "test"}
CLAVES_SPLIT = {"n", "media", "desv", "percentiles", "histograma", "bandas"}
CLAVES_HISTOGRAMA = {"ancho", "bordes", "conteos"}
CLAVES_BANDA = {"banda", "desde", "hasta", "n", "puntaje_medio", "pd_media", "tasa_malos"}


# ------------------------------------------------------------------------------------------------------------------
# Modelo de juguete (bloque 1). Todas las cuentas se hicieron con Python (float64).
# ------------------------------------------------------------------------------------------------------------------

# params de C1 con 600 / 50 / 20:
# factor = 20 / ln 2 = 28.85390081777927
# offset = 600 − factor · ln 50 = 487.1228762045055
FACTOR_20 = 28.85390081777927
OFFSET_20 = 487.1228762045055

# Preprocesador de juguete con el formato de Preprocesador.to_dict(). Orden de variables: a (num, log1p y recorte),
# c (cat de 3 categorías + OTRA), b (num sin log1p). Así c queda entre dos numéricas y los índices no son triviales.
PRE_JUGUETE = {
    "version": 1,
    "percentiles_recorte": [1.0, 99.0],
    "categoricas": {"limpieza": "strip", "vacio": "DESCONOCIDO", "no_vista": "OTRA"},
    "orden_numericas": ["imputar_mediana", "recortar", "log1p_si_aplica", "estandarizar"],
    "columnas_salida": ["a", "c=x", "c=y", "c=z", "c=OTRA", "b"],
    "variables": [
        {"nombre": "a", "tipo": "num", "mediana": 9.0, "recorte": [1.0, 99.0], "log1p": True, "media": 2.0,
         "desv": 0.5},
        {"nombre": "c", "tipo": "cat", "categorias": ["x", "y", "z", "OTRA"]},
        {"nombre": "b", "tipo": "num", "mediana": 4.0, "recorte": [0.0, 10.0], "log1p": False, "media": 5.0,
         "desv": 2.0},
    ],
}
F_A = {"nombre": "a", "tipo": "num", "etiqueta_es": "Variable A"}
F_B = {"nombre": "b", "tipo": "num", "etiqueta_es": "Variable B"}
# categorias_es en un orden distinto del del preprocesador (z, x, y): la tabla debe seguir este orden.
F_C = {"nombre": "c", "tipo": "cat", "etiqueta_es": "Variable C", "categorias_es": {"z": "Zeta", "x": "Equis",
                                                                                   "y": "Ye"}}
B0_JUGUETE = -1.5
COEF_JUGUETE = np.array([0.4, 0.3, -0.2, 0.9, 0.5, -0.25])  # a, c=x, c=y, c=z, c=OTRA, b
# Fila 0: a = 1, c = x, b = 0.5; fila 1: a = −2, c = OTRA, b = −1 (valores ya transformados).
X_JUGUETE = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.5],
                      [-2.0, 0.0, 0.0, 0.0, 1.0, -1.0]])
# percentiles_rango [50, 75] es DISTINTO de los extremos de la tabla [25, 75]: así una implementación que tomara el
# rango de la tabla (pct[0], pct[-1]) falla (mutante M16, candidato H-20; registrado primero como H-13).
CFG_DIST_JUGUETE = {"percentiles_tabla": [25, 50, 75], "percentiles_rango": [50, 75], "fraccion_min_categoria": 0.1,
                    "ancho_bin": 10, "n_bandas": 3}


def params_20() -> dict:
    return parametros_scorecard({"base": 600, "odds_base": 50, "pdo": 20})


def v_pre(nombre: str, pre_json: dict = PRE_JUGUETE) -> dict:
    return next(v for v in pre_json["variables"] if v["nombre"] == nombre)


def test_00_constantes_publicas():
    assert CLAVES_RAIZ == ("scorecard", "modelo", "rutas", "distribucion", "salida")
    assert CLAVES_RUTAS == ("modelo", "preprocesador", "datos", "features")
    assert CLAVES_DISTRIBUCION == ("percentiles_tabla", "percentiles_rango", "fraccion_min_categoria", "ancho_bin",
                                   "n_bandas")
    assert MODELOS == ("logistica",)
    assert ARCHIVO_SCORECARD == "scorecard.json"
    assert SPLITS == ("train", "val", "test")
    assert TOL_ADITIVIDAD == 1e-9


def test_00b_config_base_de_c3():
    """configs/scorecard/base.yaml queda exactamente con los bloques de C3 y la valida validar_cfg_puntos."""
    cfg = load_config(RUTA_CONFIG)
    assert cfg == CFG_ESPERADA
    validar_cfg_puntos(cfg)


def test_01a_params_de_juguete_coinciden_con_c1():
    p = params_20()
    assert p["factor"] == pytest.approx(FACTOR_20, abs=TOL)
    assert p["offset"] == pytest.approx(OFFSET_20, abs=TOL)


def test_01b_puntos_base_a_mano():
    # offset − factor · β₀ = 487.1228762045055 + 1.5 · 28.85390081777927 = 530.4037274311744
    assert puntos_base(B0_JUGUETE, params_20()) == pytest.approx(530.4037274311744, abs=TOL)


def test_01c_puntos_columnas_a_mano():
    P = puntos_columnas(X_JUGUETE, COEF_JUGUETE, params_20())
    assert isinstance(P, np.ndarray) and P.dtype == np.float64 and P.shape == (2, 6)
    # −factor · β · x por celda:
    # fila 0: a −28.8539·0.4·1 = −11.541560327111709; c=x −28.8539·0.3·1 = −8.656170245333781; c=y, c=z, c=OTRA 0;
    #         b −28.8539·(−0.25)·0.5 = 3.6067376022224087
    # fila 1: a −28.8539·0.4·(−2) = 23.083120654223418; c=OTRA −28.8539·0.5·1 = −14.426950408889635;
    #         b −28.8539·(−0.25)·(−1) = −7.213475204444817
    esperado = np.array([[-11.541560327111709, -8.656170245333781, 0.0, 0.0, 0.0, 3.6067376022224087],
                         [23.083120654223418, 0.0, 0.0, 0.0, -14.426950408889635, -7.213475204444817]])
    np.testing.assert_allclose(P, esperado, rtol=0, atol=TOL)


def test_01d_suma_de_puntos_es_el_score_a_mano():
    p = params_20()
    P = puntos_columnas(X_JUGUETE, COEF_JUGUETE, p)
    score = score_desde_logit(B0_JUGUETE + X_JUGUETE @ COEF_JUGUETE, p)
    # score = offset − factor·(β₀ + x·β): fila 0 → 513.8127344609513; fila 1 → 531.8464224720634
    np.testing.assert_allclose(score, [513.8127344609513, 531.8464224720634], rtol=0, atol=TOL)
    np.testing.assert_allclose(puntos_base(B0_JUGUETE, p) + P.sum(axis=1), score, rtol=0, atol=TOL)
    assert error_aditividad(puntos_base(B0_JUGUETE, p), P, score) < TOL


def test_01e_error_aditividad_a_mano():
    # pb = 1; P = [[1, 2], [0, 0]] ⇒ pb + Σ = [4, 1]; score = [4.5, 0.75] ⇒ |dif| = [0.5, 0.25] ⇒ máx 0.5
    assert error_aditividad(1.0, np.array([[1.0, 2.0], [0.0, 0.0]]), np.array([4.5, 0.75])) == pytest.approx(0.5)


def test_01f_grupos_columnas_a_mano():
    grupos = grupos_columnas(PRE_JUGUETE)
    assert [g[0] for g in grupos] == ["a", "c", "b"]
    assert [list(map(int, g[1])) for g in grupos] == [[0], [1, 2, 3, 4], [5]]


def test_01g_puntos_por_variable_a_mano():
    P = puntos_columnas(X_JUGUETE, COEF_JUGUETE, params_20())
    V = puntos_por_variable(P, grupos_columnas(PRE_JUGUETE))
    assert V.shape == (2, 3)
    # a: su columna; c: suma de c=x, c=y, c=z, c=OTRA; b: su columna (valores de test_01c)
    esperado = np.array([[-11.541560327111709, -8.656170245333781, 3.6067376022224087],
                         [23.083120654223418, -14.426950408889635, -7.213475204444817]])
    np.testing.assert_allclose(V, esperado, rtol=0, atol=TOL)


def test_01h_puntos_valor_crudo_con_log1p_y_recorte():
    va = v_pre("a")
    unidad = -FACTOR_20 * 0.4  # = −11.541560327111709
    valores = np.array([3.0, 10.0, 50.0, 0.5, 1.0, 200.0, 99.0])
    # puntos = unidad · (ln(1 + clip(x, 1, 99)) − 2) / 0.5:
    # 3 → 14.166241308446832; 10 → −9.18466458974993; 50 → −44.59256416309711;
    # 0.5 (bajo el recorte) = 1 (borde) → 30.166241308446832; 200 (sobre el recorte) = 99 (borde) → −60.13545772794878
    esperado = [14.166241308446832, -9.18466458974993, -44.59256416309711, 30.166241308446832, 30.166241308446832,
                -60.13545772794878, -60.13545772794878]
    np.testing.assert_allclose(puntos_valor_crudo(valores, va, unidad), esperado, rtol=0, atol=TOL)


def test_01i_puntos_valor_crudo_sin_log1p():
    vb = v_pre("b")
    unidad = -FACTOR_20 * -0.25  # = 7.213475204444817
    valores = np.array([2.0, 4.0, 8.0, 20.0, -3.0, 10.0, 0.0])
    # puntos = unidad · (clip(x, 0, 10) − 5) / 2:
    # 2 → −10.820212806667225; 4 → −3.6067376022224087; 8 → 10.820212806667225;
    # 20 (sobre el recorte) = 10 → 18.033688011112044; −3 (bajo el recorte) = 0 → −18.033688011112044
    esperado = [-10.820212806667225, -3.6067376022224087, 10.820212806667225, 18.033688011112044,
                -18.033688011112044, 18.033688011112044, -18.033688011112044]
    np.testing.assert_allclose(puntos_valor_crudo(valores, vb, unidad), esperado, rtol=0, atol=TOL)


def test_01i2_puntos_valor_crudo_imputa_nan_con_la_mediana():
    """Decisión de Juan José (C3, fase verde): NaN → mediana antes de recortar (imputar → recortar → log1p →
    estandarizar, el mismo orden del preprocesador)."""
    # a: mediana 9 ⇒ −11.541560327111709 · (ln(10) − 2)/0.5 = −6.9846082097509745
    np.testing.assert_allclose(puntos_valor_crudo(np.array([np.nan, 9.0]), v_pre("a"), -FACTOR_20 * 0.4),
                               [-6.9846082097509745, -6.9846082097509745], rtol=0, atol=TOL)
    # b: mediana 4 ⇒ 7.213475204444817 · (4 − 5)/2 = −3.6067376022224087
    np.testing.assert_allclose(puntos_valor_crudo(np.array([np.nan]), v_pre("b"), -FACTOR_20 * -0.25),
                               [-3.6067376022224087], rtol=0, atol=TOL)


def test_01j_entrada_numerica_a_mano():
    va = v_pre("a")
    # crudos de train con un vacío (NaN): la tabla usa solo los NO vacíos [0.5, 3, 10, 50, 200].
    crudos = np.array([np.nan, 0.5, 3.0, 10.0, 50.0, 200.0])
    e = entrada_numerica(F_A, va, 0.4, crudos, params_20(), CFG_DIST_JUGUETE)
    assert set(e) == CLAVES_NUM
    assert e["nombre"] == "a" and e["tipo"] == "num" and e["etiqueta_es"] == "Variable A"
    assert e["beta"] == pytest.approx(0.4, abs=TOL)
    assert e["puntos_por_unidad_estandarizada"] == pytest.approx(-11.541560327111709, abs=TOL)  # −factor · 0.4
    assert e["transformacion"] == {"mediana": 9.0, "recorte": [1.0, 99.0], "log1p": True, "media": 2.0, "desv": 0.5}
    # np.percentile lineal de [0.5, 3, 10, 50, 200] en 25, 50, 75 → 3, 10, 50 (índices exactos 1, 2, 3)
    assert [f["percentil"] for f in e["tabla"]] == [25, 50, 75]
    np.testing.assert_allclose([f["valor"] for f in e["tabla"]], [3.0, 10.0, 50.0], rtol=0, atol=TOL)
    np.testing.assert_allclose([f["puntos"] for f in e["tabla"]],
                               [14.166241308446832, -9.18466458974993, -44.59256416309711], rtol=0, atol=TOL)
    # rango con percentiles_rango [50, 75] = |puntos(p75) − puntos(p50)| = |−44.59256416309711 − (−9.18466458974993)|
    # = 35.40789957334718 (con la tabla [25, 75] sería 58.75880547154394)
    assert e["rango_puntos"] == pytest.approx(35.40789957334718, abs=TOL)


def test_01k_entrada_numerica_sin_log1p_a_mano():
    vb = v_pre("b")
    crudos = np.array([0.0, 2.0, 4.0, 8.0, 20.0])
    e = entrada_numerica(F_B, vb, -0.25, crudos, params_20(), CFG_DIST_JUGUETE)
    # np.percentile lineal de [0, 2, 4, 8, 20] en 25, 50, 75 → 2, 4, 8
    np.testing.assert_allclose([f["valor"] for f in e["tabla"]], [2.0, 4.0, 8.0], rtol=0, atol=TOL)
    # puntos (test_01i): −10.820212806667225, −3.6067376022224087, 10.820212806667225
    np.testing.assert_allclose([f["puntos"] for f in e["tabla"]],
                               [-10.820212806667225, -3.6067376022224087, 10.820212806667225], rtol=0, atol=TOL)
    # rango con [50, 75] = |puntos(8) − puntos(4)| = |10.820212806667225 − (−3.6067376022224087)| = 14.426950408889635
    # (con [25, 75] sería 21.64042561333445)
    assert e["rango_puntos"] == pytest.approx(14.426950408889635, abs=TOL)
    assert e["puntos_por_unidad_estandarizada"] == pytest.approx(7.213475204444817, abs=TOL)  # −factor · (−0.25)


def test_01l_entrada_categorica_a_mano():
    vc = v_pre("c")
    betas = {"x": 0.3, "y": -0.2, "z": 0.9, "OTRA": 0.5}
    fracciones = {"x": 0.6, "y": 0.35, "z": 0.05, "OTRA": 0.0}
    e = entrada_categorica(F_C, vc, betas, fracciones, params_20(), CFG_DIST_JUGUETE)
    assert set(e) == CLAVES_CAT
    assert e["nombre"] == "c" and e["tipo"] == "cat" and e["etiqueta_es"] == "Variable C"
    # Orden de categorias_es (z, x, y), sin OTRA.
    assert [c["categoria"] for c in e["categorias"]] == ["z", "x", "y"]
    assert [c["etiqueta"] for c in e["categorias"]] == ["Zeta", "Equis", "Ye"]
    assert all(set(c) == CLAVES_CATEGORIA for c in e["categorias"])
    np.testing.assert_allclose([c["beta"] for c in e["categorias"]], [0.9, 0.3, -0.2], rtol=0, atol=TOL)
    # puntos = −factor · β: z −25.968510736001342; x −8.656170245333781; y 5.770780163555854
    np.testing.assert_allclose([c["puntos"] for c in e["categorias"]],
                               [-25.968510736001342, -8.656170245333781, 5.770780163555854], rtol=0, atol=TOL)
    np.testing.assert_allclose([c["fraccion_train"] for c in e["categorias"]], [0.05, 0.6, 0.35], rtol=0, atol=TOL)
    # OTRA aparte: −factor · 0.5 = −14.426950408889635
    assert set(e["otra"]) == CLAVES_OTRA
    assert e["otra"]["categoria"] == "OTRA" and e["otra"]["interna"] is True
    assert e["otra"]["beta"] == pytest.approx(0.5, abs=TOL)
    assert e["otra"]["puntos"] == pytest.approx(-14.426950408889635, abs=TOL)
    # Rango solo entre categorías con fracción ≥ 0.1 (x e y; z = 0.05 queda fuera):
    # 5.770780163555854 − (−8.656170245333781) = 0.5 · factor = 14.426950408889634
    # (con z sería 5.7708 − (−25.9685) = 1.1 · factor = 31.739290899557197).
    assert e["rango_puntos"] == pytest.approx(14.426950408889634, abs=TOL)


def test_01l2_fraccion_igual_al_minimo_entra_en_el_rango():
    """M15: una categoría con fraccion_train EXACTAMENTE igual a fraccion_min_categoria (0.1) entra en el rango (≥)."""
    betas = {"x": 0.3, "y": -0.2, "z": 0.9, "OTRA": 0.5}
    fracciones = {"x": 0.55, "y": 0.35, "z": 0.1, "OTRA": 0.0}
    e = entrada_categorica(F_C, v_pre("c"), betas, fracciones, params_20(), CFG_DIST_JUGUETE)
    # con z: 5.770780163555854 − (−25.968510736001342) = 1.1 · factor = 31.739290899557197
    assert e["rango_puntos"] == pytest.approx(31.739290899557197, abs=TOL)


def _signo_positivo(x: float) -> bool:
    return x == 0 and math.copysign(1.0, x) == 1.0


def test_01l3_beta_cero_da_cero_positivo():
    """Con β = 0 los puntos son +0.0, no −0.0 (−factor · 0 = −0.0 en float; el JSON escribiría "-0.0")."""
    betas = {"x": 0.3, "y": 0.0, "z": 0.9, "OTRA": 0.0}
    fracciones = {"x": 0.6, "y": 0.35, "z": 0.05, "OTRA": 0.0}
    e = entrada_categorica(F_C, v_pre("c"), betas, fracciones, params_20(), CFG_DIST_JUGUETE)
    y = next(c for c in e["categorias"] if c["categoria"] == "y")
    assert _signo_positivo(y["puntos"]) and _signo_positivo(e["otra"]["puntos"])
    en = entrada_numerica(F_A, v_pre("a"), 0.0, np.array([0.5, 3.0, 10.0, 50.0, 200.0]), params_20(),
                          CFG_DIST_JUGUETE)
    assert _signo_positivo(en["puntos_por_unidad_estandarizada"])
    assert all(_signo_positivo(f["puntos"]) for f in en["tabla"])
    assert _signo_positivo(en["rango_puntos"])


def test_01m_preprocesador_de_juguete_es_coherente():
    """Control de la propia prueba: PRE_JUGUETE es un to_dict válido y transform da las columnas declaradas."""
    pre = Preprocesador.from_dict(PRE_JUGUETE)
    assert pre.nombres_salida() == PRE_JUGUETE["columnas_salida"]
    # a = e^2.5 − 1 ⇒ log1p = 2.5 ⇒ (2.5 − 2)/0.5 = 1; c = x; b = 6 ⇒ (6 − 5)/2 = 0.5 → fila 0 de X_JUGUETE
    df = pd.DataFrame({"a": [math.expm1(2.5)], "c": ["x"], "b": [6.0]})
    np.testing.assert_allclose(pre.transform(df)[0], X_JUGUETE[0], rtol=0, atol=1e-6)


# ------------------------------------------------------------------------------------------------------------------
# Fixtures del modelo real (se cargan una vez por módulo)
# ------------------------------------------------------------------------------------------------------------------


def _cfg_real(salida: Path, overrides: list[str] | None = None, datos: Path | None = None) -> dict:
    """Config de base.yaml (con overrides) con rutas absolutas y la salida en un directorio temporal."""
    cfg = apply_overrides(load_config(RUTA_CONFIG), overrides or [])
    cfg = copy.deepcopy(cfg)
    cfg["rutas"] = {k: str(RAIZ / v) for k, v in cfg["rutas"].items()}
    if datos is not None:
        cfg["rutas"]["datos"] = str(datos)
    cfg["salida"] = str(salida)
    return cfg


@pytest.fixture(scope="module")
def features() -> list[dict]:
    return load_config(RAIZ / "configs" / "features.yaml")["features"]


@pytest.fixture(scope="module")
def pre_json() -> dict:
    return load_json(RAIZ / "models" / "preprocesador_pd.json")


@pytest.fixture(scope="module")
def pre(pre_json) -> Preprocesador:
    return Preprocesador.from_dict(pre_json)


@pytest.fixture(scope="module")
def datos() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data" / "processed" / "loans_pd.parquet")


@pytest.fixture(scope="module")
def logistica() -> LogisticaPD:
    return LogisticaPD.cargar(RAIZ / "models" / "pd_logistica.joblib")


@pytest.fixture(scope="module")
def params() -> dict:
    return parametros_scorecard(bloque_scorecard(load_config(RUTA_CONFIG)))


@pytest.fixture(scope="module")
def cfg20(tmp_path_factory) -> dict:
    """Config real de base.yaml (pdo 20) con la salida en un directorio temporal."""
    return _cfg_real(tmp_path_factory.mktemp("sc20"))


@pytest.fixture(scope="module")
def preparado(cfg20) -> dict:
    """Preparación ÚNICA del módulo (carga y transform de los 3 splits); la comparten por_split, sc y sc40."""
    return preparar_splits(cfg20)


@pytest.fixture(scope="module")
def por_split(preparado, params) -> dict:
    """Por split: X (float32 de transform), y, logit, PD y score de referencia de C1 (desde el logit preparado)."""
    salida = {}
    for s in SPLITS:
        d = preparado["splits"][s]
        salida[s] = {"X": d["X"], "y": d["y"], "logit": d["logit"], "prob": d["prob"],
                     "score": score_desde_logit(d["logit"], params)}
    return salida


@pytest.fixture(scope="module")
def sc(cfg20, preparado) -> dict:
    """scorecard.json real (pdo 20) construido en memoria con el preparado compartido."""
    return construir_scorecard(cfg20, preparado)


@pytest.fixture(scope="module")
def cfg40(tmp_path_factory) -> dict:
    """Config real con --set scorecard.pdo=40 (apply_overrides) y la salida en un directorio temporal."""
    return _cfg_real(tmp_path_factory.mktemp("sc40"), ["scorecard.pdo=40"])


@pytest.fixture(scope="module")
def sc40(cfg40, preparado) -> dict:
    """scorecard.json real con pdo 40 (todo el ensamblado del JSON) usando el preparado de la config de pdo 20: el
    preparado no depende de scorecard.* (lo comprueba test_12)."""
    return construir_scorecard(cfg40, preparado)


def _entrada(sc: dict, nombre: str) -> dict:
    return next(e for e in sc["puntos_por_variable"] if e["nombre"] == nombre)


def _indice_columna(pre_json: dict, nombre_columna: str) -> int:
    return pre_json["columnas_salida"].index(nombre_columna)


# ------------------------------------------------------------------------------------------------------------------
# 2. Aditividad con el modelo real
# ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
def test_02_aditividad_real(por_split, logistica, params, pre_json, split):
    """pb + Σ puntos por columna = score de C1 (y lo mismo agregando por variable), error < TOL_ADITIVIDAD."""
    coef, b0 = logistica.coeficientes()
    d = por_split[split]
    pb = puntos_base(b0, params)
    P = puntos_columnas(d["X"], coef, params)
    err_col = error_aditividad(pb, P, d["score"])
    V = puntos_por_variable(P, grupos_columnas(pre_json))
    assert V.shape == (len(d["score"]), len(pre_json["variables"]))
    err_var = error_aditividad(pb, V, d["score"])
    print(f"\n{split}: error máximo de aditividad por columna = {err_col:.3e}; por variable = {err_var:.3e}")
    assert err_col < TOL_ADITIVIDAD
    assert err_var < TOL_ADITIVIDAD


def test_02b_puntos_base_del_json(sc, logistica, params):
    _, b0 = logistica.coeficientes()
    assert sc["puntos_base"] == pytest.approx(params["offset"] - params["factor"] * b0, abs=TOL)
    assert sc["aditividad"]["tolerancia"] == TOL_ADITIVIDAD
    assert 0 <= sc["aditividad"]["error_max"] < TOL_ADITIVIDAD


# ------------------------------------------------------------------------------------------------------------------
# 3. Signo: β > 0 ⇒ menos puntos al crecer la variable
# ------------------------------------------------------------------------------------------------------------------


def test_03_signo_de_las_numericas(sc, logistica, params, pre_json):
    coef, _ = logistica.coeficientes()
    numericas = [v["nombre"] for v in pre_json["variables"] if v["tipo"] == "num"]
    assert numericas
    for nombre in numericas:
        beta = coef[_indice_columna(pre_json, nombre)]
        e = _entrada(sc, nombre)
        assert e["beta"] == pytest.approx(beta, abs=TOL)
        assert e["puntos_por_unidad_estandarizada"] == pytest.approx(-params["factor"] * beta, abs=TOL)
        valores = np.array([f["valor"] for f in e["tabla"]])
        puntos = np.array([f["puntos"] for f in e["tabla"]])
        assert np.all(np.diff(valores) >= 0), nombre  # percentiles crecientes ⇒ valores no decrecientes
        if beta > 0:
            assert np.all(np.diff(puntos) <= 0), f"{nombre}: β > 0 y la tabla no es no creciente"
        elif beta < 0:
            assert np.all(np.diff(puntos) >= 0), f"{nombre}: β < 0 y la tabla no es no decreciente"


# ------------------------------------------------------------------------------------------------------------------
# 4. Cero en la media y categoría activa
# ------------------------------------------------------------------------------------------------------------------


def test_04a_numerica_en_su_media_da_cero(logistica, params, pre_json):
    coef, _ = logistica.coeficientes()
    for v in pre_json["variables"]:
        if v["tipo"] != "num":
            continue
        # valor crudo cuyo transformado es la media: expm1(media) si hay log1p; si no, la media (dentro del recorte).
        crudo = math.expm1(v["media"]) if v["log1p"] else v["media"]
        assert v["recorte"][0] <= crudo <= v["recorte"][1], v["nombre"]
        unidad = -params["factor"] * coef[_indice_columna(pre_json, v["nombre"])]
        assert puntos_valor_crudo(np.array([crudo]), v, unidad)[0] == pytest.approx(0.0, abs=1e-8), v["nombre"]


@pytest.mark.parametrize("split", SPLITS)
def test_04b_categorica_suma_one_hot_es_la_categoria_activa(sc, por_split, logistica, params, pre_json, split):
    coef, _ = logistica.coeficientes()
    X = por_split[split]["X"]
    P = puntos_columnas(X, coef, params)
    grupos = grupos_columnas(pre_json)
    V = puntos_por_variable(P, grupos)
    columnas = pre_json["columnas_salida"]
    for k, (nombre, idx) in enumerate(grupos):
        v = next(w for w in pre_json["variables"] if w["nombre"] == nombre)
        if v["tipo"] != "cat":
            continue
        bloque = X[:, idx]
        assert np.all(bloque.sum(axis=1) == 1), nombre  # one-hot: exactamente una activa
        activas = [columnas[idx[j]].split("=", 1)[1] for j in bloque.argmax(axis=1)]
        e = _entrada(sc, nombre)
        puntos_cat = {c["categoria"]: c["puntos"] for c in e["categorias"]}
        puntos_cat["OTRA"] = e["otra"]["puntos"]
        esperado = np.array([puntos_cat[a] for a in activas])
        np.testing.assert_allclose(V[:, k], esperado, rtol=0, atol=TOL, err_msg=nombre)


# ------------------------------------------------------------------------------------------------------------------
# 5. Fracciones de train (decisión de Juan José: media de la columna one-hot de pre.transform(train))
# ------------------------------------------------------------------------------------------------------------------


def test_05_fracciones_de_train(sc, por_split, pre_json):
    X = por_split["train"]["X"]
    for v in pre_json["variables"]:
        if v["tipo"] != "cat":
            continue
        e = _entrada(sc, v["nombre"])
        suma = 0.0
        for c in e["categorias"]:
            j = _indice_columna(pre_json, f"{v['nombre']}={c['categoria']}")
            assert c["fraccion_train"] == pytest.approx(float(np.mean(X[:, j], dtype=np.float64)),
                                                        abs=TOL_FRACCION), (v["nombre"], c["categoria"])
            suma += c["fraccion_train"]
        frac_otra = float(np.mean(X[:, _indice_columna(pre_json, f"{v['nombre']}=OTRA")], dtype=np.float64))
        assert suma + frac_otra == pytest.approx(1.0, abs=TOL_FRACCION), v["nombre"]
        # Todas las categorías de train (sin OTRA) aparecen, en el orden de categorias_es.
        assert sorted(c["categoria"] for c in e["categorias"]) == sorted(c for c in v["categorias"] if c != "OTRA")


# ------------------------------------------------------------------------------------------------------------------
# 6. --set scorecard.pdo=40
# ------------------------------------------------------------------------------------------------------------------


def _doble(a, b) -> None:
    assert b == pytest.approx(2 * a, rel=TOL_REL_PDO, abs=1e-12)


def test_06_pdo_40_duplica_todos_los_puntos(sc, sc40, por_split, logistica):
    _doble(sc["factor"], sc40["factor"])
    _doble(sc["puntos_base"] - sc["base"], sc40["puntos_base"] - sc40["base"])
    params40 = {k: sc40[k] for k in CLAVES_PARAMETROS}
    params20 = {k: sc[k] for k in CLAVES_PARAMETROS}
    # Puntaje de las odds de referencia (p = 1/51) sigue en 600.
    assert score_desde_pd(1 / 51, params40) == pytest.approx(600.0, abs=TOL)
    coef, _ = logistica.coeficientes()
    X = por_split["test"]["X"]
    P20 = puntos_columnas(X, coef, params20)
    P40 = puntos_columnas(X, coef, params40)
    np.testing.assert_allclose(P40, 2 * P20, rtol=TOL_REL_PDO, atol=1e-12)
    for e20, e40 in zip(sc["puntos_por_variable"], sc40["puntos_por_variable"], strict=True):
        assert e20["nombre"] == e40["nombre"]
        _doble(e20["rango_puntos"], e40["rango_puntos"])
        if e20["tipo"] == "num":
            _doble(e20["puntos_por_unidad_estandarizada"], e40["puntos_por_unidad_estandarizada"])
            for f20, f40 in zip(e20["tabla"], e40["tabla"], strict=True):
                assert f20["valor"] == f40["valor"]
                _doble(f20["puntos"], f40["puntos"])
        else:
            for c20, c40 in zip(e20["categorias"], e40["categorias"], strict=True):
                _doble(c20["puntos"], c40["puntos"])
            _doble(e20["otra"]["puntos"], e40["otra"]["puntos"])


# ------------------------------------------------------------------------------------------------------------------
# 7. Bandas e histograma
# ------------------------------------------------------------------------------------------------------------------


def test_07a_cortes_a_mano():
    # np.quantile lineal de 0, 1, …, 10 en 1/4, 2/4, 3/4 → 2.5, 5.0, 7.5
    np.testing.assert_allclose(cortes_bandas(np.arange(11.0), 4), [2.5, 5.0, 7.5], rtol=0, atol=TOL)


def test_07b_cortes_con_empates_es_error():
    # np.quantile de [1, 1, 1, 1, 2] en 1/4, 2/4, 3/4 → 1, 1, 1: los cortes colapsan
    with pytest.raises(ValueError):
        cortes_bandas(np.array([1.0, 1.0, 1.0, 1.0, 2.0]), 4)


def test_07c_tabla_bandas_a_mano():
    score = np.array([450.0, 500.0, 550.0, 600.0, 700.0])
    pd_ = np.array([0.3, 0.2, 0.1, 0.05, 0.01])
    y = np.array([1, 1, 0, 0, 0])
    bandas = tabla_bandas(score, pd_, y, [500.0, 600.0])
    assert all(set(b) == CLAVES_BANDA for b in bandas)
    assert [b["banda"] for b in bandas] == [0, 1, 2]
    assert [(b["desde"], b["hasta"]) for b in bandas] == [(None, 500.0), (500.0, 600.0), (600.0, None)]
    # banda 0 (−∞, 500): {450}; banda 1 [500, 600): {500, 550} (500 está en el corte → banda superior);
    # banda 2 [600, +∞): {600, 700}
    assert [b["n"] for b in bandas] == [1, 2, 2]
    # puntaje medio: 450; (500 + 550)/2 = 525; (600 + 700)/2 = 650
    np.testing.assert_allclose([b["puntaje_medio"] for b in bandas], [450.0, 525.0, 650.0], rtol=0, atol=TOL)
    # pd media: 0.3; (0.2 + 0.1)/2 = 0.15; (0.05 + 0.01)/2 = 0.03
    np.testing.assert_allclose([b["pd_media"] for b in bandas], [0.3, 0.15, 0.03], rtol=0, atol=TOL)
    # tasa de malos: 1/1; 1/2; 0/2
    np.testing.assert_allclose([b["tasa_malos"] for b in bandas], [1.0, 0.5, 0.0], rtol=0, atol=TOL)


def test_07d_banda_vacia():
    bandas = tabla_bandas(np.array([450.0, 700.0]), np.array([0.3, 0.01]), np.array([1, 0]), [500.0, 600.0])
    assert [b["n"] for b in bandas] == [1, 0, 1]
    assert bandas[1]["puntaje_medio"] is None and bandas[1]["pd_media"] is None and bandas[1]["tasa_malos"] is None


def test_07e_histograma_a_mano():
    # mín 3 → floor(0.3)·10 = 0; máx 20 es múltiplo exacto → ceil = 20 y un borde más: 30 ⇒ bordes 0, 10, 20, 30
    # [0, 10): {3}; [10, 20): {10, 15}; [20, 30): {20}
    h = histograma(np.array([3.0, 10.0, 15.0, 20.0]), 10)
    assert set(h) == CLAVES_HISTOGRAMA
    assert h["ancho"] == 10
    np.testing.assert_allclose(h["bordes"], [0.0, 10.0, 20.0, 30.0], rtol=0, atol=TOL)
    assert list(h["conteos"]) == [1, 2, 1]
    # mín −5 → floor(−0.5)·10 = −10; máx 7 → ceil(0.7)·10 = 10 ⇒ bordes −10, 0, 10; [−10, 0): {−5}; [0, 10): {0, 7}
    h = histograma(np.array([-5.0, 0.0, 7.0]), 10)
    np.testing.assert_allclose(h["bordes"], [-10.0, 0.0, 10.0], rtol=0, atol=TOL)
    assert list(h["conteos"]) == [1, 2]


def test_07e2_histograma_un_solo_valor():
    # 605 / 10 = 60.5 → k = 60 ⇒ bordes 600, 610; [600, 610): {605}
    h = histograma(np.array([605.0]), 10)
    np.testing.assert_allclose(h["bordes"], [600.0, 610.0], rtol=0, atol=TOL)
    assert list(h["conteos"]) == [1]


def test_07e3_histograma_ancho_no_entero():
    """[b_i, b_{i+1}) también con ancho 0.1: en float 0.3/0.1 = 2.9999999999999996 y 0.7/0.1 = 6.999999999999999,
    pero 0.3 va en [0.3, 0.4) y 0.7 en [0.7, 0.8)."""
    h = histograma(np.array([0.3, 0.7]), 0.1)
    # bordes 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 (redondeados a 1 decimal, los del ancho)
    assert h["bordes"] == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert list(h["conteos"]) == [1, 0, 0, 0, 1]


def test_07e4_histograma_demasiados_intervalos_es_error():
    # 0 … 1e9 con ancho 1 serían 10⁹ + 1 intervalos: por encima del tope del módulo
    with pytest.raises(ValueError):
        histograma(np.array([0.0, 1e9]), 1)


def test_07f_distribucion_split_a_mano():
    score = np.array([450.0, 500.0, 550.0, 600.0, 700.0])
    pd_ = np.array([0.3, 0.2, 0.1, 0.05, 0.01])
    y = np.array([1, 1, 0, 0, 0])
    d = distribucion_split(score, pd_, y, [500.0, 600.0], 10)
    assert set(d) == CLAVES_SPLIT
    assert d["n"] == 5
    # media 2800/5 = 560; desv muestral (n − 1) = 96.17692030835673
    assert d["media"] == pytest.approx(560.0, abs=TOL)
    assert d["desv"] == pytest.approx(96.17692030835673, abs=TOL)
    assert len(d["percentiles"]) == 101
    # p0 = 450, p50 = 550, p100 = 700 (np.percentile lineal)
    assert (d["percentiles"][0], d["percentiles"][50], d["percentiles"][100]) == pytest.approx((450.0, 550.0, 700.0))
    assert sum(d["histograma"]["conteos"]) == 5
    assert [b["n"] for b in d["bandas"]] == [1, 2, 2]
    # n = 1 → desv 0 por convención (núcleo §5.4)
    assert distribucion_split(np.array([600.0]), np.array([0.1]), np.array([0]), [500.0], 10)["desv"] == 0


def test_07g_bandas_reales(sc, por_split):
    ds = sc["distribucion_scores"]
    cortes = ds["cortes_bandas"]
    assert ds["n_bandas"] == 10 and len(cortes) == 9
    assert np.all(np.diff(cortes) > 0)
    np.testing.assert_allclose(cortes, cortes_bandas(por_split["train"]["score"], 10), rtol=0, atol=TOL)
    for s in SPLITS:
        bandas = ds[s]["bandas"]
        assert len(bandas) == 10
        assert sum(b["n"] for b in bandas) == len(por_split[s]["score"]) == ds[s]["n"]
        pd_media = [b["pd_media"] for b in bandas]
        assert None not in pd_media, s
        assert np.all(np.diff(pd_media) < 0), f"{s}: pd_media no es estrictamente decreciente con la banda"
    n_train = ds["train"]["n"]
    for b in ds["train"]["bandas"]:
        assert abs(b["n"] / n_train - 0.1) <= TOL_BANDA, b


# ------------------------------------------------------------------------------------------------------------------
# 8. Formato de scorecard.json
# ------------------------------------------------------------------------------------------------------------------


def test_08_formato(sc, features, por_split, datos, params):
    assert set(sc) == CLAVES_SCORECARD_JSON
    assert {k: sc[k] for k in CLAVES_PARAMETROS} == pytest.approx(params)
    assert sc["modelo"] == "logistica"
    assert set(sc["aditividad"]) == {"error_max", "tolerancia"}
    # Las 15 variables en el orden de features.yaml, con claves exactas por tipo.
    assert [e["nombre"] for e in sc["puntos_por_variable"]] == [f["nombre"] for f in features]
    for f, e in zip(features, sc["puntos_por_variable"], strict=True):
        assert e["tipo"] == f["tipo"] and e["etiqueta_es"] == f["etiqueta_es"]
        if f["tipo"] == "num":
            assert set(e) == CLAVES_NUM
            assert set(e["transformacion"]) == CLAVES_TRANSFORMACION
            assert len(e["tabla"]) == 7 and all(set(t) == CLAVES_FILA_TABLA for t in e["tabla"])
            assert [t["percentil"] for t in e["tabla"]] == [1, 5, 25, 50, 75, 95, 99]
            crudos = datos.loc[datos["split"] == "train", f["nombre"]].dropna().to_numpy(dtype=np.float64)
            np.testing.assert_allclose([t["valor"] for t in e["tabla"]],
                                       np.percentile(crudos, [1, 5, 25, 50, 75, 95, 99]), rtol=0, atol=TOL)
        else:
            assert set(e) == CLAVES_CAT
            assert set(e["otra"]) == CLAVES_OTRA
            assert all(set(c) == CLAVES_CATEGORIA for c in e["categorias"])
            orden_es = [c for c in f["categorias_es"] if c in {x["categoria"] for x in e["categorias"]}]
            assert [c["categoria"] for c in e["categorias"]] == orden_es
            assert all(c["etiqueta"] == f["categorias_es"][c["categoria"]] for c in e["categorias"])
    ds = sc["distribucion_scores"]
    assert set(ds) == CLAVES_DIST_SCORES
    assert ds["ancho_bin"] == 10
    for s in SPLITS:
        assert set(ds[s]) == CLAVES_SPLIT
        score = por_split[s]["score"]
        assert ds[s]["n"] == len(score)
        assert ds[s]["media"] == pytest.approx(float(np.mean(score)), abs=1e-9)
        assert ds[s]["desv"] == pytest.approx(float(np.std(score, ddof=1)), abs=1e-9)  # muestral
        pct = np.array(ds[s]["percentiles"])
        assert len(pct) == 101 and np.all(np.diff(pct) >= 0)
        np.testing.assert_allclose(pct, np.percentile(score, np.arange(101)), rtol=0, atol=1e-9)
        assert set(ds[s]["histograma"]) == CLAVES_HISTOGRAMA
        assert sum(ds[s]["histograma"]["conteos"]) == len(score)
    ev = sc["evidencia"]
    assert set(ev) == {"sha256_particion", "sha256_preprocesador"}
    for h in ev.values():
        assert isinstance(h, str) and len(h) == 64 and all(ch in "0123456789abcdef" for ch in h)
    assert ev["sha256_particion"] == huella_particion(datos["id"], datos["split"])
    assert ev["sha256_preprocesador"] == huella_archivo(RAIZ / "models" / "preprocesador_pd.json")
    json.dumps(to_jsonable(sc), allow_nan=False)  # sin NaN ni inf


# ------------------------------------------------------------------------------------------------------------------
# 9. Fuga: lo que se estima con train no cambia si se alteran val y test
# ------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sc_alterado(tmp_path_factory, datos, features) -> dict:
    """scorecard.json con las variables de val y test reemplazadas por filas de train sorteadas (semilla fija)."""
    carpeta = tmp_path_factory.mktemp("fuga")
    df = datos.copy()
    fuera = (df["split"] != "train").to_numpy()
    train = df[~fuera].reset_index(drop=True)
    idx = make_rng(SEMILLA_PRUEBA, 0).integers(0, len(train), size=int(fuera.sum()))
    for f in features:
        df.loc[fuera, f["nombre"]] = train[f["nombre"]].to_numpy()[idx]
    ruta = carpeta / "loans_pd_alterado.parquet"
    df.to_parquet(ruta, index=False)
    return construir_scorecard(_cfg_real(carpeta / "salida", datos=ruta))


def test_09_fuga(sc, sc_alterado):
    # Control: la alteración sí cambió val y test.
    for s in ("val", "test"):
        assert sc_alterado["distribucion_scores"][s]["media"] != sc["distribucion_scores"][s]["media"]
    # Las aserciones parciales (cortes, train, valores de la tabla y fracciones) están solo para dar un mensaje de
    # error más preciso; la comprobación completa es `e == e_alt`.
    assert sc_alterado["distribucion_scores"]["cortes_bandas"] == sc["distribucion_scores"]["cortes_bandas"]
    assert sc_alterado["distribucion_scores"]["train"] == sc["distribucion_scores"]["train"]
    for e, e_alt in zip(sc["puntos_por_variable"], sc_alterado["puntos_por_variable"], strict=True):
        if e["tipo"] == "num":
            assert [t["valor"] for t in e["tabla"]] == [t["valor"] for t in e_alt["tabla"]], e["nombre"]
        else:
            assert ([c["fraccion_train"] for c in e["categorias"]]
                    == [c["fraccion_train"] for c in e_alt["categorias"]]), e["nombre"]
        assert e == e_alt, e["nombre"]


# ------------------------------------------------------------------------------------------------------------------
# 10. Errores
# ------------------------------------------------------------------------------------------------------------------


def test_10a_columnas_distintas_es_error():
    with pytest.raises(ValueError):
        puntos_columnas(X_JUGUETE, COEF_JUGUETE[:-1], params_20())


def test_10b_pre_json_incoherente_es_error():
    malo = copy.deepcopy(PRE_JUGUETE)
    malo["columnas_salida"] = ["a", "c=x", "c=y", "c=z", "c=OTRA", "B"]
    with pytest.raises(ValueError):
        grupos_columnas(malo)
    malo = copy.deepcopy(PRE_JUGUETE)
    malo["columnas_salida"] = ["a", "c=x", "c=y", "c=z", "c=OTRA"]  # falta b
    with pytest.raises(ValueError):
        grupos_columnas(malo)


@pytest.mark.parametrize("categorias_es", [
    {"z": "Zeta", "x": "Equis"},  # falta y
    {"z": "Zeta", "x": "Equis", "y": "Ye", "w": "Doble"},  # sobra w (no está en el preprocesador)
    {"z": "Zeta", "x": "Equis", "y": "Ye", "OTRA": "Otra"},  # OTRA es interna: no va en categorias_es
], ids=["falta", "sobra", "con_otra"])
def test_10b2_categorias_es_incoherente_es_error(categorias_es):
    """Decisión de Juan José (C3, fase verde; como C2): categorias_es debe ser exactamente el conjunto de categorías
    del preprocesador sin OTRA."""
    f = {**F_C, "categorias_es": categorias_es}
    betas = {"x": 0.3, "y": -0.2, "z": 0.9, "OTRA": 0.5}
    fracciones = {"x": 0.6, "y": 0.35, "z": 0.05, "OTRA": 0.0}
    with pytest.raises(ValueError):
        entrada_categorica(f, v_pre("c"), betas, fracciones, params_20(), CFG_DIST_JUGUETE)


def test_10c_cfg_valida_no_se_muta():
    cfg = copy.deepcopy(CFG_ESPERADA)
    validar_cfg_puntos(cfg)
    assert cfg == CFG_ESPERADA


def _cambiar(ruta: tuple, valor) -> dict:
    cfg = copy.deepcopy(CFG_ESPERADA)
    nodo = cfg
    for k in ruta[:-1]:
        nodo = nodo[k]
    nodo[ruta[-1]] = valor
    return cfg


def _sin(ruta: tuple) -> dict:
    cfg = copy.deepcopy(CFG_ESPERADA)
    nodo = cfg
    for k in ruta[:-1]:
        nodo = nodo[k]
    del nodo[ruta[-1]]
    return cfg


CFG_INVALIDAS = {
    "falta_raiz": _sin(("salida",)),
    "sobra_raiz": _cambiar(("extra",), 1),
    "scorecard_invalido": _cambiar(("scorecard", "pdo"), 0),
    "modelo_mlp": _cambiar(("modelo",), "mlp"),
    "falta_ruta": _sin(("rutas", "features")),
    "sobra_ruta": _cambiar(("rutas", "otra"), "x"),
    "falta_distribucion": _sin(("distribucion", "n_bandas")),
    "sobra_distribucion": _cambiar(("distribucion", "otra"), 1),
    "percentiles_vacios": _cambiar(("distribucion", "percentiles_tabla"), []),
    "percentil_negativo": _cambiar(("distribucion", "percentiles_tabla"), [-1, 50]),
    "percentil_mayor_100": _cambiar(("distribucion", "percentiles_tabla"), [50, 101]),
    "percentiles_no_crecientes": _cambiar(("distribucion", "percentiles_tabla"), [5, 50, 50, 95]),
    "percentiles_decrecientes": _cambiar(("distribucion", "percentiles_tabla"), [95, 5]),
    "percentil_bool": _cambiar(("distribucion", "percentiles_tabla"), [True, 50]),
    "rango_invertido": _cambiar(("distribucion", "percentiles_rango"), [95, 5]),
    "rango_igual": _cambiar(("distribucion", "percentiles_rango"), [50, 50]),
    "rango_fuera": _cambiar(("distribucion", "percentiles_rango"), [5, 101]),
    "fraccion_1": _cambiar(("distribucion", "fraccion_min_categoria"), 1),
    "fraccion_negativa": _cambiar(("distribucion", "fraccion_min_categoria"), -0.1),
    "ancho_0": _cambiar(("distribucion", "ancho_bin"), 0),
    "ancho_negativo": _cambiar(("distribucion", "ancho_bin"), -10),
    "n_bandas_1": _cambiar(("distribucion", "n_bandas"), 1),
    "n_bandas_true": _cambiar(("distribucion", "n_bandas"), True),
    "n_bandas_real": _cambiar(("distribucion", "n_bandas"), 2.5),
    "salida_vacia": _cambiar(("salida",), ""),
    "salida_espacios": _cambiar(("salida",), "   "),  # decisión de Juan José (C3, fase verde)
    "salida_no_texto": _cambiar(("salida",), 1),
}


@pytest.mark.parametrize("nombre", sorted(CFG_INVALIDAS))
def test_10d_cfg_invalida_es_error(nombre):
    cfg = CFG_INVALIDAS[nombre]
    copia = copy.deepcopy(cfg)
    with pytest.raises(ValueError):
        validar_cfg_puntos(cfg)
    assert cfg == copia  # no muta ni siquiera al fallar


# ------------------------------------------------------------------------------------------------------------------
# 11. manejador
# ------------------------------------------------------------------------------------------------------------------


def test_11_manejador_escribe_scorecard_json(tmp_path, monkeypatch, sc):
    """manejador escribe scorecard.json y devuelve 0. construir_scorecard se sustituye por el fixture `sc` (ya
    construido): el recorrido de extremo a extremo lo cubre tests/test_no_network.py::...[scorecard]."""
    monkeypatch.setattr(puntos, "construir_scorecard", lambda cfg, preparado=None: sc)
    assert manejador(_cfg_real(tmp_path)) == 0
    sc_leido = load_json(tmp_path / ARCHIVO_SCORECARD)
    assert set(sc_leido) == CLAVES_SCORECARD_JSON
    assert sorted(p.name for p in tmp_path.iterdir()) == [ARCHIVO_SCORECARD]


# ------------------------------------------------------------------------------------------------------------------
# 12. preparar_splits: la preparación compartida no cambia el resultado
# ------------------------------------------------------------------------------------------------------------------


def test_12_preparado_compartido_da_el_mismo_json(cfg20, cfg40, sc, sc40):
    """construir_scorecard(cfg) (prepara por su cuenta) == construir_scorecard(cfg, preparar_splits(cfg)) (el fixture
    `sc`, con el preparado que comparten por_split, sc y sc40): la preparación no depende de scorecard.* ni de
    distribucion.*."""
    assert construir_scorecard(cfg20) == sc
    # sc40 usa el preparado de la config de pdo 20; preparado por su cuenta con pdo 40 da lo mismo.
    assert construir_scorecard(cfg40) == sc40


def test_12b_preparado_de_solo_lectura(preparado):
    """Los arrays del preparado compartido no se pueden modificar por error (lo comparten varios fixtures)."""
    for s in SPLITS:
        for clave, arr in preparado["splits"][s].items():
            assert not arr.flags.writeable, (s, clave)
    for nombre, arr in preparado["crudos_train"].items():
        assert not arr.flags.writeable, nombre


def test_12c_rutas_distintas_del_preparado_es_error(cfg20, preparado):
    """Un preparado hecho con otras rutas no se puede usar con esta config."""
    otra = copy.deepcopy(cfg20)
    otra["rutas"]["datos"] = str(RAIZ / "data" / "processed" / "otro.parquet")
    with pytest.raises(ValueError):
        construir_scorecard(otra, preparado)


def test_12d_logit_alterado_rompe_la_aditividad(cfg20, preparado):
    """M24: si el logit del preparado no es el de X·β + β₀ (+1e-6 en una fila de test ⇒ el puntaje se mueve
    factor · 1e-6 ≈ 2.9e-5 > TOL_ADITIVIDAD), construir_scorecard falla en vez de escribir puntos incoherentes."""
    logit = preparado["splits"]["test"]["logit"].copy()
    logit[0] += 1e-6
    alterado = {**preparado, "splits": {**preparado["splits"],
                                        "test": {**preparado["splits"]["test"], "logit": logit}}}
    with pytest.raises(ValueError, match="[Aa]ditividad"):
        construir_scorecard(cfg20, alterado)
