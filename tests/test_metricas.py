"""Pruebas de caja negra de las métricas de la PD (Parte B §3.4, tarea B1).

Solo se usa la interfaz pública de ``src.common.metricas``. Se contrasta con scikit-learn y SciPy
(``roc_auc_score``, ``roc_curve``, ``brier_score_loss``, ``calibration_curve``, ``ks_2samp``) y con
casos hechos a mano cuya cuenta va en un comentario.
"""

import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from src.common.metricas import (
    CurvaKS,
    CurvaROC,
    area_trapecio,
    auc,
    bordes_cuantiles,
    brier,
    conteos_acumulados,
    curva_calibracion,
    curva_ks,
    curva_roc,
    gini,
    ks,
    tasa_falsos_positivos,
    tasa_verdaderos_positivos,
    validar_entradas,
)

TOL = 1e-12
SEMILLA = 20261024  # semilla declarada de los datos sintéticos de contraste
N = 2000
TASA_MALOS = 0.15


# --------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------
def _datos_aleatorios(con_empates: bool):
    """n = 2 000 créditos con ~15 % de malos; la PD depende de y para que haya señal."""
    rng = np.random.default_rng(SEMILLA)
    y = rng.binomial(1, TASA_MALOS, size=N)
    p = expit(rng.normal(-1.8 + 1.0 * y, 1.0))
    if con_empates:
        p = np.round(p, 2)
    return y, p


@pytest.fixture(params=[False, True], ids=["sin_empates", "con_empates"])
def datos(request):
    return _datos_aleatorios(request.param)


# Ejemplo a mano (6 créditos, 3 malos y 3 buenos; un empate en 0.4 entre un malo y un bueno).
Y_MANO = np.array([0, 0, 1, 1, 0, 1])
P_MANO = np.array([0.1, 0.4, 0.4, 0.8, 0.3, 0.6])


# --------------------------------------------------------------------------------------------
# 1. Contraste con referencias
# --------------------------------------------------------------------------------------------
def test_auc_coincide_con_roc_auc_score(datos):
    y, p = datos
    np.testing.assert_allclose(auc(y, p), roc_auc_score(y, p), rtol=0, atol=TOL)


def test_curva_roc_coincide_con_roc_curve(datos):
    y, p = datos
    fpr_ref, tpr_ref, umb_ref = roc_curve(y, p, drop_intermediate=False)
    curva = curva_roc(y, p)
    np.testing.assert_allclose(curva.fpr, fpr_ref, rtol=0, atol=TOL)
    np.testing.assert_allclose(curva.tpr, tpr_ref, rtol=0, atol=TOL)
    np.testing.assert_array_equal(curva.umbrales, umb_ref)


def test_brier_coincide_con_brier_score_loss(datos):
    y, p = datos
    np.testing.assert_allclose(brier(y, p), brier_score_loss(y, p), rtol=0, atol=TOL)


def test_ks_coincide_con_ks_2samp(datos):
    y, p = datos
    ref = ks_2samp(p[y == 1], p[y == 0]).statistic
    np.testing.assert_allclose(ks(y, p), ref, rtol=0, atol=TOL)


def test_curva_calibracion_coincide_con_calibration_curve(datos):
    # Con empates, casi todos los bordes por cuantiles coinciden con datos: esta variante fija la
    # convención de sklearn (dato en un borde -> bin inferior) y la omisión de bins vacíos.
    y, p = datos
    tasa_ref, pd_ref = calibration_curve(y, p, n_bins=10, strategy="quantile")
    cal = curva_calibracion(y, p, n_bins=10)
    np.testing.assert_allclose(cal.pd_media, pd_ref, rtol=0, atol=TOL)
    np.testing.assert_allclose(cal.tasa_observada, tasa_ref, rtol=0, atol=TOL)


# --------------------------------------------------------------------------------------------
# 2. Casos hechos a mano
# --------------------------------------------------------------------------------------------
def test_clasificador_perfecto():
    # Todos los malos tienen PD mayor que todos los buenos:
    # cada par (malo, bueno) está bien ordenado -> AUC = 4/4 = 1; Gini = 2·1 − 1 = 1.
    # KS en t = 0.2: F₁ = 0/2 (ningún malo con p <= 0.2), F₀ = 2/2 -> |0 − 1| = 1.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(y, p) == pytest.approx(1.0, abs=TOL)
    assert gini(y, p) == pytest.approx(1.0, abs=TOL)
    assert ks(y, p) == pytest.approx(1.0, abs=TOL)


def test_clasificador_invertido():
    # Todos los malos tienen PD menor que todos los buenos:
    # ningún par (malo, bueno) bien ordenado -> AUC = 0/4 = 0; Gini = 2·0 − 1 = −1.
    # KS en t = 0.2: F₁ = 2/2, F₀ = 0/2 -> |1 − 0| = 1 (el KS usa valor absoluto).
    y = np.array([1, 1, 0, 0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(y, p) == pytest.approx(0.0, abs=TOL)
    assert gini(y, p) == pytest.approx(-1.0, abs=TOL)
    assert ks(y, p) == pytest.approx(1.0, abs=TOL)


def test_puntaje_constante():
    # Un solo umbral: la ROC es (0,0) -> (1,1); trapecio = 1·(0 + 1)/2 = 0.5.
    # KS: único t = 0.5, F₁ = 1 y F₀ = 1 -> |1 − 1| = 0.
    y = np.array([0, 1, 0, 1, 0])
    p = np.full(5, 0.5)
    assert auc(y, p) == pytest.approx(0.5, abs=TOL)
    assert ks(y, p) == pytest.approx(0.0, abs=TOL)


def test_conteos_acumulados_a_mano():
    # Umbrales únicos descendentes de P_MANO: 0.8, 0.6, 0.4, 0.3, 0.1.
    # p >= 0.8: {0.8 (malo)}                         -> vp = 1, fp = 0
    # p >= 0.6: + {0.6 (malo)}                       -> vp = 2, fp = 0
    # p >= 0.4: + {0.4 (bueno), 0.4 (malo)} (empate) -> vp = 3, fp = 1
    # p >= 0.3: + {0.3 (bueno)}                      -> vp = 3, fp = 2
    # p >= 0.1: + {0.1 (bueno)}                      -> vp = 3, fp = 3
    umbrales, vp, fp = conteos_acumulados(Y_MANO, P_MANO)
    np.testing.assert_allclose(umbrales, [0.8, 0.6, 0.4, 0.3, 0.1], rtol=0, atol=TOL)
    np.testing.assert_array_equal(vp, [1, 2, 3, 3, 3])
    np.testing.assert_array_equal(fp, [0, 0, 1, 2, 3])


def test_tasas_a_mano():
    # TPR = vp / n_pos con n_pos = 3; FPR = fp / n_neg con n_neg = 3.
    np.testing.assert_allclose(
        tasa_verdaderos_positivos(np.array([1, 2, 3, 3, 3]), 3),
        [1 / 3, 2 / 3, 1, 1, 1], rtol=0, atol=TOL,
    )
    np.testing.assert_allclose(
        tasa_falsos_positivos(np.array([0, 0, 1, 2, 3]), 3),
        [0, 0, 1 / 3, 2 / 3, 1], rtol=0, atol=TOL,
    )


def test_curva_roc_a_mano():
    # Con los conteos de test_conteos_acumulados_a_mano y n_pos = n_neg = 3, más el punto
    # inicial (0, 0) con umbral +inf:
    # FPR = [0, 0, 0, 1/3, 2/3, 1]; TPR = [0, 1/3, 2/3, 1, 1, 1].
    curva = curva_roc(Y_MANO, P_MANO)
    assert isinstance(curva, CurvaROC)
    np.testing.assert_allclose(curva.fpr, [0, 0, 0, 1 / 3, 2 / 3, 1], rtol=0, atol=TOL)
    np.testing.assert_allclose(curva.tpr, [0, 1 / 3, 2 / 3, 1, 1, 1], rtol=0, atol=TOL)
    assert np.isposinf(curva.umbrales[0])
    np.testing.assert_allclose(curva.umbrales[1:], [0.8, 0.6, 0.4, 0.3, 0.1], rtol=0, atol=TOL)


def test_area_trapecio_a_mano():
    # Puntos (0,0), (1,1), (2,1): 1·(0+1)/2 + 1·(1+1)/2 = 0.5 + 1 = 1.5.
    assert area_trapecio(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 1.0])) == pytest.approx(1.5, abs=TOL)


def test_auc_y_gini_a_mano():
    # Trapecio sobre la ROC de test_curva_roc_a_mano:
    # tramos con ancho 0 aportan 0; (0 -> 1/3): (1/3)·(2/3 + 1)/2 = 5/18; (1/3 -> 2/3): (1/3)·1 = 6/18;
    # (2/3 -> 1): (1/3)·1 = 6/18. AUC = 17/18.
    # Comprobación por pares: malos {0.4, 0.6, 0.8}, buenos {0.1, 0.3, 0.4}: 0.8 gana 3, 0.6 gana 3,
    # 0.4 gana 2 y empata 1 (vale 1/2) -> 8.5/9 = 17/18.
    # Gini = 2·17/18 − 1 = 16/18 = 8/9.
    assert auc(Y_MANO, P_MANO) == pytest.approx(17 / 18, abs=TOL)
    assert gini(Y_MANO, P_MANO) == pytest.approx(8 / 9, abs=TOL)


def test_curva_ks_y_ks_a_mano():
    # Umbrales únicos ascendentes: 0.1, 0.3, 0.4, 0.6, 0.8.
    # Malos (p = 0.4, 0.6, 0.8): F₁ = fracción con p <= t = [0, 0, 1/3, 2/3, 1].
    # Buenos (p = 0.1, 0.3, 0.4): F₀ = [1/3, 2/3, 1, 1, 1].
    # |F₁ − F₀| = [1/3, 2/3, 2/3, 1/3, 0] -> KS = 2/3.
    curva = curva_ks(Y_MANO, P_MANO)
    assert isinstance(curva, CurvaKS)
    np.testing.assert_allclose(curva.umbrales, [0.1, 0.3, 0.4, 0.6, 0.8], rtol=0, atol=TOL)
    np.testing.assert_allclose(curva.f_malos, [0, 0, 1 / 3, 2 / 3, 1], rtol=0, atol=TOL)
    np.testing.assert_allclose(curva.f_buenos, [1 / 3, 2 / 3, 1, 1, 1], rtol=0, atol=TOL)
    assert ks(Y_MANO, P_MANO) == pytest.approx(2 / 3, abs=TOL)


def test_brier_a_mano():
    # (0.1−0)² + (0.4−0)² + (0.4−1)² + (0.8−1)² + (0.3−0)² + (0.6−1)²
    # = 0.01 + 0.16 + 0.36 + 0.04 + 0.09 + 0.16 = 0.82; Brier = 0.82 / 6.
    assert brier(Y_MANO, P_MANO) == pytest.approx(0.82 / 6, abs=TOL)


def test_bordes_cuantiles_a_mano():
    # p = 0, 0.25, 0.5, 0.75, 1 y 4 bins: los cuantiles 0, 1/4, 2/4, 3/4, 1 caen exactamente
    # sobre los datos (posiciones 0, 1, 2, 3, 4 de la muestra ordenada) -> los bordes son p mismo.
    p = np.array([0.5, 0.0, 1.0, 0.25, 0.75])
    bordes = bordes_cuantiles(p, 4)
    assert bordes.shape == (5,)
    np.testing.assert_allclose(bordes, [0.0, 0.25, 0.5, 0.75, 1.0], rtol=0, atol=TOL)


def test_bordes_cuantiles_interpolacion_lineal():
    # p ordenado = [0, 1, 2, 4], 2 bins -> cuantiles 0, 1/2, 1. Con interpolación lineal la mediana
    # está en la posición (4 − 1)·0.5 = 1.5, entre 1 y 2 -> 1 + 0.5·(2 − 1) = 1.5. Bordes = [0, 1.5, 4].
    np.testing.assert_allclose(
        bordes_cuantiles(np.array([0.0, 1.0, 2.0, 4.0]), 2), [0.0, 1.5, 4.0], rtol=0, atol=TOL
    )


def test_curva_calibracion_a_mano():
    # 3 bins por cuantiles de P_MANO ordenado = [0.1, 0.3, 0.4, 0.4, 0.6, 0.8]: los bordes internos
    # (cuantiles 1/3 y 2/3, ≈ 0.367 y ≈ 0.467 por interpolación) no coinciden con ningún dato, así que
    # bin 1 = {0.1, 0.3} (y = 0, 0), bin 2 = {0.4, 0.4} (y = 0, 1), bin 3 = {0.6, 0.8} (y = 1, 1).
    # pd_media = [(0.1+0.3)/2, (0.4+0.4)/2, (0.6+0.8)/2] = [0.2, 0.4, 0.7]
    # tasa_observada = [0/2, 1/2, 2/2] = [0, 0.5, 1]; n_por_bin = [2, 2, 2].
    cal = curva_calibracion(Y_MANO, P_MANO, n_bins=3)
    np.testing.assert_allclose(cal.pd_media, [0.2, 0.4, 0.7], rtol=0, atol=TOL)
    np.testing.assert_allclose(cal.tasa_observada, [0.0, 0.5, 1.0], rtol=0, atol=TOL)
    np.testing.assert_array_equal(cal.n_por_bin, [2, 2, 2])


# --------------------------------------------------------------------------------------------
# 3. Propiedades
# --------------------------------------------------------------------------------------------
def test_auc_y_ks_invariantes_ante_transformacion_monotona(datos):
    y, p = datos
    p = np.clip(p, 1e-6, 1 - 1e-6)  # logit finito (con empates hay p redondeados a 0 o 1)
    np.testing.assert_allclose(auc(y, logit(p)), auc(y, p), rtol=0, atol=TOL)
    np.testing.assert_allclose(ks(y, logit(p)), ks(y, p), rtol=0, atol=TOL)


def test_gini_es_dos_auc_menos_uno(datos):
    y, p = datos
    np.testing.assert_allclose(gini(y, p), 2 * auc(y, p) - 1, rtol=0, atol=TOL)


def test_tpr_y_fpr_no_decrecientes_de_0_a_1(datos):
    y, p = datos
    curva = curva_roc(y, p)
    for tasa in (curva.tpr, curva.fpr):
        assert np.all(np.diff(tasa) >= 0)
        assert tasa[0] == pytest.approx(0.0, abs=TOL)
        assert tasa[-1] == pytest.approx(1.0, abs=TOL)


def test_n_por_bin_suma_n(datos):
    y, p = datos
    cal = curva_calibracion(y, p, n_bins=10)
    assert int(np.sum(cal.n_por_bin)) == len(y)


# --------------------------------------------------------------------------------------------
# 4. Inversión de dependencias
# --------------------------------------------------------------------------------------------
def test_gini_usa_auc_fn_inyectado():
    # auc falso = 0.75 -> Gini = 2·0.75 − 1 = 0.5.
    llamadas = []

    def auc_falso(y, p):
        llamadas.append(1)
        return 0.75

    assert gini(Y_MANO, P_MANO, auc_fn=auc_falso) == pytest.approx(0.5, abs=TOL)
    assert llamadas


def test_auc_usa_curva_inyectada():
    # Curva falsa (0,0) -> (0,1) -> (1,1): trapecio = 0·(0+1)/2 + 1·(1+1)/2 = 1,
    # distinto de la AUC real de los datos a mano (17/18).
    def curva_falsa(y, p):
        return CurvaROC(
            fpr=np.array([0.0, 0.0, 1.0]),
            tpr=np.array([0.0, 1.0, 1.0]),
            umbrales=np.array([np.inf, 0.5, 0.1]),
        )

    assert auc(Y_MANO, P_MANO, curva=curva_falsa) == pytest.approx(1.0, abs=TOL)


def test_auc_usa_integrador_inyectado():
    # El integrador falso registra lo que recibe y devuelve una constante: auc debe devolver esa
    # constante y pasarle (fpr, tpr) de la curva.
    recibido = {}

    def integrador_falso(x, y):
        recibido["x"], recibido["y"] = np.asarray(x), np.asarray(y)
        return 0.123

    assert auc(Y_MANO, P_MANO, integrador=integrador_falso) == pytest.approx(0.123, abs=TOL)
    curva = curva_roc(Y_MANO, P_MANO)
    np.testing.assert_allclose(recibido["x"], curva.fpr, rtol=0, atol=TOL)
    np.testing.assert_allclose(recibido["y"], curva.tpr, rtol=0, atol=TOL)


def test_ks_usa_curva_inyectada():
    # Curva falsa: |F₁ − F₀| = [|0.2 − 0.9|, |1 − 1|] = [0.7, 0] -> KS = 0.7.
    def curva_falsa(y, p):
        return CurvaKS(
            umbrales=np.array([0.0, 1.0]),
            f_malos=np.array([0.2, 1.0]),
            f_buenos=np.array([0.9, 1.0]),
        )

    assert ks(Y_MANO, P_MANO, curva=curva_falsa) == pytest.approx(0.7, abs=TOL)


def test_curva_calibracion_usa_binning_inyectado():
    # Bordes falsos [0, 0.5, 1] (0.5 no coincide con ningún dato):
    # bin 1 = {0.1, 0.4, 0.4, 0.3} con y = {0, 0, 1, 0}; bin 2 = {0.8, 0.6} con y = {1, 1}.
    # pd_media = [1.2/4, 1.4/2] = [0.3, 0.7]; tasa = [1/4, 2/2] = [0.25, 1]; n = [4, 2].
    recibido = {}

    def binning_falso(p, n_bins):
        recibido["n_bins"] = n_bins
        return np.array([0.0, 0.5, 1.0])

    cal = curva_calibracion(Y_MANO, P_MANO, n_bins=7, binning=binning_falso)
    assert recibido["n_bins"] == 7
    np.testing.assert_allclose(cal.pd_media, [0.3, 0.7], rtol=0, atol=TOL)
    np.testing.assert_allclose(cal.tasa_observada, [0.25, 1.0], rtol=0, atol=TOL)
    np.testing.assert_array_equal(cal.n_por_bin, [4, 2])


# --------------------------------------------------------------------------------------------
# 5. Errores
# --------------------------------------------------------------------------------------------
def test_validar_entradas_devuelve_arrays():
    y, p = validar_entradas([0, 1, 0], [0.2, 0.7, 0.1])
    assert isinstance(y, np.ndarray) and isinstance(p, np.ndarray)
    np.testing.assert_array_equal(y, [0, 1, 0])
    np.testing.assert_allclose(p, [0.2, 0.7, 0.1], rtol=0, atol=TOL)


@pytest.mark.parametrize(
    "y, p",
    [
        pytest.param([0, 1, 0], [0.2, 0.7], id="longitudes_distintas"),
        pytest.param([], [], id="vacio"),
        pytest.param([0, 1, 0], [0.2, np.nan, 0.1], id="nan_en_p"),
        pytest.param([0, np.nan, 1], [0.2, 0.7, 0.1], id="nan_en_y"),
        pytest.param([1, 1, 1], [0.2, 0.7, 0.1], id="una_sola_clase"),
        pytest.param([0, 2, 1], [0.2, 0.7, 0.1], id="y_fuera_de_0_1"),
    ],
)
def test_validar_entradas_rechaza(y, p):
    with pytest.raises(ValueError):
        validar_entradas(y, p)


@pytest.mark.parametrize(
    "funcion", [auc, gini, ks, brier, curva_roc, curva_ks, curva_calibracion]
)
def test_funciones_validan_entradas(funcion):
    with pytest.raises(ValueError):
        funcion(np.array([0, 1, 0]), np.array([0.2, 0.7]))  # longitudes distintas
    with pytest.raises(ValueError):
        funcion(np.array([0, 0, 0]), np.array([0.2, 0.7, 0.1]))  # una sola clase


@pytest.mark.parametrize("p_malo", [-0.1, 1.1], ids=["negativo", "mayor_que_1"])
def test_brier_rechaza_p_fuera_de_0_1(p_malo):
    with pytest.raises(ValueError):
        brier(np.array([0, 1, 0]), np.array([0.2, p_malo, 0.1]))


@pytest.mark.parametrize("p_malo", [-0.1, 1.1], ids=["negativo", "mayor_que_1"])
def test_curva_calibracion_rechaza_p_fuera_de_0_1(p_malo):
    with pytest.raises(ValueError):
        curva_calibracion(np.array([0, 1, 0]), np.array([0.2, p_malo, 0.1]))
