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
# # Pruebas de `src/pd/umbral.py`: de la PD a la decisión (B11, B-pd §3.9)
#
# Lógica pura, con datos sintéticos generados aquí: matrices de confusión a mano, conservación de malos y buenos,
# cuotas extremas, monotonía, empates en el borde de la cuota (orden estable), relación con el KS, costo y
# $p^\* = 1/(1 + L/G)$, desempate de la cuota óptima, metas de sensibilidad, Brier del trivial contra su forma cerrada
# y los errores de entrada (NaN, $\pm\infty$, cuota fuera de $[0, 1]$, razón $\le 0$, vacío, una sola clase).
#
# Se usa el módulo (`u.<nombre>`): cada prueba falla por separado mientras su función no existe.

# %% [markdown]
# ## Importaciones

# %%
import math  # infinito y comparaciones de flotantes

import numpy as np  # datos sintéticos
import pytest  # aserciones y parametrización

from src.common.metricas import brier, ks  # métricas de B1 contra las que se contrasta
from src.pd import umbral as u  # módulo bajo prueba

# %% [markdown]
# ## Datos de prueba
#
# Semilla de los datos sintéticos (valor de prueba, no del dominio) y un caso pequeño hecho a mano: 6 créditos, 3
# malos, PD sin empates.

# %%
SEMILLA = 20261030  # fecha de la tarea
Y_MANO = np.array([1, 0, 1, 0, 1, 0])  # 3 malos y 3 buenos
P_MANO = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])  # PD decreciente: orden = posición


# %% [markdown]
# Muestra sintética **calibrada**: $p_i \sim U(0, 1)$ y $y_i \sim \mathrm{Bernoulli}(p_i)$, de modo que
# $P(y = 1 \mid p) = p$.

# %%
def _calibrada(n: int = 200_000, semilla: int = SEMILLA) -> tuple[np.ndarray, np.ndarray]:
    """(y, p) con P(y = 1 | p) = p."""
    rng = np.random.default_rng(semilla)  # generador reproducible
    p = rng.uniform(0.0, 1.0, n)  # PD uniforme
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int)  # Bernoulli(p)
    return y, p  # muestra calibrada


# %% [markdown]
# ## $k$ de una cuota
#
# $k = \lceil \mathrm{round}(r \cdot n, 9) \rceil$: 0,07·100 da 7 (sin el redondeo daría 8) y una fracción
# redondea hacia arriba.

# %%
@pytest.mark.parametrize("cuota, n, k", [(0.07, 100, 7), (0.0, 10, 0), (1.0, 10, 10), (0.15, 10, 2),  # casos
                                         (0.29, 100, 29), (0.57, 100, 57), (0.01, 1, 1)])  # flotantes delicados
def test_k_de_cuota(cuota, n, k):
    """k = ⌈round(r·n, 9)⌉."""
    assert u.k_de_cuota(cuota, n) == k  # igual al esperado


# %% [markdown]
# ## Rechazo por cuota
#
# Rechaza los $k$ de mayor PD; con empates en el borde, primero la fila que aparece antes (orden estable).

# %%
def test_rechazos_por_cuota_a_mano():
    """Cuota 0,5 de 6 → las 3 primeras (mayor PD)."""
    assert u.rechazos_por_cuota(P_MANO, 0.5).tolist() == [True, True, True, False, False, False]  # 3 rechazos


# %% [markdown]
# Empate en el borde: tres PD iguales y $k = 2$ → se rechazan las dos que aparecen antes.

# %%
def test_rechazos_por_cuota_empate_estable():
    """Con PD empatadas en el borde, rechaza las primeras en aparecer."""
    p = np.array([0.2, 0.5, 0.9, 0.5, 0.5])  # 0,9 y tres empates en 0,5
    assert u.rechazos_por_cuota(p, 0.6).tolist() == [False, True, True, True, False]  # k = 3: 0,9 y dos 0,5


# %% [markdown]
# Cuota 0 no rechaza a nadie; cuota 1 rechaza a todos.

# %%
def test_rechazos_por_cuota_extremos():
    """r = 0 → ninguno; r = 1 → todos."""
    assert not u.rechazos_por_cuota(P_MANO, 0.0).any()  # ninguno
    assert u.rechazos_por_cuota(P_MANO, 1.0).all()  # todos


# %% [markdown]
# ## Umbral de una cuota
#
# La menor PD rechazada; $+\infty$ si $k = 0$.

# %%
def test_umbral_de_cuota():
    """Menor PD rechazada y +inf con k = 0."""
    assert u.umbral_de_cuota(P_MANO, 0.5) == 0.7  # la tercera
    assert u.umbral_de_cuota(P_MANO, 0.0) == math.inf  # nadie rechazado
    assert u.umbral_de_cuota(P_MANO, 1.0) == 0.1  # la menor


# %% [markdown]
# Sin empates, el umbral de la cuota aplicado a los mismos datos reproduce la cuota; con empates en el borde rechaza
# de más (efecto que se documenta en el blog).

# %%
def test_umbral_de_cuota_reproduce_la_cuota_sin_empates_y_rechaza_de_mas_con_empates():
    """p ≥ umbral_de_cuota(p, r) = rechazos_por_cuota(p, r) sin empates; con empates, más rechazos."""
    t = u.umbral_de_cuota(P_MANO, 0.5)  # 0,7
    assert np.array_equal(u.rechazos_por_umbral(P_MANO, t), u.rechazos_por_cuota(P_MANO, 0.5))  # iguales
    p = np.array([0.2, 0.5, 0.9, 0.5, 0.5])  # empates en el borde
    assert u.rechazos_por_umbral(p, u.umbral_de_cuota(p, 0.6)).sum() == 4  # 4 > k = 3


# %% [markdown]
# ## Rechazo por umbral
#
# Rechaza si $p \ge t$ (el borde se rechaza); $t = +\infty$ no rechaza a nadie.

# %%
def test_rechazos_por_umbral():
    """p ≥ t, con el borde incluido."""
    assert u.rechazos_por_umbral(P_MANO, 0.7).tolist() == [True, True, True, False, False, False]  # borde dentro
    assert not u.rechazos_por_umbral(P_MANO, math.inf).any()  # +inf: ninguno
    assert u.rechazos_por_umbral(P_MANO, 0.0).all()  # 0: todos


# %% [markdown]
# Umbral inválido: NaN, $-\infty$, negativo o mayor que 1 (finito) → `ValueError`.

# %%
@pytest.mark.parametrize("t", [math.nan, -math.inf, -0.1, 1.5, True, "0.5"])  # inválidos
def test_rechazos_por_umbral_umbral_invalido(t):
    """t fuera de [0, 1] ∪ {+inf} → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        u.rechazos_por_umbral(P_MANO, t)  # inválido


# %% [markdown]
# ## Matriz de confusión
#
# Positivo = malo; VP = malos rechazados, FN = malos aprobados, FP = buenos rechazados, VN = buenos aprobados.

# %%
def test_matriz_confusion_a_mano():
    """Cuota 0,5 en el caso a mano: rechaza 0,9 (malo), 0,8 (bueno) y 0,7 (malo)."""
    m = u.matriz_confusion(Y_MANO, u.rechazos_por_cuota(P_MANO, 0.5))  # matriz
    assert m == {"vp": 2, "fn": 1, "fp": 1, "vn": 2}  # a mano


# %% [markdown]
# Conservación: $VP + FN$ = malos y $FP + VN$ = buenos, en cualquier cuota; los conteos son `int` nativos.

# %%
@pytest.mark.parametrize("cuota", [0.0, 0.13, 0.5, 0.87, 1.0])  # cuotas variadas
def test_matriz_conserva_malos_y_buenos(cuota):
    """vp + fn = P y fp + vn = N."""
    y, p = _calibrada(1000)  # muestra pequeña
    m = u.matriz_confusion(y, u.rechazos_por_cuota(p, cuota))  # matriz
    assert m["vp"] + m["fn"] == y.sum()  # malos
    assert m["fp"] + m["vn"] == (1 - y).sum()  # buenos
    assert all(type(v) is int for v in m.values())  # int nativos (van al JSON)


# %% [markdown]
# Cuota 0 → FN = malos (se aprueba a todos); cuota 1 → FN = 0 (se rechaza a todos).

# %%
def test_matriz_cuotas_extremas():
    """r = 0 → fn = P; r = 1 → fn = 0 y fp = N."""
    m0 = u.matriz_confusion(Y_MANO, u.rechazos_por_cuota(P_MANO, 0.0))  # aprueba a todos
    m1 = u.matriz_confusion(Y_MANO, u.rechazos_por_cuota(P_MANO, 1.0))  # rechaza a todos
    assert m0 == {"vp": 0, "fn": 3, "fp": 0, "vn": 3}  # todos aprobados
    assert m1 == {"vp": 3, "fn": 0, "fp": 3, "vn": 0}  # todos rechazados


# %% [markdown]
# Monotonía: al subir la cuota, FN no sube y FP no baja.

# %%
def test_monotonia_en_la_cuota():
    """fn no creciente y fp no decreciente en la cuota."""
    y, p = _calibrada(2000)  # muestra
    ms = [u.matriz_confusion(y, u.rechazos_por_cuota(p, r)) for r in np.linspace(0, 1, 51)]  # una por cuota
    assert all(a["fn"] >= b["fn"] for a, b in zip(ms, ms[1:]))  # fn no sube
    assert all(a["fp"] <= b["fp"] for a, b in zip(ms, ms[1:]))  # fp no baja


# %% [markdown]
# La máscara debe ser booleana y del mismo largo que $y$.

# %%
@pytest.mark.parametrize("rechazo", [np.array([1, 0, 1, 0, 1, 0]), np.array([True, False])])  # no bool / largo
def test_matriz_rechazo_invalido(rechazo):
    """Máscara no booleana o de otro largo → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        u.matriz_confusion(Y_MANO, rechazo)  # inválida


# %% [markdown]
# ## Tasas
#
# $\mathrm{TPR} = VP/(VP+FN)$, $\mathrm{TNR} = VN/(VN+FP)$, tasa de rechazo $=(VP+FP)/n$ y de aprobación
# $= 1 -$ rechazo.

# %%
def test_tasas_a_mano():
    """Tasas de la matriz a mano."""
    t = u.tasas({"vp": 2, "fn": 1, "fp": 1, "vn": 2})  # matriz de la cuota 0,5
    assert t == pytest.approx({"tpr": 2 / 3, "tnr": 2 / 3, "tasa_rechazo": 0.5, "tasa_aprobacion": 0.5})  # a mano


# %% [markdown]
# Sin malos o sin buenos la TPR o la TNR no están definidas → `ValueError`.

# %%
@pytest.mark.parametrize("m", [{"vp": 0, "fn": 0, "fp": 1, "vn": 2}, {"vp": 1, "fn": 1, "fp": 0, "vn": 0},  # 1 clase
                               {"vp": -1, "fn": 2, "fp": 1, "vn": 1}, {"vp": 1, "fn": 1, "fp": 1}])  # negativo, falta
def test_tasas_invalidas(m):
    """Una clase vacía, un conteo negativo o una clave de menos → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        u.tasas(m)  # inválida


# %% [markdown]
# ## Relación con el KS
#
# $\mathrm{KS} = \max_t (\mathrm{TPR}(t) + \mathrm{TNR}(t) - 1)$ al recorrer todos los umbrales $t$ (los valores
# únicos de $p$), contra `metricas.ks`.

# %%
def test_ks_es_el_maximo_de_tpr_mas_tnr_menos_uno():
    """máx_t (TPR + TNR − 1) = KS de metricas.py en un modelo que discrimina."""
    y, p = _calibrada(5000)  # calibrada: discrimina
    curva = u.curva_sensibilidad(y, p, np.unique(p))  # todos los umbrales
    j = np.asarray(curva["tpr"]) + np.asarray(curva["tnr"]) - 1  # Youden por umbral
    assert j.max() == pytest.approx(ks(y, p), abs=1e-12)  # igual al KS


# %% [markdown]
# ## Curva de sensibilidad al umbral
#
# Para cada umbral, las tasas de la matriz con `rechazos_por_umbral` (camino lento, una máscara por umbral).

# %%
def test_curva_sensibilidad_igual_a_la_matriz():
    """Cada punto de la curva = tasas(matriz(p ≥ t))."""
    y, p = _calibrada(3000)  # muestra
    umbrales = np.linspace(0, 1, 21)  # rejilla
    curva = u.curva_sensibilidad(y, p, umbrales)  # camino rápido
    for i, t in enumerate(umbrales):  # cada umbral
        esperado = u.tasas(u.matriz_confusion(y, u.rechazos_por_umbral(p, t)))  # camino lento
        assert curva["tpr"][i] == pytest.approx(esperado["tpr"])  # TPR
        assert curva["tnr"][i] == pytest.approx(esperado["tnr"])  # TNR
        assert curva["tasa_rechazo"][i] == pytest.approx(esperado["tasa_rechazo"])  # rechazo
    assert set(curva) == {"umbrales", "tpr", "tnr", "tasa_rechazo"}  # claves exactas
    assert all(isinstance(v, float) for v in curva["tpr"])  # float nativos (van al JSON)


# %% [markdown]
# ## Costo
#
# $\text{costo} = (L/G) \cdot FN + FP$ (en unidades de $G$) y por crédito $/n$.

# %%
def test_costo_a_mano():
    """costo = razón·fn + fp; por crédito /n."""
    c = u.costo({"vp": 2, "fn": 1, "fp": 1, "vn": 2}, 5.0)  # razón 5
    assert c == pytest.approx({"costo": 6.0, "costo_por_credito": 1.0})  # 5·1 + 1 y /6


# %% [markdown]
# Razón inválida: $\le 0$, NaN, $\pm\infty$, booleana o texto → `ValueError`.

# %%
@pytest.mark.parametrize("razon", [0.0, -1.0, math.nan, math.inf, -math.inf, True, "5"])  # inválidas
def test_costo_razon_invalida(razon):
    """razón fuera de (0, ∞) → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        u.costo({"vp": 2, "fn": 1, "fp": 1, "vn": 2}, razon)  # inválida


# %% [markdown]
# ## Umbral teórico
#
# $p^\* = 1/(1 + L/G)$: con $L/G = 1$, 0,5; con $L/G = 9$, 0,1.

# %%
def test_umbral_teorico():
    """p* = 1/(1 + razón)."""
    assert u.umbral_teorico(1.0) == pytest.approx(0.5)  # simétrico
    assert u.umbral_teorico(9.0) == pytest.approx(0.1)  # malo 9 veces más caro
    with pytest.raises(ValueError):  # razón inválida
        u.umbral_teorico(0.0)  # no positiva


# %% [markdown]
# ## Conteos por cuota (camino rápido)
#
# Iguales a la matriz de cada cuota calculada con la máscara (camino lento), también con empates.

# %%
def test_conteos_por_cuota_igual_a_la_matriz():
    """conteos_por_cuota = matriz_confusion(rechazos_por_cuota) en cada cuota."""
    rng = np.random.default_rng(SEMILLA)  # generador
    p = np.round(rng.uniform(0, 1, 500), 1)  # muchos empates
    y = (rng.uniform(0, 1, 500) < p).astype(int)  # objetivo
    rejilla = np.round(np.linspace(0, 1, 101), 9)  # cuotas
    conteos = u.conteos_por_cuota(y, p, rejilla)  # rápido
    for i, r in enumerate(rejilla):  # cada cuota
        m = u.matriz_confusion(y, u.rechazos_por_cuota(p, r))  # lento
        assert {k: int(conteos[k][i]) for k in m} == m  # iguales


# %% [markdown]
# ## Cuota óptima
#
# El argmin del costo observado; un empate exacto → la menor cuota.

# %%
def test_cuota_optima_a_mano_y_desempate():
    """Razón 1: mínimo único en 0,5; razón 0,5: empate exacto entre 0 y 0,5 → la menor."""
    rejilla = [0.0, 0.5, 1.0]  # tres cuotas
    # r = 0: fn 3 fp 0 → 3; r = 0,5: fn 1 fp 1 → 2; r = 1: fn 0 fp 3 → 3 (razón 1).
    assert u.cuota_optima(Y_MANO, P_MANO, 1.0, rejilla) == 0.5  # mínimo único
    # razón 0,5: r = 0 → 1,5; r = 0,5 → 1,5; r = 1 → 3: empate exacto entre 0 y 0,5 → la menor.
    assert u.cuota_optima(Y_MANO, P_MANO, 0.5, rejilla) == 0.0  # desempate


# %% [markdown]
# Razones extremas: $L/G \to \infty$ lleva a la menor cuota sin FN (la que atrapa a todos los malos: cuota 1 si el
# crédito de menor PD es malo) y $L/G \to 0$ a la cuota sin FP (cuota 0 si el crédito de mayor PD es bueno).

# %%
def test_cuota_optima_razones_extremas():
    """razón enorme → menor cuota con TPR = 1; razón diminuta → FP = 0."""
    y, p = _calibrada(2000)  # muestra
    rejilla = np.round(np.linspace(0, 1, 101), 9)  # cuotas
    assert u.cuota_optima(y, p, 1e9, rejilla) == u.cuota_para_meta(y, p, 1.0, rejilla)  # FN carísimo: FN = 0
    r0 = u.cuota_optima(y, p, 1e-9, rejilla)  # FN casi gratis
    assert u.matriz_confusion(y, u.rechazos_por_cuota(p, r0))["fp"] == 0  # no rechaza a ningún bueno
    y_alto = y.copy()  # el de mayor PD pasa a ser bueno
    y_alto[np.argmax(p)] = 0  # bueno con la mayor PD
    assert u.cuota_optima(y_alto, p, 1e-9, rejilla) == 0.0  # aprobar a todos
    y_bajo = y.copy()  # el de menor PD pasa a ser malo
    y_bajo[np.argmin(p)] = 1  # malo con la menor PD
    assert u.cuota_optima(y_bajo, p, 1e9, rejilla) == 1.0  # hay que rechazar a todos


# %% [markdown]
# Con PD calibrada, el corte óptimo observado coincide con $p^\*$: la menor PD rechazada en la cuota óptima queda
# cerca de $1/(1 + L/G)$.

# %%
@pytest.mark.parametrize("razon", [2.0, 5.0, 10.0])  # escenarios de la config
def test_p_estrella_con_pd_calibrada(razon):
    """umbral_de_cuota(p, r*) ≈ 1/(1 + razón) con PD calibrada."""
    y, p = _calibrada()  # 200 000 créditos calibrados
    rejilla = np.round(np.linspace(0, 1, 1001), 9)  # rejilla fina
    r = u.cuota_optima(y, p, razon, rejilla)  # cuota óptima observada
    assert u.umbral_de_cuota(p, r) == pytest.approx(u.umbral_teorico(razon), abs=0.02)  # cerca de p*


# %% [markdown]
# La rejilla debe ser estrictamente creciente y estar en $[0, 1]$; vacía → error.

# %%
@pytest.mark.parametrize("rejilla", [[], [0.5, 0.2], [0.1, 0.1], [-0.1, 0.5], [0.5, 1.1], [0.0, math.nan]])  # malas
def test_cuota_optima_rejilla_invalida(rejilla):
    """Rejilla vacía, no creciente o fuera de [0, 1] → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        u.cuota_optima(Y_MANO, P_MANO, 1.0, rejilla)  # inválida


# %% [markdown]
# ## Cuota para una meta de sensibilidad
#
# La menor cuota de la rejilla con $\mathrm{TPR} \ge m$; si ninguna la alcanza, `None`.

# %%
def test_cuota_para_meta():
    """Menor cuota con TPR ≥ meta en el caso a mano."""
    rejilla = [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]  # cuotas de 0 a 6 créditos
    assert u.cuota_para_meta(Y_MANO, P_MANO, 1 / 3, rejilla) == 1 / 6  # 1 malo de 3
    assert u.cuota_para_meta(Y_MANO, P_MANO, 0.5, rejilla) == 3 / 6  # 2 malos: hace falta 0,5
    assert u.cuota_para_meta(Y_MANO, P_MANO, 1.0, rejilla) == 5 / 6  # el último malo está en la quinta
    assert u.cuota_para_meta(Y_MANO, P_MANO, 1.0, [0.0, 0.5]) is None  # inalcanzable


# %% [markdown]
# Meta fuera de $[0, 1]$ o no finita → `ValueError`.

# %%
@pytest.mark.parametrize("meta", [-0.1, 1.1, math.nan, True])  # inválidas
def test_cuota_para_meta_invalida(meta):
    """meta fuera de [0, 1] → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        u.cuota_para_meta(Y_MANO, P_MANO, meta, [0.0, 1.0])  # inválida


# %% [markdown]
# ## Brier del modelo trivial
#
# Con PD constante $q$ y tasa $\pi$ del conjunto: $\mathrm{Brier} = \pi(1-\pi) + (q-\pi)^2$; con $q = \pi$ queda el
# piso $\pi(1 - \pi)$.

# %%
@pytest.mark.parametrize("q", [0.0, 0.163, 0.262, 0.5, 1.0])  # PD constantes
def test_brier_trivial_forma_cerrada(q):
    """brier_trivial(y, q) = π(1−π) + (q−π)² = brier con PD constante."""
    y, _ = _calibrada(10_000)  # objetivo
    pi = y.mean()  # tasa del conjunto
    assert u.brier_trivial(y, q) == pytest.approx(pi * (1 - pi) + (q - pi) ** 2, abs=1e-12)  # forma cerrada
    assert u.brier_trivial(y, q) == brier(y, np.full(y.shape, q))  # misma función de B1


# %% [markdown]
# $q$ fuera de $[0, 1]$ → `ValueError` (lo rechaza `metricas.brier`).

# %%
@pytest.mark.parametrize("q", [-0.1, 1.1, math.nan])  # inválidas
def test_brier_trivial_q_invalida(q):
    """q fuera de [0, 1] → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        u.brier_trivial(Y_MANO, q)  # inválida


# %% [markdown]
# ## Errores de entrada
#
# PD con NaN, $\pm\infty$, fuera de $[0, 1]$, vacía o 2-D → `ValueError` en las funciones que solo reciben $p$; con
# $y$ además: una sola clase o largo distinto.

# %%
@pytest.mark.parametrize("p", [[0.1, math.nan], [0.1, math.inf], [0.1, -math.inf], [0.1, 1.2], [-0.1, 0.2], [],
                               [[0.1, 0.2]]])  # PD inválidas
def test_pd_invalida(p):
    """rechazos_por_cuota, rechazos_por_umbral y umbral_de_cuota rechazan la PD inválida."""
    for f in (lambda: u.rechazos_por_cuota(p, 0.5), lambda: u.rechazos_por_umbral(p, 0.5),  # por cuota y umbral
              lambda: u.umbral_de_cuota(p, 0.5)):  # umbral de la cuota
        with pytest.raises(ValueError):  # error esperado
            f()  # inválida


# %% [markdown]
# Cuota fuera de $[0, 1]$, NaN, booleana o texto → `ValueError`.

# %%
@pytest.mark.parametrize("cuota", [-0.01, 1.01, math.nan, math.inf, True, "0.5", None])  # inválidas
def test_cuota_invalida(cuota):
    """cuota fuera de [0, 1] → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        u.rechazos_por_cuota(P_MANO, cuota)  # inválida


# %% [markdown]
# Con $y$: una sola clase, largos distintos o $y$ fuera de $\{0, 1\}$ → `ValueError` (de `metricas.validar_entradas`).

# %%
@pytest.mark.parametrize("y, p", [([0, 0, 0], [0.1, 0.2, 0.3]), ([1, 1], [0.1, 0.2]), ([0, 1], [0.1]),  # malas
                                  ([0, 2], [0.1, 0.2]), ([0, 1], [0.1, math.nan])])  # y o p inválidos
def test_entradas_con_y_invalidas(y, p):
    """cuota_optima, cuota_para_meta, curva_sensibilidad y conteos_por_cuota rechazan (y, p) inválidos."""
    for f in (lambda: u.cuota_optima(y, p, 1.0, [0.0, 1.0]), lambda: u.cuota_para_meta(y, p, 0.5, [0.0, 1.0]),  # dos
              lambda: u.curva_sensibilidad(y, p, [0.5]), lambda: u.conteos_por_cuota(y, p, [0.0])):  # dos más
        with pytest.raises(ValueError):  # error esperado
            f()  # inválida


# %% [markdown]
# Un solo crédito no tiene las dos clases → error; dos créditos (uno de cada clase) funcionan.

# %%
def test_n_minimo():
    """n = 2 con las dos clases: cuota 0,5 rechaza al de mayor PD."""
    m = u.matriz_confusion([1, 0], u.rechazos_por_cuota([0.6, 0.4], 0.5))  # rechaza al de 0,6 (malo)
    assert m == {"vp": 1, "fn": 0, "fp": 0, "vn": 1}  # matriz perfecta
    with pytest.raises(ValueError):  # una sola fila: una sola clase
        u.matriz_confusion([1], np.array([True]))  # inválida


# %% [markdown]
# Las funciones no mutan sus entradas.

# %%
def test_no_mutan_entradas():
    """p e y quedan intactos."""
    p, y = P_MANO.copy(), Y_MANO.copy()  # copias
    u.rechazos_por_cuota(p, 0.5)  # cuota
    u.cuota_optima(y, p, 2.0, [0.0, 0.5, 1.0])  # óptima
    u.curva_sensibilidad(y, p, [0.5])  # curva
    assert np.array_equal(p, P_MANO) and np.array_equal(y, Y_MANO)  # intactos


# %% [markdown]
# ## Cuotas desde conteos ya calculados
#
# `cuota_optima_conteos` y `cuota_para_meta_conteos` reciben los conteos de `conteos_por_cuota` (un solo ordenamiento
# por modelo) y dan exactamente lo mismo que las versiones directas para cualquier razón y meta.

# %%
def test_cuotas_desde_conteos_igual_a_la_directa():
    """Misma cuota que cuota_optima y cuota_para_meta en razones y metas variadas."""
    y, p = _calibrada(3000)  # muestra
    rejilla = np.round(np.linspace(0, 1, 101), 9)  # cuotas
    c = u.conteos_por_cuota(y, p, rejilla)  # una vez
    for razon in (1e-9, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 1e9):  # razones, incluidas las extremas
        assert u.cuota_optima_conteos(c, rejilla, razon) == u.cuota_optima(y, p, razon, rejilla)  # igual
    for meta in (0.0, 0.3, 0.5, 0.8, 0.9, 1.0):  # metas
        assert u.cuota_para_meta_conteos(c, rejilla, meta) == u.cuota_para_meta(y, p, meta, rejilla)  # igual
    corta = rejilla[:2]  # cuotas 0 y 0,01
    assert u.cuota_para_meta_conteos(u.conteos_por_cuota(y, p, corta), corta, 1.0) is None  # inalcanzable


# %% [markdown]
# Conteos de otro largo que la rejilla, sin una clave, o razón y meta inválidas → `ValueError`.

# %%
def test_cuotas_desde_conteos_invalidas():
    """Largo distinto, clave de menos, razón ≤ 0 o meta > 1 → ValueError."""
    rejilla = [0.0, 0.5, 1.0]  # tres cuotas
    c = u.conteos_por_cuota(Y_MANO, P_MANO, rejilla)  # conteos
    for f in (lambda: u.cuota_optima_conteos(c, [0.0, 1.0], 1.0),  # largo distinto
              lambda: u.cuota_optima_conteos({"fn": c["fn"]}, rejilla, 1.0),  # falta fp
              lambda: u.cuota_optima_conteos(c, rejilla, 0.0),  # razón inválida
              lambda: u.cuota_para_meta_conteos(c, rejilla, 1.5)):  # meta inválida
        with pytest.raises(ValueError):  # error esperado
            f()  # inválida
