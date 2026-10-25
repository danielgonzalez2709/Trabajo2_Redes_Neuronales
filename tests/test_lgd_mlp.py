"""Pruebas de D3: referencia lineal y MLP de LGD (Parte D §3.3; núcleo §5.3).

Datos sintéticos pequeños generados aquí (no se construye la variable objetivo real: eso es D1/D2).
"""
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

from src.common.config import load_config
from src.lgd.lineal import LinealLGD
from src.lgd.mlp import CLAVES_MLP, MLPLGD
from src.lgd.perdidas import PERDIDAS_VALIDAS, obtener_perdida, perdida_bce_fraccional, perdida_mse

RAIZ = Path(__file__).resolve().parents[1]
N_FEATURES = 5


def _cfg(**cambios) -> dict:
    """Config pequeña y rápida para las pruebas (no son los valores por defecto del proyecto)."""
    mlp = {
        "capas": [8, 4],
        "activacion": "relu",
        "dropout": 0.0,
        "l2": 0.0,
        "optimizador": "adam",
        "lr": 0.01,
        "momento": 0.9,
        "batch": 64,
        "max_epocas": 15,
        "paciencia": 5,
        "device": "cpu",
        "perdida": "bce_fraccional",
    }
    mlp.update(cambios)
    return {"mlp": mlp}


def _datos(n: int, semilla: int = 123):
    """E[y|x] = σ(w·x) con y fraccional ~ Beta de esa media."""
    rng = np.random.default_rng(semilla)
    w = np.array([1.5, -2.0, 1.0, 0.0, 0.5])
    X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    mu = 1.0 / (1.0 + np.exp(-(X @ w)))
    phi = 20.0
    y = rng.beta(mu * phi, (1.0 - mu) * phi)
    return X, y


@pytest.fixture(scope="module")
def datos():
    X_tr, y_tr = _datos(1500, 1)
    X_val, y_val = _datos(500, 2)
    return X_tr, y_tr, X_val, y_val


# 1. predict en [0, 1] con shape (n,)
def test_predict_mlp_shape_y_rango(datos):
    X_tr, y_tr, X_val, y_val = datos
    m = MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(max_epocas=3), seed=0)
    p = m.predict(X_val)
    assert p.shape == (len(X_val),)
    assert np.all((p >= 0) & (p <= 1))
    extremos = np.full((4, N_FEATURES), 1e6, dtype=np.float32)
    extremos[1] *= -1
    pe = m.predict(extremos)
    assert pe.shape == (4,)
    assert np.all(np.isfinite(pe)) and np.all((pe >= 0) & (pe <= 1))


# 2. pérdida fraccional
def test_bce_fraccional_formula_y_gradiente():
    z = torch.tensor([0.2, -1.0, 2.5], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.3, 0.0, 0.85], dtype=torch.float64)
    perdida = perdida_bce_fraccional(z, y)
    mu = 1.0 / (1.0 + np.exp(-z.detach().numpy()))
    yn = y.numpy()
    a_mano = -np.mean(yn * np.log(mu) + (1 - yn) * np.log(1 - mu))
    assert perdida.item() == pytest.approx(a_mano, rel=1e-12)
    perdida.backward()
    # d(-(1/n)Σℓ)/dz_i = (μ_i − y_i)/n
    np.testing.assert_allclose(z.grad.numpy() * len(yn), mu - yn, rtol=1e-10)


def test_perdida_mse_y_desconocida():
    z = torch.tensor([0.0, 1.0], dtype=torch.float64)
    y = torch.tensor([0.5, 0.2], dtype=torch.float64)
    mu = 1.0 / (1.0 + np.exp(-z.numpy()))
    assert perdida_mse(z, y).item() == pytest.approx(np.mean((mu - y.numpy()) ** 2))
    assert obtener_perdida("mse") is perdida_mse
    assert obtener_perdida("bce_fraccional") is perdida_bce_fraccional
    with pytest.raises(ValueError, match="perdida"):
        obtener_perdida("huber")


# 3. aprende
def test_mlp_aprende(datos):
    X_tr, y_tr, X_val, y_val = datos
    m = MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(max_epocas=30), seed=0)
    mae_mlp = np.mean(np.abs(m.predict(X_val) - y_val))
    mae_media = np.mean(np.abs(np.mean(y_tr) - y_val))
    assert mae_mlp < 0.8 * mae_media


# 4. reproducibilidad
def test_reproducibilidad(datos):
    X_tr, y_tr, X_val, y_val = datos
    cfg = _cfg(max_epocas=3, dropout=0.2)
    p1 = MLPLGD().fit(X_tr, y_tr, X_val, y_val, cfg, seed=7).predict(X_val)
    p2 = MLPLGD().fit(X_tr, y_tr, X_val, y_val, cfg, seed=7).predict(X_val)
    p3 = MLPLGD().fit(X_tr, y_tr, X_val, y_val, cfg, seed=8).predict(X_val)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, p3)


# 5. n_parametros
def test_n_parametros(datos):
    X_tr, y_tr, X_val, y_val = datos
    m = MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(max_epocas=1, capas=[8, 4]), seed=0)
    assert m.n_parametros() == (5 * 8 + 8) + (8 * 4 + 4) + (4 * 1 + 1)
    lin = LinealLGD().fit(X_tr, y_tr, X_val, y_val, {}, seed=0)
    assert lin.n_parametros() == N_FEATURES + 1


# 6. guardar / cargar
def test_guardar_cargar(datos, tmp_path):
    X_tr, y_tr, X_val, y_val = datos
    m = MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(max_epocas=2, dropout=0.3), seed=0)
    ruta = tmp_path / "lgd_mlp.pt"
    m.guardar(ruta)
    m2 = MLPLGD.cargar(ruta)
    assert np.array_equal(m.predict(X_val), m2.predict(X_val))
    assert m2.n_parametros() == m.n_parametros()
    assert m2.epocas_ == m.epocas_ and m2.device_ == m.device_

    lin = LinealLGD().fit(X_tr, y_tr, X_val, y_val, {}, seed=0)
    ruta_l = tmp_path / "lgd_lineal.joblib"
    lin.guardar(ruta_l)
    lin2 = LinealLGD.cargar(ruta_l)
    assert np.array_equal(lin.predict(X_val), lin2.predict(X_val))


# 7. lineal fuera de [0, 1]
def test_lineal_fuera_de_rango():
    """Ajuste exacto y = 0,1·x con y ∈ [0, 1]; al extrapolar, la recta sale de [0, 1] y no se recorta."""
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = 0.1 * X[:, 0]  # 0,0 … 0,9
    lin = LinealLGD().fit(X, y, X, y, {}, seed=0)
    X_nuevo = np.array([-5.0, -1.0, 3.0, 5.0, 12.0, 15.0]).reshape(-1, 1)
    p = lin.predict(X_nuevo)
    np.testing.assert_allclose(p, 0.1 * X_nuevo[:, 0], atol=1e-10)  # no recorta en silencio
    conteo = lin.contar_fuera_de_rango(X_nuevo)
    assert conteo == {"debajo_de_0": 2, "encima_de_1": 2, "total": 4, "n": 6}


# 8. parada temprana
def test_parada_temprana_y_restauracion():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(80, N_FEATURES)).astype(np.float32)
    y_tr = rng.uniform(size=80)
    X_val = rng.normal(size=(80, N_FEATURES)).astype(np.float32)
    y_val = rng.uniform(size=80)  # sin relación con X: la validación deja de mejorar pronto
    cfg = _cfg(capas=[64, 64], lr=0.05, batch=16, max_epocas=300, paciencia=3)
    m = MLPLGD().fit(X_tr, y_tr, X_val, y_val, cfg, seed=0)
    assert m.epocas_ < 300
    assert m.epocas_ == m.mejor_epoca_ + 3
    hist = m.historial_["val"]
    assert len(hist) == m.epocas_
    mejor = min(hist)
    assert hist[m.mejor_epoca_ - 1] == mejor
    # la pérdida de validación del modelo final es la de la mejor época
    p = m.predict(X_val)
    mu = np.clip(p, 1e-12, 1 - 1e-12)
    val_final = -np.mean(y_val * np.log(mu) + (1 - y_val) * np.log(1 - mu))
    assert val_final == pytest.approx(mejor, rel=1e-4)
    assert m.tiempo_s_ > 0 and m.device_ == "cpu"


# 9. mse y errores de config
def test_mse_funciona_y_errores_claros(datos):
    X_tr, y_tr, X_val, y_val = datos
    m = MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(max_epocas=2, perdida="mse"), seed=0)
    p = m.predict(X_val)
    assert np.all((p >= 0) & (p <= 1))
    with pytest.raises(ValueError, match="perdida"):
        MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(perdida="huber"), seed=0)
    cfg = _cfg()
    del cfg["mlp"]["paciencia"]
    with pytest.raises(KeyError, match="paciencia"):
        MLPLGD().fit(X_tr, y_tr, X_val, y_val, cfg, seed=0)
    with pytest.raises(KeyError, match="mlp"):
        MLPLGD().fit(X_tr, y_tr, X_val, y_val, {}, seed=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MLPLGD().fit(X_tr, y_tr * 2, X_val, y_val, _cfg(max_epocas=1), seed=0)


def test_sgd_y_activaciones(datos):
    X_tr, y_tr, X_val, y_val = datos
    for act in ("leaky_relu", "tanh"):
        m = MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(max_epocas=1, activacion=act, optimizador="sgd"), seed=0)
        assert m.predict(X_val).shape == (len(X_val),)
    with pytest.raises(ValueError, match="activacion"):
        MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(activacion="gelu"), seed=0)


def test_config_base_lgd_tiene_bloque_mlp_completo():
    cfg = load_config(RAIZ / "configs" / "lgd" / "base.yaml")
    claves = set(CLAVES_MLP)
    assert claves <= set(cfg["mlp"])
    assert cfg["mlp"]["perdida"] in PERDIDAS_VALIDAS


# ---------------------------------------------------------------- ronda 2
def test_bce_fraccional_minimo_en_mu_igual_y():
    """Para y fraccional fijo, la pérdida en función de μ es mínima en μ = y."""
    mus = np.linspace(0.001, 0.999, 999)
    z = torch.logit(torch.tensor(mus, dtype=torch.float64))
    for y in (0.1, 0.3, 0.5, 0.77, 0.95):
        perdidas = [perdida_bce_fraccional(zi.reshape(1), torch.tensor([y], dtype=torch.float64)).item() for zi in z]
        assert mus[int(np.argmin(perdidas))] == pytest.approx(y, abs=1e-3)


def test_bce_fraccional_estable_con_logits_extremos():
    z = torch.tensor([50.0, -50.0, 50.0, -50.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    perdida = perdida_bce_fraccional(z, y)
    perdida.backward()
    assert torch.isfinite(perdida) and torch.all(torch.isfinite(z.grad))


def test_fit_con_y_en_0_y_1(datos):
    X_tr, y_tr, X_val, y_val = datos
    m = MLPLGD().fit(X_tr, (y_tr > 0.5).astype(float), X_val, (y_val > 0.5).astype(float), _cfg(max_epocas=3), seed=0)
    p = m.predict(X_val)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))


def test_x_no_finita_da_error(datos):
    X_tr, y_tr, X_val, y_val = datos
    X_mal = X_tr.copy()
    X_mal[0, 0] = np.nan
    with pytest.raises(ValueError, match="finit"):
        MLPLGD().fit(X_mal, y_tr, X_val, y_val, _cfg(max_epocas=1), seed=0)
    X_mal[0, 0] = np.inf
    with pytest.raises(ValueError, match="finit"):
        LinealLGD().fit(X_mal, y_tr, X_val, y_val, {}, seed=0)


def test_divergencia_da_runtime_error(datos):
    X_tr, y_tr, X_val, y_val = datos
    cfg = _cfg(optimizador="sgd", lr=1e30, max_epocas=5)
    with pytest.raises(RuntimeError, match="lr"):
        MLPLGD().fit(X_tr * 1e6, y_tr, X_val * 1e6, y_val, cfg, seed=0)


def test_errores_de_forma_y_de_estado(datos, tmp_path):
    X_tr, y_tr, X_val, y_val = datos
    with pytest.raises(ValueError, match="columnas"):
        MLPLGD().fit(X_tr, y_tr, X_val[:, :3], y_val, _cfg(max_epocas=1), seed=0)
    with pytest.raises(ValueError, match="filas"):
        MLPLGD().fit(X_tr, y_tr[:-1], X_val, y_val, _cfg(max_epocas=1), seed=0)
    with pytest.raises(ValueError, match="filas"):
        LinealLGD().fit(X_tr, y_tr[:-1], X_val, y_val, {}, seed=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LinealLGD().fit(X_tr, y_tr + 2, X_val, y_val, {}, seed=0)
    m = MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(max_epocas=1), seed=0)
    with pytest.raises(ValueError, match="columnas"):
        m.predict(X_val[:, :3])
    lin = LinealLGD().fit(X_tr, y_tr, X_val, y_val, {}, seed=0)
    with pytest.raises(ValueError, match="columnas"):
        lin.predict(X_val[:, :3])
    for modelo in (MLPLGD(), LinealLGD()):
        with pytest.raises(RuntimeError, match="fit"):
            modelo.predict(X_val)
        with pytest.raises(RuntimeError, match="fit"):
            modelo.n_parametros()

    ruta = tmp_path / "ajeno.joblib"
    joblib.dump({"no": "es un modelo"}, ruta)
    with pytest.raises(TypeError, match="LinealLGD"):
        LinealLGD.cargar(ruta)


@pytest.mark.parametrize("clave,valor", [("batch", 0), ("max_epocas", 1.5), ("paciencia", True), ("capas", [8, 0]),
                                         ("capas", [8, 2.0]), ("lr", 0.0), ("l2", -1.0), ("momento", -0.1),
                                         ("dropout", 1.0), ("optimizador", "rmsprop"), ("device", "tpu")])
def test_valores_invalidos_de_cfg(datos, clave, valor):
    X_tr, y_tr, X_val, y_val = datos
    with pytest.raises((ValueError, TypeError), match=clave if clave != "capas" else "capa"):
        MLPLGD().fit(X_tr, y_tr, X_val, y_val, _cfg(**{clave: valor}), seed=0)
