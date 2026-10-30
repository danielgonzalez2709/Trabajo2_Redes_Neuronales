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
# # Pruebas de `src/viz/figs_umbral_pd.py` (B11, B-pd §3.9)
#
# Las figuras se dibujan desde un `umbral.json` **sintético** (el análisis de `umbral_experimento.analizar` sobre PD
# generadas aquí): `generar` escribe exactamente sus `ARCHIVOS` en `tmp_path`, lee `SALIDA` al llamarse (la convención
# de los `figs_*`, para poder redirigirla) y las funciones puras validan su entrada. Nunca se escribe en `blog/`.

# %% [markdown]
# ## Importaciones

# %%
import math  # NaN de los umbrales nulos

import numpy as np  # PD sintéticas
import pytest  # fixtures y aserciones
from scipy.special import expit  # σ

from src.common.config import load_config  # config versionada (solo lectura)
from src.common.results import save_json  # umbral.json sintético en disco
from src.pd import umbral_experimento as ue  # analizar y validación del bloque umbral
from src.viz import figs_umbral_pd as fu  # módulo bajo prueba

# %% [markdown]
# ## Datos sintéticos
#
# PD de dos modelos y una semilla en val y test (valores de prueba, no del dominio) y el bloque `umbral` de la config
# versionada.

# %%
SEMILLA = 20261030  # fecha de la tarea
N = {"val": 400, "test": 500}  # filas por split
RUTA_CFG = ue.RAIZ_REPOSITORIO / "configs" / "pd" / "umbral.yaml"  # config versionada


def _umbral() -> dict:
    """Resultado de ue.analizar sobre PD sintéticas (los bloques que leen las figuras)."""
    rng = np.random.default_rng(SEMILLA)  # generador
    preds = {}  # por split
    for s, n in N.items():  # val y test
        z = rng.normal(size=n)  # puntaje latente
        y = (rng.uniform(size=n) < expit(z - 1)).astype(int)  # objetivo
        preds[s] = {"y": y, "ids": np.arange(n).astype(str).astype(object),  # y e ids
                    "logistica": expit(0.8 * z - 1), "mlp": expit(0.9 * z - 1.1)}  # dos modelos
    sem = [{"seed": 0, "tiempo_s": 1.0, "val": preds["val"]["mlp"], "test": preds["test"]["mlp"]}]  # una semilla
    ucfg = ue.validar_cfg_umbral(load_config(RUTA_CFG))["umbral"]  # bloque umbral validado (solo lectura)
    return ue.analizar(preds, 0.2, sem, ucfg)  # bloques


# %% [markdown]
# ## 1. `generar` escribe exactamente sus archivos
#
# Con `datos` en memoria y `destino` en `tmp_path`.

# %%
def test_generar_con_datos_escribe_sus_archivos(tmp_path):
    """Las dos figuras, ni de más ni de menos."""
    rutas = fu.generar(destino=tmp_path, datos=_umbral())  # dibuja
    assert sorted(r.name for r in tmp_path.iterdir()) == sorted(fu.ARCHIVOS)  # exactos
    assert [r.name for r in rutas] == list(fu.ARCHIVOS) and all(r.stat().st_size > 0 for r in rutas)  # orden y bytes


# %% [markdown]
# ## 2. Lectura desde `umbral.json` y `SALIDA` leído al llamar
#
# Sin `datos`, lee `raiz/results/data/pd_umbral/umbral.json`; sin `destino`, usa `raiz/SALIDA` (redirigible).

# %%
def test_generar_lee_el_json_y_usa_salida(tmp_path, monkeypatch):
    """Lee ENTRADA/umbral.json y escribe en SALIDA."""
    save_json(tmp_path / fu.ENTRADA / fu.ARCHIVO_UMBRAL, _umbral())  # JSON sintético en la raíz falsa
    monkeypatch.setattr(fu, "SALIDA", tmp_path / "figs")  # redirige
    rutas = fu.generar(raiz=tmp_path)  # lee y dibuja
    assert {r.parent for r in rutas} == {tmp_path / "figs"}  # destino = SALIDA
    assert sorted(r.name for r in (tmp_path / "figs").iterdir()) == sorted(fu.ARCHIVOS)  # exactos


# %% [markdown]
# Sin `umbral.json` → `FileNotFoundError` que dice cómo generarlo.

# %%
def test_generar_sin_json_explica(tmp_path):
    """FileNotFoundError con el comando."""
    with pytest.raises(FileNotFoundError, match="scripts/umbral_pd.py"):  # error esperado
        fu.generar(raiz=tmp_path, destino=tmp_path / "figs")  # sin JSON


# %% [markdown]
# ## 3. Funciones puras
#
# `datos_sensibilidad`: eje de umbrales, curvas de los dos modelos en val y test y el umbral inicial.

# %%
def test_datos_sensibilidad():
    """Curvas del bloque sensibilidad_umbral y umbral inicial."""
    u = _umbral()  # resultado
    d = fu.datos_sensibilidad(u)  # puro
    assert d["umbrales"] == u["sensibilidad_umbral"]["umbrales"] and d["umbral_inicial"] == 0.5  # eje y 0,5
    assert d["modelos"]["mlp"]["test"]["tpr"] == u["sensibilidad_umbral"]["modelos"]["mlp"]["test"]["tpr"]  # TPR


# %% [markdown]
# `datos_costo`: cuota óptima, tasas de rechazo de prueba por cuota y por umbral, umbral de val (los `null` como NaN
# para dibujar) y $p^\*$.

# %%
def test_datos_costo_convierte_nulos_en_nan():
    """umbral_val null → NaN; el resto, igual al JSON."""
    u = _umbral()  # resultado
    u["curva_costo"]["modelos"]["mlp"]["umbral_val"][0] = None  # un umbral nulo (cuota 0)
    d = fu.datos_costo(u)  # puro
    assert math.isnan(d["modelos"]["mlp"]["umbral_val"][0])  # NaN
    assert d["razones"] == u["curva_costo"]["razones"] and d["p_estrella"] == u["curva_costo"]["p_estrella"]  # eje
    assert d["modelos"]["logistica"]["cuota"] == u["curva_costo"]["modelos"]["logistica"]["cuota"]  # r*


# %% [markdown]
# Un bloque que falta o curvas de largo distinto al eje → `ValueError`.

# %%
@pytest.mark.parametrize("dano", ["sin_sensibilidad", "sin_costo", "largo_sensibilidad", "largo_costo"])  # daños
def test_datos_invalidos(tmp_path, dano):
    """Bloque ausente o largos incoherentes → ValueError."""
    u = _umbral()  # resultado válido
    if dano == "sin_sensibilidad":  # sin el bloque
        del u["sensibilidad_umbral"]  # falta
    elif dano == "sin_costo":  # sin el bloque
        del u["curva_costo"]  # falta
    elif dano == "largo_sensibilidad":  # curva corta
        u["sensibilidad_umbral"]["modelos"]["mlp"]["val"]["tpr"].pop()  # un punto menos
    else:  # curva corta
        u["curva_costo"]["modelos"]["logistica"]["cuota"].pop()  # un punto menos
    with pytest.raises(ValueError):  # error esperado
        fu.generar(destino=tmp_path, datos=u)  # no dibuja


# %% [markdown]
# Una entrada inválida no deja figuras a medias: se calculan todos los datos antes de dibujar.

# %%
def test_no_deja_figuras_a_medias(tmp_path):
    """Con curva_costo dañada no se escribe ni la primera figura."""
    u = _umbral()  # resultado válido
    del u["curva_costo"]  # dañado
    with pytest.raises(ValueError):  # error esperado
        fu.generar(destino=tmp_path, datos=u)  # no dibuja
    assert list(tmp_path.iterdir()) == []  # nada escrito
