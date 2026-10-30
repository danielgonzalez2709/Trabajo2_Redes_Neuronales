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
# # Pruebas de `src/pd/tablas_umbral.py` (B11, B-pd §3.9)
#
# Las tablas del notebook 02 §5 y del blog 06 §6.8 se leen de `umbral.json`: aquí, de un resultado **sintético**
# (`umbral_experimento.analizar` sobre PD generadas aquí). Se comprueba que cada celda es la del JSON, el orden de
# filas y columnas, que no hay NaN y que un bloque que falta es `ValueError`.

# %% [markdown]
# ## Importaciones

# %%
import numpy as np  # PD sintéticas
import pandas as pd  # tablas
import pytest  # aserciones
from scipy.special import expit, logit  # σ y su inversa

from src.common.config import load_config  # config versionada (solo lectura)
from src.pd import tablas_umbral as tu  # módulo bajo prueba
from src.pd import umbral_experimento as ue  # analizar y validación del bloque umbral

# %% [markdown]
# ## Resultado sintético
#
# Dos modelos y dos semillas sobre PD sintéticas (valores de prueba, no del dominio), con el bloque `umbral` de la
# config versionada y un control de semilla mediana falso.

# %%
SEMILLA = 20261030  # fecha de la tarea
RUTA_CFG = ue.RAIZ_REPOSITORIO / "configs" / "pd" / "umbral.yaml"  # config versionada


def _umbral() -> dict:
    """umbral.json sintético (bloques de analizar + control)."""
    rng = np.random.default_rng(SEMILLA)  # generador
    preds = {}  # por split
    for s, n in {"val": 400, "test": 500}.items():  # val y test
        z = rng.normal(size=n)  # puntaje latente
        y = (rng.uniform(size=n) < expit(z - 1)).astype(int)  # objetivo
        preds[s] = {"y": y, "ids": np.arange(n).astype(str).astype(object),  # y e ids
                    "logistica": expit(0.8 * z - 1), "mlp": expit(0.9 * z - 1.1)}  # dos modelos
    def desplazada(s: str, k: int) -> np.ndarray:  # PD del MLP con el logit desplazado k/10 (otra "semilla")
        return expit(logit(preds[s]["mlp"]) + k / 10)  # σ(logit(p) + k/10)
    sem = [{"seed": k, "tiempo_s": 1.0, "val": desplazada("val", k), "test": desplazada("test", k)}  # una semilla
           for k in (0, 1)]  # dos semillas
    res = ue.analizar(preds, 0.2, sem, ue.validar_cfg_umbral(load_config(RUTA_CFG))["umbral"])  # bloques
    res["control_semilla_mediana"] = {"identica_bit_a_bit": True, "max_abs_dif": 0.0,  # control falso
                                      "dentro_tolerancia": True}  # dentro
    return res  # resultado


# %% [markdown]
# ## 1. Trivial
#
# Filas: los dos triviales y los dos modelos; columnas: $q$ (vacío en los modelos), Brier de val y de test.

# %%
def test_tabla_trivial():
    """Valores del bloque trivial."""
    u = _umbral()  # resultado
    t = tu.tabla_trivial(u)  # tabla
    assert list(t.index) == list(tu.FILAS_TRIVIAL.values())  # filas
    assert list(t.columns) == ["PD constante", "Brier val", "Brier test"]  # columnas
    assert t.loc[tu.FILAS_TRIVIAL["principal"], "Brier test"] == u["trivial"]["principal"]["brier_test"]  # principal
    assert t.loc[tu.FILAS_TRIVIAL["mlp"], "Brier val"] == u["trivial"]["modelos"]["mlp"]["brier_val"]  # MLP
    assert t.loc[tu.FILAS_TRIVIAL["logistica"], "PD constante"] == tu.VACIO  # sin q


# %% [markdown]
# ## 2. Matriz inicial
#
# Una fila por (modelo, split) con los conteos y las tasas.

# %%
def test_tabla_matriz_inicial():
    """Conteos y tasas de la matriz con 0,5."""
    u = _umbral()  # resultado
    t = tu.tabla_matriz_inicial(u)  # tabla
    assert t.shape[0] == 4 and list(t.columns) == list(tu.COLUMNAS_MATRIZ.values())  # 2 modelos × 2 splits
    fila = t.loc[(tu.NOMBRE_MODELO["mlp"], "test")]  # MLP en prueba
    d = u["matriz_inicial"]["modelos"]["mlp"]["test"]  # JSON
    assert fila["FN"] == d["fn"] and fila["TPR"] == d["tpr"]  # mismas celdas
    assert pd.api.types.is_integer_dtype(t["VP"])  # conteos enteros


# %% [markdown]
# ## 3. Escenarios
#
# Una fila por ($L/G$, modelo): $p^\*$, cuota, umbral de val y la medida de prueba por cuota y por umbral.

# %%
def test_tabla_escenarios():
    """Celdas de los escenarios."""
    u = _umbral()  # resultado
    t = tu.tabla_escenarios(u)  # tabla
    assert t.shape[0] == 3 * 2  # 3 razones × 2 modelos
    e = u["escenarios"][1]  # L/G = 5
    fila = t.loc[(e["razon"], tu.NOMBRE_MODELO["logistica"])]  # fila
    d = e["modelos"]["logistica"]  # JSON
    assert fila["p*"] == e["p_estrella"] and fila["cuota"] == d["cuota"]  # p* y cuota
    assert fila["TPR prueba (cuota)"] == d["test_cuota"]["tpr"]  # cuota en prueba
    assert fila["rechazo prueba (umbral val)"] == d["test_umbral"]["tasa_rechazo"]  # PD absoluta
    assert fila["costo/crédito prueba (cuota)"] == d["test_cuota"]["costo_por_credito"]  # costo


# %% [markdown]
# Un umbral de val nulo (cuota 0) queda vacío, no NaN.

# %%
def test_tabla_escenarios_umbral_nulo_vacio():
    """umbral_val None → VACIO."""
    u = _umbral()  # resultado
    u["escenarios"][0]["modelos"]["mlp"]["umbral_val"] = None  # cuota 0
    t = tu.tabla_escenarios(u)  # tabla
    assert t.loc[(u["escenarios"][0]["razon"], tu.NOMBRE_MODELO["mlp"]), "umbral val"] == tu.VACIO  # vacío
    assert not t.isna().any().any()  # sin NaN


# %% [markdown]
# ## 3b. PD absoluta frente a cuota
#
# Una fila por ($L/G$, modelo): $p^\*$, el umbral de val de la cuota elegida, su brecha y la tasa de rechazo en val,
# en prueba por cuota y en prueba con el umbral de val (notebook 02 §5.5).

# %%
def test_tabla_pd_absoluta():
    """Celdas de p*, umbral de val, brecha y tasas de rechazo."""
    u = _umbral()  # resultado
    t = tu.tabla_pd_absoluta(u)  # tabla
    assert t.shape[0] == 3 * 2 and list(t.columns) == list(tu.COLUMNAS_PD_ABSOLUTA)  # 3 razones × 2 modelos
    e = u["escenarios"][2]  # L/G = 10
    d = e["modelos"]["mlp"]  # JSON
    fila = t.loc[(e["razon"], tu.NOMBRE_MODELO["mlp"])]  # fila
    assert fila["p*"] == e["p_estrella"] and fila["umbral val"] == d["umbral_val"]  # referencia y corte observado
    assert fila["p* − umbral val"] == e["p_estrella"] - d["umbral_val"]  # brecha
    assert fila["rechazo val"] == d["val"]["tasa_rechazo"]  # val
    assert fila["rechazo prueba (cuota)"] == d["test_cuota"]["tasa_rechazo"]  # cuota
    assert fila["rechazo prueba (umbral val)"] == d["test_umbral"]["tasa_rechazo"]  # PD absoluta


# %% [markdown]
# Un umbral de val nulo (cuota 0) deja vacíos el umbral y la brecha, nunca NaN.

# %%
def test_tabla_pd_absoluta_umbral_nulo_vacio():
    """umbral_val None → umbral y brecha VACIO."""
    u = _umbral()  # resultado
    u["escenarios"][0]["modelos"]["logistica"]["umbral_val"] = None  # cuota 0
    fila = tu.tabla_pd_absoluta(u).loc[(u["escenarios"][0]["razon"], tu.NOMBRE_MODELO["logistica"])]  # fila
    assert fila["umbral val"] == tu.VACIO and fila["p* − umbral val"] == tu.VACIO  # vacíos
    assert not tu.tabla_pd_absoluta(u).isna().any().any()  # sin NaN


# %% [markdown]
# ## 4. Metas
#
# Una fila por (meta, modelo); una meta inalcanzable deja la fila vacía.

# %%
def test_tabla_metas():
    """Celdas de las metas y meta inalcanzable."""
    u = _umbral()  # resultado
    t = tu.tabla_metas(u)  # tabla
    assert t.shape[0] == 4 * 2  # 4 metas × 2 modelos
    m = u["metas"][2]  # meta 0,8
    fila = t.loc[(m["meta"], tu.NOMBRE_MODELO["mlp"])]  # fila
    assert fila["TPR val"] == m["modelos"]["mlp"]["val"]["tpr"]  # TPR de val
    assert fila["cuota"] == m["modelos"]["mlp"]["cuota"]  # cuota
    u["metas"][3]["modelos"]["logistica"] = dict.fromkeys(m["modelos"]["logistica"])  # inalcanzable
    t2 = tu.tabla_metas(u)  # tabla
    assert (t2.loc[(u["metas"][3]["meta"], tu.NOMBRE_MODELO["logistica"])] == tu.VACIO).all()  # vacía
    assert not t2.isna().any().any()  # sin NaN


# %% [markdown]
# ## 5. Semillas
#
# Una fila por (escenario o meta, métrica) con media, desv, mejor, peor y n del resumen.

# %%
def test_tabla_semillas():
    """Resumen entre semillas."""
    u = _umbral()  # resultado
    t = tu.tabla_semillas(u)  # tabla
    assert t.shape[0] == (3 + 4) * 3  # 7 políticas × 3 métricas
    r = u["semillas"]["escenarios"][0]["resumen"]["tpr_test"]  # JSON
    fila = t.loc[(f"L/G = {u['semillas']['escenarios'][0]['razon']:g}", "TPR prueba")]  # fila
    assert fila["media"] == r["media"] and fila["n"] == r["n"] == 2  # media y n
    assert list(t.columns) == ["media", "desv", "mejor", "peor", "n"]  # columnas


# %% [markdown]
# ## 6. Control y bloques que faltan
#
# El control se muestra como una ficha (o "no aplica" si es null); un bloque que falta → `ValueError`.

# %%
def test_ficha_control():
    """Ficha del control y null."""
    u = _umbral()  # resultado
    assert tu.ficha_control(u) == u["control_semilla_mediana"]  # ficha
    u["control_semilla_mediana"] = None  # demo
    assert tu.ficha_control(u) == {"control": tu.NO_APLICA}  # no aplica


# %% [markdown]
# Cada función rechaza un resultado sin su bloque.

# %%
@pytest.mark.parametrize("f, bloque", [(tu.tabla_trivial, "trivial"), (tu.tabla_matriz_inicial, "matriz_inicial"),
                                       (tu.tabla_escenarios, "escenarios"), (tu.tabla_metas, "metas"),  # bloques
                                       (tu.tabla_pd_absoluta, "escenarios"),  # PD absoluta
                                       (tu.tabla_semillas, "semillas"),  # semillas
                                       (tu.ficha_control, "control_semilla_mediana")])  # control
def test_bloque_que_falta(f, bloque):
    """Sin el bloque → ValueError."""
    u = _umbral()  # resultado
    del u[bloque]  # falta
    with pytest.raises(ValueError, match=bloque):  # error esperado
        f(u)  # tabla


# %% [markdown]
# ## 7. Un solo lugar para el contrato de `umbral.json`
#
# `tablas_umbral` define `ARCHIVO_UMBRAL`, `MODELOS`, `COMANDO_UMBRAL` y `bloque`; la orquestación y las figuras los
# importan de allí (los mismos objetos, sin copias), y `MODELOS` coincide con los nombres de `experimento.py`.

# %%
def test_contrato_en_un_solo_lugar():
    """Mismos objetos en umbral_experimento y src/viz/umbral_pd; MODELOS = los de B5."""
    from src.pd import experimento as ex  # nombres de los modelos de B5
    from src.viz import umbral_pd as vu  # figuras de B11
    assert tu.MODELOS == (ex.MODELO_LOGISTICA, ex.MODELO_MLP)  # orden y nombres
    assert ue.ARCHIVO_UMBRAL is tu.ARCHIVO_UMBRAL and vu.ARCHIVO_UMBRAL is tu.ARCHIVO_UMBRAL  # un solo nombre
    assert ue.MODELOS is tu.MODELOS and vu.MODELOS is tu.MODELOS  # un solo orden
    assert vu.bloque is tu.bloque and not hasattr(vu, "_bloque") and not hasattr(tu, "_bloque")  # una sola función
    with pytest.raises(ValueError, match="scripts/umbral_pd.py"):  # mensaje con el comando
        tu.bloque({}, "trivial")  # falta
