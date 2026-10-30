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
# # Pruebas de `src/pd/umbral_experimento.py` y `scripts/umbral_pd.py` (B11, B-pd §3.9)
#
# Orquestación con dependencias **falsas** inyectadas (datos sintéticos, modelos lineales falsos, entrenador falso,
# summary falso) y una raíz falsa en `tmp_path`: la config (claves exactas, `datos.muestra` null, bloque `umbral`), la
# guarda de rutas (lección H-26), el contenido de `umbral.json` y `pred_umbral.parquet`, el control de la semilla
# mediana (idéntica / dentro de tolerancia / fuera → error / ausente → null), que no escribe nada fuera de `salida` y
# el código 2 del script ante un error esperado. La única prueba con datos reales está marcada `lento`.
#
# **Nunca** se escribe en `data/`, `models/`, `results/`, `configs/` ni `blog/` reales.

# %% [markdown]
# ## Importaciones

# %%
import copy  # copias de configs
import math  # raíz cuadrada de la tolerancia
import os  # enlaces simbólicos (guarda y prueba lenta)
from pathlib import Path  # rutas

import numpy as np  # datos sintéticos
import pandas as pd  # parquets
import pytest  # fixtures y aserciones
from scipy.special import expit  # σ de los modelos falsos

import umbral_pd  # script bajo prueba (conftest pone scripts/ en sys.path)
from src.common.config import load_config  # mismo cargador que el script
from src.common.results import load_json  # lectura del JSON escrito
from src.pd import experimento as ex  # evaluar y resumir de B5
from src.pd import umbral as u  # lógica pura (B11)
from src.pd import umbral_experimento as ue  # módulo bajo prueba

# %% [markdown]
# ## Constantes de las pruebas
#
# Valores de prueba, no del dominio: tamaños pequeños, pesos de los modelos falsos y la semilla mediana falsa.

# %%
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio
RUTA_CFG = ROOT / "configs" / "pd" / "umbral.yaml"  # config bajo prueba
SEMILLA = 20261030  # semilla de los datos sintéticos (fecha de la tarea)
N_SINT = {"train": 200, "val": 150, "test": 180}  # filas por split
W_VERDAD = np.array([1.2, -0.7])  # pesos que generan y
W_LOG = np.array([1.0, -0.5])  # logística falsa
W_MLP = np.array([1.1, -0.6])  # MLP falso (semilla 0); cada semilla suma 0,01·seed
SALIDA_REL = "results/data/pd_umbral_prueba"  # hermana permitida dentro de la raíz falsa
FIGURAS_REL = f"{SALIDA_REL}/figuras"  # figuras dentro de salida
EVIDENCIA_FALSA = {"sha256_particion": "a" * 64, "sha256_preprocesador": "b" * 64}  # evidencia de la partición


# %% [markdown]
# ## Datos y modelos falsos
#
# Matrices `{split: (X float32, y, ids)}` como las de `experimento.cargar_datos`, un DataFrame con `id` y `split` y un
# modelo lineal falso con `predict_proba` $= \sigma(Xw + b)$.

# %%
def _matrices() -> tuple[pd.DataFrame, dict]:
    """(df con id y split, {split: (X, y, ids)}) sintéticos y reproducibles."""
    rng = np.random.default_rng(SEMILLA)  # generador
    matrices, filas = {}, []  # resultado y filas del df
    for split, n in N_SINT.items():  # train, val, test
        X = rng.normal(size=(n, 2)).astype(np.float32)  # dos variables
        y = (rng.uniform(size=n) < expit(X @ W_VERDAD)).astype(np.int64)  # objetivo
        ids = np.array([f"{split}{i}" for i in range(n)], dtype=object)  # ids únicos
        matrices[split] = (X, y, ids)  # tripleta
        filas += [(i, split) for i in ids]  # id y split
    return pd.DataFrame(filas, columns=["id", "split"]), matrices  # df y matrices


class _Modelo:
    """Modelo lineal falso: PD = σ(Xw + b)."""

    def __init__(self, w, b: float = 0.0):
        self.w, self.b = np.asarray(w, dtype=float), b  # parámetros

    def predict_proba(self, X) -> np.ndarray:
        """σ(Xw + b) en float64."""
        return expit(np.asarray(X, dtype=float) @ self.w + self.b)  # PD


# %% [markdown]
# Entrenador falso con la firma `entrenar(datos, cfg_mlp, seed, reloj) → (modelo, tiempo)`: la semilla $s$ da pesos
# $w_{\text{MLP}} + 0{,}01\,s$. Registra las semillas que entrena.

# %%
def _entrenador(registro: list):
    """Entrenador falso que anota cada semilla."""
    def entrenar(datos, cfg_mlp, seed, reloj):  # misma firma que la inyectada por defecto
        registro.append(seed)  # semilla entrenada
        return _Modelo(W_MLP + 0.01 * seed), 0.25  # modelo y tiempo del fit
    return entrenar  # función


# %% [markdown]
# Summary falso: semilla mediana y, si se pide, las métricas de prueba de las semillas falsas (media exacta) con una
# desviación dada (controla la tolerancia $3\sqrt2\,s/\sqrt n$).

# %%
def _summary(seed_med: int, semillas=(0, 1, 2), desv: float = 1.0, desplazar: float = 0.0) -> dict:
    """summary.json falso con la entrada mlp (semilla_mediana y métricas de test)."""
    _, mats = _matrices()  # datos
    X, y, _ = mats["test"]  # prueba
    por_semilla = [ex.evaluar(y, _Modelo(W_MLP + 0.01 * s).predict_proba(X)) for s in semillas]  # métricas
    metricas = {f"{k}_test": {"media": float(np.mean([m[k] for m in por_semilla])) + desplazar, "desv": desv,  # media
                              "mejor": 0.0, "peor": 0.0, "n": 30} for k in por_semilla[0]}  # resumen falso
    return {"mlp": {"semilla_mediana": seed_med, "metricas": metricas}}  # entrada mlp


# %% [markdown]
# Dependencias falsas completas (la lectura del summary y el MLP cargado se pueden cambiar por prueba).

# %%
def _dependencias(seed_med: int = 1, summary=None, mlp_cargado=None, registro=None, figuras=None, reloj=None):
    """ue.Dependencias con datos, modelos, entrenador y summary falsos."""
    df, mats = _matrices()  # datos sintéticos
    summary = summary if summary is not None else _summary(seed_med)  # summary falso
    mlp_cargado = mlp_cargado if mlp_cargado is not None else _Modelo(W_MLP + 0.01 * seed_med)  # pd_mlp.pt falso
    return ue.Dependencias(  # todas inyectadas
        cargar_datos=lambda rutas: (df, mats),  # datos sintéticos
        cargar_logistica=lambda ruta: _Modelo(W_LOG),  # logística falsa
        cargar_mlp=lambda ruta: mlp_cargado,  # pd_mlp.pt falso
        entrenar=_entrenador(registro if registro is not None else []),  # entrenador falso
        leer_summary=lambda ruta: copy.deepcopy(summary),  # summary falso
        evidencia=lambda df_, ruta: dict(EVIDENCIA_FALSA),  # huellas falsas
        huella=lambda ruta: "c" * 64,  # huella falsa de los modelos
        generar_figuras=figuras if figuras is not None else (lambda destino, datos: []),  # sin figuras
        reloj=reloj if reloj is not None else iter(range(1000)).__next__,  # reloj determinista
    )


# %% [markdown]
# Config de prueba: la versionada con `semillas.n = 3` y salida y figuras en la hermana de prueba (relativas a la raíz
# falsa).

# %%
def _cfg(**cambios) -> dict:
    """configs/pd/umbral.yaml con 3 semillas, salida y figuras de prueba y ``cambios`` en la raíz."""
    cfg = load_config(RUTA_CFG)  # config real (solo lectura)
    cfg["semillas"]["n"] = 3  # semillas 0, 1 y 2
    cfg["salida"], cfg["figuras"] = SALIDA_REL, FIGURAS_REL  # hermana de prueba
    cfg.update(cambios)  # cambios de la prueba
    return cfg  # config


# %% [markdown]
# Todos los archivos que hay bajo una carpeta (para comprobar que no se escribe fuera de `salida`).

# %%
def _archivos(carpeta: Path) -> set[Path]:
    """Archivos (recursivo) bajo ``carpeta``."""
    return {r for r in carpeta.rglob("*") if r.is_file()}  # solo archivos


# %% [markdown]
# ## 1. La config versionada
#
# `configs/pd/umbral.yaml` hereda `base.yaml` y declara los valores de B-pd §3.9; pasa la validación con la raíz real.

# %%
def test_config_versionada_tiene_los_valores_del_spec():
    """Valores de B-pd §3.9 y validación con la raíz real."""
    cfg = load_config(RUTA_CFG)  # config real
    b = cfg["umbral"]  # bloque umbral
    assert cfg["salida"] == "results/data/pd_umbral" and cfg["figuras"] == "blog/docs/figures"  # rutas
    assert b["umbral_inicial"] == 0.5 and b["escenarios_razon"] == [2, 5, 10]  # 0,5 y L/G
    assert b["curva_razon"] == {"min": 1, "max": 20, "n": 25}  # curva logarítmica
    assert b["metas_sensibilidad"] == [0.5, 0.7, 0.8, 0.9]  # metas
    assert cfg["semillas"]["n"] == 30 and cfg["datos"]["muestra"] is None  # de base.yaml
    v = ue.validar_cfg_umbral(cfg, raiz=ROOT)  # validación con la raíz real
    assert np.allclose(v["umbral"]["rejilla_cuotas"], np.arange(101) / 100)  # 0 a 1 en pasos de 0,01
    assert np.allclose(v["umbral"]["rejilla_umbrales"], np.arange(101) / 100)  # ídem
    assert v["umbral"]["rejilla_cuotas"][7] == 0.07  # redondeada: 0,07 exacto
    razones = v["umbral"]["curva_razon"]  # rejilla logarítmica
    assert razones[0] == pytest.approx(1) and razones[-1] == pytest.approx(20) and len(razones) == 25  # extremos
    assert np.allclose(np.diff(np.log(razones)), np.log(20) / 24)  # paso constante en log


# %% [markdown]
# ## 2. Claves exactas
#
# Una clave de más o de menos en la raíz o en `umbral` (o en `curva_razon` y las rejillas) → `ValueError`.

# %%
@pytest.mark.parametrize("ruta, accion", [((), "extra"), ((), "falta_figuras"), (("umbral",), "extra"),  # raíz y umbral
                                          (("umbral",), "falta_metas"), (("umbral", "curva_razon"), "extra"),  # curva
                                          (("umbral", "rejilla_cuotas"), "falta_n")])  # rejilla
def test_claves_exactas(tmp_path, ruta, accion):
    """Claves de más o de menos → ValueError."""
    cfg = _cfg()  # config válida
    nodo = cfg  # nodo a cambiar
    for clave in ruta:  # baja por la ruta
        nodo = nodo[clave]  # sub-bloque
    if accion == "extra":  # clave desconocida
        nodo["desconocida"] = 1  # de más
    else:  # clave que falta
        del nodo[{"falta_figuras": "figuras", "falta_metas": "metas_sensibilidad", "falta_n": "n"}[accion]]  # de menos
    with pytest.raises(ValueError):  # error esperado
        ue.validar_cfg_umbral(cfg, raiz=tmp_path)  # inválida


# %% [markdown]
# ## 3. `datos.muestra` debe ser null
#
# B11 usa todos los datos: una muestra → `ValueError`.

# %%
def test_datos_muestra_no_null_es_error(tmp_path):
    """datos.muestra = 5000 → ValueError."""
    with pytest.raises(ValueError, match="muestra"):  # error esperado
        ue.validar_cfg_umbral(_cfg(datos={"muestra": 5000}), raiz=tmp_path)  # muestra


# %% [markdown]
# ## 4. Valores inválidos del bloque `umbral`
#
# Umbral inicial fuera de $[0, 1]$, razones $\le 0$ o vacías, curva con min ≥ max o $n < 2$, rejillas fuera de
# $[0, 1]$, metas fuera de $[0, 1]$, booleanos y textos → `ValueError`.

# %%
@pytest.mark.parametrize("clave, valor", [
    ("umbral_inicial", 1.5), ("umbral_inicial", True), ("umbral_inicial", "0.5"),  # umbral inicial
    ("escenarios_razon", []), ("escenarios_razon", [0]), ("escenarios_razon", [-2]), ("escenarios_razon", 5),  # razones
    ("escenarios_razon", [float("nan")]), ("escenarios_razon", [float("inf")]),  # no finitas
    ("curva_razon", {"min": 20, "max": 1, "n": 25}), ("curva_razon", {"min": 0, "max": 20, "n": 25}),  # curva
    ("curva_razon", {"min": 1, "max": 20, "n": 1}), ("curva_razon", {"min": 1, "max": 20, "n": 2.5}),  # n
    ("rejilla_cuotas", {"min": 0, "max": 1.5, "n": 101}), ("rejilla_cuotas", {"min": 0.5, "max": 0.5, "n": 3}),  # cuo.
    ("rejilla_umbrales", {"min": -0.1, "max": 1, "n": 101}), ("rejilla_umbrales", [0, 0.5]),  # umbrales
    ("metas_sensibilidad", []), ("metas_sensibilidad", [1.2]), ("metas_sensibilidad", [True]),  # metas
])
def test_valores_invalidos_del_bloque_umbral(tmp_path, clave, valor):
    """Valores fuera de rango o de tipo → ValueError."""
    cfg = _cfg()  # config válida
    cfg["umbral"][clave] = valor  # valor inválido
    with pytest.raises(ValueError):  # error esperado
        ue.validar_cfg_umbral(cfg, raiz=tmp_path)  # inválida


# %% [markdown]
# Bloques reutilizados de B5 (`semillas`, `base_seed`, `mlp`): sus reglas se aplican igual.

# %%
@pytest.mark.parametrize("cambio", [{"semillas": {"n": 0}}, {"base_seed": -1}, {"mlp": "x"}])  # inválidos
def test_bloques_de_b5_se_validan(tmp_path, cambio):
    """semillas.n = 0, base_seed < 0 o mlp no mapeo → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        ue.validar_cfg_umbral(_cfg(**cambio), raiz=tmp_path)  # inválida


# %% [markdown]
# Una config que no es un mapeo → `ValueError`.

# %%
@pytest.mark.parametrize("cfg", [None, [], "x"])  # no mapeos
def test_config_no_mapeo(tmp_path, cfg):
    """cfg no dict → ValueError."""
    with pytest.raises(ValueError):  # error esperado
        ue.validar_cfg_umbral(cfg, raiz=tmp_path)  # inválida


# %% [markdown]
# ## 5. Guarda de rutas (lección H-26)
#
# `salida` = `results/data/pd_umbral` o su hermana `results/data/pd_umbral_<algo>`; `figuras` = `blog/docs/figures`
# o una carpeta dentro de `salida` (sin pisar `umbral.json` ni `pred_umbral.parquet`). Nunca `results/data/pd`,
# `models/`, rutas con `..`, absolutas ni enlaces simbólicos que salgan de la raíz.

# %%
@pytest.mark.parametrize("salida, figuras", [
    ("results/data/pd_umbral", "blog/docs/figures"), (SALIDA_REL, FIGURAS_REL),  # las dos de la spec
    ("results/data/pd_umbral_demo", "results/data/pd_umbral_demo/figuras"),  # la demo
    ("results/data/pd_umbral_demo", "results/data/pd_umbral_demo/a/b"),  # anidada dentro de salida
    ("./results/data/pd_umbral", "blog/docs/figures/"),  # con ./ y barra final
])
def test_guarda_acepta(tmp_path, salida, figuras):
    """Rutas permitidas: devuelve las absolutas dentro de la raíz."""
    s, f = ue.validar_rutas(_cfg(salida=salida, figuras=figuras), raiz=tmp_path)  # guarda
    assert tmp_path.resolve() in s.parents and tmp_path.resolve() in f.parents  # dentro de la raíz


# %% [markdown]
# Rutas prohibidas para `salida`.

# %%
@pytest.mark.parametrize("salida", [
    "results/data/pd", "models", "results/data/pd/umbral", "results/data/pd_umbral/../pd",  # B5, modelos y ..
    "results/data/pd_umbralx", "results/data/pd_umbral_", "results/data/pd_umbral/sub",  # sin _, sufijo vacío, sub
    "results/data/lgd", ".", "", "results", "/tmp/pd_umbral", "../results/data/pd_umbral",  # otras, vacía, fuera
    "results/data/pd_umbral_x/..", 5, None,  # .. al final y no texto
])
def test_guarda_rechaza_salida(tmp_path, salida):
    """salida fuera de la lista blanca → ValueError que nombra la clave."""
    with pytest.raises(ValueError, match="salida"):  # error esperado
        ue.validar_rutas(_cfg(salida=salida, figuras="blog/docs/figures"), raiz=tmp_path)  # prohibida


# %% [markdown]
# Rutas prohibidas para `figuras`.

# %%
@pytest.mark.parametrize("figuras", [
    "blog/docs", "blog/docs/figures/sub", SALIDA_REL, f"{SALIDA_REL}/umbral.json",  # otra, sub, salida, JSON
    f"{SALIDA_REL}/pred_umbral.parquet", "results/data/pd/figs", "models/figs", "/tmp/figs",  # parquet y prohibidas
    f"{SALIDA_REL}/../pd", "", 3,  # .., vacía y no texto
])
def test_guarda_rechaza_figuras(tmp_path, figuras):
    """figuras fuera de blog/docs/figures o de salida → ValueError que nombra la clave."""
    with pytest.raises(ValueError, match="figuras"):  # error esperado
        ue.validar_rutas(_cfg(figuras=figuras), raiz=tmp_path)  # prohibida


# %% [markdown]
# Un enlace simbólico con un nombre permitido que apunta fuera de la raíz se rechaza (la ruta se resuelve).

# %%
def test_guarda_rechaza_enlace_que_sale_de_la_raiz(tmp_path):
    """results/data/pd_umbral_link → /fuera: ValueError."""
    raiz = tmp_path / "repo"  # raíz falsa
    (raiz / "results" / "data").mkdir(parents=True)  # carpeta padre
    fuera = tmp_path / "fuera"  # destino del enlace
    fuera.mkdir()  # existe
    os.symlink(fuera, raiz / "results" / "data" / "pd_umbral_link")  # enlace con nombre permitido
    with pytest.raises(ValueError, match="salida"):  # error esperado
        ue.validar_rutas(_cfg(salida="results/data/pd_umbral_link", figuras="blog/docs/figures"), raiz=raiz)  # sale


# %% [markdown]
# Con la raíz real del repositorio (sin inyectar) también se rechaza `results/data/pd`.

# %%
def test_guarda_con_la_raiz_por_defecto():
    """Raíz por defecto = la del repositorio."""
    with pytest.raises(ValueError, match="salida"):  # error esperado
        ue.validar_rutas(_cfg(salida="results/data/pd"))  # prohibida
    s, _ = ue.validar_rutas(_cfg())  # la de prueba es válida
    assert s == ROOT / SALIDA_REL  # absoluta en la raíz real


# %% [markdown]
# `modelos` (solo lectura) también debe ser relativa, sin `..` y dentro de la raíz.

# %%
@pytest.mark.parametrize("modelos", ["/tmp/models", "../models", "", 3])  # inválidas
def test_modelos_invalido(tmp_path, modelos):
    """modelos absoluto, con .., vacío o no texto → ValueError."""
    with pytest.raises(ValueError, match="modelos"):  # error esperado
        ue.validar_cfg_umbral(_cfg(modelos=modelos), raiz=tmp_path)  # inválida


# %% [markdown]
# ## 6. Corrida completa con dependencias falsas
#
# Escribe exactamente `umbral.json` y `pred_umbral.parquet` en `salida` (nada fuera), entrena las 3 semillas, y el JSON
# tiene los bloques de B-pd §3.9 más `config`, `tiempo_s` y `evidencia` (la de la partición + los dos modelos).

# %%
def test_corrida_escribe_solo_en_salida_con_los_bloques(tmp_path):
    """Archivos exactos, bloques exactos y evidencia."""
    registro = []  # semillas entrenadas
    res = ue.correr_umbral(_cfg(), raiz=tmp_path, dep=_dependencias(registro=registro))  # corrida
    salida = tmp_path / SALIDA_REL  # carpeta de resultados
    assert _archivos(tmp_path) == {salida / ue.ARCHIVO_UMBRAL, salida / ue.ARCHIVO_PRED}  # nada más
    assert registro == [0, 1, 2]  # base_seed + 0…2
    j = load_json(salida / ue.ARCHIVO_UMBRAL)  # JSON escrito
    assert list(j) == list(ue.BLOQUES_UMBRAL)  # bloques exactos y en orden
    assert set(ue.BLOQUES_UMBRAL) >= {"trivial", "matriz_inicial", "sensibilidad_umbral", "escenarios",  # §3.9
                                      "curva_costo", "metas", "semillas", "control_semilla_mediana", "config",
                                      "tiempo_s", "evidencia"}  # todos
    assert j["evidencia"] == {**EVIDENCIA_FALSA, "sha256_modelo_logistica": "c" * 64,  # partición y modelos
                              "sha256_modelo_mlp": "c" * 64}  # los dos modelos leídos
    assert j["config"]["umbral"] == _cfg()["umbral"] and j["tiempo_s"] > 0  # config y tiempo
    assert res["rutas"] == [salida / ue.ARCHIVO_UMBRAL, salida / ue.ARCHIVO_PRED]  # rutas devueltas


# %% [markdown]
# `pred_umbral.parquet`: columnas exactas, val y luego test en el orden de las matrices, y las PD de los modelos
# cargados (no de las semillas reentrenadas).

# %%
def test_pred_umbral(tmp_path):
    """id, split, y, pred_logistica, pred_mlp."""
    ue.correr_umbral(_cfg(), raiz=tmp_path, dep=_dependencias(seed_med=1))  # corrida
    pred = pd.read_parquet(tmp_path / SALIDA_REL / ue.ARCHIVO_PRED)  # escrito
    _, mats = _matrices()  # datos
    assert list(pred.columns) == ["id", "split", "y", "pred_logistica", "pred_mlp"]  # columnas
    X = np.vstack([mats["val"][0], mats["test"][0]])  # val y test
    assert pred["split"].tolist() == ["val"] * N_SINT["val"] + ["test"] * N_SINT["test"]  # orden
    assert pred["id"].tolist() == list(mats["val"][2]) + list(mats["test"][2])  # ids
    assert np.array_equal(pred["y"], np.concatenate([mats["val"][1], mats["test"][1]]))  # y
    assert np.array_equal(pred["pred_logistica"], _Modelo(W_LOG).predict_proba(X))  # logística cargada
    assert np.array_equal(pred["pred_mlp"], _Modelo(W_MLP + 0.01).predict_proba(X))  # pd_mlp.pt


# %% [markdown]
# ## 7. Contenido del análisis
#
# Escenarios: la cuota se elige en **val** con `cuota_optima` y se aplica en **test** como cuota; el contraste aplica
# el umbral de val tal cual en test. $p^\* = 1/(1 + L/G)$.

# %%
def test_escenarios(tmp_path):
    """Cuota de val, test por cuota y por umbral, p*."""
    res = ue.correr_umbral(_cfg(), raiz=tmp_path, dep=_dependencias())["umbral"]  # resultado en memoria
    _, mats = _matrices()  # datos
    rejilla = np.round(np.linspace(0, 1, 101), 9)  # cuotas
    p_val, p_te = _Modelo(W_LOG).predict_proba(mats["val"][0]), _Modelo(W_LOG).predict_proba(mats["test"][0])  # PD
    assert [e["razon"] for e in res["escenarios"]] == [2, 5, 10]  # escenarios de la config
    for e in res["escenarios"]:  # cada razón
        d = e["modelos"]["logistica"]  # logística
        r = u.cuota_optima(mats["val"][1], p_val, e["razon"], rejilla)  # cuota de val
        assert e["p_estrella"] == pytest.approx(1 / (1 + e["razon"]))  # referencia teórica
        assert d["cuota"] == r and d["umbral_val"] == u.umbral_de_cuota(p_val, r)  # cuota y su umbral
        m = u.matriz_confusion(mats["test"][1], u.rechazos_por_cuota(p_te, r))  # test por cuota
        assert {k: d["test_cuota"][k] for k in m} == m  # misma matriz
        mu = u.matriz_confusion(mats["test"][1], p_te >= d["umbral_val"])  # test por umbral de val
        assert {k: d["test_umbral"][k] for k in mu} == mu  # misma matriz
        assert d["test_cuota"]["costo"] == pytest.approx(e["razon"] * m["fn"] + m["fp"])  # costo
        assert set(d["val"]) == set(ue.CLAVES_MEDIDA)  # matriz + tasas + costo


# %% [markdown]
# Con una razón diminuta y un MLP que ordena al revés (los de mayor PD son buenos), la cuota óptima es 0: el umbral
# equivalente sería $+\infty$, se guarda como `null` y el contraste por umbral no rechaza a nadie.

# %%
def test_cuota_cero_umbral_null(tmp_path):
    """razón 1e-9 → cuota 0, umbral_val null y test_umbral sin rechazos."""
    cfg = _cfg()  # config
    cfg["umbral"]["escenarios_razon"] = [1e-9]  # FN casi gratis
    al_reves = _Modelo(-3 * W_VERDAD)  # ordena al revés: los de mayor PD son buenos → rechazar cuesta
    dep = _dependencias(seed_med=20, mlp_cargado=al_reves)  # sin control (semilla 20 ausente)
    res = ue.correr_umbral(cfg, raiz=tmp_path, dep=dep)["umbral"]  # corrida
    d = res["escenarios"][0]["modelos"]["mlp"]  # MLP
    assert d["cuota"] == 0.0 and d["umbral_val"] is None  # +inf → null
    assert d["test_umbral"]["tasa_rechazo"] == 0.0  # no rechaza a nadie
    escrito = load_json(tmp_path / SALIDA_REL / ue.ARCHIVO_UMBRAL)  # JSON en disco
    assert escrito["escenarios"][0]["modelos"]["mlp"]["umbral_val"] is None  # null en el JSON


# %% [markdown]
# Trivial: $q$ principal = tasa de train y piso = tasa de test; Brier con `brier_trivial`. Matriz inicial con 0,5.
# Curva de sensibilidad con la rejilla de umbrales. Curva de costo con 25 razones. Metas con `cuota_para_meta`.

# %%
def test_trivial_matriz_inicial_curvas_y_metas(tmp_path):
    """Bloques trivial, matriz_inicial, sensibilidad_umbral, curva_costo y metas."""
    res = ue.correr_umbral(_cfg(), raiz=tmp_path, dep=_dependencias())["umbral"]  # corrida
    _, mats = _matrices()  # datos
    q = float(np.mean(mats["train"][1]))  # tasa de train
    pi_te = float(np.mean(mats["test"][1]))  # tasa de test
    t = res["trivial"]  # bloque trivial
    assert t["principal"]["q"] == q and t["piso_teorico"]["q"] == pi_te  # q de cada trivial
    assert t["principal"]["brier_test"] == u.brier_trivial(mats["test"][1], q)  # Brier del trivial
    assert t["piso_teorico"]["brier_test"] == pytest.approx(pi_te * (1 - pi_te))  # piso π(1−π)
    p_te = _Modelo(W_LOG).predict_proba(mats["test"][0])  # PD de la logística
    mi = res["matriz_inicial"]  # matriz con 0,5
    assert mi["umbral"] == 0.5  # umbral inicial
    m = u.matriz_confusion(mats["test"][1], p_te >= 0.5)  # a mano
    assert {k: mi["modelos"]["logistica"]["test"][k] for k in m} == m  # misma matriz
    s = res["sensibilidad_umbral"]  # curva
    assert len(s["umbrales"]) == 101 and len(s["modelos"]["mlp"]["val"]["tpr"]) == 101  # rejilla completa
    c = res["curva_costo"]  # curva de costo
    assert len(c["razones"]) == 25 and len(c["modelos"]["logistica"]["cuota"]) == 25  # 25 razones
    assert c["p_estrella"] == pytest.approx([1 / (1 + r) for r in c["razones"]])  # p*
    for meta in res["metas"]:  # cada meta
        r = u.cuota_para_meta(mats["val"][1], _Modelo(W_LOG).predict_proba(mats["val"][0]), meta["meta"],  # a mano
                              np.round(np.linspace(0, 1, 101), 9))  # rejilla
        assert meta["modelos"]["logistica"]["cuota"] == r  # misma cuota
        assert meta["modelos"]["logistica"]["val"]["tpr"] >= meta["meta"]  # alcanza la meta en val


# %% [markdown]
# Semillas: para cada escenario y meta, `experimento.resumir` de TPR, TNR y tasa de aprobación de prueba, con
# "mayor es mejor" en las tres.

# %%
def test_resumen_de_semillas(tmp_path):
    """Media, desv, mejor, peor y n de las 3 semillas."""
    res = ue.correr_umbral(_cfg(), raiz=tmp_path, dep=_dependencias())["umbral"]  # corrida
    _, mats = _matrices()  # datos
    rejilla = np.round(np.linspace(0, 1, 101), 9)  # cuotas
    sem = res["semillas"]  # bloque
    assert sem["n"] == 3 and sem["semillas"] == [0, 1, 2]  # semillas
    e = sem["escenarios"][0]  # razón 2
    tprs = []  # TPR de test por semilla
    for s in (0, 1, 2):  # cada semilla
        m = _Modelo(W_MLP + 0.01 * s)  # modelo falso
        r = u.cuota_optima(mats["val"][1], m.predict_proba(mats["val"][0]), e["razon"], rejilla)  # cuota en val
        mt = u.matriz_confusion(mats["test"][1], u.rechazos_por_cuota(m.predict_proba(mats["test"][0]), r))  # test
        tprs.append(u.tasas(mt)["tpr"])  # TPR
    assert e["resumen"]["tpr_test"]["media"] == pytest.approx(np.mean(tprs))  # media
    assert e["resumen"]["tpr_test"]["mejor"] == max(tprs) and e["resumen"]["tpr_test"]["n"] == 3  # mayor es mejor
    assert set(e["resumen"]) == {"tpr_test", "tnr_test", "tasa_aprobacion_test"}  # tres métricas
    assert len(sem["metas"]) == 4 and len(sem["por_semilla"]) == 3  # metas y detalle


# %% [markdown]
# ## 8. Control de la semilla mediana
#
# Idéntica bit a bit; distinta pero dentro de la tolerancia $3\sqrt2\,s/\sqrt n$ (§8); fuera → error sin escribir;
# ausente (demo con pocas semillas) → `null`.

# %%
def test_control_identica(tmp_path):
    """Semilla 1 reentrenada = pd_mlp.pt falso."""
    res = ue.correr_umbral(_cfg(), raiz=tmp_path, dep=_dependencias(seed_med=1))["umbral"]  # corrida
    assert res["control_semilla_mediana"] == {"identica_bit_a_bit": True, "max_abs_dif": 0.0,  # idéntica
                                              "dentro_tolerancia": True}  # dentro


# %% [markdown]
# Distinta por $10^{-9}$ en el intercepto, con las medias del summary iguales a las de las semillas: dentro.

# %%
def test_control_dentro_de_tolerancia(tmp_path):
    """No idéntica pero dentro de la tolerancia."""
    dep = _dependencias(seed_med=1, mlp_cargado=_Modelo(W_MLP + 0.01, b=1e-9))  # pd_mlp.pt de otra máquina
    c = ue.correr_umbral(_cfg(), raiz=tmp_path, dep=dep)["umbral"]["control_semilla_mediana"]  # control
    assert c["identica_bit_a_bit"] is False and 0 < c["max_abs_dif"] < 1e-8  # distinta y cerca
    assert c["dentro_tolerancia"] is True  # medias dentro de 3√2·EE


# %% [markdown]
# Fuera de tolerancia: medias del summary desplazadas 0,1 con $s = 10^{-6}$ → `RuntimeError` y no se escribe nada.

# %%
def test_control_fuera_de_tolerancia_es_error_sin_escribir(tmp_path):
    """Fuera de tolerancia → RuntimeError; ningún archivo."""
    dep = _dependencias(seed_med=1, mlp_cargado=_Modelo(W_MLP + 0.01, b=1e-9),  # no idéntica
                        summary=_summary(1, desv=1e-6, desplazar=0.1))  # referencia lejana
    with pytest.raises(RuntimeError, match="tolerancia"):  # error esperado
        ue.correr_umbral(_cfg(), raiz=tmp_path, dep=dep)  # corrida
    assert _archivos(tmp_path) == set()  # nada escrito


# %% [markdown]
# La tolerancia es $3\sqrt2\,s/\sqrt n$ con $s$ y $n$ del summary de referencia (función pura).

# %%
def test_tolerancia_formula():
    """dentro si |media − ref| < 3·√2·desv/√n."""
    ref = {"auc_test": {"media": 0.5, "desv": 0.3, "mejor": 0, "peor": 0, "n": 9}}  # 3√2·0,3/3 = 0,4243
    assert ue.dentro_de_tolerancia({"auc_test": 0.5 + 0.42}, ref) is True  # dentro
    assert ue.dentro_de_tolerancia({"auc_test": 0.5 - 0.43}, ref) is False  # fuera
    assert ue.TOLERANCIA_SIGMAS == pytest.approx(3 * math.sqrt(2))  # 3·√2


# %% [markdown]
# Ausente: la semilla mediana (20) no está entre las 3 reentrenadas → `null`.

# %%
def test_control_ausente_es_null(tmp_path):
    """Semilla mediana fuera de las reentrenadas → None."""
    res = ue.correr_umbral(_cfg(), raiz=tmp_path, dep=_dependencias(seed_med=20))["umbral"]  # corrida
    assert res["control_semilla_mediana"] is None  # demo


# %% [markdown]
# Summary sin `mlp.semilla_mediana` entera → `ValueError` (antes de entrenar).

# %%
@pytest.mark.parametrize("summary", [{}, {"mlp": {}}, {"mlp": {"semilla_mediana": "20"}}, [1]])  # inválidos
def test_summary_invalido(tmp_path, summary):
    """summary sin semilla mediana entera → ValueError y sin entrenar."""
    registro = []  # semillas entrenadas
    with pytest.raises(ValueError, match="semilla_mediana"):  # error esperado
        ue.correr_umbral(_cfg(), raiz=tmp_path, dep=_dependencias(summary=summary, registro=registro))  # corrida
    assert registro == []  # no entrenó


# %% [markdown]
# ## 9. Validación antes de leer
#
# Con una config inválida (guarda), no se lee ni entrena nada: los cargadores falsos lanzan si se llaman.

# %%
def test_valida_antes_de_leer(tmp_path):
    """La guarda corre antes de cualquier lectura."""
    def no_llamar(*a, **k):  # cargador que no debe usarse
        raise AssertionError("no debía leer")  # falla si se llama
    dep = _dependencias()  # dependencias falsas
    dep = ue.Dependencias(**{**dep.__dict__, "cargar_datos": no_llamar, "leer_summary": no_llamar})  # trampas
    with pytest.raises(ValueError, match="salida"):  # error esperado (no AssertionError)
        ue.correr_umbral(_cfg(salida="results/data/pd"), raiz=tmp_path, dep=dep)  # prohibida


# %% [markdown]
# Las figuras se piden con el destino `figuras` y el resultado en memoria.

# %%
def test_figuras_reciben_destino_y_datos(tmp_path):
    """generar_figuras(destino, datos) con destino = figuras absoluta."""
    llamadas = []  # argumentos recibidos
    def figuras(destino, datos):  # generador falso que anota sus argumentos
        llamadas.append((destino, datos))  # argumentos
        return [destino / "x.png"]  # una ruta
    dep = _dependencias(figuras=figuras)  # inyecta
    res = ue.correr_umbral(_cfg(), raiz=tmp_path, dep=dep)  # corrida
    assert llamadas[0][0] == (tmp_path / FIGURAS_REL).resolve()  # destino
    assert llamadas[0][1] is res["umbral"]  # mismo resultado
    assert res["rutas"][-1] == (tmp_path / FIGURAS_REL).resolve() / "x.png"  # rutas de las figuras al final


# %% [markdown]
# ## 10. `manejador` y el script
#
# Errores esperados → una línea `scripts/umbral_pd.py: error: <Tipo>: <mensaje>` en stderr y código 2; los de
# programación (`NotImplementedError`, `TypeError`) se propagan con su traza.

# %%
def test_manejador_error_esperado(tmp_path, capsys):
    """ValueError → código 2 y una línea en stderr."""
    codigo = ue.manejador(_cfg(salida="models"), raiz=tmp_path, dep=_dependencias())  # prohibida
    err = capsys.readouterr().err  # stderr
    assert codigo == 2 and err.startswith(ue.PREFIJO_ERROR + "ValueError: ")  # código y prefijo
    assert err.count("\n") == 1  # una línea


# %% [markdown]
# Un error de programación del entrenador se propaga.

# %%
@pytest.mark.parametrize("exc", [NotImplementedError, TypeError])  # errores de programación
def test_manejador_propaga_errores_de_programacion(tmp_path, exc):
    """NotImplementedError y TypeError salen con traza."""
    dep = _dependencias()  # dependencias falsas
    def entrenar(*a, **k):  # entrenador roto
        raise exc("bug")  # error de programación
    dep = ue.Dependencias(**{**dep.__dict__, "entrenar": entrenar})  # inyecta
    with pytest.raises(exc):  # se propaga
        ue.manejador(_cfg(), raiz=tmp_path, dep=dep)  # corrida


# %% [markdown]
# Éxito: código 0, imprime las rutas escritas y el tiempo.

# %%
def test_manejador_exito_imprime_rutas_y_tiempo(tmp_path, capsys):
    """Código 0 con rutas y tiempo en stdout."""
    assert ue.manejador(_cfg(), raiz=tmp_path, dep=_dependencias()) == 0  # éxito
    out = capsys.readouterr().out  # stdout
    assert (tmp_path / SALIDA_REL / ue.ARCHIVO_UMBRAL).as_posix() in out and "Tiempo de ejecución" in out  # informe


# %% [markdown]
# El script: config inexistente, `--set` de una clave que no existe y una salida prohibida → código 2 (sin leer
# datos); con dependencias falsas y una config válida → 0.

# %%
@pytest.mark.parametrize("argv", [["--config", "configs/pd/no_existe.yaml"],  # archivo que no existe
                                  ["--config", str(RUTA_CFG), "--set", "umbral.nada=1"],  # clave inexistente
                                  ["--config", str(RUTA_CFG), "--set", "salida=results/data/pd"],  # prohibida
                                  ["--config", str(RUTA_CFG), "--set", "datos.muestra=100"]])  # muestra
def test_script_error_esperado_codigo_2(tmp_path, capsys, argv):
    """Código 2 y una línea en stderr."""
    assert umbral_pd.main(argv, raiz=tmp_path, dep=_dependencias()) == 2  # código
    err = capsys.readouterr().err  # stderr
    assert err.startswith(ue.PREFIJO_ERROR) and err.count("\n") == 1  # una línea


# %% [markdown]
# Éxito del script con `--set` repetible.

# %%
def test_script_exito(tmp_path):
    """--set repetible y código 0."""
    argv = ["--config", str(RUTA_CFG), "--set", "semillas.n=3", "--set", f"salida={SALIDA_REL}",  # demo
            "--set", f"figuras={FIGURAS_REL}"]  # figuras dentro de salida
    assert umbral_pd.main(argv, raiz=tmp_path, dep=_dependencias()) == 0  # éxito
    assert (tmp_path / SALIDA_REL / ue.ARCHIVO_UMBRAL).is_file()  # escribió


# %% [markdown]
# ## 11. Prueba lenta con datos reales
#
# Raíz temporal con enlaces de solo lectura a `data/processed`, `models` y `results/data/pd` reales, y la salida en
# una carpeta propia: `base_seed = 20` y `semillas.n = 1` reentrena solo la semilla mediana, así que el control se
# ejerce con la red real. Comprueba que las PD de prueba de `pd_mlp.pt` y de la logística coinciden con
# `pred_test.parquet` y que nada del repositorio cambia.

# %%
@pytest.mark.lento  # entrena una red con todos los datos
def test_lento_datos_reales(tmp_path):
    """Corrida real con la semilla mediana: control, pred_test y archivos."""
    resumen = load_json(ROOT / "results" / "data" / "pd" / "summary.json")  # semilla mediana real
    raiz = tmp_path / "repo"  # raíz temporal
    for rel in ("data/processed", "models", "results/data/pd"):  # entradas reales
        (raiz / rel).parent.mkdir(parents=True, exist_ok=True)  # carpeta padre
        os.symlink(ROOT / rel, raiz / rel)  # enlace de solo lectura
    cfg = load_config(RUTA_CFG)  # config real
    cfg.update(base_seed=resumen["mlp"]["semilla_mediana"], semillas={"n": 1},  # solo la mediana
               salida="results/data/pd_umbral_lento", figuras="results/data/pd_umbral_lento/figuras")  # carpeta propia
    antes = {r: r.stat().st_mtime_ns for r in _archivos(ROOT / "results" / "data" / "pd")}  # B5 intacto
    res = ue.correr_umbral(cfg, raiz=raiz)  # dependencias reales
    c = res["umbral"]["control_semilla_mediana"]  # control
    print(c)  # se informa en la salida de pytest
    assert c is not None and c["dentro_tolerancia"] is True  # reproduce pd_mlp.pt
    pred = pd.read_parquet(raiz / "results/data/pd_umbral_lento" / ue.ARCHIVO_PRED)  # escrito
    pred_test = pd.read_parquet(ROOT / "results" / "data" / "pd" / "pred_test.parquet").set_index("id")  # B5
    te = pred[pred["split"] == "test"].set_index("id")  # test
    assert set(te.index) == set(pred_test.index)  # mismos ids
    assert np.array_equal(te.loc[pred_test.index, "pred_mlp"], pred_test["pred_mlp"])  # pd_mlp.pt = pred_mlp
    assert np.array_equal(te.loc[pred_test.index, "pred_logistica"], pred_test["pred_logistica"])  # logística
    assert {r.name for r in (raiz / "results/data/pd_umbral_lento" / "figuras").iterdir()} == {  # figuras
        "pd-umbral-sensibilidad.png", "pd-umbral-costo.png"}  # las dos
    assert {r: r.stat().st_mtime_ns for r in _archivos(ROOT / "results" / "data" / "pd")} == antes  # B5 intacto
