"""Pruebas de SHAP del MLP, importancia por permutación y scorecard del MLP (Parte C §3.2 y §3.3, tarea C4).

Aceptación de C4 (C §4): **aditividad comprobada numéricamente**. Se explica el **logit** (no la probabilidad):
logit(x) = φ₀ + Σⱼ φⱼ(x) con φ₀ = E_fondo[f] y, como score = offset − factor · logit (C §3.1, tarea C1):

- puntos base = offset − factor · φ₀;
- puntosⱼ = −factor · φⱼ (φ > 0, más log-odds de malo ⇒ MENOS puntos);
- score = puntos base + Σⱼ puntosⱼ.

Jugadores = las 15 variables de ``features.yaml`` (no las 48 columnas): matriz Z (n, 15) con las numéricas
transformadas por el preprocesador y las categóricas como índice de la categoría activa en
``pre_json["variables"][j]["categorias"]``; ``a_columnas`` la decodifica a las 48 columnas one-hot.

Modelo explicado del MLP: el forward en float64 de las capas de ``models/pd_mlp.pt``
(``exportar_app.capas_mlp`` + ``exportar_app.forward_desde_json``, la misma función que calcula la app).
Logística: ``LogisticaPD.predict_logit``. Método: ``shap.explainers.Permutation`` con
``shap.maskers.Independent`` (shap 0.54.1).

Solo se usa la interfaz pública de ``src.scorecard.shap_pd`` (más ``src.scorecard.puntos`` y ``scorecard`` de C1, el
exportador de C2/C5, los modelos de B, las métricas comunes y el cargador de configuración). Toda escritura va a
``tmp_path``; nunca a ``results/``. Las corridas completas usan tamaños pequeños por ``--set`` (``SHAP_PEQUENO``).

Bloques (numeración de la tarea C4):
0. constantes públicas;
1. Z ↔ 48 columnas (ida y vuelta exacta; índices inválidos; bloque no one-hot);
2. juguete a mano: SHAP lineal, interacción, eficiencia; H-24 (valores isclose), estado global de numpy, φ₀ no
   finito, más de MAX_CODIGOS valores (ConfigInvalida), dirección no definida;
3. logística real: SHAP por permutación = forma cerrada;
4. MLP real: eficiencia y diferencia con MLPPD; H-24 con filas reales (revol_bal junto al recorte);
5. puntos: signo, puntos base y suma = score; ruta de construir: jugadores agrupados y fondo de train (H-25);
6. ``--set scorecard.pdo=40`` duplica los puntos (reutiliza la preparación);
7. importancia por permutación (juguete; caso a mano con media, desviación muestral y signos, H-25);
8. fuga: fondo, tramos y cortes solo de train; explicados y permutación solo de test (lento);
9. formato de ``scorecard_mlp.json`` y ``shap_mlp.json``; tramos con bordes repetidos; categorías sin explicados;
10. configuración (incluida rutas.logistica solo con mlp);
11. estabilidad;
12. manejadores (resumen impreso; config inválida → código 2).

Tiempos medidos con ``--durations`` (Windows, CPU, 26-oct, ronda 2):
- módulo sin las lentas (``-m "not lento"``): 61 pruebas en ~30 s;
- fixture ``corrida`` (``preparar_mlp`` + corrida pequeña): ~9,7 s; ``corrida40`` (reutiliza la preparación): ~8,6 s;
- ``test_01a`` (transformar train y test): ~1,9 s de preparación; el resto de pruebas, < 0,4 s cada una.
Marcada ``lento`` solo la 8 (fuga): necesita su propia preparación (escribir el parquet alterado, cargar, transformar
y hacer el forward de los 3 splits) además de la corrida: ~11,3 s de preparación (~28 s el módulo con ``-m lento``).
La 6 reutiliza la preparación del módulo (``preparar_mlp``), sin marca.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap
from scipy.special import expit

from src.common.config import apply_overrides, load_config
from src.common.metricas import auc
from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json, to_jsonable
from src.common.seeds import make_rng
from src.export.exportar_app import capas_mlp, forward_desde_json
from src.pd.logistica import LogisticaPD, logloss
import src.scorecard.shap_pd as shap_pd
from src.scorecard import puntos
from src.scorecard.puntos import CLAVES_RAIZ, cortes_bandas, grupos_columnas, validar_cfg_puntos
from src.scorecard.scorecard import CLAVES_PARAMETROS, parametros_scorecard, score_desde_logit
from src.scorecard.shap_pd import (
    ARCHIVO_SCORECARD_MLP,
    ARCHIVO_SHAP,
    CLAVES_SHAP,
    TOL_EFICIENCIA,
    a_columnas,
    a_variables,
    construir_scorecard_mlp,
    direccion,
    error_eficiencia,
    estabilidad,
    funcion_logit,
    importancia_global,
    importancia_permutacion,
    manejador_mlp,
    preparar_mlp,
    shap_cerrado_lineal,
    validar_cfg_shap,
    valores_shap,
)

RAIZ = Path(__file__).resolve().parents[1]
RUTA_BASE = RAIZ / "configs" / "scorecard" / "base.yaml"
RUTA_MLP = RAIZ / "configs" / "scorecard" / "mlp.yaml"
RUTA_SCORECARD_C3 = RAIZ / "results" / "data" / "scorecard" / "scorecard.json"
RUTA_SUMMARY_PD = RAIZ / "results" / "data" / "pd" / "summary.json"
RUTA_PD_MLP = RAIZ / "models" / "pd_mlp.pt"

TOL = 1e-9  # cuentas en float64 sobre puntajes del orden de 10² y logits del orden de 1
TOL_JUGUETE = 1e-12  # juguetes con valores diádicos pequeños
TOL_REL_PDO = 1e-12  # relación ×2 al pasar de pdo 20 a pdo 40
TOL_PD_MLPPD = 1e-6  # cota holgada |PD float64 − PD de MLPPD (float32)| (la de la app, núcleo §5.5)
SEMILLA_PRUEBA = 20261026  # semilla propia de las pruebas (no la de la config)
N_JUGADORES = 15  # variables de features.yaml

# Bloque shap de configs/scorecard/mlp.yaml (tamaños aprobados por Juan José en el plan de C4).
SHAP_APROBADO = {"base_seed": 0, "n_fondo": 100, "n_explicar": 1000, "n_estabilidad": 200, "max_evals": 500,
                 "k_permutacion": 5}
# Tamaños pequeños para las pruebas: 62 = 2 · 31 ⇒ 2 permutaciones antitéticas por fila.
N_FONDO, N_EXPLICAR, N_ESTABILIDAD, MAX_EVALS, K_PERMUTACION = 20, 30, 10, 62, 2
SHAP_PEQUENO = [f"shap.n_fondo={N_FONDO}", f"shap.n_explicar={N_EXPLICAR}", f"shap.n_estabilidad={N_ESTABILIDAD}",
                f"shap.max_evals={MAX_EVALS}", f"shap.k_permutacion={K_PERMUTACION}"]

# Config resuelta que debe dar configs/scorecard/mlp.yaml (hereda base.yaml; plan de C4).
CFG_MLP_ESPERADA = {
    "scorecard": {"base": 600, "odds_base": 50, "pdo": 20},
    "modelo": "mlp",
    "rutas": {
        "modelo": "models/pd_mlp.pt",
        "preprocesador": "models/preprocesador_pd.json",
        "datos": "data/processed/loans_pd.parquet",
        "features": "configs/features.yaml",
        "logistica": "models/pd_logistica.joblib",
    },
    "distribucion": {
        "percentiles_tabla": [1, 5, 25, 50, 75, 95, 99],
        "percentiles_rango": [1, 99],
        "fraccion_min_categoria": 0.01,
        "ancho_bin": 10,
        "n_bandas": 10,
    },
    "salida": "results/data/scorecard",
    "shap": SHAP_APROBADO,
}

CLAVES_SHAP_JSON = {"modelo", "metodo", "phi0", "eficiencia", "verificacion_logistica", "diferencia_mlppd",
                    "importancia_global", "direccion", "estabilidad", "permutacion", "explicados", "evidencia",
                    "tiempos_s"}
CLAVES_TIEMPOS = {"preparacion", "shap_base", "estabilidad", "verificacion_logistica", "permutacion_logistica",
                  "permutacion_mlp"}
CLAVES_METODO = {"explainer", "masker", "salida", "version_shap", "jugadores", "fondo", "explicados", "n_fondo",
                 "n_explicar", "n_estabilidad", "max_evals", "n_permutaciones", "semillas"}
CLAVES_NUM_MLP = {"nombre", "tipo", "etiqueta_es", "tramos", "n_sin_valor", "rango_puntos"}
CLAVES_TRAMO = {"desde", "hasta", "n", "puntos_medios"}
CLAVES_CAT_MLP = {"nombre", "tipo", "etiqueta_es", "categorias", "rango_puntos"}
CLAVES_CATEGORIA_MLP = {"categoria", "etiqueta", "n", "puntos_medios", "fraccion_train"}
CLAVES_DIST_SCORES = {"cortes_bandas", "n_bandas", "ancho_bin", "train", "val", "test"}
CLAVES_SPLIT = {"n", "media", "desv", "percentiles", "histograma", "bandas"}
CLAVES_PERMUTACION_RES = {"auc_base", "logloss_base", "variables"}
CLAVES_PERMUTACION_VAR = {"nombre", "caida_auc_media", "caida_auc_desv", "aumento_logloss_media",
                          "aumento_logloss_desv"}
SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------------------------------------------------------
# Juguete (bloques 1, 2 y 7). Variables: a (num), c (cat de 3 categorías), b (num); columnas [a, c=0, c=1, c=2, b].
# Todas las cuentas se hicieron con Python (float64) y se contrastaron con shap 0.54.1.
# ------------------------------------------------------------------------------------------------------------------

GRUPOS_JUGUETE = [("a", np.array([0])), ("c", np.array([1, 2, 3])), ("b", np.array([4]))]  # formato grupos_columnas
D_JUGUETE = 5
NOMBRES_JUGUETE = ["a", "c", "b"]
# Fondo de 4 filas en Z (a, índice de c, b): media de a = (0 + 1 + 2 − 1)/4 = 0.5; media de b = (1 + 0 − 1 + 2)/4 = 0.5;
# frecuencias de c: 0 → 2/4, 1 → 1/4, 2 → 1/4.
FONDO_JUGUETE = np.array([[0.0, 0, 1], [1, 1, 0], [2, 2, -1], [-1, 0, 2]])
EXPLICADOS_JUGUETE = np.array([[3.0, 2, -2], [-2, 1, 1]])
# logit lineal sobre las 5 columnas: 2·a + γ[c] − 3·b + 0.7 con γ = (0.5, −1, 2).
COEF_JUGUETE = np.array([2.0, 0.5, -1.0, 2.0, -3.0])
B0_JUGUETE = 0.7


def logit48_juguete(X) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ COEF_JUGUETE + B0_JUGUETE


# ------------------------------------------------------------------------------------------------------------------
# 0. Constantes públicas
# ------------------------------------------------------------------------------------------------------------------


def test_00_constantes_publicas():
    assert CLAVES_SHAP == ("base_seed", "n_fondo", "n_explicar", "n_estabilidad", "max_evals", "k_permutacion")
    assert ARCHIVO_SCORECARD_MLP == "scorecard_mlp.json"
    assert ARCHIVO_SHAP == "shap_mlp.json"
    assert TOL_EFICIENCIA == 1e-9
    assert puntos.MODELOS == ("logistica", "mlp")


# ------------------------------------------------------------------------------------------------------------------
# Fixtures del modelo real (se cargan una vez por módulo)
# ------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def features() -> list[dict]:
    return load_config(RAIZ / "configs" / "features.yaml")["features"]


@pytest.fixture(scope="module")
def nombres(features) -> list[str]:
    return [f["nombre"] for f in features]


@pytest.fixture(scope="module")
def pre_json() -> dict:
    return load_json(RAIZ / "models" / "preprocesador_pd.json")


@pytest.fixture(scope="module")
def pre(pre_json) -> Preprocesador:
    return Preprocesador.from_dict(pre_json)


@pytest.fixture(scope="module")
def grupos(pre_json) -> list:
    return grupos_columnas(pre_json)


@pytest.fixture(scope="module")
def d(pre_json) -> int:
    return len(pre_json["columnas_salida"])


@pytest.fixture(scope="module")
def datos() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data" / "processed" / "loans_pd.parquet")


@pytest.fixture(scope="module")
def por_split(datos, pre) -> dict:
    """Por split: filas crudas (índice reiniciado), X (float32 de transform) e y."""
    salida = {}
    for s in SPLITS:
        df = datos[datos["split"] == s].reset_index(drop=True)
        salida[s] = {"df": df, "X": pre.transform(df), "y": df["target"].to_numpy()}
    return salida


@pytest.fixture(scope="module")
def logistica() -> LogisticaPD:
    return LogisticaPD.cargar(RAIZ / "models" / "pd_logistica.joblib")


@pytest.fixture(scope="module")
def red_y_mlppd() -> tuple:
    capas, mlppd = capas_mlp(RUTA_PD_MLP)
    return {"capas": capas, "salida": "sigmoid"}, mlppd


@pytest.fixture(scope="module")
def red(red_y_mlppd) -> dict:
    return red_y_mlppd[0]


@pytest.fixture(scope="module")
def mlppd(red_y_mlppd):
    return red_y_mlppd[1]


@pytest.fixture(scope="module")
def z_fondo_y_explicados(por_split, grupos) -> tuple[np.ndarray, np.ndarray]:
    """Fondo (N_FONDO filas de train) y explicados (N_EXPLICAR filas de test) propios de la prueba, en Z."""
    rng = make_rng(SEMILLA_PRUEBA, 0)
    i_f = rng.choice(len(por_split["train"]["X"]), size=N_FONDO, replace=False)
    i_e = rng.choice(len(por_split["test"]["X"]), size=N_EXPLICAR, replace=False)
    return (a_variables(por_split["train"]["X"][i_f], grupos), a_variables(por_split["test"]["X"][i_e], grupos))


def _cfg_mlp(salida: Path, overrides: list[str] | None = None, datos: Path | None = None) -> dict:
    """Config de mlp.yaml con tamaños pequeños (y overrides), rutas absolutas y la salida en un directorio temporal."""
    cfg = apply_overrides(load_config(RUTA_MLP), [*SHAP_PEQUENO, *(overrides or [])])
    cfg = copy.deepcopy(cfg)
    cfg["rutas"] = {k: str(RAIZ / v) for k, v in cfg["rutas"].items()}
    if datos is not None:
        cfg["rutas"]["datos"] = str(datos)
    cfg["salida"] = str(salida)
    return cfg


@pytest.fixture(scope="module")
def cfg_mlp(tmp_path_factory) -> dict:
    return _cfg_mlp(tmp_path_factory.mktemp("mlp20"))


@pytest.fixture(scope="module")
def preparado(cfg_mlp) -> dict:
    """Preparación ÚNICA (carga, transform y forward de los 3 splits); la comparten corrida y corrida40."""
    return preparar_mlp(cfg_mlp)


@pytest.fixture(scope="module")
def corrida(cfg_mlp, preparado) -> tuple[dict, dict]:
    """(scorecard_mlp, shap_mlp) con pdo 20 y tamaños pequeños, construidos en memoria UNA vez por módulo."""
    return construir_scorecard_mlp(cfg_mlp, preparado)


@pytest.fixture(scope="module")
def sc_mlp(corrida) -> dict:
    return corrida[0]


@pytest.fixture(scope="module")
def shap_mlp(corrida) -> dict:
    return corrida[1]


@pytest.fixture(scope="module")
def summary_pd() -> dict:
    return load_json(RUTA_SUMMARY_PD)


def _explicados_crudos(por_split, ids: list[str]) -> pd.DataFrame:
    """Filas crudas de test de los explicados, en el orden de ``ids``."""
    test = por_split["test"]["df"].set_index("id")
    return test.loc[list(ids)].reset_index()


# ------------------------------------------------------------------------------------------------------------------
# 1. Z ↔ 48 columnas
# ------------------------------------------------------------------------------------------------------------------


def test_01a_ida_y_vuelta_exacta_en_train_y_test(por_split, grupos, d, pre_json):
    rng = make_rng(SEMILLA_PRUEBA, 1)
    muestra_train = por_split["train"]["X"][rng.choice(len(por_split["train"]["X"]), size=5000, replace=False)]
    for X in (muestra_train, por_split["test"]["X"]):
        Z = a_variables(X, grupos)
        assert isinstance(Z, np.ndarray) and Z.dtype == np.float64 and Z.shape == (len(X), N_JUGADORES)
        for j, (v, (_, idx)) in enumerate(zip(pre_json["variables"], grupos, strict=True)):
            if v["tipo"] == "num":  # su columna transformada
                np.testing.assert_array_equal(Z[:, j], X[:, idx[0]].astype(np.float64))
            else:  # índice de la categoría activa
                np.testing.assert_array_equal(Z[:, j], np.argmax(X[:, idx], axis=1).astype(np.float64))
        X48 = a_columnas(Z, grupos, d)
        assert isinstance(X48, np.ndarray) and X48.dtype == np.float64 and X48.shape == X.shape
        np.testing.assert_array_equal(X48, X.astype(np.float64))  # exacto


def test_01b_a_columnas_a_mano():
    # Z = (a, índice de c, b): [3, 2, −2] → [a, c=0, c=1, c=2, b] = [3, 0, 0, 1, −2]; [−2, 0, 1.5] → [−2, 1, 0, 0, 1.5].
    # El índice puede llegar como float (el masker de shap trabaja en float64): 2.0 vale como 2.
    Z = np.array([[3.0, 2.0, -2.0], [-2.0, 0.0, 1.5]])
    np.testing.assert_array_equal(a_columnas(Z, GRUPOS_JUGUETE, D_JUGUETE),
                                  [[3.0, 0.0, 0.0, 1.0, -2.0], [-2.0, 1.0, 0.0, 0.0, 1.5]])
    np.testing.assert_array_equal(a_variables(a_columnas(Z, GRUPOS_JUGUETE, D_JUGUETE), GRUPOS_JUGUETE), Z)


@pytest.mark.parametrize("valor", ["n_categorias", -1.0, 0.5, np.nan], ids=["fuera_por_arriba", "negativo",
                                                                           "no_entero", "nan"])
def test_01c_indice_categorico_invalido_es_error(por_split, grupos, d, pre_json, valor):
    j = next(j for j, v in enumerate(pre_json["variables"]) if v["nombre"] == "term")
    n_categorias = len(pre_json["variables"][j]["categorias"])  # 36 months, 60 months, OTRA ⇒ índices 0, 1, 2
    Z = a_variables(por_split["test"]["X"][:5], grupos)
    Z[0, j] = float(n_categorias) if valor == "n_categorias" else valor
    with pytest.raises(ValueError):
        a_columnas(Z, grupos, d)


@pytest.mark.parametrize("bloque", [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
                         ids=["ninguno", "dos", "fraccion"])
def test_01d_a_variables_bloque_no_one_hot_es_error(bloque):
    X = np.array([[3.0, *bloque, -2.0]])
    with pytest.raises(ValueError):
        a_variables(X, GRUPOS_JUGUETE)


# ------------------------------------------------------------------------------------------------------------------
# 2. Juguete a mano: SHAP del logit con el masker Independent
# ------------------------------------------------------------------------------------------------------------------


def test_02a_lineal_a_mano():
    f = funcion_logit(logit48_juguete, GRUPOS_JUGUETE, D_JUGUETE)
    # f(Z) = 2a + γ[c] − 3b + 0.7: [3, 2, −2] → 6 + 2 + 6 + 0.7 = 14.7; [−2, 1, 1] → −4 − 1 − 3 + 0.7 = −7.3
    np.testing.assert_allclose(f(EXPLICADOS_JUGUETE), [14.7, -7.3], rtol=0, atol=TOL_JUGUETE)
    phi, phi0 = valores_shap(f, FONDO_JUGUETE, EXPLICADOS_JUGUETE, 62, 0)
    assert isinstance(phi, np.ndarray) and phi.shape == (2, 3)
    # φ₀ = E_fondo[f] = 0.7 + 2·0.5 + E[γ] − 3·0.5, E[γ] = 0.5·0.5 + 0.25·(−1) + 0.25·2 = 0.5 ⇒ φ₀ = 0.7
    assert phi0 == pytest.approx(0.7, abs=TOL_JUGUETE)
    # Aditivo ⇒ φⱼ = gⱼ(xⱼ) − E_fondo[gⱼ]:
    # fila 0: a 2·(3 − 0.5) = 5; c γ[2] − 0.5 = 1.5; b −3·(−2 − 0.5) = 7.5   (0.7 + 14 = 14.7)
    # fila 1: a 2·(−2 − 0.5) = −5; c γ[1] − 0.5 = −1.5; b −3·(1 − 0.5) = −1.5 (0.7 − 8 = −7.3)
    esperado = np.array([[5.0, 1.5, 7.5], [-5.0, -1.5, -1.5]])
    np.testing.assert_allclose(phi, esperado, rtol=0, atol=TOL_JUGUETE)
    assert error_eficiencia(phi, phi0, f(EXPLICADOS_JUGUETE)) < TOL_JUGUETE
    # Forma cerrada: numérica βⱼ·(zⱼ − media del fondo); categórica Σₖ βₖ·(1[z = k] − frecuencia de k):
    # c en fila 0: 0.5·(0 − 0.5) + (−1)·(0 − 0.25) + 2·(1 − 0.25) = −0.25 + 0.25 + 1.5 = 1.5
    cerrado = shap_cerrado_lineal(EXPLICADOS_JUGUETE, FONDO_JUGUETE, COEF_JUGUETE, GRUPOS_JUGUETE)
    np.testing.assert_allclose(cerrado, esperado, rtol=0, atol=TOL_JUGUETE)


def test_02b_interaccion_a_mano():
    """f = a·b: con el masker Independent las variables ocultas toman la MISMA fila del fondo, y una permutación con
    su inversa (antitética) da el Shapley exacto con interacciones de orden 2."""
    def f(Z):
        return Z[:, 0] * Z[:, 2]

    phi, phi0 = valores_shap(f, FONDO_JUGUETE, EXPLICADOS_JUGUETE, 62, 0)
    # v(∅) = media de a·b en el fondo = (0 + 0 − 2 − 2)/4 = −1 = φ₀
    assert phi0 == pytest.approx(-1.0, abs=TOL_JUGUETE)
    # fila 0 (a = 3, b = −2): v(a) = 3·0.5 = 1.5; v(b) = 0.5·(−2) = −1; v(ab) = −6
    #   φ_a = ½[(1.5 + 1) + (−6 + 1)] = −1.25; φ_b = ½[(−1 + 1) + (−6 − 1.5)] = −3.75; φ_c = 0
    # fila 1 (a = −2, b = 1): v(a) = −1; v(b) = 0.5; v(ab) = −2
    #   φ_a = ½[(−1 + 1) + (−2 − 0.5)] = −1.25; φ_b = ½[(0.5 + 1) + (−2 + 1)] = 0.25; φ_c = 0
    np.testing.assert_allclose(phi, [[-1.25, 0.0, -3.75], [-1.25, 0.0, 0.25]], rtol=0, atol=TOL_JUGUETE)
    assert error_eficiencia(phi, phi0, f(EXPLICADOS_JUGUETE)) < TOL_JUGUETE


def test_02c_eficiencia_exacta_con_interaccion_de_orden_3_y_misma_semilla():
    """Con interacción de orden 3 los φ dependen de las permutaciones sorteadas, pero la eficiencia es exacta (suma
    telescópica de cada recorrido) y la misma semilla da los mismos φ."""
    def f(Z):
        return Z[:, 0] * Z[:, 2] * (Z[:, 1] + 1) + Z[:, 0] ** 3

    phi_0a, phi0_0a = valores_shap(f, FONDO_JUGUETE, EXPLICADOS_JUGUETE, 62, 0)
    phi_0b, phi0_0b = valores_shap(f, FONDO_JUGUETE, EXPLICADOS_JUGUETE, 62, 0)
    phi_5, phi0_5 = valores_shap(f, FONDO_JUGUETE, EXPLICADOS_JUGUETE, 62, 5)
    for phi, phi0 in ((phi_0a, phi0_0a), (phi_5, phi0_5)):
        assert error_eficiencia(phi, phi0, f(EXPLICADOS_JUGUETE)) < TOL_JUGUETE
    np.testing.assert_array_equal(phi_0a, phi_0b)
    assert phi0_0a == phi0_0b
    # Control: con otra semilla las permutaciones cambian (semilla 0: φ_a fila 0 = 20.1875; semilla 5: 21.875).
    assert not np.array_equal(phi_0a, phi_5)


def test_02d_error_eficiencia_a_mano():
    # φ₀ + Σφ = [0.5 + 3, 0.5 − 1] = [3.5, −0.5]; logit = [3.5, −0.25] ⇒ |dif| = [0, 0.25] ⇒ máx 0.25
    assert error_eficiencia(np.array([[1.0, 2.0], [0.0, -1.0]]), 0.5, np.array([3.5, -0.25])) == pytest.approx(0.25)


def test_02e_importancia_global_a_mano():
    # media |φ|: a (1 + 3)/2 = 2; c (4 + 2)/2 = 3; b (0.5 + 0.5)/2 = 0.5 ⇒ c, a, b
    phi = np.array([[1.0, -4.0, 0.5], [-3.0, 2.0, 0.5]])
    ig = importancia_global(phi, NOMBRES_JUGUETE)
    assert [e["nombre"] for e in ig] == ["c", "a", "b"]
    assert all(set(e) == {"nombre", "media_abs_phi", "puesto"} for e in ig)
    np.testing.assert_allclose([e["media_abs_phi"] for e in ig], [3.0, 2.0, 0.5], rtol=0, atol=TOL_JUGUETE)
    assert [e["puesto"] for e in ig] == [1, 2, 3]  # supuesto: el puesto 1 es el de mayor media |φ|


def test_02f_direccion_a_mano():
    crudos = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, np.nan], "c": ["x", "y", "x", "z", "y"],
                           "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
    feats = [{"nombre": "a", "tipo": "num"},
             {"nombre": "c", "tipo": "cat", "categorias_es": {"z": "Zeta", "x": "Equis", "y": "Ye"}},
             {"nombre": "b", "tipo": "num"}]
    phi = np.array([[0.1, 1.0, 0.5], [0.3, -2.0, 0.4], [0.2, 3.0, 0.3], [0.5, 0.5, 0.2], [9.0, -1.0, 0.1]])
    res = direccion(phi, crudos, feats)
    # a: solo los 4 crudos no vacíos; rangos [1, 2, 3, 4] frente a [1, 3, 2, 4] ⇒ ρ = 1 − 6·2/(4·15) = 0.8
    assert res["a"]["tipo"] == "num" and res["a"]["n"] == 4
    assert res["a"]["spearman"] == pytest.approx(0.8, abs=TOL_JUGUETE)
    # b: crudo decreciente y φ decreciente ⇒ ρ = 1
    assert res["b"]["tipo"] == "num" and res["b"]["n"] == 5
    assert res["b"]["spearman"] == pytest.approx(1.0, abs=TOL_JUGUETE)
    # c: φ medio por categoría: x (1 + 3)/2 = 2; y (−2 − 1)/2 = −1.5; z 0.5
    assert res["c"]["tipo"] == "cat"
    assert res["c"]["n"] == {"x": 2, "y": 2, "z": 1}
    assert res["c"]["phi_medio"] == pytest.approx({"x": 2.0, "y": -1.5, "z": 0.5}, abs=TOL_JUGUETE)


def test_02g_h21_valores_casi_iguales_se_reevaluan():
    """H-24: el masker de shap decide qué variable "no cambia" con np.isclose (rtol 1e-5), no con igualdad. Con el
    fondo en b = 1 y el explicado en b·(1 + 1e-6) (isclose, pero distintos), shap no reevaluaba b: φ_b = 0 y la suma no
    telescopiaba (error ≈ 1). Con la codificación de valores_shap, b se reevalúa y la eficiencia es exacta."""
    fondo = np.array([[0.0, 0, 1.0], [0.0, 0, 1.0], [0.0, 0, 1.0], [0.0, 0, 1.0]])
    expl = np.array([[0.0, 0, 1.0 * (1 + 1e-6)]])
    assert np.isclose(expl[0, 2], fondo[0, 2]) and expl[0, 2] != fondo[0, 2]  # control: el caso de H-24

    def f(Z):
        return 1e6 * Z[:, 2]  # muy sensible a b

    phi, phi0 = valores_shap(f, fondo, expl, 62, 0)
    # φ₀ = 1e6; f(x) − φ₀ = 1e6 · 1e-6 ≈ 1 (0.9999999999177334 en float64) y todo es de b
    assert error_eficiencia(phi, phi0, f(expl)) <= TOL_JUGUETE
    assert phi[0, 2] == pytest.approx(f(expl)[0] - phi0, abs=TOL_JUGUETE) and phi[0, 2] != 0
    assert phi[0, 0] == 0 and phi[0, 1] == 0  # a y c son iguales al fondo


def logit_juguete_z(Z) -> np.ndarray:
    return funcion_logit(logit48_juguete, GRUPOS_JUGUETE, D_JUGUETE)(Z)


def _mismo_estado(a, b) -> None:
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def test_02h_restaura_el_estado_global_de_numpy():
    """shap siembra el generador global (np.random.seed en PermutationExplainer.__init__); valores_shap lo restaura,
    también si la función explicada falla."""
    np.random.seed(SEMILLA_PRUEBA)
    antes = np.random.get_state()
    valores_shap(logit_juguete_z, FONDO_JUGUETE, EXPLICADOS_JUGUETE, 62, 0)
    _mismo_estado(antes, np.random.get_state())

    def falla(Z):
        raise RuntimeError("falla a propósito")

    with pytest.raises(RuntimeError):
        valores_shap(falla, FONDO_JUGUETE, EXPLICADOS_JUGUETE, 62, 0)
    _mismo_estado(antes, np.random.get_state())


def test_02i_phi0_no_finito_es_error():
    with pytest.raises(ValueError, match="finit"):
        valores_shap(lambda Z: np.full(len(Z), np.nan), FONDO_JUGUETE, EXPLICADOS_JUGUETE, 62, 0)


def test_02i2_demasiados_valores_distintos_es_config_invalida():
    """Con más de MAX_CODIGOS valores distintos en un jugador, dos códigos vecinos serían np.isclose: error de config
    (ConfigInvalida: lo provocan n_fondo y n_explicar), antes de llamar a shap."""
    fondo = np.arange(shap_pd.MAX_CODIGOS + 1, dtype=np.float64)[:, None]
    with pytest.raises(puntos.ConfigInvalida):
        valores_shap(lambda Z: Z[:, 0], fondo, np.array([[0.5]]), 3, 0)


def test_02j_direccion_no_definida_es_none():
    crudos = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0], "e": [5.0, 5.0, 5.0]})
    feats = [{"nombre": "a", "tipo": "num"}, {"nombre": "b", "tipo": "num"}, {"nombre": "e", "tipo": "num"}]
    phi = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.1], [0.3, 0.2, 0.2]])
    res = direccion(phi, crudos, feats)
    assert res["a"] == {"tipo": "num", "spearman": None, "n": 0}  # crudo todo vacío
    assert res["b"]["spearman"] is None and res["b"]["n"] == 3  # φ constante
    assert res["e"]["spearman"] is None and res["e"]["n"] == 3  # crudo constante


# ------------------------------------------------------------------------------------------------------------------
# 3. Logística real: el MISMO código de SHAP reproduce la forma cerrada
# ------------------------------------------------------------------------------------------------------------------


def test_03_logistica_real_shap_igual_a_forma_cerrada(logistica, grupos, d, z_fondo_y_explicados):
    fondo, expl = z_fondo_y_explicados
    f = funcion_logit(logistica.predict_logit, grupos, d)
    np.testing.assert_array_equal(f(expl), logistica.predict_logit(a_columnas(expl, grupos, d)))
    phi, phi0 = valores_shap(f, fondo, expl, MAX_EVALS, 0)
    assert phi.shape == (N_EXPLICAR, N_JUGADORES)
    coef, _ = logistica.coeficientes()
    cerrado = shap_cerrado_lineal(expl, fondo, coef, grupos)
    assert np.max(np.abs(phi - cerrado)) < TOL
    assert phi0 == pytest.approx(float(np.mean(logistica.predict_logit(a_columnas(fondo, grupos, d)))), abs=TOL)
    assert error_eficiencia(phi, phi0, f(expl)) < TOL_EFICIENCIA


# ------------------------------------------------------------------------------------------------------------------
# 4. MLP real: eficiencia del forward float64 y diferencia con MLPPD
# ------------------------------------------------------------------------------------------------------------------


def test_04_mlp_real_eficiencia_y_diferencia_con_mlppd(red, mlppd, grupos, d, z_fondo_y_explicados):
    fondo, expl = z_fondo_y_explicados
    f = funcion_logit(lambda X: forward_desde_json(red, X), grupos, d)
    X_expl = a_columnas(expl, grupos, d)
    np.testing.assert_array_equal(f(expl), forward_desde_json(red, X_expl))  # la misma función que la app
    phi, phi0 = valores_shap(f, fondo, expl, MAX_EVALS, 0)
    assert phi.shape == (N_EXPLICAR, N_JUGADORES)
    assert error_eficiencia(phi, phi0, f(expl)) < TOL_EFICIENCIA
    assert phi0 == pytest.approx(float(np.mean(f(fondo))), abs=TOL)
    # Forward float64 frente a MLPPD (torch float32): diferencia reportada, cota holgada en PD.
    dif_pd = np.max(np.abs(expit(f(expl)) - mlppd.predict_proba(X_expl.astype(np.float32))))
    assert dif_pd < TOL_PD_MLPPD


def test_04b_h21_caso_real_revol_bal_cerca_del_recorte(red, grupos, d, pre_json, por_split):
    """H-24 con filas reales de train: explicado = primera fila con revol_bal crudo a menos de 2 $ por debajo del
    recorte superior (p99 de train); fondo = las primeras N_FONDO filas por encima del recorte (quedan recortadas al
    mismo valor). Transformados, los dos valores son np.isclose pero distintos: antes del arreglo φ(revol_bal) = 0 y
    el error de eficiencia era 2,2e-6."""
    j = [v["nombre"] for v in pre_json["variables"]].index("revol_bal")
    techo = pre_json["variables"][j]["recorte"][1]
    crudo = por_split["train"]["df"]["revol_bal"].to_numpy(dtype=np.float64)
    i_x = int(np.flatnonzero((crudo < techo) & (crudo > techo - 2))[0])
    i_fondo = np.flatnonzero(crudo > techo)[:N_FONDO]
    Z = a_variables(por_split["train"]["X"][np.r_[i_x, i_fondo]], grupos)
    expl, fondo = Z[:1], Z[1:]
    assert np.all(np.isclose(expl[0, j], fondo[:, j])) and np.all(expl[0, j] != fondo[:, j])  # control de H-24
    f = funcion_logit(lambda X: forward_desde_json(red, X), grupos, d)
    phi, phi0 = valores_shap(f, fondo, expl, MAX_EVALS, 0)
    assert error_eficiencia(phi, phi0, f(expl)) < TOL_EFICIENCIA
    assert phi[0, j] != 0


# ------------------------------------------------------------------------------------------------------------------
# 5. Puntos del MLP: signo, puntos base y suma = score de C1
# ------------------------------------------------------------------------------------------------------------------


def test_05_puntos_signo_base_y_suma(sc_mlp, shap_mlp, por_split, pre, red):
    params = {k: sc_mlp[k] for k in CLAVES_PARAMETROS}
    factor = sc_mlp["factor"]
    phi = np.asarray(shap_mlp["explicados"]["phi"], dtype=np.float64)
    logit = np.asarray(shap_mlp["explicados"]["logit"], dtype=np.float64)
    # El logit del JSON es f(x) = forward float64 de los explicados.
    X_e = pre.transform(_explicados_crudos(por_split, shap_mlp["explicados"]["ids"]))
    np.testing.assert_allclose(logit, forward_desde_json(red, X_e), rtol=0, atol=1e-12)
    P = -factor * phi
    assert np.all(P[phi > 0] < 0) and np.all(P[phi < 0] > 0)  # más log-odds de malo ⇒ menos puntos
    assert sc_mlp["puntos_base"] == pytest.approx(sc_mlp["offset"] - factor * shap_mlp["phi0"], abs=TOL)
    score = np.asarray(score_desde_logit(logit, params), dtype=np.float64)
    assert np.max(np.abs(sc_mlp["puntos_base"] + P.sum(axis=1) - score)) < TOL
    assert sc_mlp["aditividad"]["error_max"] < TOL
    assert sc_mlp["aditividad"]["error_max"] <= sc_mlp["aditividad"]["tolerancia"]


def test_05b_ruta_de_construir_jugadores_agrupados_y_fondo_de_train(shap_mlp, por_split, red, grupos, d):
    """H-25: por la ruta de construir_scorecard_mlp, los φ son los de los 15 jugadores agrupados (no los de las 48
    columnas sumadas por grupo) y φ₀ es la media de f sobre las filas de TRAIN sorteadas con make_rng(base_seed, 0)
    (primero el fondo, después los explicados de test). Se recalcula con la interfaz pública y los mismos índices."""
    rng = make_rng(SHAP_APROBADO["base_seed"], 0)
    i_fondo = rng.choice(len(por_split["train"]["X"]), size=N_FONDO, replace=False)
    i_expl = rng.choice(len(por_split["test"]["X"]), size=N_EXPLICAR, replace=False)
    assert shap_mlp["explicados"]["ids"] == por_split["test"]["df"]["id"].astype(str).to_numpy()[i_expl].tolist()
    fondo = a_variables(por_split["train"]["X"][i_fondo], grupos)
    expl = a_variables(por_split["test"]["X"][i_expl], grupos)
    f = funcion_logit(lambda X: forward_desde_json(red, X), grupos, d)
    # (c) fondo de train: φ₀ = media de f sobre esas filas de train.
    assert shap_mlp["phi0"] == pytest.approx(float(np.mean(f(fondo))), abs=1e-12)
    # (b) jugadores agrupados: los mismos φ que valores_shap con Z (15 columnas) y la semilla base.
    phi, phi0 = valores_shap(f, fondo, expl, MAX_EVALS, SHAP_APROBADO["base_seed"])
    np.testing.assert_allclose(np.asarray(shap_mlp["explicados"]["phi"]), phi, rtol=0, atol=1e-12)
    assert shap_mlp["phi0"] == phi0


# ------------------------------------------------------------------------------------------------------------------
# 6. --set scorecard.pdo=40 duplica los puntos (misma preparación, otra corrida de SHAP y permutación)
# ------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corrida40(tmp_path_factory, preparado) -> tuple[dict, dict]:
    """Con --set scorecard.pdo=40 y el preparado de pdo 20: la preparación no depende de scorecard.*."""
    return construir_scorecard_mlp(_cfg_mlp(tmp_path_factory.mktemp("mlp40"), ["scorecard.pdo=40"]), preparado)


def _doble(a, b) -> None:
    if a is None:
        assert b is None
    else:
        assert b == pytest.approx(2 * a, rel=TOL_REL_PDO, abs=1e-12)


def test_06_pdo_40_duplica_los_puntos(sc_mlp, shap_mlp, corrida40):
    sc40, shap40 = corrida40
    _doble(sc_mlp["factor"], sc40["factor"])
    _doble(sc_mlp["puntos_base"] - sc_mlp["base"], sc40["puntos_base"] - sc40["base"])
    assert shap40["explicados"]["phi"] == shap_mlp["explicados"]["phi"]  # φ no depende de la scorecard
    assert shap40["phi0"] == shap_mlp["phi0"]
    for e20, e40 in zip(sc_mlp["puntos_por_variable"], sc40["puntos_por_variable"], strict=True):
        assert e20["nombre"] == e40["nombre"]
        _doble(e20["rango_puntos"], e40["rango_puntos"])
        filas20 = e20["tramos"] if e20["tipo"] == "num" else e20["categorias"]
        filas40 = e40["tramos"] if e40["tipo"] == "num" else e40["categorias"]
        for f20, f40 in zip(filas20, filas40, strict=True):
            assert f20["n"] == f40["n"]
            _doble(f20["puntos_medios"], f40["puntos_medios"])


# ------------------------------------------------------------------------------------------------------------------
# 7. Importancia por permutación (juguete)
# ------------------------------------------------------------------------------------------------------------------

# 8 filas en Z (a, índice de c, b); y con las dos clases. β de b = 0 ⇒ permutar b no cambia ninguna PD.
Z_PERM = np.array([[0.5, 0, 1], [-1, 1, 2], [2, 2, 3], [0, 0, 4], [1.5, 1, 5], [-0.5, 2, 6], [1, 0, 7], [-2, 1, 8]])
Y_PERM = np.array([1, 0, 1, 0, 1, 0, 1, 0])
COEF_PERM = np.array([1.0, 0.3, -0.4, 0.8, 0.0])  # a, c=0, c=1, c=2, b (β_b = 0)
B0_PERM = -0.2


def prob_perm(X) -> np.ndarray:
    return expit(np.asarray(X, dtype=np.float64) @ COEF_PERM + B0_PERM)


def test_07a_permutacion_formato_base_y_variable_irrelevante():
    res = importancia_permutacion(prob_perm, Z_PERM, Y_PERM, GRUPOS_JUGUETE, D_JUGUETE, 3, SEMILLA_PRUEBA)
    assert set(res) == CLAVES_PERMUTACION_RES
    p_base = prob_perm(a_columnas(Z_PERM, GRUPOS_JUGUETE, D_JUGUETE))
    assert res["auc_base"] == pytest.approx(auc(Y_PERM, p_base), abs=TOL_JUGUETE)
    assert res["logloss_base"] == pytest.approx(logloss(Y_PERM, p_base), abs=TOL_JUGUETE)  # la log-loss de B
    assert [v["nombre"] for v in res["variables"]] == NOMBRES_JUGUETE
    assert all(set(v) == CLAVES_PERMUTACION_VAR for v in res["variables"])
    b = res["variables"][2]
    for clave in CLAVES_PERMUTACION_VAR - {"nombre"}:
        assert b[clave] == pytest.approx(0.0, abs=1e-15), clave  # β_b = 0: la permutación no cambia nada
    json.dumps(to_jsonable(res), allow_nan=False)


def test_07b_permuta_cada_variable_entera_y_solo_esa():
    """Cada llamada a prob_fn recibe filas one-hot válidas; respecto de Z cambia como mucho una variable y sus valores
    son una permutación de los originales (las categóricas se permutan enteras, por índice)."""
    vistas = []

    def prob_registrada(X):
        vistas.append(np.array(X, dtype=np.float64, copy=True))
        return prob_perm(X)

    importancia_permutacion(prob_registrada, Z_PERM, Y_PERM, GRUPOS_JUGUETE, D_JUGUETE, 2, SEMILLA_PRUEBA)
    assert vistas
    permutadas = set()
    for X in vistas:
        bloque = X[:, 1:4]
        assert np.all((bloque == 0) | (bloque == 1)) and np.all(bloque.sum(axis=1) == 1)  # one-hot válido
        Z = a_variables(X, GRUPOS_JUGUETE)
        distintas = [j for j in range(3) if not np.array_equal(Z[:, j], Z_PERM[:, j])]
        assert len(distintas) <= 1
        for j in distintas:
            np.testing.assert_array_equal(np.sort(Z[:, j]), np.sort(Z_PERM[:, j]))
            permutadas.add(j)
    assert permutadas == {0, 1, 2}  # se permutó cada una de las tres variables


def test_07c_misma_semilla_mismo_resultado_y_desv_con_k_1():
    a = importancia_permutacion(prob_perm, Z_PERM, Y_PERM, GRUPOS_JUGUETE, D_JUGUETE, 3, SEMILLA_PRUEBA)
    b = importancia_permutacion(prob_perm, Z_PERM, Y_PERM, GRUPOS_JUGUETE, D_JUGUETE, 3, SEMILLA_PRUEBA)
    assert a == b
    uno = importancia_permutacion(prob_perm, Z_PERM, Y_PERM, GRUPOS_JUGUETE, D_JUGUETE, 1, SEMILLA_PRUEBA)
    for v in uno["variables"]:  # desviación muestral con n = 1: 0 por convención (núcleo §5.4)
        assert v["caida_auc_desv"] == 0.0 and v["aumento_logloss_desv"] == 0.0


# Caso a mano (H-25): 8 filas, a informativa (y = 1 sobre todo con a > 0), c categórica, b sin efecto (β_b = 0).
Z_PERM_MANO = np.array([[-2.0, 0, 1], [-1.5, 1, 2], [-1, 2, 3], [-0.5, 0, 4], [0.5, 1, 5], [1, 2, 6], [1.5, 0, 7],
                        [2, 1, 8]])
Y_PERM_MANO = np.array([0, 0, 1, 0, 1, 0, 1, 1])


def test_07d_permutacion_a_mano_media_desv_y_signos():
    """Cuentas hechas con Python (sklearn roc_auc_score y log_loss): AUC base 0.75, log-loss base 0.5766940422539151.
    El código saca las permutaciones de make_rng(semilla, 0).permutation en el orden (variable, repetición); con
    semilla 20261026:
      a: [7,0,5,4,3,2,1,6], [3,4,1,5,2,7,6,0], [7,4,1,6,3,5,0,2] → ΔAUC = 0.25, 0.5625, 0.75;
         ΔLL = 0.3446153038320314, 0.8104299550002172, 1.0332427532521875
      c: [1,3,5,6,7,4,0,2], [0,3,5,6,2,4,7,1], [7,2,5,3,4,1,6,0] → ΔAUC = −0.09375, −0.15625, −0.09375;
         ΔLL = −0.12281279825197022, −0.13322447535801024, −0.10223638624875497
      b: β = 0 → todo 0.
    Media y desviación MUESTRAL (ddof = 1): a ΔAUC 0.5208333333333334 ± 0.2525907427704613, ΔLL 0.7294293373614787 ±
    0.3513869178941075; c ΔAUC −0.11458333333333333 ± 0.03608439182435161, ΔLL −0.11942455328624514 ±
    0.015769450517427903 (con ddof = 0 serían 0.2062…, 0.2869…, 0.0295…, 0.0129…)."""
    res = importancia_permutacion(prob_perm, Z_PERM_MANO, Y_PERM_MANO, GRUPOS_JUGUETE, D_JUGUETE, 3, SEMILLA_PRUEBA)
    assert res["auc_base"] == pytest.approx(0.75, abs=TOL_JUGUETE)
    assert res["logloss_base"] == pytest.approx(0.5766940422539151, abs=TOL_JUGUETE)
    a, c, b = res["variables"]
    esperado = {
        "a": (0.5208333333333334, 0.2525907427704613, 0.7294293373614787, 0.3513869178941075),
        "c": (-0.11458333333333333, 0.03608439182435161, -0.11942455328624514, 0.015769450517427903),
        "b": (0.0, 0.0, 0.0, 0.0),
    }
    for v in (a, c, b):
        obtenido = (v["caida_auc_media"], v["caida_auc_desv"], v["aumento_logloss_media"], v["aumento_logloss_desv"])
        np.testing.assert_allclose(obtenido, esperado[v["nombre"]], rtol=0, atol=TOL_JUGUETE, err_msg=v["nombre"])
    # La variable informativa: permutarla EMPEORA el modelo (caída de AUC > 0 y aumento de log-loss > 0).
    assert a["caida_auc_media"] > 0 and a["aumento_logloss_media"] > 0


# ------------------------------------------------------------------------------------------------------------------
# 8. Fuga (lento: otra corrida completa con val y test alterados)
# ------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corrida_alterada(tmp_path_factory, datos, features) -> tuple[dict, dict]:
    """Corrida con las variables de val y test reemplazadas por filas de train sorteadas (semilla fija)."""
    carpeta = tmp_path_factory.mktemp("fuga_mlp")
    df = datos.copy()
    fuera = (df["split"] != "train").to_numpy()
    train = df[~fuera].reset_index(drop=True)
    idx = make_rng(SEMILLA_PRUEBA, 2).integers(0, len(train), size=int(fuera.sum()))
    for f in features:
        df.loc[fuera, f["nombre"]] = train[f["nombre"]].to_numpy()[idx]
    ruta = carpeta / "loans_pd_alterado.parquet"
    df.to_parquet(ruta, index=False)
    return construir_scorecard_mlp(_cfg_mlp(carpeta / "salida", datos=ruta))


@pytest.mark.lento
def test_08_fuga(sc_mlp, shap_mlp, corrida_alterada):
    sc_alt, shap_alt = corrida_alterada
    # Control: la alteración sí cambió val, test y los φ de los explicados (que son de test).
    for s in ("val", "test"):
        assert sc_alt["distribucion_scores"][s]["media"] != sc_mlp["distribucion_scores"][s]["media"]
    assert shap_alt["explicados"]["phi"] != shap_mlp["explicados"]["phi"]
    assert shap_alt["permutacion"]["mlp"]["auc_base"] != shap_mlp["permutacion"]["mlp"]["auc_base"]
    # Fondo de train: φ₀ = E_fondo[f] no cambia. Mismos explicados (filas de test elegidas por posición).
    assert shap_alt["phi0"] == pytest.approx(shap_mlp["phi0"], abs=1e-12)
    assert shap_alt["explicados"]["ids"] == shap_mlp["explicados"]["ids"]
    # Cortes de bandas, distribución de train, bordes de los tramos y fracciones de train: solo de train.
    assert sc_alt["distribucion_scores"]["cortes_bandas"] == sc_mlp["distribucion_scores"]["cortes_bandas"]
    assert sc_alt["distribucion_scores"]["train"] == sc_mlp["distribucion_scores"]["train"]
    for e, e_alt in zip(sc_mlp["puntos_por_variable"], sc_alt["puntos_por_variable"], strict=True):
        if e["tipo"] == "num":
            assert ([(t["desde"], t["hasta"]) for t in e["tramos"]]
                    == [(t["desde"], t["hasta"]) for t in e_alt["tramos"]]), e["nombre"]
        else:
            assert ([c["fraccion_train"] for c in e["categorias"]]
                    == [c["fraccion_train"] for c in e_alt["categorias"]]), e["nombre"]


def test_08b_explicados_y_permutacion_son_de_test(shap_mlp, por_split, red, logistica):
    ids_test = set(por_split["test"]["df"]["id"])
    ids = shap_mlp["explicados"]["ids"]
    assert len(ids) == N_EXPLICAR and len(set(ids)) == N_EXPLICAR and set(ids) <= ids_test
    perm = shap_mlp["permutacion"]
    assert perm["conjunto"] == "test"
    X_te, y_te = por_split["test"]["X"], por_split["test"]["y"]
    p_log = logistica.predict_proba(X_te)
    assert perm["logistica"]["auc_base"] == pytest.approx(auc(y_te, p_log), abs=TOL)
    assert perm["logistica"]["logloss_base"] == pytest.approx(logloss(y_te, p_log), abs=TOL)
    p_mlp = expit(forward_desde_json(red, X_te))
    assert perm["mlp"]["auc_base"] == pytest.approx(auc(y_te, p_mlp), abs=TOL_PD_MLPPD)
    assert perm["mlp"]["logloss_base"] == pytest.approx(logloss(y_te, p_mlp), abs=TOL_PD_MLPPD)


# ------------------------------------------------------------------------------------------------------------------
# 9. Formato de scorecard_mlp.json y shap_mlp.json
# ------------------------------------------------------------------------------------------------------------------


def test_09a_formato_scorecard_mlp(sc_mlp, features, summary_pd, por_split, red, shap_mlp, pre, grupos):
    assert set(sc_mlp) == set(load_json(RUTA_SCORECARD_C3))  # mismas claves de raíz que scorecard.json de C3
    assert sc_mlp["modelo"] == "mlp"
    assert {k: sc_mlp[k] for k in CLAVES_PARAMETROS} == pytest.approx(
        parametros_scorecard({"base": 600, "odds_base": 50, "pdo": 20}))
    assert set(sc_mlp["aditividad"]) == {"error_max", "tolerancia"}
    assert sc_mlp["evidencia"] == summary_pd["mlp"]["evidencia"]
    factor = sc_mlp["factor"]
    phi = np.asarray(shap_mlp["explicados"]["phi"], dtype=np.float64)
    crudos = _explicados_crudos(por_split, shap_mlp["explicados"]["ids"])
    Z_e = a_variables(pre.transform(crudos), grupos)
    train = por_split["train"]["df"]
    assert [e["nombre"] for e in sc_mlp["puntos_por_variable"]] == [f["nombre"] for f in features]
    for j, (f, e) in enumerate(zip(features, sc_mlp["puntos_por_variable"], strict=True)):
        assert e["tipo"] == f["tipo"] and e["etiqueta_es"] == f["etiqueta_es"]
        P_j = -factor * phi[:, j]
        if f["tipo"] == "num":
            assert set(e) == CLAVES_NUM_MLP
            assert all(set(t) == CLAVES_TRAMO for t in e["tramos"])
            bordes = np.percentile(train[f["nombre"]].dropna().to_numpy(dtype=np.float64), [1, 5, 25, 50, 75, 95, 99])
            assert e["tramos"][0]["desde"] is None and e["tramos"][-1]["hasta"] is None  # colas abiertas
            np.testing.assert_allclose([t["desde"] for t in e["tramos"][1:]], bordes, rtol=0, atol=1e-12)
            np.testing.assert_allclose([t["hasta"] for t in e["tramos"][:-1]], bordes, rtol=0, atol=1e-12)
            crudo = crudos[f["nombre"]].to_numpy(dtype=np.float64)
            con_valor = ~np.isnan(crudo)
            assert e["n_sin_valor"] == int((~con_valor).sum())
            assert sum(t["n"] for t in e["tramos"]) == int(con_valor.sum())
            # Σ n·puntos_medios de los tramos = Σ puntos de los explicados con valor (vale con cualquier convención
            # de borde; la convención [desde, hasta) frente a (desde, hasta] no está fijada en el plan).
            suma = sum(t["n"] * t["puntos_medios"] for t in e["tramos"] if t["n"] > 0)
            assert suma == pytest.approx(float(P_j[con_valor].sum()), abs=1e-6)
            interiores = [t["puntos_medios"] for t in e["tramos"]
                          if t["desde"] is not None and t["hasta"] is not None and t["n"] > 0]
            if interiores:
                assert e["rango_puntos"] == pytest.approx(max(interiores) - min(interiores), abs=TOL)
        else:
            assert set(e) == CLAVES_CAT_MLP
            assert all(set(c) == CLAVES_CATEGORIA_MLP for c in e["categorias"])
            assert [c["categoria"] for c in e["categorias"]] == list(f["categorias_es"])  # orden de la app
            assert all(c["etiqueta"] == f["categorias_es"][c["categoria"]] for c in e["categorias"])
            cats_pre = next(v for v in pre.to_dict()["variables"] if v["nombre"] == f["nombre"])["categorias"]
            frecuentes = []
            for c in e["categorias"]:
                activos = Z_e[:, j] == cats_pre.index(c["categoria"])
                assert c["n"] == int(activos.sum()), (f["nombre"], c["categoria"])
                if c["n"] > 0:
                    assert c["puntos_medios"] == pytest.approx(float(P_j[activos].mean()), abs=TOL)
                    if c["fraccion_train"] >= 0.01:
                        frecuentes.append(c["puntos_medios"])
            if frecuentes:
                assert e["rango_puntos"] == pytest.approx(max(frecuentes) - min(frecuentes), abs=TOL)
    # fraccion_train de las categóricas = la de scorecard.json de C3 (misma media one-hot de train).
    c3 = {e["nombre"]: e for e in load_json(RUTA_SCORECARD_C3)["puntos_por_variable"]}
    for e in sc_mlp["puntos_por_variable"]:
        if e["tipo"] == "cat":
            esperado = {c["categoria"]: c["fraccion_train"] for c in c3[e["nombre"]]["categorias"]}
            assert {c["categoria"]: c["fraccion_train"] for c in e["categorias"]} == pytest.approx(esperado,
                                                                                                    abs=1e-12)
    # Distribución del puntaje del MLP (forward float64) en los tres splits completos; cortes = deciles de train.
    params = {k: sc_mlp[k] for k in CLAVES_PARAMETROS}
    ds = sc_mlp["distribucion_scores"]
    assert set(ds) == CLAVES_DIST_SCORES and ds["n_bandas"] == 10 and ds["ancho_bin"] == 10
    scores = {s: np.asarray(score_desde_logit(forward_desde_json(red, por_split[s]["X"]), params)) for s in SPLITS}
    np.testing.assert_allclose(ds["cortes_bandas"], cortes_bandas(scores["train"], 10), rtol=0, atol=TOL)
    for s in SPLITS:
        assert set(ds[s]) == CLAVES_SPLIT
        assert ds[s]["n"] == len(scores[s])
        assert ds[s]["media"] == pytest.approx(float(np.mean(scores[s])), abs=TOL)
        np.testing.assert_allclose(ds[s]["percentiles"], np.percentile(scores[s], np.arange(101)), rtol=0, atol=TOL)
        assert sum(ds[s]["histograma"]["conteos"]) == len(scores[s])
    json.dumps(to_jsonable(sc_mlp), allow_nan=False)  # sin NaN ni inf


def test_09b_formato_shap_mlp(shap_mlp, nombres, features, summary_pd, por_split, pre, red, mlppd):
    assert set(shap_mlp) == CLAVES_SHAP_JSON
    assert shap_mlp["modelo"] == "mlp"
    m = shap_mlp["metodo"]
    assert set(m) == CLAVES_METODO
    assert "Permutation" in m["explainer"] and "Independent" in m["masker"]
    assert m["salida"] == "logit" and m["fondo"] == "train" and m["explicados"] == "test"
    assert m["version_shap"] == shap.__version__
    assert m["jugadores"] == nombres
    assert (m["n_fondo"], m["n_explicar"], m["n_estabilidad"], m["max_evals"]) == (N_FONDO, N_EXPLICAR,
                                                                                   N_ESTABILIDAD, MAX_EVALS)
    assert m["n_permutaciones"] == MAX_EVALS // (2 * N_JUGADORES + 1) == 2
    assert m["semillas"] == [SHAP_APROBADO["base_seed"], SHAP_APROBADO["base_seed"] + 1]
    ex = shap_mlp["explicados"]
    assert set(ex) == {"ids", "phi", "logit"}
    phi = np.asarray(ex["phi"], dtype=np.float64)
    logit = np.asarray(ex["logit"], dtype=np.float64)
    assert phi.shape == (N_EXPLICAR, N_JUGADORES) and logit.shape == (N_EXPLICAR,)
    # Eficiencia (aceptación de C4) recalculada desde el JSON.
    assert set(shap_mlp["eficiencia"]) == {"error_max", "tolerancia"}
    assert shap_mlp["eficiencia"]["tolerancia"] == TOL_EFICIENCIA
    assert shap_mlp["eficiencia"]["error_max"] < TOL_EFICIENCIA
    assert error_eficiencia(phi, shap_mlp["phi0"], logit) < TOL_EFICIENCIA
    vl = shap_mlp["verificacion_logistica"]
    assert set(vl) == {"error_max", "tolerancia", "n"}
    assert vl["error_max"] < TOL and vl["error_max"] <= vl["tolerancia"]
    assert isinstance(vl["n"], int) and vl["n"] >= 1
    # Diferencia forward float64 frente a MLPPD en los explicados (reportada).
    X_e = pre.transform(_explicados_crudos(por_split, ex["ids"]))
    l64, l32 = forward_desde_json(red, X_e), mlppd.predict_logit(X_e)
    dm = shap_mlp["diferencia_mlppd"]
    assert set(dm) == {"max_abs_pd", "max_abs_logit"}
    assert dm["max_abs_logit"] == pytest.approx(float(np.max(np.abs(l64 - l32))), abs=1e-12)
    assert dm["max_abs_pd"] == pytest.approx(float(np.max(np.abs(expit(l64) - expit(l32)))), abs=1e-12)
    assert dm["max_abs_pd"] < TOL_PD_MLPPD
    # Importancia global = la de los φ del JSON; dirección con las 15 variables.
    ig = shap_mlp["importancia_global"]
    esperado = importancia_global(phi, nombres)
    assert [e["nombre"] for e in ig] == [e["nombre"] for e in esperado]
    np.testing.assert_allclose([e["media_abs_phi"] for e in ig], np.sort(np.abs(phi).mean(axis=0))[::-1],
                               rtol=0, atol=TOL)
    assert set(shap_mlp["direccion"]) == set(nombres)
    for f in features:
        assert shap_mlp["direccion"][f["nombre"]]["tipo"] == f["tipo"]
    # Permutación de los dos modelos.
    perm = shap_mlp["permutacion"]
    assert set(perm) == {"k", "semilla", "conjunto", "logistica", "mlp"}
    assert perm["k"] == K_PERMUTACION and isinstance(perm["semilla"], int)
    for modelo in ("logistica", "mlp"):
        assert set(perm[modelo]) == CLAVES_PERMUTACION_RES
        assert [v["nombre"] for v in perm[modelo]["variables"]] == nombres
        assert all(set(v) == CLAVES_PERMUTACION_VAR for v in perm[modelo]["variables"])
    assert shap_mlp["evidencia"] == summary_pd["mlp"]["evidencia"]
    assert set(shap_mlp["tiempos_s"]) == CLAVES_TIEMPOS
    assert all(isinstance(v, float) and v >= 0 for v in shap_mlp["tiempos_s"].values())
    json.dumps(to_jsonable(shap_mlp), allow_nan=False)  # sin NaN ni inf


def test_09c_tramos_con_bordes_repetidos_pub_rec(sc_mlp, shap_mlp, por_split):
    """pub_rec tiene bordes de train [0, 0, 0, 0, 0, 0, 1]: con tramos [desde, hasta), el 0 cae en [0, 1) (el único
    tramo interior con datos), el 1 o más en la cola [1, ∞) y los tramos [0, 0) quedan vacíos (puntos_medios None);
    por eso su rango_puntos en el MLP es 0."""
    e = next(x for x in sc_mlp["puntos_por_variable"] if x["nombre"] == "pub_rec")
    crudo = _explicados_crudos(por_split, shap_mlp["explicados"]["ids"])["pub_rec"].to_numpy(dtype=np.float64)
    esperados = [(None, 0.0)] + [(0.0, 0.0)] * 5 + [(0.0, 1.0), (1.0, None)]
    assert [(t["desde"], t["hasta"]) for t in e["tramos"]] == esperados
    assert all(t["n"] == 0 and t["puntos_medios"] is None for t in e["tramos"][:6])
    assert e["tramos"][6]["n"] == int((crudo == 0).sum()) > 0
    assert e["tramos"][7]["n"] == int((crudo >= 1).sum())
    assert e["rango_puntos"] == 0.0


def test_09d_categoria_sin_explicados(sc_mlp):
    """Una categoría sin explicados tiene puntos_medios None y no entra en el rango."""
    vacias = [(e, c) for e in sc_mlp["puntos_por_variable"] if e["tipo"] == "cat" for c in e["categorias"]
              if c["n"] == 0]
    assert vacias  # control: con 30 explicados hay categorías raras sin casos (p. ej. home_ownership NONE)
    for e, c in vacias:
        assert c["puntos_medios"] is None, (e["nombre"], c["categoria"])
        con_datos = [x["puntos_medios"] for x in e["categorias"] if x["n"] > 0 and x["fraccion_train"] >= 0.01]
        if con_datos:
            assert e["rango_puntos"] == pytest.approx(max(con_datos) - min(con_datos), abs=TOL)


# ------------------------------------------------------------------------------------------------------------------
# 10. Configuración
# ------------------------------------------------------------------------------------------------------------------


def test_10a_config_mlp_yaml():
    cfg = load_config(RUTA_MLP)
    assert cfg == CFG_MLP_ESPERADA
    validar_cfg_puntos(cfg)
    assert validar_cfg_shap(cfg["shap"]) == SHAP_APROBADO
    assert load_config(RUTA_BASE)["modelo"] == "logistica" and "shap" not in load_config(RUTA_BASE)


def test_10b_validar_cfg_shap_no_muta():
    cfg = copy.deepcopy(SHAP_APROBADO)
    assert validar_cfg_shap(cfg) == SHAP_APROBADO
    assert cfg == SHAP_APROBADO


def _shap_con(**cambios) -> dict:
    return {**SHAP_APROBADO, **cambios}


def _shap_sin(clave: str) -> dict:
    return {k: v for k, v in SHAP_APROBADO.items() if k != clave}


SHAP_INVALIDOS = {
    "no_dict": [1, 2],
    "falta_clave": _shap_sin("k_permutacion"),
    "sobra_clave": _shap_con(extra=1),
    "base_seed_negativa": _shap_con(base_seed=-1),
    "n_fondo_0": _shap_con(n_fondo=0),
    "n_explicar_0": _shap_con(n_explicar=0),
    "n_estabilidad_0": _shap_con(n_estabilidad=0),
    "n_estabilidad_mayor": _shap_con(n_explicar=10, n_estabilidad=11),
    "max_evals_30": _shap_con(max_evals=30),  # 2·15 + 1 = 31: con 30 shap no completa una permutación
    "k_0": _shap_con(k_permutacion=0),
    "bool": _shap_con(n_fondo=True),
    "bool_base_seed": _shap_con(base_seed=False),
    "real": _shap_con(n_explicar=1000.0),
    "texto": _shap_con(max_evals="500"),
    "none": _shap_con(k_permutacion=None),
}


@pytest.mark.parametrize("nombre", sorted(SHAP_INVALIDOS))
def test_10c_validar_cfg_shap_invalida_es_error(nombre):
    cfg = SHAP_INVALIDOS[nombre]
    copia = copy.deepcopy(cfg)
    with pytest.raises(ValueError):
        validar_cfg_shap(cfg)
    assert cfg == copia


def test_10d_validar_cfg_shap_bordes_validos():
    # max_evals = 31 (una permutación), n_estabilidad = n_explicar y mínimos en 1 son válidos.
    validar_cfg_shap(_shap_con(max_evals=31, n_fondo=1, n_explicar=1, n_estabilidad=1, k_permutacion=1, base_seed=0))
    validar_cfg_shap(_shap_con(n_explicar=7, n_estabilidad=7))


def test_10e_raiz_segun_el_modelo():
    mlp_sin_shap = {k: v for k, v in copy.deepcopy(CFG_MLP_ESPERADA).items() if k != "shap"}
    with pytest.raises(ValueError):
        validar_cfg_puntos(mlp_sin_shap)
    logistica_con_shap = copy.deepcopy(CFG_MLP_ESPERADA)
    logistica_con_shap["modelo"] = "logistica"
    logistica_con_shap["rutas"]["modelo"] = "models/pd_logistica.joblib"
    del logistica_con_shap["rutas"]["logistica"]
    with pytest.raises(ValueError):
        validar_cfg_puntos(logistica_con_shap)
    assert set(validar_cfg_puntos(copy.deepcopy(CFG_MLP_ESPERADA))) == {*CLAVES_RAIZ, "shap"}


def test_10f_ruta_de_la_logistica_solo_con_mlp():
    """Con modelo mlp, rutas exige además `logistica` (verificación y permutación); con logística, rutas no cambia."""
    mlp_sin_ruta = copy.deepcopy(CFG_MLP_ESPERADA)
    del mlp_sin_ruta["rutas"]["logistica"]
    with pytest.raises(ValueError):
        validar_cfg_puntos(mlp_sin_ruta)
    logistica_con_ruta = {k: v for k, v in copy.deepcopy(CFG_MLP_ESPERADA).items() if k != "shap"}
    logistica_con_ruta["modelo"] = "logistica"
    logistica_con_ruta["rutas"]["modelo"] = "models/pd_logistica.joblib"
    with pytest.raises(ValueError):
        validar_cfg_puntos(logistica_con_ruta)
    del logistica_con_ruta["rutas"]["logistica"]
    validar_cfg_puntos(logistica_con_ruta)  # control: sin la clave extra es válida


# ------------------------------------------------------------------------------------------------------------------
# 11. Estabilidad
# ------------------------------------------------------------------------------------------------------------------


def test_11a_estabilidad_a_mano():
    # media |φ|: a [3, 2, 1] en el orden (a, c, b) ⇒ ranking a, c, b; b [1, 2, 3] ⇒ ranking b, c, a.
    # Spearman entre las importancias [3, 2, 1] y [1, 2, 3] = −1.
    phi_a = np.array([[3.0, -2.0, 1.0], [-3.0, 2.0, -1.0]])
    phi_b = np.array([[1.0, 2.0, -3.0], [-1.0, -2.0, 3.0]])
    res = estabilidad(phi_a, phi_b, NOMBRES_JUGUETE)
    assert set(res) == {"ranking_a", "ranking_b", "spearman_importancia"}
    assert res["ranking_a"] == ["a", "c", "b"]  # supuesto: lista de nombres de mayor a menor media |φ|
    assert res["ranking_b"] == ["b", "c", "a"]
    assert res["spearman_importancia"] == pytest.approx(-1.0, abs=TOL_JUGUETE)
    assert estabilidad(phi_a, phi_a, NOMBRES_JUGUETE)["spearman_importancia"] == pytest.approx(1.0, abs=TOL_JUGUETE)


def test_11b_estabilidad_en_el_json_usa_los_primeros_n_estabilidad(shap_mlp, nombres):
    est = shap_mlp["estabilidad"]
    assert {"ranking_a", "ranking_b", "spearman_importancia"} <= set(est)
    phi = np.asarray(shap_mlp["explicados"]["phi"], dtype=np.float64)
    media = np.abs(phi[:N_ESTABILIDAD]).mean(axis=0)  # solo los primeros n_estabilidad explicados
    assert est["ranking_a"] == [nombres[j] for j in np.argsort(-media, kind="stable")]
    assert sorted(est["ranking_b"]) == sorted(nombres)
    assert -1.0 <= est["spearman_importancia"] <= 1.0


# ------------------------------------------------------------------------------------------------------------------
# 12. Manejadores
# ------------------------------------------------------------------------------------------------------------------


def test_12a_manejador_mlp_escribe_los_dos_json(tmp_path, monkeypatch, corrida, capsys):
    """manejador_mlp escribe scorecard_mlp.json y shap_mlp.json en `salida` y devuelve 0. construir_scorecard_mlp se
    sustituye por la corrida ya construida: el recorrido de extremo a extremo lo cubre
    tests/test_no_network.py::...[scorecard-mlp]."""
    monkeypatch.setattr(shap_pd, "construir_scorecard_mlp", lambda cfg: corrida)
    assert manejador_mlp(_cfg_mlp(tmp_path)) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([ARCHIVO_SCORECARD_MLP, ARCHIVO_SHAP])
    assert set(load_json(tmp_path / ARCHIVO_SCORECARD_MLP)) == set(corrida[0])
    assert set(load_json(tmp_path / ARCHIVO_SHAP)) == CLAVES_SHAP_JSON
    salida = capsys.readouterr().out
    for texto in ("Error de aditividad", "Error de eficiencia", "Verificación con la logística", "rango de puntos",
                  "Tiempos por etapa", ARCHIVO_SCORECARD_MLP, ARCHIVO_SHAP):
        assert texto in salida, texto
    rango_max = max((e for e in corrida[0]["puntos_por_variable"] if e["rango_puntos"] is not None),
                    key=lambda e: e["rango_puntos"])
    assert rango_max["etiqueta_es"] in salida  # el top por rango lleva la etiqueta


def test_12b_manejador_de_puntos_deriva_al_mlp(tmp_path, monkeypatch):
    llamadas = []
    monkeypatch.setattr(shap_pd, "manejador_mlp", lambda cfg: llamadas.append(cfg) or 0)
    assert puntos.manejador(_cfg_mlp(tmp_path)) == 0
    assert len(llamadas) == 1 and llamadas[0]["modelo"] == "mlp"
    assert not any(tmp_path.iterdir())  # el manejador de la logística no escribió nada


def test_12c_signo_de_cero_en_los_puntos(sc_mlp):
    """Ningún valor del JSON es −0.0 (el JSON escribiría "-0.0"; misma regla que C3)."""
    def recorrer(x):
        if isinstance(x, dict):
            for v in x.values():
                recorrer(v)
        elif isinstance(x, list):
            for v in x:
                recorrer(v)
        elif isinstance(x, float) and x == 0:
            assert math.copysign(1.0, x) == 1.0

    recorrer(to_jsonable(sc_mlp["puntos_por_variable"]))


@pytest.mark.parametrize("override", ["shap.max_evals=30", "scorecard.pdo=-1"])
def test_12d_config_invalida_da_codigo_2_sin_traceback(tmp_path, monkeypatch, capsys, override):
    from src.common import cli

    monkeypatch.chdir(RAIZ)
    argv = ["scorecard", "--config", "configs/scorecard/mlp.yaml", "--set", f"salida={tmp_path.as_posix()}",
            "--set", override]
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert "Config inválida" in err and "Traceback" not in err
    assert not any(tmp_path.iterdir())


def test_12e_config_invalida_tras_cargar_los_datos(preparado, tmp_path, monkeypatch, capsys):
    """Pedir más explicados que filas de test es un error de config (ConfigInvalida, subclase de ValueError) que solo
    se detecta con los datos cargados; el manejador lo convierte en código 2."""
    grande = _cfg_mlp(tmp_path, ["shap.n_explicar=1000000"])
    with pytest.raises(puntos.ConfigInvalida):
        construir_scorecard_mlp(grande, preparado)
    assert issubclass(puntos.ConfigInvalida, ValueError)

    def lanza(cfg):
        raise puntos.ConfigInvalida("shap pide más filas de test de las que hay")

    monkeypatch.setattr(shap_pd, "manejador_mlp", lanza)
    assert puntos.manejador(_cfg_mlp(tmp_path)) == 2
    assert "Config inválida" in capsys.readouterr().err
