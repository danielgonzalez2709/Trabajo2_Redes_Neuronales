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
# # Pruebas de las 30 semillas del MLP de PD y de `run.py pd` (Parte B §3.5, tarea B5)
#
# Se prueban nuestras decisiones (Andrés, 26-oct; Spec v0.16): la semilla mediana por (log-loss de val, semilla) sin
# mirar test, la procedencia de la config (`config_de_busqueda`) y su presupuesto, la evidencia de la búsqueda frente
# a la de la corrida (antes de borrar y de entrenar), el borrado acotado a `mlp_seed<entero>.json`, la validación antes
# de leer datos, los archivos del núcleo §5.4 (semillas, entrada `mlp` de `summary.json`, `pred_mlp` y `pd_mlp.pt`)
# conservando lo de la logística, el tiempo de solo el `fit`, el determinismo y el subcomando `run.py pd`.
#
# **Datos:** sintéticos, generados aquí con una logística conocida (núcleo §0 regla 8: solo para desarrollar), con un
# preprocesador ajustado solo con su train. Las únicas excepciones leen los datos reales de A: la prueba 26 (huella del
# parquet real) y la 27 (marcada `lento`, corre `correr_pd` con 2 semillas).
#
# **Nunca** se escribe en `results/`, `models/` ni `configs/` reales: todo va a `tmp_path` (la prueba 24 lo comprueba).
#
# Se usa el módulo (`ex.<nombre>`) y no `from … import`: cada prueba falla por separado mientras su función no existe.

# %%
"""Pruebas de las celdas de B5 en src/pd/experimento.py y del subcomando run.py pd (Parte B §3.5)."""

import copy  # copias profundas de configs y bloques
import itertools  # reloj falso que avanza a paso fijo
import re  # formato de la línea del MLP en la consola
from pathlib import Path  # rutas de los archivos de prueba

import numpy as np  # datos sintéticos y comparaciones
import pandas as pd  # parquet sintético y pred_test
import pytest  # fixtures, parametrización y pytest.raises
import torch  # hilos y versión registrados en los JSON
import yaml  # base.yaml temporal
from scipy.special import expit  # σ de la logística generadora y del modelo falso

from src.common import cli  # CLI de run.py (prueba 25)
from src.common.config import apply_overrides, load_config  # mismo cargador que el CLI (--set)
from src.common.preprocesamiento import Preprocesador  # preprocesador común (núcleo §5.2)
from src.common.results import load_json, save_json  # JSON del núcleo §5.4
from src.pd import experimento as ex  # módulo bajo prueba
from src.pd import hpo  # bloque busqueda y TIPO_BUSQUEDA (prueba 4) y validar_cfg (prueba 26)
from src.pd.entrenar import ResultadoEntrenamiento  # resultado del modelo falso
from src.pd.mlp import MLPPD, validar_cfg_mlp  # modelo real y reglas del bloque mlp (B2)

# %% [markdown]
# ## Constantes de las pruebas
#
# Son valores de prueba, no constantes del dominio: tamaños pequeños para que cada corrida tarde poco.

# %%
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio (configs versionadas y datos reales)
SEMILLA_DATOS = 20261026  # semilla de los datos sintéticos (fecha de la tarea)
N_EXP = {"train": 300, "val": 150, "test": 150}  # filas por split del parquet sintético
ANIO_SPLIT = {"train": 2012, "val": 2013, "test": 2014}  # año de originación de cada split (como la partición real)
CATEGORIAS_EXP = ["A", "B", "C"]  # valores de la categórica sintética
FEATURES_EXP = [  # variables del parquet sintético: 2 numéricas + 1 categórica
    {"nombre": "monto", "tipo": "num", "transformacion": "log1p"},  # numérica con log1p
    {"nombre": "razon", "tipo": "num", "transformacion": "ninguna"},  # numérica sin transformar
    {"nombre": "clase", "tipo": "cat", "transformacion": "ninguna"},  # categórica
]
BASE_SEED = 7  # base_seed de la corrida sintética (no 0: atrapa semillas que ignoran la base)
N_SEMILLAS = 3  # semillas de la corrida sintética
MLP_ESCENARIO = {  # bloque mlp de mlp_final.yaml temporal: válido y rápido
    "capas": [4], "activacion": "relu", "dropout": 0.0, "l2": 1.0e-4, "optimizador": "adam", "lr": 1.0e-2,
    "momento": 0.9, "batch": 64, "max_epocas": 2, "paciencia": 2, "device": "cpu", "pos_weight": 1.0,
}
BUSQUEDA_SIN_EVIDENCIA = {  # bloque busqueda del YAML temporal (la evidencia se agrega con la del escenario)
    "tipo": "optuna", "ensayo": 3, "n_ensayos": 5, "tiempo_s": 12.5, "fecha": "2026-10-26T00:00:00",
}
REJILLA_CORTA = [1.0e-2, 1.0]  # rejilla de C del escenario (rápido)
CAPAS_OTRAS = [8]  # capas del --set de la prueba 20
PASO_RELOJ = 1.0  # avance del reloj falso por llamada (exacto en binario)
PENDIENTE_FALSA = 0.5  # pendiente de la PD del modelo falso sobre la columna 0
DESPLAZAMIENTO_FALSO = 0.01  # cambio del logit (o de la PD constante) del modelo falso por unidad de semilla
PD_BASE_FALSA = 0.6  # PD constante del modelo falso monótono: 0.67…0.69 > ȳ de val → log-loss creciente en la semilla
N_PARAM_FALSO = 11  # parámetros que declara el modelo falso
MENSAJE_DIVERGE = "Pérdida de train no finita (nan) en la época 1"  # FloatingPointError simulado
AVISO = ("summary.json conserva la entrada mlp anterior, que ya no corresponde a semillas/; "
         "vuelve a correr run.py pd")  # texto exacto del aviso (Spec v0.16)
CLAVES_SEMILLA = {  # núcleo §5.4 + evidencia + extras de PD mlp (v0.16)
    "modelo", "seed", "config", "metricas", "n_parametros", "tiempo_s", "device", "epocas", "evidencia",
    "mejor_epoca", "mejor_perdida_val", "historial", "hilos_torch", "version_torch", "calibracion_global", "n",
}
CLAVES_SUMMARY_MLP = {  # núcleo §5.4 + extras de PD mlp (v0.16)
    "n_parametros", "n_configuraciones_busqueda", "tiempo_total_s", "tiempo_busqueda_s", "tiempo_semillas_s",
    "metricas", "evidencia", "semillas", "semilla_mediana", "criterio_semilla_mediana", "config",
    "config_de_busqueda", "busqueda", "hilos_torch", "version_torch",
}
CLAVES_CALIBRACION = {"pd_media_val", "tasa_val", "pd_media_test", "tasa_test"}  # calibracion_global
TIEMPOS_SEMILLA = {"tiempo_s"}  # campos de tiempo de una semilla
TIEMPOS_SUMMARY = {"tiempo_total_s", "tiempo_semillas_s"}  # campos de tiempo del summary que cambian


# %% [markdown]
# ## Escenario sintético y dobles
#
# - `escenario` (módulo): parquet sintético + preprocesador ajustado con su train, `mlp_final.yaml` temporal escrito
#   con `hpo.texto_mlp_final` (el mismo texto que produce `pd-hpo`) y su `busqueda` con la evidencia del escenario, y
#   un `base.yaml` temporal que lo hereda; `cfg` es lo que devuelve `load_config`.
# - `_fabrica_falsa`: modelo falso que cuenta sus `fit` y puede lanzar `FloatingPointError` en la k-ésima llamada.
# - `_CargarContador`: envuelve `cargar_datos` y cuenta sus llamadas.
# - `_reloj`: 0, 1, 2, … en llamadas sucesivas.

# %%
def _parquet_sintetico(ruta):
    """Escribe un loans_pd.parquet sintético con el esquema del núcleo §5.1 y devuelve el DataFrame."""
    rng = np.random.default_rng(SEMILLA_DATOS)  # generador del parquet sintético
    partes = []  # un DataFrame por split
    for split, n in N_EXP.items():  # train, val, test
        monto = rng.lognormal(mean=9.0, sigma=0.5, size=n)  # numérica positiva (admite log1p)
        razon = rng.normal(size=n)  # numérica centrada
        clase = rng.choice(CATEGORIAS_EXP, size=n)  # categórica
        logit = 0.8 * razon + 0.5 * (clase == "C") - 1.0  # logística generadora
        partes.append(pd.DataFrame({  # filas del split
            "issue_d": pd.Timestamp(f"{ANIO_SPLIT[split]}-06-01"),  # fecha de originación
            "anio_origen": np.int32(ANIO_SPLIT[split]),  # año de originación (int)
            "target": rng.binomial(1, expit(logit)).astype(np.int8),  # objetivo 0/1
            "split": split,  # partición
            "monto": monto, "razon": razon, "clase": clase,  # variables crudas
        }))
    df = pd.concat(partes, ignore_index=True)  # un solo DataFrame en orden train, val, test
    df.insert(0, "id", [f"id{i:05d}" for i in rng.permutation(len(df))])  # id str, no ordenado
    df.to_parquet(ruta, index=False)  # escribe el parquet
    return df  # DataFrame escrito


def _escribir_mlp_final(ruta, mlp, busqueda):
    """Escribe un mlp_final.yaml con el mismo texto que genera pd-hpo y devuelve su ruta."""
    Path(ruta).write_text(hpo.texto_mlp_final(mlp, busqueda), encoding="utf-8")  # texto de pd-hpo
    return Path(ruta)  # ruta escrita


def _escribir_base(carpeta, n=N_SEMILLAS, base_seed=BASE_SEED):
    """Escribe base.yaml (hereda mlp_final.yaml de la misma carpeta) con semillas, salida y modelos temporales."""
    cuerpo = {  # raíz de la config, como configs/pd/base.yaml
        "hereda": "mlp_final.yaml",  # bloque mlp y busqueda
        "base_seed": base_seed,  # semilla base
        "logistica": {"rejilla_C": list(REJILLA_CORTA), "max_iter": 1000, "tol": 1.0e-6},  # logística rápida
        "semillas": {"n": n},  # número de semillas
        "salida": (carpeta / "res").as_posix(),  # resultados temporales
        "modelos": (carpeta / "models").as_posix(),  # modelos temporales
    }
    ruta = carpeta / "base.yaml"  # junto a mlp_final.yaml
    ruta.write_text(yaml.safe_dump(cuerpo, sort_keys=False), encoding="utf-8")  # YAML
    return ruta  # ruta escrita


@pytest.fixture(scope="module")
def escenario(tmp_path_factory):
    """Parquet sintético, preprocesador, mlp_final.yaml y base.yaml temporales; cfg resuelta y su evidencia."""
    base = tmp_path_factory.mktemp("esc_b5")  # carpeta del escenario
    df = _parquet_sintetico(base / "loans_pd.parquet")  # parquet sintético
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES_EXP)  # preprocesador ajustado SOLO con train
    save_json(base / "preprocesador_pd.json", pre.to_dict())  # guardado como lo deja A
    huellas = ex.evidencia(df, base / "preprocesador_pd.json")  # evidencia del escenario
    busqueda = {**BUSQUEDA_SIN_EVIDENCIA, "evidencia": huellas}  # bloque busqueda coherente con los datos
    ruta_final = _escribir_mlp_final(base / "mlp_final.yaml", MLP_ESCENARIO, busqueda)  # YAML de pd-hpo
    ruta_base = _escribir_base(base)  # base.yaml temporal
    return {  # todo lo que usan las pruebas
        "base": base, "df": df, "huellas": huellas, "busqueda": busqueda, "ruta_final": ruta_final,
        "ruta_base": ruta_base, "cfg": load_config(ruta_base),  # cfg como la deja load_config
        "parquet": base / "loans_pd.parquet", "pre": base / "preprocesador_pd.json",  # entradas
        "n_columnas": len(pre.nombres_salida()),  # columnas de X
    }


def _rutas(escenario, carpeta: Path) -> ex.RutasPD:
    """RutasPD con las entradas del escenario y las salidas en ``carpeta``."""
    return ex.RutasPD(escenario["parquet"], escenario["pre"], carpeta / "res", carpeta / "models")  # rutas


def _cfg(escenario, n=N_SEMILLAS, base_seed=BASE_SEED) -> dict:
    """Copia de la cfg del escenario con ``semillas.n`` y ``base_seed`` pedidos."""
    cfg = copy.deepcopy(escenario["cfg"])  # no muta la del módulo
    cfg["semillas"]["n"], cfg["base_seed"] = n, base_seed  # cambios pedidos
    return cfg  # config completa


def _reloj(paso=PASO_RELOJ):
    """Reloj falso: 0, paso, 2·paso, … en llamadas sucesivas."""
    contador = itertools.count(start=0.0, step=paso)  # flotantes exactos con paso 1.0
    return lambda: next(contador)  # misma firma que time.perf_counter


class _CargarContador:
    """Doble de ``cargar``: delega en ``cargar_datos`` y cuenta sus llamadas."""

    def __init__(self):
        """Sin llamadas al inicio."""
        self.llamadas = 0  # llamadas recibidas

    def __call__(self, rutas):
        """Cuenta la llamada y devuelve (df, matrices) del escenario."""
        self.llamadas += 1  # registra la llamada
        return ex.cargar_datos(rutas)  # datos reales del escenario


def _fabrica_falsa(falla_en=None, constante=False):
    """(clase de modelo falso, contador): fit cuenta llamadas y lanza FloatingPointError en la llamada ``falla_en``;
    con ``constante`` la PD es PD_BASE_FALSA + DESPLAZAMIENTO_FALSO·semilla en todas las filas."""
    contador = {"fit": 0, "predict": 0}  # llamadas a fit y a predict_proba (compartidas por todas las instancias)

    class _ModeloFalso:
        """Modelo con la interfaz que usa correr_mlp (fit, predict_proba, n_parametros, resultado, guardar)."""

        def fit(self, X_tr, y_tr, X_val, y_val, cfg, seed):
            """Cuenta la llamada, diverge si toca y guarda la semilla."""
            k = contador["fit"]  # número de esta llamada (desde 0)
            contador["fit"] += 1  # cuenta la llamada
            if k == falla_en:  # llamada que debe diverger
                raise FloatingPointError(MENSAJE_DIVERGE)  # como entrenar ante una pérdida no finita
            self._seed = seed  # la PD depende de la semilla
            return self  # como MLPPD

        def predict_proba(self, X):
            """PD que depende de la semilla: constante (monótona en la semilla) o logística de la columna 0."""
            contador["predict"] += 1  # cuenta la predicción
            if constante:  # log-loss de val monótona en la semilla
                return np.full(np.shape(X)[0], PD_BASE_FALSA + DESPLAZAMIENTO_FALSO * self._seed)  # una PD por fila
            return expit(PENDIENTE_FALSA * np.asarray(X, dtype=np.float64)[:, 0]
                         + DESPLAZAMIENTO_FALSO * self._seed)  # PD en (0, 1)

        def n_parametros(self):
            """Parámetros declarados (fijo)."""
            return N_PARAM_FALSO  # fijo

        @property
        def resultado(self):
            """ResultadoEntrenamiento de una época."""
            historial = [{"epoca": 1, "perdida_train": 0.6, "perdida_val": 0.6}]  # una época
            return ResultadoEntrenamiento(historial, 1, 1, 0.6)  # mismo tipo que MLPPD

        def guardar(self, ruta):
            """Escribe la semilla del modelo como texto (identifica qué red se guardó)."""
            Path(ruta).write_text(str(self._seed), encoding="utf-8")  # archivo no vacío con la semilla

    return _ModeloFalso, contador  # modelo_cls inyectable y su contador


def _sin_tiempos(d: dict, tiempos) -> dict:
    """Copia de ``d`` sin las claves de ``tiempos``."""
    return {k: v for k, v in d.items() if k not in tiempos}  # comparable entre corridas


def _corrida_registro(seed, logloss_val, logloss_test=0.5):
    """Corrida mínima para semilla_mediana: seed y métricas de val y test."""
    return {"seed": seed, "metricas": {"logloss_val": logloss_val, "logloss_test": logloss_test}}  # registro


# %% [markdown]
# ## 1–2. Semilla mediana
#
# Orden por (`logloss_val`, `seed`) y posición ⌊(n − 1)/2⌋. Test nunca decide.

# %%
@pytest.mark.parametrize("perdidas, esperada", [  # (log-loss de val por semilla 0…n−1, semilla esperada)
    ([0.3, 0.1, 0.2], 2),  # n impar: 0.1 (s1), 0.2 (s2), 0.3 (s0) → posición 1 = s2
    ([0.4, 0.1, 0.3, 0.2], 3),  # n par: s1, s3, s2, s0 → posición (4 − 1)//2 = 1 = s3 (mediana inferior)
    ([0.2, 0.2, 0.2], 1),  # empate total: orden por semilla → posición 1 = s1
    ([0.5], 0),  # n = 1: la única
], ids=["impar", "par", "empate", "una"])
def test_01_semilla_mediana(perdidas, esperada):
    """Mediana por (logloss_val, seed), posición (n−1)//2; empate → semilla menor."""
    corridas = [_corrida_registro(s, p) for s, p in enumerate(perdidas)]  # una corrida por semilla
    assert ex.semilla_mediana(corridas) == esperada  # semilla esperada
    assert ex.semilla_mediana(list(reversed(corridas))) == esperada  # no depende del orden de la lista


def test_02_semilla_mediana_ignora_test():
    """Con logloss_test en orden inverso al de val el resultado no cambia (atrapa ordenar por test)."""
    val = [0.1, 0.2, 0.3, 0.4]  # val creciente por semilla (n par: por test saldría la semilla 2, no la 1)
    con_test = [_corrida_registro(s, v, logloss_test=1.0 - v) for s, v in enumerate(val)]  # test decreciente
    sin_test = [_corrida_registro(s, v) for s, v in enumerate(val)]  # test constante
    assert ex.semilla_mediana(con_test) == ex.semilla_mediana(sin_test) == 1  # decide val


# %% [markdown]
# ## 3–5. Procedencia y presupuesto de la búsqueda

# %%
def test_03_procedencia(escenario, tmp_path):
    """Igual → (True, []); capas distintas → (False, ["capas"]); busqueda distinta → False con "busqueda"."""
    mlp, busqueda, ruta = validar_cfg_mlp(MLP_ESCENARIO), escenario["busqueda"], escenario["ruta_final"]  # base
    assert ex.claves_distintas(mlp, mlp) == []  # pura: nada difiere
    assert ex.procedencia(mlp, busqueda, ruta) == (True, [])  # la config es la de la búsqueda
    otra = {**mlp, "capas": CAPAS_OTRAS}  # --set mlp.capas=[8]
    assert ex.claves_distintas(otra, mlp) == ["capas"]  # nombra la clave
    assert ex.procedencia(otra, busqueda, ruta) == (False, ["capas"])  # no es la de la búsqueda
    otra_busqueda = {**busqueda, "ensayo": 4}  # otro ensayo ganador
    de_busqueda, difieren = ex.procedencia(mlp, otra_busqueda, ruta)  # busqueda distinta
    assert de_busqueda is False and "busqueda" in difieren  # False y nombrada


def test_04_constantes_coinciden_con_hpo():
    """TIPO_OPTUNA == hpo.TIPO_BUSQUEDA y CLAVES_BUSQUEDA == claves que escribe hpo.armar_busqueda."""
    resumen = {"mejor": {"numero": 1}, "n_ensayos": 2, "tiempo_total_s": 3.0}  # resumen mínimo
    bloque = hpo.armar_busqueda(resumen, {"sha256_particion": "a", "sha256_preprocesador": "b"}, "f")  # bloque
    assert ex.TIPO_OPTUNA == hpo.TIPO_BUSQUEDA  # misma constante sin importar hpo en experimento
    assert tuple(ex.CLAVES_BUSQUEDA) == tuple(bloque)  # mismas claves y orden


def test_05_presupuesto_busqueda(escenario):
    """True coherente → n_ensayos y tiempo_s; evidencia distinta → ValueError; False → None (el tipo lo decide
    procedencia: con sin_busqueda nunca llega true, prueba 30)."""
    busqueda, huellas = escenario["busqueda"], escenario["huellas"]  # coherentes
    p = ex.presupuesto_busqueda(busqueda, True, huellas)  # config de la búsqueda
    assert p == {"n_configuraciones_busqueda": 5, "tiempo_busqueda_s": 12.5, "busqueda": busqueda}  # presupuesto
    otras = {**huellas, "sha256_particion": "0" * 64}  # otra partición
    with pytest.raises(ValueError, match="repite pd-hpo"):  # otros datos
        ex.presupuesto_busqueda(busqueda, True, otras)  # evidencia distinta
    assert ex.presupuesto_busqueda(busqueda, False, otras) == {  # no es la config de la búsqueda
        "n_configuraciones_busqueda": None, "tiempo_busqueda_s": None, "busqueda": None}  # tres None


# %% [markdown]
# ## 6. Borrado acotado

# %%
def test_06_borrar_semillas_mlp(tmp_path):
    """Borra solo los archivos mlp_seed<entero>.json; carpeta inexistente → 0."""
    carpeta = tmp_path / "semillas"  # semillas/
    carpeta.mkdir()  # existe
    for nombre in ("mlp_seed99.json", "mlp_seed0.json", "logistica_seed0.json", "mlp_seed_notas.json",
                   "mlp_seed7.json.bak"):  # candidatos
        (carpeta / nombre).write_text("{}", encoding="utf-8")  # archivos
    (carpeta / "mlp_seed3.json").mkdir()  # subcarpeta con nombre de semilla
    assert ex.borrar_semillas_mlp(carpeta) == 2  # mlp_seed99 y mlp_seed0
    quedan = sorted(p.name for p in carpeta.iterdir())  # lo que sigue
    assert quedan == ["logistica_seed0.json", "mlp_seed3.json", "mlp_seed7.json.bak", "mlp_seed_notas.json"]  # ajenos
    assert ex.borrar_semillas_mlp(tmp_path / "no_existe") == 0  # sin carpeta


# %% [markdown]
# ## 7–9. Validación antes de leer datos y salidas previas incompatibles

# %%
@pytest.mark.parametrize("n", [0, -1, 2.5, True, "30"], ids=["cero", "negativo", "flotante", "bool", "texto"])
def test_07_semillas_n_invalido(escenario, tmp_path, n):
    """semillas.n inválido → ValueError y cargar no se llamó."""
    cfg = _cfg(escenario)  # config válida
    cfg["semillas"]["n"] = n  # valor inválido
    cargar = _CargarContador()  # cuenta las lecturas
    with pytest.raises(ValueError):  # antes de leer datos
        ex.correr_mlp(cfg, _rutas(escenario, tmp_path), cargar=cargar,
                      ruta_mlp_final=escenario["ruta_final"])  # corrida con la config rota
    assert cargar.llamadas == 0  # no leyó datos


def _mutar(cambio):
    """Mutación de la config: ("quitar", ruta) o ("poner", ruta, valor) con ruta = tupla de claves."""
    def mutar(cfg):
        """Aplica la mutación a ``cfg`` en el sitio."""
        nodo = cfg  # baja hasta el padre
        for clave in cambio[1][:-1]:  # tramos intermedios
            nodo = nodo[clave]  # siguiente nivel
        if cambio[0] == "quitar":  # quitar la hoja
            del nodo[cambio[1][-1]]  # sin la clave
        else:  # poner un valor
            nodo[cambio[1][-1]] = cambio[2]  # reemplaza
    return mutar  # función que muta en el sitio


@pytest.mark.parametrize("cambio", [  # cada caso rompe una regla
    ("poner", ("base_seed",), -1), ("poner", ("base_seed",), True), ("poner", ("base_seed",), "0"),  # base_seed
    ("poner", ("base_seed",), 1.5),  # base_seed flotante
    ("quitar", ("semillas",)), ("quitar", ("busqueda",)), ("quitar", ("busqueda", "evidencia")),  # bloques
    ("poner", ("busqueda", "extra"), 1), ("poner", ("busqueda",), "optuna"),  # busqueda con clave de más o no mapeo
    ("poner", ("mlp", "dropout"), 1.0), ("quitar", ("mlp", "capas")), ("poner", ("mlp",), [1]),  # mlp inválido
    ("poner", ("semillas",), 3), ("quitar", ("semillas", "n")),  # semillas no es mapeo o sin n
    ("poner", ("semillas", "extra"), 1),  # E3: semillas con una clave de más
    ("poner", ("busqueda", "tipo"), "otro"),  # E2: tipo desconocido con las 6 claves
    ("poner", ("busqueda",), {"tipo": "sin_busqueda", "ensayo": 1}),  # E2: sin_busqueda con clave extra
    ("poner", ("busqueda",), {"ensayo": 1}),  # E2: sin tipo
], ids=["seed_neg", "seed_bool", "seed_txt", "seed_float", "sin_semillas", "sin_busqueda", "sin_evidencia",
        "busqueda_extra", "busqueda_no_mapeo", "dropout_1", "sin_capas", "mlp_no_mapeo", "semillas_no_mapeo",
        "sin_n", "semillas_extra", "tipo_otro", "sin_busqueda_extra", "busqueda_sin_tipo"])
def test_08_config_invalida_antes_de_cargar(escenario, tmp_path, cambio):
    """base_seed, semillas, busqueda o mlp inválidos → ValueError sin leer datos."""
    cfg = _cfg(escenario)  # config válida
    _mutar(cambio)(cfg)  # la rompe
    cargar = _CargarContador()  # cuenta las lecturas
    with pytest.raises(ValueError):  # antes de leer datos
        ex.correr_mlp(cfg, _rutas(escenario, tmp_path), cargar=cargar,
                      ruta_mlp_final=escenario["ruta_final"])  # corrida con la config rota
    assert cargar.llamadas == 0  # no leyó datos


def test_09_salidas_previas_incompatibles(escenario, tmp_path):
    """summary.json con raíz lista o pred_test con otros ids → ValueError antes del 1.er fit."""
    rutas = _rutas(escenario, tmp_path)  # salidas temporales
    save_json(rutas.resultados / ex.ARCHIVO_SUMMARY, [1, 2])  # raíz no dict
    modelo_cls, contador = _fabrica_falsa()  # modelo falso
    with pytest.raises(ValueError):  # falla rápido
        ex.correr_mlp(_cfg(escenario), rutas, modelo_cls=modelo_cls,
                      ruta_mlp_final=escenario["ruta_final"])  # corrida con el modelo falso
    assert contador["fit"] == 0  # ningún fit
    (rutas.resultados / ex.ARCHIVO_SUMMARY).unlink()  # quita el summary ajeno
    pd.DataFrame({"id": ["otro"], "y": [0], "pred_logistica": [0.1]}).to_parquet(
        rutas.resultados / ex.ARCHIVO_PRED_TEST, index=False)  # pred_test de otros ids
    with pytest.raises(ValueError):  # falla rápido
        ex.correr_mlp(_cfg(escenario), rutas, modelo_cls=modelo_cls,
                      ruta_mlp_final=escenario["ruta_final"])  # corrida con el modelo falso
    assert contador["fit"] == 0  # ningún fit
    assert not (rutas.resultados / ex.CARPETA_SEMILLAS).exists()  # no escribió semillas


# %% [markdown]
# ## 10–13. Flujo con el modelo falso

# %%
def test_10_evidencia_distinta_antes_de_borrar(escenario, tmp_path):
    """Config de la búsqueda con otra evidencia → ValueError "repite pd-hpo", 0 fit y las semillas viejas siguen."""
    rutas = _rutas(escenario, tmp_path)  # salidas temporales
    otra = {**escenario["busqueda"], "evidencia": {**escenario["huellas"], "sha256_particion": "0" * 64}}  # otros datos
    ruta_final = _escribir_mlp_final(tmp_path / "mlp_final.yaml", MLP_ESCENARIO, otra)  # búsqueda con otros datos
    cfg = _cfg(escenario)  # config del escenario
    cfg["busqueda"] = copy.deepcopy(otra)  # la misma busqueda que el archivo: config de la búsqueda
    vieja = rutas.resultados / ex.CARPETA_SEMILLAS / "mlp_seed42.json"  # semilla de una corrida anterior
    save_json(vieja, {"viejo": True})  # existe antes
    modelo_cls, contador = _fabrica_falsa()  # modelo falso
    with pytest.raises(ValueError, match="repite pd-hpo"):  # otros datos
        ex.correr_mlp(cfg, rutas, modelo_cls=modelo_cls, ruta_mlp_final=ruta_final)  # corrida
    assert contador["fit"] == 0  # ningún fit
    assert vieja.exists()  # el chequeo va antes de borrar


def test_10b_dos_lecturas_una_huella(escenario):
    """La huella de comprobar_antes (solo id, split) es idéntica a la de evidencia(df completo)."""
    rapida = ex.evidencia_rapida(escenario["parquet"], escenario["pre"])  # lee solo id, split
    df, _ = ex.cargar_datos(_rutas(escenario, escenario["base"]))  # lectura completa de correr_mlp
    assert rapida == ex.evidencia(df, escenario["pre"]) == escenario["huellas"]  # misma huella
    ex.comprobar_antes(_cfg(escenario), _rutas(escenario, escenario["base"]),
                       ruta_mlp_final=escenario["ruta_final"])  # config de la búsqueda: no lanza


def test_10c_comprobar_antes_usa_el_lector_inyectado(escenario, tmp_path):
    """comprobar_antes lee la partición con ``leer_particion`` (rutas del parquet y del preprocesador) y falla si
    la huella no es la de la búsqueda."""
    rutas = _rutas(escenario, tmp_path)  # rutas temporales
    llamadas = []  # argumentos recibidos

    def leer(parquet, pre):
        """Lector espía de la partición: registra la llamada."""
        llamadas.append((parquet, pre))  # registra
        return {**escenario["huellas"], "sha256_particion": "0" * 64}  # otra partición

    with pytest.raises(ValueError, match="repite pd-hpo"):  # otros datos
        ex.comprobar_antes(_cfg(escenario), rutas, ruta_mlp_final=escenario["ruta_final"],
                           leer_particion=leer)  # pre-chequeo con el lector espía
    assert llamadas == [(rutas.parquet, rutas.preprocesador)]  # una lectura con las rutas


def test_11_floating_point_error_en_la_segunda_semilla(escenario, tmp_path):
    """FloatingPointError en la 2.ª semilla → mensaje con la semilla y el aviso; solo queda la 1.ª semilla; summary,
    pred_mlp y pd_mlp.pt no se escriben."""
    rutas = _rutas(escenario, tmp_path)  # salidas temporales
    ruta_summary = rutas.resultados / ex.ARCHIVO_SUMMARY  # summary previo
    save_json(ruta_summary, {"logistica": {"x": 1}, "mlp": {"viejo": True}})  # entradas anteriores
    antes = ruta_summary.read_bytes()  # bytes previos
    modelo_cls, _ = _fabrica_falsa(falla_en=1)  # diverge en la 2.ª llamada
    with pytest.raises(FloatingPointError) as exc:  # mismo tipo
        ex.correr_mlp(_cfg(escenario), rutas, modelo_cls=modelo_cls,
                      ruta_mlp_final=escenario["ruta_final"])  # corrida con el modelo falso
    assert f"semilla {BASE_SEED + 1}" in str(exc.value) and AVISO in str(exc.value)  # semilla y aviso
    assert MENSAJE_DIVERGE in str(exc.value)  # conserva el mensaje original
    assert isinstance(exc.value.__cause__, FloatingPointError)  # from exc
    semillas = sorted(p.name for p in (rutas.resultados / ex.CARPETA_SEMILLAS).iterdir())  # escritas
    assert semillas == [f"mlp_seed{BASE_SEED}.json"]  # solo la terminada
    assert ruta_summary.read_bytes() == antes  # summary intacto
    assert not (rutas.resultados / ex.ARCHIVO_PRED_TEST).exists()  # sin pred_mlp
    assert not (rutas.modelos / ex.ARCHIVO_MODELO_MLP).exists()  # sin pd_mlp.pt


def test_12_reloj_solo_el_fit(escenario, tmp_path):
    """tiempo_s = un paso del reloj por semilla; tiempo_semillas_s = Σ; tiempo_total_s = búsqueda + Σ (o Σ)."""
    modelo_cls, _ = _fabrica_falsa()  # modelo falso
    entrada = ex.correr_mlp(_cfg(escenario), _rutas(escenario, tmp_path / "a"), modelo_cls=modelo_cls,
                            reloj=_reloj(), ruta_mlp_final=escenario["ruta_final"])  # config de la búsqueda
    for s in range(BASE_SEED, BASE_SEED + N_SEMILLAS):  # cada semilla
        registro = load_json(tmp_path / "a" / "res" / ex.CARPETA_SEMILLAS / f"mlp_seed{s}.json")  # su archivo
        assert registro["tiempo_s"] == PASO_RELOJ  # solo el fit
    assert entrada["tiempo_semillas_s"] == N_SEMILLAS * PASO_RELOJ  # Σ
    assert entrada["tiempo_busqueda_s"] == BUSQUEDA_SIN_EVIDENCIA["tiempo_s"]  # de mlp_final.yaml
    assert entrada["tiempo_total_s"] == BUSQUEDA_SIN_EVIDENCIA["tiempo_s"] + N_SEMILLAS * PASO_RELOJ  # suma
    cfg = apply_overrides(_cfg(escenario), [f"mlp.capas={CAPAS_OTRAS}"])  # no es la config de la búsqueda
    entrada = ex.correr_mlp(cfg, _rutas(escenario, tmp_path / "b"), modelo_cls=modelo_cls, reloj=_reloj(),
                            ruta_mlp_final=escenario["ruta_final"])  # corrida
    assert entrada["tiempo_total_s"] == N_SEMILLAS * PASO_RELOJ  # solo Σ
    assert entrada["tiempo_busqueda_s"] is None  # sin presupuesto de búsqueda


def test_12b_reloj_rodea_solo_el_fit(escenario, tmp_path):
    """Entre los dos tics de cada semilla hay exactamente un fit y ninguna predicción (atrapa medir o predecir dentro
    de la zona cronometrada, que el reloj de paso fijo no ve)."""
    modelo_cls, contador = _fabrica_falsa()  # modelo falso con contadores
    tics = []  # (fits, predicciones) en cada llamada al reloj

    def reloj():
        """Reloj espía: registra (fits, predicciones) y avanza."""
        tics.append((contador["fit"], contador["predict"]))  # estado al leer el reloj
        return float(len(tics))  # avanza

    ex.correr_mlp(_cfg(escenario), _rutas(escenario, tmp_path), modelo_cls=modelo_cls, reloj=reloj,
                  ruta_mlp_final=escenario["ruta_final"])  # corrida
    assert len(tics) == 2 * N_SEMILLAS  # dos tics por semilla
    for k in range(N_SEMILLAS):  # cada semilla
        (fit_0, pred_0), (fit_1, pred_1) = tics[2 * k], tics[2 * k + 1]  # antes y después
        assert (fit_0, fit_1) == (k, k + 1) and pred_0 == pred_1  # solo el fit en medio


def test_13_no_muta_la_config(escenario, tmp_path):
    """correr_mlp no muta cfg (el CLI la reutiliza)."""
    cfg = _cfg(escenario)  # config
    copia = copy.deepcopy(cfg)  # antes
    modelo_cls, _ = _fabrica_falsa()  # modelo falso
    ex.correr_mlp(cfg, _rutas(escenario, tmp_path), modelo_cls=modelo_cls,
                  ruta_mlp_final=escenario["ruta_final"])  # corrida con el modelo falso
    assert cfg == copia  # intacta


# %% [markdown]
# ## 14–21. Flujo real con `MLPPD`
#
# La fixture `corrida` escribe primero la logística (como `correr_pd`) y luego corre las 3 semillas del MLP.

# %%
@pytest.fixture(scope="module")
def corrida(escenario, tmp_path_factory):
    """correr_logistica y luego correr_mlp con MLPPD (base_seed 7, n 3); devuelve rutas, entrada y lo previo."""
    rutas = _rutas(escenario, tmp_path_factory.mktemp("corrida_b5"))  # salidas temporales
    cfg = _cfg(escenario)  # config del escenario
    ex.correr_logistica(cfg, rutas, seed=BASE_SEED)  # la logística primero
    log_previa = load_json(rutas.resultados / ex.ARCHIVO_SUMMARY)["logistica"]  # entrada previa
    pred_previa = pd.read_parquet(rutas.resultados / ex.ARCHIVO_PRED_TEST)  # pred_logistica previa
    entrada = ex.correr_mlp(cfg, rutas, ruta_mlp_final=escenario["ruta_final"])  # MLP real
    semillas = {s: load_json(rutas.resultados / ex.CARPETA_SEMILLAS / f"mlp_seed{s}.json")
                for s in range(BASE_SEED, BASE_SEED + N_SEMILLAS)}  # archivos escritos
    return {"rutas": rutas, "cfg": cfg, "entrada": entrada, "log_previa": log_previa, "pred_previa": pred_previa,
            "semillas": semillas, "datos": ex.cargar_datos(rutas)[1]}  # todo lo que se inspecciona


def test_14_semillas_escritas(corrida):
    """mlp_seed7/8/9.json y summary.mlp.semillas == [7, 8, 9]."""
    carpeta = corrida["rutas"].resultados / ex.CARPETA_SEMILLAS  # semillas/
    nombres = sorted(p.name for p in carpeta.glob("mlp_seed*.json"))  # escritas
    assert nombres == [f"mlp_seed{s}.json" for s in range(BASE_SEED, BASE_SEED + N_SEMILLAS)]  # 7, 8, 9
    summary = load_json(corrida["rutas"].resultados / ex.ARCHIVO_SUMMARY)  # summary escrito
    assert summary["mlp"]["semillas"] == list(range(BASE_SEED, BASE_SEED + N_SEMILLAS))  # semillas efectivas
    assert summary["mlp"] == corrida["entrada"]  # lo devuelto es lo escrito


def test_15_contenido_de_cada_semilla(escenario, corrida):
    """Claves del núcleo §5.4 + extras; epocas = len(historial); config validada; evidencia; n por split;
    calibracion_global de cada semilla con ȳ exacta y p̄ en (0, 1), y la de la mediana = media de predict_proba."""
    datos = corrida["datos"]  # matrices por split
    for s, r in corrida["semillas"].items():  # cada semilla
        assert set(r) == CLAVES_SEMILLA  # claves exactas
        assert (r["modelo"], r["seed"], r["device"]) == ("mlp", s, "cpu")  # identidad
        assert r["epocas"] == len(r["historial"]) and 1 <= r["mejor_epoca"] <= r["epocas"]  # épocas
        assert r["config"] == validar_cfg_mlp(MLP_ESCENARIO)  # bloque validado
        assert r["evidencia"] == escenario["huellas"]  # misma evidencia
        assert r["n"] == {k: int(len(datos[k][1])) for k in ("train", "val", "test")}  # filas por split
        assert set(r["calibracion_global"]) == CLAVES_CALIBRACION  # claves
        for split in ("val", "test"):  # cada split medido
            assert r["calibracion_global"][f"tasa_{split}"] == float(np.mean(datos[split][1]))  # ȳ exacta
            assert 0.0 < r["calibracion_global"][f"pd_media_{split}"] < 1.0  # p̄ es una probabilidad
    mediana = corrida["semillas"][corrida["entrada"]["semilla_mediana"]]  # registro de la mediana
    modelo = MLPPD.cargar(corrida["rutas"].modelos / ex.ARCHIVO_MODELO_MLP)  # red guardada
    for split in ("val", "test"):  # p̄ de cada split
        p = modelo.predict_proba(datos[split][0])  # PD de la red mediana
        assert mediana["calibracion_global"][f"pd_media_{split}"] == pytest.approx(float(np.mean(p)))  # p̄


def test_16_entrada_mlp_del_summary(corrida):
    """n = 3, desv con ddof = 1, mejor/peor según SENTIDO; la entrada logistica previa queda idéntica."""
    summary = load_json(corrida["rutas"].resultados / ex.ARCHIVO_SUMMARY)  # summary escrito
    mlp = summary["mlp"]  # entrada mlp
    assert set(mlp) == CLAVES_SUMMARY_MLP  # claves exactas
    for nombre, resumen in mlp["metricas"].items():  # cada métrica
        x = np.array([r["metricas"][nombre] for r in corrida["semillas"].values()])  # un valor por semilla
        assert resumen["n"] == N_SEMILLAS  # tres corridas
        assert resumen["media"] == pytest.approx(float(np.mean(x)))  # x̄
        assert resumen["desv"] == pytest.approx(float(np.std(x, ddof=1)))  # s muestral
        mayor = ex.SENTIDO[nombre.rsplit("_", 1)[0]] == "max"  # sentido
        assert resumen["mejor"] == (x.max() if mayor else x.min())  # mejor
        assert resumen["peor"] == (x.min() if mayor else x.max())  # peor
    assert summary["logistica"] == corrida["log_previa"]  # entrada ajena intacta
    assert {r["n_parametros"] for r in corrida["semillas"].values()} == {mlp["n_parametros"]}  # igual en todas


def test_17_pred_test(corrida):
    """pred_logistica previa intacta y pred_mlp alineada por id con la red guardada."""
    pred = pd.read_parquet(corrida["rutas"].resultados / ex.ARCHIVO_PRED_TEST)  # archivo final
    previa = corrida["pred_previa"]  # antes del MLP
    pd.testing.assert_frame_equal(pred[list(previa.columns)], previa)  # columnas previas intactas
    X_te, _, ids_te = corrida["datos"]["test"]  # test
    p = MLPPD.cargar(corrida["rutas"].modelos / ex.ARCHIVO_MODELO_MLP).predict_proba(X_te)  # PD por id de test
    esperado = pd.Series(p, index=list(ids_te))  # id → PD
    assert np.array_equal(pred["pred_mlp"].to_numpy(), esperado.loc[pred["id"]].to_numpy())  # alineada, exacta


def test_18_modelo_guardado_es_la_semilla_mediana(corrida):
    """pd_mlp.pt reproduce exactamente las métricas de la semilla mediana; config_de_busqueda y presupuesto."""
    entrada, datos = corrida["entrada"], corrida["datos"]  # summary y matrices
    corridas = list(corrida["semillas"].values())  # registros escritos
    assert entrada["semilla_mediana"] == ex.semilla_mediana(corridas)  # regla sobre los archivos
    modelo = MLPPD.cargar(corrida["rutas"].modelos / ex.ARCHIVO_MODELO_MLP)  # red guardada
    assert ex.medir(modelo, datos) == corrida["semillas"][entrada["semilla_mediana"]]["metricas"]  # exacto
    assert entrada["config_de_busqueda"] is True  # config de la búsqueda
    assert entrada["n_configuraciones_busqueda"] == BUSQUEDA_SIN_EVIDENCIA["n_ensayos"]  # presupuesto
    assert entrada["busqueda"]["ensayo"] == BUSQUEDA_SIN_EVIDENCIA["ensayo"]  # bloque de mlp_final.yaml
    assert entrada["config"] == validar_cfg_mlp(MLP_ESCENARIO)  # bloque validado


def test_19_determinismo(escenario, corrida, tmp_path):
    """Otra corrida idéntica en otra carpeta → mismas semillas y misma entrada mlp salvo los tiempos."""
    rutas = _rutas(escenario, tmp_path)  # otra carpeta
    entrada = ex.correr_mlp(corrida["cfg"], rutas, ruta_mlp_final=escenario["ruta_final"])  # misma config
    assert _sin_tiempos(entrada, TIEMPOS_SUMMARY) == _sin_tiempos(corrida["entrada"], TIEMPOS_SUMMARY)  # summary
    for s, r in corrida["semillas"].items():  # cada semilla
        otra = load_json(rutas.resultados / ex.CARPETA_SEMILLAS / f"mlp_seed{s}.json")  # nueva
        assert _sin_tiempos(otra, TIEMPOS_SEMILLA) == _sin_tiempos(r, TIEMPOS_SEMILLA)  # idéntica


def test_20_set_cambia_la_config(escenario, tmp_path, capsys):
    """--set mlp.capas=[8] → config_de_busqueda false, presupuesto null, capas [8] y aviso con "capas"."""
    cfg = apply_overrides(_cfg(escenario, n=1), [f"mlp.capas={CAPAS_OTRAS}"])  # como el CLI
    entrada = ex.correr_mlp(cfg, _rutas(escenario, tmp_path), ruta_mlp_final=escenario["ruta_final"])  # corrida
    assert entrada["config_de_busqueda"] is False  # no es la de la búsqueda
    assert (entrada["n_configuraciones_busqueda"], entrada["tiempo_busqueda_s"], entrada["busqueda"]) == (
        None, None, None)  # presupuesto null
    assert entrada["config"]["capas"] == CAPAS_OTRAS  # la del --set
    d, k = escenario["n_columnas"], CAPAS_OTRAS[0]  # entradas y anchura
    registro = load_json(tmp_path / "res" / ex.CARPETA_SEMILLAS / f"mlp_seed{BASE_SEED}.json")  # su semilla
    assert registro["n_parametros"] == d * k + k + k + 1  # Linear(d, 8) + Linear(8, 1)
    assert "capas" in capsys.readouterr().out  # el aviso nombra la clave


def test_21_crea_la_carpeta_de_modelos(escenario, tmp_path):
    """modelos inexistente (anidada) se crea; pd_mlp.pt pesa > 0 y carga."""
    rutas = ex.RutasPD(escenario["parquet"], escenario["pre"], tmp_path / "res", tmp_path / "a" / "b")  # anidada
    ex.correr_mlp(_cfg(escenario, n=1), rutas, ruta_mlp_final=escenario["ruta_final"])  # corrida
    archivo = tmp_path / "a" / "b" / ex.ARCHIVO_MODELO_MLP  # modelo
    assert archivo.stat().st_size > 0  # no vacío
    assert MLPPD.cargar(archivo).n_parametros() > 0  # carga


# %% [markdown]
# ## 22–25. Orquestación, rutas y CLI

# %%
def test_22_correr_pd_orden(escenario, tmp_path):
    """comprobar → logística → MLP con las mismas rutas y seed = base_seed; si comprobar falla, nada corre."""
    cfg = _cfg(escenario)  # config
    cfg["salida"], cfg["modelos"] = str(tmp_path / "r"), str(tmp_path / "m")  # salidas
    orden = []  # llamadas en orden

    def comprobar(c, rutas):
        """Pre-chequeo espía: registra la llamada."""
        orden.append(("comprobar", rutas))  # primero

    def log(c, rutas, seed):
        """Logística espía: registra la llamada."""
        orden.append(("logistica", rutas, seed))  # segundo
        return {"n": "log"}  # entrada falsa

    def red(c, rutas):
        """MLP espía: registra la llamada."""
        orden.append(("mlp", rutas))  # tercero
        return {"n": "mlp"}  # entrada falsa

    res = ex.correr_pd(cfg, correr_log=log, correr_red=red, comprobar=comprobar)  # orquestación
    rutas = ex.rutas_de(cfg)  # rutas esperadas
    assert orden == [("comprobar", rutas), ("logistica", rutas, BASE_SEED), ("mlp", rutas)]  # orden y argumentos
    assert list(res) == ["logistica", "mlp", "rutas"] and res["rutas"] == rutas  # resultado
    orden.clear()  # de nuevo, con un pre-chequeo que falla

    def falla(c, rutas):
        """Pre-chequeo que falla como con otros datos."""
        raise ValueError("la búsqueda se hizo con otros datos; repite pd-hpo")  # pre-chequeo

    with pytest.raises(ValueError):  # se propaga
        ex.correr_pd(cfg, correr_log=log, correr_red=red, comprobar=falla)  # orquestación
    assert orden == []  # ni la logística ni el MLP


@pytest.mark.parametrize("cambio", [("salida", ""), ("modelos", "  "), ("salida", 3), ("modelos", None)],
                         ids=["salida_vacia", "modelos_blanco", "salida_numero", "modelos_none"])
def test_23_rutas_de_invalidas(escenario, cambio):
    """salida o modelos vacíos o no texto → ValueError; válidas → RutasPD con las entradas de RUTAS_PD."""
    cfg = _cfg(escenario)  # config válida
    assert ex.rutas_de(cfg) == ex.RutasPD(ex.RUTAS_PD.parquet, ex.RUTAS_PD.preprocesador, Path(cfg["salida"]),
                                          Path(cfg["modelos"]))  # válidas
    cfg[cambio[0]] = cambio[1]  # inválida
    with pytest.raises(ValueError):  # rechazo
        ex.rutas_de(cfg)  # rutas


def _huella_carpetas():
    """{ruta relativa: mtime_ns} de results/data/pd y models reales (vacío si no existen)."""
    huella = {}  # resultado
    for carpeta in (ROOT / "results" / "data" / "pd", ROOT / "models"):  # carpetas reales
        if carpeta.exists():  # puede no existir en un clon limpio
            huella.update({p.relative_to(ROOT).as_posix(): p.stat().st_mtime_ns for p in carpeta.rglob("*")})  # todo
    return huella  # listado + mtime


def _apuntar_al_escenario(escenario, monkeypatch):
    """RUTAS_PD y RUTA_MLP_FINAL del módulo apuntan al escenario sintético (se leen al llamar)."""
    monkeypatch.setattr(ex, "RUTAS_PD", ex.RutasPD(escenario["parquet"], escenario["pre"], Path("x"),
                                                   Path("y")))  # entradas sintéticas (salidas: las de la config)
    monkeypatch.setattr(ex, "RUTA_MLP_FINAL", escenario["ruta_final"])  # mlp_final.yaml coherente con el escenario


def _cli_pd(escenario, tmp_path, monkeypatch, n=2, extra=()):
    """Corre run.py pd con el escenario sintético y salidas en tmp_path; ``extra`` = más --set."""
    _apuntar_al_escenario(escenario, monkeypatch)  # datos y mlp_final.yaml del escenario
    argv = ["pd", "--config", str(escenario["ruta_base"]), "--set", f"salida={(tmp_path / 'res').as_posix()}",
            "--set", f"modelos={(tmp_path / 'models').as_posix()}", "--set", f"semillas.n={n}"]  # salidas temporales
    for s in extra:  # overrides adicionales
        argv += ["--set", s]  # uno por uno
    return cli.main(argv)  # código de salida


def test_24_no_escribe_en_las_carpetas_reales(escenario, tmp_path, monkeypatch):
    """Con salida y modelos en tmp_path, results/data/pd y models reales no cambian."""
    antes = _huella_carpetas()  # listado + mtime
    assert _cli_pd(escenario, tmp_path, monkeypatch, n=1) == 0  # corrida completa
    assert _huella_carpetas() == antes  # nada cambió
    assert (tmp_path / "models" / ex.ARCHIVO_MODELO_MLP).exists()  # escribió en tmp_path


def test_25_cli_pd(escenario, tmp_path, monkeypatch, capsys):
    """run.py pd imprime la config resuelta, una línea por modelo, la semilla mediana y el tiempo."""
    assert cli.SUBCOMANDOS["pd"].manejador is cli._pd  # conectado (no _pendiente)
    assert _cli_pd(escenario, tmp_path, monkeypatch) == 0  # éxito
    salida = capsys.readouterr().out  # consola
    for texto in (cli.CABECERA_CONFIG, "logistica:", "mlp:", "Semilla mediana", "Tiempo de ejecución"):  # piezas
        assert texto in salida  # impresas
    mlp = load_json(tmp_path / "res" / ex.ARCHIVO_SUMMARY)["mlp"]  # entrada escrita
    assert mlp["metricas"]["auc_test"]["n"] == 2 and mlp["config_de_busqueda"] is True  # 2 semillas, de la búsqueda
    assert "Config de la búsqueda: sí" in salida  # procedencia en consola
    linea_mlp = next(linea for linea in salida.splitlines() if linea.startswith("mlp: "))  # línea del MLP (no el YAML)
    assert re.search(r"AUC test \d\.\d{4} ± \d\.\d{4}", linea_mlp)  # media ± desv con 2 semillas
    assert f"{BUSQUEDA_SIN_EVIDENCIA['n_ensayos']} configuraciones" in salida  # presupuesto de la búsqueda


def test_25b_cli_pd_con_set_fuera_de_la_busqueda(escenario, tmp_path, monkeypatch, capsys):
    """run.py pd --set mlp.capas=[8] → config_de_busqueda false, aviso y sin presupuesto de búsqueda."""
    assert _cli_pd(escenario, tmp_path, monkeypatch, n=1, extra=[f"mlp.capas={CAPAS_OTRAS}"]) == 0  # éxito
    salida = capsys.readouterr().out  # consola
    assert "Config de la búsqueda: no" in salida and "sin presupuesto de búsqueda" in salida  # procedencia
    assert "difieren: capas" in salida  # aviso
    assert load_json(tmp_path / "res" / ex.ARCHIVO_SUMMARY)["mlp"]["config_de_busqueda"] is False  # escrito


# %% [markdown]
# ## 28–32. Decisiones de la ronda 2 (B-pd §3.5, "Bloques y pre-chequeo")
#
# - E1: sin `mlp_final.yaml` → `ValueError` ("corre pd-hpo") antes de leer datos.
# - E2: `busqueda` = bloque de pd-hpo con `tipo: optuna` o exactamente `{tipo: sin_busqueda}`; con `sin_busqueda`,
#   `config_de_busqueda` false, presupuesto null, aviso y sin comprobar la evidencia.
# - E4: el pre-chequeo valida también el bloque `logistica` y las salidas previas, antes de la logística.

# %%
def test_28_sin_mlp_final(escenario, tmp_path):
    """ruta_mlp_final inexistente → ValueError "corre pd-hpo" y cargar no se llamó."""
    cargar = _CargarContador()  # cuenta las lecturas
    with pytest.raises(ValueError, match="corre pd-hpo"):  # antes de leer datos
        ex.correr_mlp(_cfg(escenario), _rutas(escenario, tmp_path), cargar=cargar,
                      ruta_mlp_final=tmp_path / "no_existe.yaml")  # sin el archivo de pd-hpo
    assert cargar.llamadas == 0  # no leyó datos


def _escenario_sin_busqueda(escenario, tmp_path):
    """(cfg, ruta de un mlp_final.yaml) con busqueda = {tipo: sin_busqueda} y el mismo bloque mlp."""
    ruta = _escribir_mlp_final(tmp_path / "mlp_final.yaml", MLP_ESCENARIO, {"tipo": ex.TIPO_SIN_BUSQUEDA})  # a mano
    cfg = _cfg(escenario)  # config del escenario
    cfg["busqueda"] = {"tipo": ex.TIPO_SIN_BUSQUEDA}  # misma procedencia que el archivo
    return cfg, ruta  # config y YAML


def test_29_busquedas_validas():
    """El bloque de pd-hpo y {tipo: sin_busqueda} se aceptan; la constante vale "sin_busqueda"."""
    assert ex.TIPO_SIN_BUSQUEDA == "sin_busqueda"  # decisión de Andrés (E2)
    assert ex._validar_busqueda({"tipo": "sin_busqueda"}) == {"tipo": "sin_busqueda"}  # config fijada a mano
    bloque = {**BUSQUEDA_SIN_EVIDENCIA, "evidencia": {"sha256_particion": "a", "sha256_preprocesador": "b"}}  # pd-hpo
    assert ex._validar_busqueda(bloque) == bloque  # se acepta
    with pytest.raises(ValueError):  # optuna con solo el tipo
        ex._validar_busqueda({"tipo": "optuna"})  # faltan claves


def test_30_sin_busqueda_flujo(escenario, tmp_path, capsys):
    """sin_busqueda con el mismo mlp → config_de_busqueda false, presupuesto y busqueda null, tiempo = Σ, aviso, y
    sin comprobar la evidencia (ni en correr_mlp ni en comprobar_antes)."""
    cfg, ruta = _escenario_sin_busqueda(escenario, tmp_path)  # procedencia a mano
    assert ex.procedencia(validar_cfg_mlp(MLP_ESCENARIO), cfg["busqueda"], ruta)[0] is False  # aunque mlp coincida
    modelo_cls, _ = _fabrica_falsa()  # modelo falso
    entrada = ex.correr_mlp(cfg, _rutas(escenario, tmp_path), modelo_cls=modelo_cls, reloj=_reloj(),
                            ruta_mlp_final=ruta)  # corrida
    assert entrada["config_de_busqueda"] is False  # no viene de una búsqueda
    assert (entrada["n_configuraciones_busqueda"], entrada["tiempo_busqueda_s"], entrada["busqueda"]) == (
        None, None, None)  # null (núcleo §5.4 v0.16)
    assert entrada["tiempo_total_s"] == N_SEMILLAS * PASO_RELOJ  # solo Σ
    assert "sin_busqueda" in capsys.readouterr().out  # aviso

    def otra_particion(parquet, pre):
        """Lector que devuelve otra partición."""
        return {**escenario["huellas"], "sha256_particion": "0" * 64}  # huella distinta

    ex.comprobar_antes(cfg, _rutas(escenario, tmp_path), ruta_mlp_final=ruta,
                       leer_particion=otra_particion)  # no lanza: sin_busqueda no compara evidencia


def test_31_comprobar_antes_valida_la_logistica(escenario, tmp_path):
    """Bloque logistica inválido → ValueError en el pre-chequeo, sin leer la partición."""
    cfg = _cfg(escenario)  # config válida
    cfg["logistica"]["rejilla_C"] = []  # rejilla vacía
    llamadas = []  # lecturas de la partición

    def leer(parquet, pre):
        """Lector espía de la partición: registra la llamada."""
        llamadas.append(parquet)  # registra
        return escenario["huellas"]  # coherente

    with pytest.raises(ValueError):  # logística inválida
        ex.comprobar_antes(cfg, _rutas(escenario, tmp_path), ruta_mlp_final=escenario["ruta_final"],
                           leer_particion=leer)  # pre-chequeo
    assert llamadas == []  # antes de leer datos


@pytest.mark.parametrize("previo", ["summary_lista", "pred_otros_ids"])
def test_32_correr_pd_salidas_previas_antes_de_la_logistica(escenario, tmp_path, monkeypatch, previo):
    """summary.json con raíz lista o pred_test con otros ids → ValueError en el pre-chequeo; la logística (doble) y
    el MLP (doble) no se llaman."""
    _apuntar_al_escenario(escenario, monkeypatch)  # datos y mlp_final.yaml del escenario
    cfg = _cfg(escenario)  # config
    cfg["salida"], cfg["modelos"] = (tmp_path / "res").as_posix(), (tmp_path / "m").as_posix()  # salidas
    if previo == "summary_lista":  # raíz no dict
        save_json(tmp_path / "res" / ex.ARCHIVO_SUMMARY, [1, 2])  # summary ajeno
    else:  # otros ids
        (tmp_path / "res").mkdir()  # carpeta de resultados
        pd.DataFrame({"id": ["otro"], "y": [0], "pred_logistica": [0.1]}).to_parquet(
            tmp_path / "res" / ex.ARCHIVO_PRED_TEST, index=False)  # pred_test ajeno
    llamadas = []  # dobles llamados

    def log(c, rutas, seed):
        """Logística espía: registra la llamada."""
        llamadas.append("logistica")  # no debe ocurrir

    def red(c, rutas):
        """MLP espía: registra la llamada."""
        llamadas.append("mlp")  # no debe ocurrir

    with pytest.raises(ValueError):  # pre-chequeo
        ex.correr_pd(cfg, correr_log=log, correr_red=red)  # comprobar por defecto
    assert llamadas == []  # nada entrenó


def test_33_leer_test_inyectable(escenario, tmp_path):
    """comprobar_antes obtiene ids e y de test con ``leer_test`` (ruta del parquet); por defecto coinciden con los de
    cargar_datos."""
    rutas = _rutas(escenario, tmp_path)  # rutas temporales
    _, datos = ex.cargar_datos(rutas)  # lectura completa
    ids, y = ex.leer_test_rapido(rutas.parquet)  # solo id, split y target
    assert list(ids) == list(datos["test"][2]) and np.array_equal(y, datos["test"][1])  # mismos ids e y
    llamadas = []  # lecturas

    def leer_test(parquet):
        """Lector espía de ids e y de test."""
        llamadas.append(parquet)  # registra
        return ids, y  # los del parquet

    ex.comprobar_antes(_cfg(escenario), rutas, ruta_mlp_final=escenario["ruta_final"], leer_test=leer_test)  # ok
    assert llamadas == [rutas.parquet]  # una lectura


# %% [markdown]
# ## 34–38. Ronda 3 (revisor-calidad)
#
# Borrado dentro del flujo y su orden respecto de `_verificar_salidas`, red guardada = la de la semilla mediana cuando
# no es ni la primera ni la última, config que no es un mapeo, e hilos y versión de torch registrados.

# %%
def test_34_correr_mlp_borra_las_semillas_viejas(escenario, tmp_path, capsys):
    """Corrida correcta: mlp_seed42.json desaparece, logistica_seed0.json queda y la consola informa 1 borrado."""
    rutas = _rutas(escenario, tmp_path)  # salidas temporales
    vieja = rutas.resultados / ex.CARPETA_SEMILLAS / "mlp_seed42.json"  # semilla de una corrida anterior
    ajena = rutas.resultados / ex.CARPETA_SEMILLAS / "logistica_seed0.json"  # semilla de la logística
    save_json(vieja, {"viejo": True})  # existe antes
    save_json(ajena, {"ajena": True})  # existe antes
    modelo_cls, _ = _fabrica_falsa()  # modelo falso
    ex.correr_mlp(_cfg(escenario), rutas, modelo_cls=modelo_cls,
                  ruta_mlp_final=escenario["ruta_final"])  # corrida correcta
    assert not vieja.exists() and ajena.exists()  # solo se borra la del MLP
    assert "Semillas anteriores del MLP borradas: 1" in capsys.readouterr().out  # informa cuántas


def test_35_verificar_salidas_antes_de_borrar(escenario, tmp_path):
    """Si _verificar_salidas falla (summary con raíz lista), mlp_seed42.json sigue existiendo."""
    rutas = _rutas(escenario, tmp_path)  # salidas temporales
    vieja = rutas.resultados / ex.CARPETA_SEMILLAS / "mlp_seed42.json"  # semilla de una corrida anterior
    save_json(vieja, {"viejo": True})  # existe antes
    save_json(rutas.resultados / ex.ARCHIVO_SUMMARY, [1, 2])  # raíz no dict
    modelo_cls, _ = _fabrica_falsa()  # modelo falso
    with pytest.raises(ValueError):  # falla rápido
        ex.correr_mlp(_cfg(escenario), rutas, modelo_cls=modelo_cls,
                      ruta_mlp_final=escenario["ruta_final"])  # corrida
    assert vieja.exists()  # la verificación va antes del borrado


def test_36_red_guardada_es_la_mediana_del_medio(escenario, tmp_path):
    """Log-loss de val creciente en la semilla: con {7, 8, 9} la mediana es 8 (ni la primera ni la última) y
    pd_mlp.pt y pred_mlp son los de la 8."""
    rutas = _rutas(escenario, tmp_path)  # salidas temporales
    modelo_cls, _ = _fabrica_falsa(constante=True)  # PD constante creciente en la semilla
    entrada = ex.correr_mlp(_cfg(escenario), rutas, modelo_cls=modelo_cls,
                            ruta_mlp_final=escenario["ruta_final"])  # corrida
    perdidas = [load_json(rutas.resultados / ex.CARPETA_SEMILLAS / f"mlp_seed{s}.json")["metricas"]["logloss_val"]
                for s in (BASE_SEED, BASE_SEED + 1, BASE_SEED + 2)]  # log-loss de val de 7, 8 y 9
    assert perdidas == sorted(perdidas) and len(set(perdidas)) == 3  # estrictamente creciente
    medio = BASE_SEED + 1  # semilla 8
    assert entrada["semilla_mediana"] == medio  # la del medio
    assert (rutas.modelos / ex.ARCHIVO_MODELO_MLP).read_text(encoding="utf-8") == str(medio)  # red de la 8
    pred = pd.read_parquet(rutas.resultados / ex.ARCHIVO_PRED_TEST)["pred_mlp"]  # PD de test escrita
    assert np.allclose(pred, PD_BASE_FALSA + DESPLAZAMIENTO_FALSO * medio, rtol=0, atol=1e-12)  # PD de la 8


@pytest.mark.parametrize("cfg", [None, [1, 2], "base.yaml"], ids=["none", "lista", "texto"])
def test_37_config_que_no_es_mapeo(cfg):
    """correr_pd y manejador con una config que no es mapeo → ValueError antes de rutas_de."""
    with pytest.raises(ValueError, match="la config de pd debe ser un mapeo"):  # mensaje claro
        ex.correr_pd(cfg)  # orquestación
    with pytest.raises(ValueError, match="la config de pd debe ser un mapeo"):  # también desde el CLI
        ex.manejador(cfg)  # run.py pd


def test_38_hilos_y_version_de_torch(corrida):
    """hilos_torch y version_torch de cada semilla y del summary son los del proceso."""
    esperado = (torch.get_num_threads(), str(torch.__version__))  # proceso actual
    for r in corrida["semillas"].values():  # cada semilla
        assert (r["hilos_torch"], r["version_torch"]) == esperado  # registrados
    assert (corrida["entrada"]["hilos_torch"], corrida["entrada"]["version_torch"]) == esperado  # summary


# %% [markdown]
# ## 26. Configs versionadas y huella del parquet real

# %%
def test_26_configs_versionadas():
    """base.yaml: semillas.n 30, salida y modelos del proyecto, procedencia True con el mlp_final.yaml real;
    hpo.yaml sigue válido; y la huella rápida del parquet real es la del DataFrame completo."""
    base = load_config(ROOT / "configs" / "pd" / "base.yaml")  # config de B5
    assert (base["semillas"]["n"], base["salida"], base["modelos"]) == (30, "results/data/pd", "models")  # claves
    _, _, mlp, busqueda = ex.validar_cfg_mlp_final(base)  # validación completa
    assert ex.procedencia(mlp, busqueda, ROOT / "configs" / "pd" / "mlp_final.yaml") == (True, [])  # de la búsqueda
    hpo.validar_cfg(load_config(ROOT / "configs" / "pd" / "hpo.yaml"))  # pd-hpo sigue válido
    parquet, pre = ROOT / "data" / "processed" / "loans_pd.parquet", ROOT / "models" / "preprocesador_pd.json"  # A
    assert ex.evidencia_rapida(parquet, pre) == ex.evidencia(pd.read_parquet(parquet), pre)  # misma huella


# %% [markdown]
# ## 27. Datos reales (lenta)

# %%
@pytest.mark.lento
def test_27_datos_reales(tmp_path, monkeypatch):
    """correr_pd con los datos reales, 2 semillas, rejilla de un C y salidas en tmp_path: la entrada mlp tiene n = 2."""
    monkeypatch.chdir(ROOT)  # rutas del proyecto relativas a la raíz
    cfg = apply_overrides(load_config(ROOT / "configs" / "pd" / "base.yaml"), [  # config versionada
        "semillas.n=2", "logistica.rejilla_C=[1.0e-2]",  # corto
        f"salida={(tmp_path / 'res').as_posix()}", f"modelos={(tmp_path / 'models').as_posix()}"])  # temporales
    res = ex.correr_pd(cfg)  # corrida completa
    assert res["mlp"]["metricas"]["auc_test"]["n"] == 2  # dos semillas
    assert res["mlp"]["config_de_busqueda"] is True  # la config es la de la búsqueda
    assert (tmp_path / "models" / ex.ARCHIVO_MODELO_MLP).exists()  # modelo guardado
