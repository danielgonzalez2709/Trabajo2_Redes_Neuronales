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
# # Pruebas del análisis de sensibilidad con `grade` e `int_rate` (B10, B-pd §3.8)
#
# Se prueban las reglas de la unión (mismos ids y orden, sin repetidos ni vacíos), la validación de las 17 variables
# (ninguna prohibida del núcleo §5.7; solo `grade` e `int_rate` como adicionales), el preprocesador ajustado **solo con
# train**, las rutas y configs de cada modo (nunca las del modelo principal), la comparación con el principal y la
# orquestación con dependencias inyectadas.
#
# **Datos:** sintéticos, generados aquí (núcleo §0 regla 8: solo para desarrollar). Las únicas excepciones leen los
# datos reales de A: la prueba 10 (huella de la unión real; se omite si faltan) y la 13 (marcada `lento`). La 9 lee
# las configs versionadas.
#
# **Nunca** se escribe en `data/`, `models/`, `results/`, `configs/` ni `blog/` reales: todo va a `tmp_path`.
#
# Se usa el módulo (`sens.<nombre>`) y no `from … import`: cada prueba falla por separado mientras su función no existe.

# %% [markdown]
# ## Importaciones

# %%
import ast  # imports de src/pd/sensibilidad.py (prueba 2)
import copy  # copias de configs
import functools  # partial de correr_mlp y comprobar_antes (prueba 6)
import hashlib  # huellas de los archivos que no deben cambiar
import importlib.util  # carga de tests/test_fuga.py por ruta (prueba 2)
import json  # bytes del preprocesador (prueba 3)
from pathlib import Path  # rutas

import numpy as np  # datos sintéticos
import pandas as pd  # parquets sintéticos
import pytest  # fixtures y aserciones
import yaml  # YAML temporales
from scipy.special import expit  # σ de la logística generadora

from src.common import cli  # CLI de run.py (pruebas 11 y 13)
from src.common.config import apply_overrides, load_config  # mismo cargador que el CLI
from src.common.preprocesamiento import OTRA  # categoría de las no vistas (núcleo §5.2)
from src.common.results import load_json, save_json  # JSON del núcleo §5.4
from src.pd import experimento as ex  # pipeline de B que se reutiliza
from src.pd import hpo  # búsqueda que se reutiliza (prueba 6)
from src.pd import sensibilidad as sens  # módulo bajo prueba

# %% [markdown]
# ## Constantes de las pruebas
#
# Valores de prueba, no del dominio: tamaños pequeños para que cada corrida tarde poco.

# %%
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio
RUTA_CFG = ROOT / "configs" / "pd" / "sensibilidad.yaml"  # config bajo prueba
RUTA_FINAL_REAL = ROOT / "configs" / "pd" / "mlp_final.yaml"  # la del modelo principal (no se toca)
RUTA_BASE = ROOT / "configs" / "pd" / "base.yaml"  # logística del principal
RUTA_HPO = ROOT / "configs" / "pd" / "hpo.yaml"  # búsqueda del principal
RUTA_DATOS_A = ROOT / "configs" / "datos.yaml"  # config de la Parte A
RESULTADOS_PD = ROOT / "results" / "data" / "pd"  # resultados del principal (no se tocan)
MODELOS = ROOT / "models"  # modelos del principal (no se tocan)
SEMILLA_DATOS = 20261027  # semilla de los datos sintéticos (fecha de la tarea)
N_SINT = {"train": 300, "val": 150, "test": 150}  # filas por split del parquet sintético
ANIO_SPLIT = {"train": 2012, "val": 2013, "test": 2014}  # año de originación por split
CATEGORIAS = ["A", "B", "C"]  # valores de la categórica sintética
GRADOS = list("ABCDEFG")  # grados de Lending Club (sensibilidad_lc.parquet real)
FEATURES_SINT = [  # 2 numéricas + 1 categórica, con los campos del contrato que se validan
    {"nombre": "monto", "tipo": "num", "transformacion": "log1p", "disponible_en_originacion": True},  # numérica
    {"nombre": "razon", "tipo": "num", "transformacion": "ninguna", "disponible_en_originacion": True},  # numérica
    {"nombre": "clase", "tipo": "cat", "transformacion": "ninguna", "disponible_en_originacion": True},  # categórica
]
ADICIONALES_SINT = [  # las dos de configs/features_sensibilidad.yaml (solo los campos que se usan)
    {"nombre": "grade", "tipo": "cat", "transformacion": "ninguna", "disponible_en_originacion": True},  # grado
    {"nombre": "int_rate", "tipo": "num", "transformacion": "ninguna", "disponible_en_originacion": True},  # tasa
]
PERCENTILES = [1.0, 99.0]  # percentiles del datos.yaml sintético (los de configs/datos.yaml)
MLP_RAPIDO = {  # bloque mlp válido y rápido
    "capas": [4], "activacion": "relu", "dropout": 0.0, "l2": 1.0e-4, "optimizador": "adam", "lr": 1.0e-2,
    "momento": 0.9, "batch": 64, "max_epocas": 2, "paciencia": 2, "device": "cpu", "pos_weight": 1.0,
}
REJILLA_CORTA = [1.0e-2, 1.0]  # rejilla de C rápida
N_SEMILLAS = 2  # semillas sintéticas
N_ENSAYOS = 2  # ensayos de la búsqueda sintética
METRICAS_TEST = ("auc_test", "gini_test", "ks_test", "brier_test")  # métricas comparadas (B-pd §3.8)


# %% [markdown]
# ## Datos sintéticos: `loans_pd`
#
# Mismo esquema que `loans_pd.parquet` (núcleo §5.1) con 3 variables; ids de texto, no ordenados.

# %%
def _loans_sint() -> pd.DataFrame:
    """loans_pd sintético: id, issue_d, anio_origen, target, split y 3 variables."""
    rng = np.random.default_rng(SEMILLA_DATOS)  # generador de los datos
    partes = []  # un DataFrame por split
    for split, n in N_SINT.items():  # train, val, test
        razon = rng.normal(size=n)  # numérica centrada
        clase = rng.choice(CATEGORIAS, size=n)  # categórica
        partes.append(pd.DataFrame({  # filas del split
            "issue_d": pd.Timestamp(f"{ANIO_SPLIT[split]}-06-01"),  # fecha de originación
            "anio_origen": np.int32(ANIO_SPLIT[split]),  # año
            "target": rng.binomial(1, expit(0.8 * razon - 1.0)).astype(np.int8),  # objetivo 0/1
            "split": split,  # partición
            "monto": rng.lognormal(mean=9.0, sigma=0.5, size=n), "razon": razon, "clase": clase,  # variables
        }))
    df = pd.concat(partes, ignore_index=True)  # en orden train, val, test
    df.insert(0, "id", [f"id{i:05d}" for i in rng.permutation(len(df))])  # id str no ordenado
    return df  # DataFrame


# %% [markdown]
# ## Datos sintéticos: `sensibilidad_lc`
#
# `id`, `grade` e `int_rate` en el mismo orden que `loans`; train contiene los 7 grados (así el preprocesador los ve
# todos) y la tasa crece con el grado.

# %%
def _lc_sint(loans: pd.DataFrame) -> pd.DataFrame:
    """sensibilidad_lc sintético alineado con ``loans``."""
    rng = np.random.default_rng(SEMILLA_DATOS + 1)  # otro generador
    grados = np.resize(GRADOS, len(loans))  # A…G repetidos: todos aparecen en train
    tasa = 6.0 + 3.0 * np.array([GRADOS.index(g) for g in grados]) + rng.normal(scale=0.5, size=len(loans))  # %
    return pd.DataFrame({"id": loans["id"].to_numpy(), "grade": grados, "int_rate": tasa})  # mismo orden


# %% [markdown]
# ## Entorno sintético (módulo)
#
# Escribe en una carpeta temporal los insumos de `preparar` (los dos parquets y los tres YAML) y corre el **modelo
# principal** sintético con el mismo pipeline de B (logística + MLP sobre las 3 variables), para que la comparación
# tenga un `summary.json` y un `pred_test.parquet` reales.

# %%
@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    """Insumos sintéticos y el modelo principal sintético; devuelve sus rutas."""
    base = tmp_path_factory.mktemp("sens_b10")  # carpeta del entorno
    loans = _loans_sint()  # loans_pd sintético
    loans.to_parquet(base / "loans_pd.parquet", index=False)  # parquet de A
    _lc_sint(loans).to_parquet(base / "sensibilidad_lc.parquet", index=False)  # parquet de A
    (base / "features.yaml").write_text(yaml.safe_dump({"features": FEATURES_SINT}), encoding="utf-8")  # 3 variables
    (base / "adicionales.yaml").write_text(yaml.safe_dump({"adicionales": ADICIONALES_SINT}),
                                          encoding="utf-8")  # grade e int_rate
    (base / "datos.yaml").write_text(yaml.safe_dump({"preprocesamiento": {"percentiles_recorte": PERCENTILES}}),
                                    encoding="utf-8")  # percentiles
    pre = sens.ajustar_preprocesador(loans, FEATURES_SINT, tuple(PERCENTILES))  # preprocesador del principal
    save_json(base / "preprocesador_pd.json", pre.to_dict())  # como lo deja A
    rutas = ex.RutasPD(base / "loans_pd.parquet", base / "preprocesador_pd.json", base / "pd", base / "models")  # pd
    cfg = _cfg_principal()  # logística + MLP rápidos, sin_busqueda
    ex.correr_logistica(cfg, rutas, seed=cfg["base_seed"])  # logística del principal
    ex.correr_mlp(cfg, rutas)  # MLP del principal
    return {"base": base, "loans": loans, "principal": base / "pd" / ex.ARCHIVO_SUMMARY}  # rutas del entorno


# %% [markdown]
# ## Configs sintéticas
#
# `_cfg_principal`: la del modelo principal sintético. `_cfg`: la config real de sensibilidad con bloques rápidos y
# todas las rutas de `sensibilidad` en el entorno y en ``carpeta``.

# %%
def _cfg_principal() -> dict:
    """Config del principal sintético (sin búsqueda, 2 semillas)."""
    return {"base_seed": 0, "datos": {"muestra": None},  # todos los datos
            "logistica": {"rejilla_C": list(REJILLA_CORTA), "max_iter": 1000, "tol": 1.0e-6},  # rápida
            "mlp": dict(MLP_RAPIDO), "busqueda": {"tipo": ex.TIPO_SIN_BUSQUEDA},  # fijada a mano
            "semillas": {"n": N_SEMILLAS}}  # semillas


def _cfg(entorno, carpeta: Path, modos=("reusar", "optuna")) -> dict:
    """Config real de sensibilidad con bloques rápidos y rutas en el entorno y en ``carpeta``."""
    cfg = load_config(RUTA_CFG)  # config real
    cfg["logistica"]["rejilla_C"] = list(REJILLA_CORTA)  # rápida
    cfg["mlp"] = dict(MLP_RAPIDO)  # rápido
    cfg["semillas"]["n"] = N_SEMILLAS  # pocas semillas
    cfg["hpo"]["n_ensayos"] = N_ENSAYOS  # pocos ensayos
    b = entorno["base"]  # insumos
    cfg["sensibilidad"].update({  # rutas temporales
        "modos": list(modos), "loans_pd": str(b / "loans_pd.parquet"), "features": str(b / "features.yaml"),  # A
        "datos_lc": str(b / "sensibilidad_lc.parquet"), "adicionales": str(b / "adicionales.yaml"),  # A
        "config_datos": str(b / "datos.yaml"), "union": str(carpeta / "res" / "union.parquet"),  # entradas
        "preprocesador": str(carpeta / "models" / "pre.json"), "salida": str(carpeta / "res"),  # salidas
        "modelos": str(carpeta / "models"), "principal": str(entorno["principal"]),  # salidas y principal
    })
    return cfg  # config completa


# %% [markdown]
# ## Huellas de carpetas reales
#
# SHA-256 de cada archivo (recursivo) de las carpetas y archivos del modelo principal: deben quedar iguales.

# %%
def _huellas_principal() -> dict:
    """{ruta: sha256} de results/data/pd, models/pd_*, models/preprocesador_pd.json y configs/pd/mlp_final.yaml."""
    archivos = [RUTA_FINAL_REAL, *MODELOS.glob("pd_*.*"), MODELOS / "preprocesador_pd.json"]  # archivos sueltos
    if RESULTADOS_PD.exists():  # puede no existir en un clon limpio
        archivos += [p for p in RESULTADOS_PD.rglob("*") if p.is_file()]  # resultados del principal
    return {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in archivos if p.is_file()}  # huellas


# %% [markdown]
# ## 1. `unir_sensibilidad`: conserva filas y orden
#
# La unión agrega `grade` e `int_rate` al final, con las mismas filas, el mismo orden y el mismo índice que `loans_pd`.

# %%
def test_01_unir_conserva_filas_y_orden():
    """Columnas de loans + adicionales, mismas filas y orden; no muta las entradas."""
    loans = _loans_sint()  # loans_pd
    lc = _lc_sint(loans)  # adicionales
    copia_loans, copia_lc = loans.copy(), lc.copy()  # para ver que no se mutan
    union = sens.unir_sensibilidad(loans, lc, ["grade", "int_rate"])  # unión
    assert list(union.columns) == [*loans.columns, "grade", "int_rate"]  # adicionales al final
    assert union["id"].tolist() == loans["id"].tolist()  # mismo orden
    pd.testing.assert_frame_equal(union[list(loans.columns)], loans)  # loans intacto dentro de la unión
    assert union["grade"].tolist() == lc["grade"].tolist()  # alineado fila a fila
    pd.testing.assert_frame_equal(loans, copia_loans)  # loans sin mutar
    pd.testing.assert_frame_equal(lc, copia_lc)  # lc sin mutar


# %% [markdown]
# ## 1b. `unir_sensibilidad`: rechazos
#
# Ids en otro orden, uno que falta, un id repetido, `grade` o `int_rate` vacíos o una columna que ya existe → `ValueError`.

# %%
def _romper(caso: str):
    """(loans, lc, adicionales) con el defecto ``caso``."""
    loans = _loans_sint()  # loans_pd
    lc = _lc_sint(loans)  # adicionales correctas
    adicionales = ["grade", "int_rate"]  # nombres
    if caso == "orden":  # dos filas intercambiadas
        lc = lc.iloc[[1, 0, *range(2, len(lc))]].reset_index(drop=True)  # mismo conjunto, otro orden
    elif caso == "falta":  # una fila menos
        lc = lc.iloc[:-1]  # falta el último id
    elif caso == "duplicado":  # un id repetido en las dos tablas (mismo orden)
        loans.loc[1, "id"] = loans.loc[0, "id"]  # repetido en loans
        lc.loc[1, "id"] = lc.loc[0, "id"]  # y en lc, para que el orden coincida
    elif caso == "grade_nan":  # vacío
        lc.loc[3, "grade"] = None  # NA
    elif caso == "grade_blanco":  # texto vacío
        lc.loc[3, "grade"] = "  "  # solo espacios
    elif caso == "int_rate_nan":  # vacío numérico
        lc.loc[3, "int_rate"] = np.nan  # NaN
    elif caso == "columna_existe":  # la adicional ya está en loans
        loans["grade"] = "A"  # columna repetida
    elif caso == "falta_columna":  # la adicional no está en lc
        lc = lc.drop(columns=["int_rate"])  # sin int_rate
    return loans, lc, adicionales  # entradas rotas


@pytest.mark.parametrize("caso", ["orden", "falta", "duplicado", "grade_nan", "grade_blanco", "int_rate_nan",
                                  "columna_existe", "falta_columna"])  # defectos
def test_01b_unir_rechaza(caso):
    """Cada defecto de la unión → ValueError."""
    with pytest.raises(ValueError):  # rechazo
        sens.unir_sensibilidad(*_romper(caso))  # unión


# %% [markdown]
# ## 2. `PROHIBIDAS` = la lista de `tests/test_fuga.py`
#
# La constante vive en `src/` (que nunca importa de `tests/`); esta prueba la compara con la de la Parte A.

# %%
def _test_fuga():
    """Módulo tests/test_fuga.py cargado por ruta."""
    spec = importlib.util.spec_from_file_location("test_fuga_b10", ROOT / "tests" / "test_fuga.py")  # por ruta
    modulo = importlib.util.module_from_spec(spec)  # módulo vacío
    spec.loader.exec_module(modulo)  # ejecuta sus definiciones (sin correr las pruebas)
    return modulo  # PROHIBIDAS y PREFIJOS_PROHIBIDOS


def test_02_prohibidas_iguales_a_test_fuga():
    """PROHIBIDAS y PREFIJOS_PROHIBIDOS coinciden con tests/test_fuga.py."""
    fuga = _test_fuga()  # lista de la Parte A
    assert set(sens.PROHIBIDAS) == set(fuga.PROHIBIDAS)  # mismos nombres
    assert tuple(sens.PREFIJOS_PROHIBIDOS) == tuple(fuga.PREFIJOS_PROHIBIDOS)  # mismos prefijos


def test_02b_src_no_importa_de_tests():
    """src/pd/sensibilidad.py no importa nada de tests/ (ni test_*)."""
    arbol = ast.parse((ROOT / "src" / "pd" / "sensibilidad.py").read_text(encoding="utf-8"))  # código
    modulos = set()  # módulos importados
    for nodo in ast.walk(arbol):  # cada nodo
        if isinstance(nodo, ast.Import):  # import x
            modulos.update(a.name for a in nodo.names)  # nombres
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:  # from x import y
            modulos.add(nodo.module)  # módulo
    assert not [m for m in modulos if m.split(".")[0] in {"tests", "conftest"} or m.startswith("test_")]  # ninguno


# %% [markdown]
# ## 2c. `features_sensibilidad`: 17 en orden
#
# Las 15 (aquí 3) del modelo principal y luego las 2 adicionales; con las reales, 17.

# %%
def test_02c_features_en_orden():
    """Principales y luego adicionales, en ese orden; con las reales, 17 nombres."""
    nombres = [f["nombre"] for f in sens.features_sensibilidad(FEATURES_SINT, ADICIONALES_SINT)]  # sintéticas
    assert nombres == ["monto", "razon", "clase", "grade", "int_rate"]  # orden
    reales = load_config(ROOT / "configs" / "features.yaml")["features"]  # 15 de A
    adicionales = load_config(ROOT / "configs" / "features_sensibilidad.yaml")["adicionales"]  # 2 de A
    juntas = sens.features_sensibilidad(reales, adicionales)  # 17
    assert len(juntas) == 17 and [f["nombre"] for f in juntas[-2:]] == ["grade", "int_rate"]  # al final


# %% [markdown]
# ## 2d. `features_sensibilidad`: rechazos
#
# Nombre repetido, `disponible_en_originacion` distinto de true, una prohibida del núcleo §5.7 (por nombre o por
# prefijo) entre las principales o las adicionales, y `sub_grade` o `installment` como adicionales → `ValueError`.

# %%
def _var(nombre: str, tipo: str = "num", disponible=True) -> dict:
    """Variable mínima con ``nombre``."""
    return {"nombre": nombre, "tipo": tipo, "transformacion": "ninguna", "disponible_en_originacion": disponible}


@pytest.mark.parametrize("principales, adicionales", [  # casos rotos
    (FEATURES_SINT, [ADICIONALES_SINT[0], ADICIONALES_SINT[0]]),  # adicional repetida
    (FEATURES_SINT + [_var("monto")], ADICIONALES_SINT),  # principal repetida
    (FEATURES_SINT + [_var("grade", "cat")], ADICIONALES_SINT),  # adicional que ya es principal
    (FEATURES_SINT, [ADICIONALES_SINT[0], _var("int_rate", disponible=False)]),  # no disponible
    (FEATURES_SINT, [ADICIONALES_SINT[0], _var("int_rate", disponible="true")]),  # texto, no bool
    (FEATURES_SINT + [_var("total_pymnt")], ADICIONALES_SINT),  # prohibida entre las principales
    (FEATURES_SINT + [_var("total_rec_algo")], ADICIONALES_SINT),  # prefijo prohibido
    (FEATURES_SINT, [*ADICIONALES_SINT, _var("total_pymnt")]),  # prohibida entre las adicionales
    (FEATURES_SINT, [*ADICIONALES_SINT, _var("sub_grade", "cat")]),  # redundante con grade (A §3.4)
    (FEATURES_SINT, [*ADICIONALES_SINT, _var("installment")]),  # depende de la tasa (A §3.4)
    (FEATURES_SINT, [*ADICIONALES_SINT, _var("otra_cosa")]),  # fuera de grade e int_rate
    (FEATURES_SINT, []),  # sin adicionales no hay análisis
], ids=["adic_repetida", "princ_repetida", "adic_en_principales", "no_disponible", "disponible_texto",
        "prohibida_principal", "prefijo", "prohibida_adicional", "sub_grade", "installment", "ajena", "vacia"])
def test_02d_features_rechaza(principales, adicionales):
    """Cada defecto → ValueError."""
    with pytest.raises(ValueError):  # rechazo
        sens.features_sensibilidad(principales, adicionales)  # validación


# %% [markdown]
# ## 3. Preprocesador sin fuga
#
# Cambiar val o test no cambia el preprocesador (byte a byte del JSON); cambiar train sí. `grade` sale como A–G + OTRA
# y el recorte de `int_rate` usa los percentiles de train.

# %%
def _bytes_pre(df: pd.DataFrame) -> bytes:
    """Bytes del JSON del preprocesador ajustado con ``df``."""
    feats = sens.features_sensibilidad(FEATURES_SINT, ADICIONALES_SINT)  # 5 variables
    pre = sens.ajustar_preprocesador(df, feats, tuple(PERCENTILES))  # ajuste
    return json.dumps(pre.to_dict(), sort_keys=True).encode("utf-8")  # bytes comparables


def test_03_preprocesador_sin_fuga():
    """val/test no influyen; train sí; grade A–G + OTRA; recorte de int_rate de train."""
    loans = _loans_sint()  # loans
    union = sens.unir_sensibilidad(loans, _lc_sint(loans), ["grade", "int_rate"])  # unión
    referencia = _bytes_pre(union)  # preprocesador base
    otra = union.copy()  # val y test alterados
    fuera = otra["split"] != "train"  # filas de val y test
    otra.loc[fuera, "int_rate"] = 99.0  # tasa extrema
    otra.loc[fuera, "grade"] = "Z"  # grado nunca visto
    otra.loc[fuera, "monto"] = 1.0  # monto extremo
    assert _bytes_pre(otra) == referencia  # sin fuga
    tren = union.copy()  # train alterado
    tren.loc[tren["split"] == "train", "int_rate"] += 1.0  # tasa de train cambia
    assert _bytes_pre(tren) != referencia  # train sí cuenta
    feats = sens.features_sensibilidad(FEATURES_SINT, ADICIONALES_SINT)  # 5 variables
    pre = sens.ajustar_preprocesador(union, feats, tuple(PERCENTILES)).to_dict()  # dict
    var = {v["nombre"]: v for v in pre["variables"]}  # por nombre
    assert var["grade"]["categorias"] == [*GRADOS, OTRA]  # A–G + OTRA
    train = union.loc[union["split"] == "train", "int_rate"]  # tasa de train
    assert var["int_rate"]["recorte"] == pytest.approx(list(np.percentile(train, PERCENTILES)))  # percentiles de train


# %% [markdown]
# ## 3b. Preprocesador: sin train no hay ajuste
#
# Un DataFrame sin filas de train → `ValueError` (no se ajusta con val o test por accidente).

# %%
def test_03b_preprocesador_sin_train():
    """Sin filas split == train → ValueError."""
    loans = _loans_sint()  # loans
    sin_train = loans[loans["split"] != "train"]  # solo val y test
    with pytest.raises(ValueError):  # rechazo
        sens.ajustar_preprocesador(sin_train, FEATURES_SINT, tuple(PERCENTILES))  # ajuste


# %% [markdown]
# ## 4. Rutas de cada modo
#
# Nunca apuntan a `results/data/pd`, a `models/pd_*` ni a los insumos del principal; cada modo en su carpeta; un modo
# desconocido → `ValueError`.

# %%
def test_04_rutas_de_los_modos():
    """Con la config real: carpetas por modo, distintas y fuera de las del principal."""
    cfg = load_config(RUTA_CFG)  # config real
    rutas = {m: sens.rutas_sensibilidad(cfg, m) for m in sens.MODOS}  # una por modo
    assert rutas["reusar"].resultados != rutas["optuna"].resultados  # carpetas distintas
    assert rutas["reusar"].modelos != rutas["optuna"].modelos  # ídem
    for r in rutas.values():  # cada modo
        assert r.resultados.resolve() != RESULTADOS_PD.resolve()  # no es results/data/pd
        assert RESULTADOS_PD.resolve() not in r.resultados.resolve().parents  # ni está dentro
        assert r.modelos.resolve() != MODELOS.resolve()  # no pisa models/pd_*.{joblib,pt}
        assert r.parquet.resolve() != (ROOT / ex.RUTAS_PD.parquet).resolve()  # no es loans_pd
        assert r.preprocesador.resolve() != (ROOT / ex.RUTAS_PD.preprocesador).resolve()  # ni el de A
    destino = Path(sens.cfg_hpo_sensibilidad(cfg)["hpo"]["destino"])  # mlp_final.yaml del modo optuna
    assert destino.resolve() != RUTA_FINAL_REAL.resolve()  # nunca configs/pd/mlp_final.yaml
    assert destino.parent == rutas["optuna"].resultados  # dentro de optuna/
    with pytest.raises(ValueError):  # modo desconocido
        sens.rutas_sensibilidad(cfg, "pd")  # ni siquiera el del principal


# %% [markdown]
# ## 4b. Rutas protegidas
#
# Una config que apunte a las rutas del modelo principal → `ValueError` antes de escribir nada.

# %%
@pytest.mark.parametrize("clave, valor", [  # rutas del principal
    ("salida", "results/data/pd"),  # reusar/ quedaría dentro de results/data/pd
    ("salida", "results/data"),  # results/data/{modo}: no pisa, pero optuna/ y reusar/ fuera de B → se acepta
    ("modelos", "."),  # ./reusar: se acepta (no pisa archivos del principal)
    ("union", "data/processed/loans_pd.parquet"),  # pisaría loans_pd
    ("preprocesador", "models/preprocesador_pd.json"),  # pisaría el preprocesador de A
], ids=["salida_pd", "salida_data", "modelos_punto", "union_loans", "pre_de_A"])
def test_04b_rutas_protegidas(clave, valor, monkeypatch):
    """Solo las rutas que pisarían o anidarían el principal se rechazan."""
    monkeypatch.chdir(ROOT)  # rutas relativas a la raíz
    cfg = load_config(RUTA_CFG)  # config real
    cfg["sensibilidad"][clave] = valor  # ruta a probar
    aceptadas = {("salida", "results/data"), ("modelos", ".")}  # no pisan nada del principal
    if (clave, valor) in aceptadas:  # se aceptan
        sens.rutas_sensibilidad(cfg, "reusar")  # sin error
        return  # listo
    with pytest.raises(ValueError):  # rechazo
        sens.rutas_sensibilidad(cfg, "reusar")  # rutas


# %% [markdown]
# ## 5. `cfg_modo("reusar")`
#
# `busqueda` es `sin_busqueda`: `config_de_busqueda` false y presupuesto de la búsqueda null. Con el `busqueda` de
# `mlp_final.yaml` del principal la huella no coincide: `presupuesto_busqueda` da `ValueError` (por eso reusar usa
# `sin_busqueda`).

# %%
def test_05_cfg_modo_reusar(entorno, tmp_path):
    """reusar: sin_busqueda, mlp de la config, salidas en reusar/; el busqueda del principal no pasa."""
    cfg = _cfg(entorno, tmp_path)  # config sintética
    original = copy.deepcopy(cfg)  # para ver que no se muta
    c = sens.cfg_modo(cfg, "reusar")  # config del modo
    assert cfg == original  # sin mutar
    assert c["busqueda"] == {"tipo": ex.TIPO_SIN_BUSQUEDA} and c["mlp"] == cfg["mlp"]  # misma arquitectura
    assert "hpo" not in c and "sensibilidad" not in c  # solo lo que usa correr_pd
    assert Path(c["salida"]) == sens.rutas_sensibilidad(cfg, "reusar").resultados  # salidas del modo
    _, _, mlp, busqueda = ex.validar_cfg_mlp_final(c)  # bloques válidos
    de_busqueda, _ = ex.procedencia(mlp, busqueda, RUTA_FINAL_REAL)  # frente al principal
    assert de_busqueda is False  # no es la de la búsqueda
    assert ex.presupuesto_busqueda(busqueda, de_busqueda, {"x": 1})["n_configuraciones_busqueda"] is None  # null
    final = load_config(RUTA_FINAL_REAL)  # mlp_final.yaml del principal
    assert ex.procedencia(final["mlp"], final["busqueda"], RUTA_FINAL_REAL)[0] is True  # sería "de la búsqueda"
    huellas_sens = {"sha256_particion": "0" * 64, "sha256_preprocesador": "1" * 64}  # otra huella
    with pytest.raises(ValueError, match="otros datos"):  # por eso reusar usa sin_busqueda
        ex.presupuesto_busqueda(final["busqueda"], True, huellas_sens)  # huella distinta


# %% [markdown]
# ## 5b. `cfg_modo("optuna")` y modos desconocidos
#
# optuna toma `mlp` y `busqueda` del `mlp_final.yaml` del modo (obligatorio); un modo desconocido → `ValueError`.

# %%
def test_05b_cfg_modo_optuna_y_desconocido(entorno, tmp_path):
    """optuna exige su mlp_final; otro modo → ValueError."""
    cfg = _cfg(entorno, tmp_path)  # config sintética
    final = {"mlp": {**MLP_RAPIDO, "capas": [8]}, "busqueda": {"tipo": "optuna", "n_ensayos": 2}}  # del modo
    c = sens.cfg_modo(cfg, "optuna", final)  # config del modo
    assert c["mlp"] == final["mlp"] and c["busqueda"] == final["busqueda"]  # del YAML del modo
    assert c["mlp"] is not final["mlp"]  # copia
    assert Path(c["modelos"]) == sens.rutas_sensibilidad(cfg, "optuna").modelos  # modelos del modo
    with pytest.raises(ValueError):  # optuna sin su YAML
        sens.cfg_modo(cfg, "optuna")  # falta mlp_final_optuna
    with pytest.raises(ValueError):  # reusar no admite YAML
        sens.cfg_modo(cfg, "reusar", final)  # se ignoraría en silencio
    with pytest.raises(ValueError):  # desconocido
        sens.cfg_modo(cfg, "principal")  # no existe


# %% [markdown]
# ## 5c. `validar_cfg_sensibilidad`
#
# Antes de leer datos: modos ⊆ {reusar, optuna}, sin repetidos y no vacíos; claves exactas de `sensibilidad` y de
# `hpo`; `busqueda` = `sin_busqueda`; raíz sin claves ajenas; bloques `mlp`, `logistica` y `datos` válidos.

# %%
@pytest.mark.parametrize("cambio", [  # (función que rompe la cfg)
    lambda c: c["sensibilidad"].update(modos=[]),  # vacío
    lambda c: c["sensibilidad"].update(modos=["reusar", "reusar"]),  # repetido
    lambda c: c["sensibilidad"].update(modos=["otro"]),  # desconocido
    lambda c: c["sensibilidad"].update(modos="reusar"),  # texto, no lista
    lambda c: c["sensibilidad"].update(salida=""),  # ruta vacía
    lambda c: c["sensibilidad"].update(extra="x"),  # clave ajena
    lambda c: c["sensibilidad"].pop("principal"),  # falta una clave
    lambda c: c["hpo"].update(destino="configs/pd/mlp_final.yaml"),  # pisaría el principal
    lambda c: c["hpo"].update(n_ensayos=0),  # búsqueda vacía
    lambda c: c.update(busqueda=load_config(RUTA_FINAL_REAL)["busqueda"]),  # búsqueda del principal
    lambda c: c.update(salida="results/data/pd"),  # clave ajena en la raíz
    lambda c: c["mlp"].update(pos_weight=0.0),  # mlp inválido (validar_cfg_mlp: pos_weight > 0, B2)
    lambda c: c["logistica"].update(rejilla_C=[]),  # logística inválida
    lambda c: c["datos"].update(muestra=2),  # muestra inválida
    lambda c: c["semillas"].update(n=0),  # semillas inválidas
], ids=["vacio", "repetido", "desconocido", "texto", "ruta_vacia", "clave_ajena", "falta_clave", "hpo_destino",
        "hpo_cero", "busqueda_principal", "raiz_ajena", "mlp", "logistica", "muestra", "semillas"])
def test_05c_validar_cfg_sensibilidad(entorno, tmp_path, cambio):
    """Cada defecto → ValueError; la config sana pasa."""
    cfg = _cfg(entorno, tmp_path)  # config sana
    sens.validar_cfg_sensibilidad(cfg)  # pasa
    cambio(cfg)  # defecto
    with pytest.raises(ValueError):  # rechazo
        sens.validar_cfg_sensibilidad(cfg)  # antes de leer datos


# %% [markdown]
# ## 5d. La validación va antes de preparar
#
# Con una config inválida, `correr_sensibilidad` no llama a `preparar_fn` ni a nada más.

# %%
def test_05d_validacion_antes_de_todo(entorno, tmp_path):
    """Config inválida → ValueError sin llamar a ninguna dependencia."""
    cfg = _cfg(entorno, tmp_path)  # config
    cfg["sensibilidad"]["modos"] = ["otro"]  # inválida
    llamadas = []  # dependencias llamadas

    def espia(*a, **k):
        """Registra cualquier llamada."""
        llamadas.append(a)  # llamada

    with pytest.raises(ValueError):  # rechazo
        sens.correr_sensibilidad(cfg, preparar_fn=espia, correr_pd_fn=espia, buscar_fn=espia,
                                 comprobar_fn=espia, comparar_fn=espia)  # orquestación
    assert llamadas == []  # nada corrió


# %% [markdown]
# ## 5e. `preparar` con lectura y escritura inyectables
#
# Lee los dos parquets y los tres YAML con `leer`, une, ajusta y escribe con `escribir` exactamente la unión y el
# preprocesador (aquí, con las 3 variables sintéticas + `grade` + `int_rate`).

# %%
def test_05e_preparar_inyectable(entorno, tmp_path):
    """preparar escribe 2 archivos en las rutas de la config, con el contenido esperado."""
    cfg = _cfg(entorno, tmp_path)  # config
    escritos = {}  # ruta → objeto

    def escribir(ruta, obj):
        """Guarda en memoria."""
        escritos[Path(ruta)] = obj  # registro

    res = sens.preparar(cfg, leer=sens.leer_archivo, escribir=escribir)  # preparación
    s = cfg["sensibilidad"]  # rutas
    assert set(escritos) == {Path(s["union"]), Path(s["preprocesador"])}  # exactamente dos
    union = escritos[Path(s["union"])]  # DataFrame
    assert union["id"].tolist() == entorno["loans"]["id"].tolist()  # mismo orden
    pre = escritos[Path(s["preprocesador"])]  # dict
    assert [v["nombre"] for v in pre["variables"]] == ["monto", "razon", "clase", "grade", "int_rate"]  # 5
    assert res["filas"] == N_SINT  # filas por split
    assert not (tmp_path / "res").exists()  # nada en disco


# %% [markdown]
# ## 5f. `preparar` en disco
#
# Con las escrituras por defecto, la unión y el preprocesador quedan en disco (atómicos) y se pueden releer.

# %%
def test_05f_preparar_en_disco(entorno, tmp_path):
    """Unión y preprocesador escritos y legibles; sin temporales sueltos."""
    cfg = _cfg(entorno, tmp_path)  # config
    sens.preparar(cfg)  # lectura y escritura por defecto
    s = cfg["sensibilidad"]  # rutas
    union = pd.read_parquet(s["union"])  # relectura
    assert list(union.columns)[-2:] == ["grade", "int_rate"] and len(union) == sum(N_SINT.values())  # completa
    assert load_json(s["preprocesador"])["variables"][-1]["nombre"] == "int_rate"  # JSON válido
    assert not list(tmp_path.rglob("*.tmp"))  # sin temporales


# %% [markdown]
# ## 6. Modo optuna y reusar de punta a punta (sintético)
#
# Con `hpo.buscar` y `correr_pd` reales envueltos en espías: la búsqueda recibe las rutas de sensibilidad y escribe en
# `optuna/`; `correr_pd` del modo optuna recibe `ruta_mlp_final` de `optuna/` y su presupuesto es el de esa búsqueda.
# La logística sale igual en los dos modos, `comparacion.json` tiene `misma_particion` true y el principal real no
# cambia.

# %%
def test_06_optuna_y_reusar_de_punta_a_punta(entorno, tmp_path):
    """Búsqueda en optuna/, presupuesto propio, logística igual en los dos modos y el principal intacto."""
    antes = _huellas_principal()  # huellas del principal real
    cfg = _cfg(entorno, tmp_path)  # config sintética
    llamadas = {"buscar": [], "pd": []}  # registros

    def buscar(c, **k):
        """Espía de hpo.buscar."""
        llamadas["buscar"].append((c, k))  # registro
        return hpo.buscar(c, **k)  # real

    def correr(c, **k):
        """Espía de correr_pd."""
        llamadas["pd"].append((c, k))  # registro
        return ex.correr_pd(c, **k)  # real

    res = sens.correr_sensibilidad(cfg, buscar_fn=buscar, correr_pd_fn=correr)  # corrida completa
    r_opt = sens.rutas_sensibilidad(cfg, "optuna")  # rutas de optuna
    c_hpo, k_hpo = llamadas["buscar"][0]  # única búsqueda
    assert len(llamadas["buscar"]) == 1 and k_hpo["rutas"] == r_opt  # rutas de sensibilidad
    assert Path(c_hpo["hpo"]["salida"]) == r_opt.resultados  # JSON en optuna/
    destino = Path(c_hpo["hpo"]["destino"])  # mlp_final.yaml del modo
    assert destino.parent == r_opt.resultados and destino.is_file()  # en optuna/
    modos_pd = [Path(c["salida"]).name for c, _ in llamadas["pd"]]  # orden de las corridas
    assert modos_pd == ["reusar", "optuna"]  # en el orden de la config
    _, k_opt = llamadas["pd"][1]  # corrida optuna
    assert k_opt["rutas"] == r_opt  # rutas inyectadas
    for clave in ("correr_red", "comprobar"):  # las dos piezas que leen mlp_final.yaml
        assert isinstance(k_opt[clave], functools.partial)  # partial
        assert Path(k_opt[clave].keywords["ruta_mlp_final"]) == destino  # el de optuna/
    assert "correr_red" not in llamadas["pd"][0][1]  # reusar: correr_mlp por defecto
    hpo_json = load_json(r_opt.resultados / hpo.ARCHIVO_HPO)  # búsqueda escrita
    mlp_opt = load_json(r_opt.resultados / ex.ARCHIVO_SUMMARY)["mlp"]  # entrada mlp de optuna
    assert mlp_opt["config_de_busqueda"] is True  # es la de su búsqueda
    assert mlp_opt["n_configuraciones_busqueda"] == N_ENSAYOS == hpo_json["n_ensayos"]  # presupuesto
    assert mlp_opt["tiempo_busqueda_s"] == hpo_json["tiempo_total_s"]  # tiempo de esa búsqueda
    r_reu = sens.rutas_sensibilidad(cfg, "reusar")  # rutas de reusar
    mlp_reu = load_json(r_reu.resultados / ex.ARCHIVO_SUMMARY)["mlp"]  # entrada mlp de reusar
    assert mlp_reu["config_de_busqueda"] is False and mlp_reu["n_configuraciones_busqueda"] is None  # sin búsqueda
    log = [load_json(r.resultados / ex.ARCHIVO_SUMMARY)["logistica"]["metricas"] for r in (r_reu, r_opt)]  # log.
    assert log[0] == log[1]  # misma logística en los dos modos
    pred = [pd.read_parquet(r.resultados / ex.ARCHIVO_PRED_TEST) for r in (r_reu, r_opt)]  # pred_test
    assert pred[0]["pred_logistica"].tolist() == pred[1]["pred_logistica"].tolist()  # idéntica
    for r in (r_reu, r_opt):  # modelos de cada modo
        assert (r.modelos / ex.ARCHIVO_MODELO_LOGISTICA).is_file() and (r.modelos / ex.ARCHIVO_MODELO_MLP).is_file()
    comp = load_json(Path(cfg["sensibilidad"]["salida"]) / sens.ARCHIVO_COMPARACION)  # comparación escrita
    assert comp == res["comparacion"] and comp["misma_particion"] is True  # misma partición que el principal
    assert comp["modos"] == ["reusar", "optuna"]  # los dos modos
    assert _huellas_principal() == antes  # el principal real intacto


# %% [markdown]
# ## 6b. Orden de la orquestación y pre-chequeo de todos los modos
#
# Con dobles: validar → preparar → `comprobar_fn` de cada modo (antes de la búsqueda) → reusar → buscar → optuna →
# comparar. Si el pre-chequeo de optuna falla, no corre ni la búsqueda ni ningún modelo.

# %%
def test_06b_orden_y_prechequeo(entorno, tmp_path):
    """Orden de las llamadas; un pre-chequeo que falla detiene todo antes de entrenar."""
    cfg = _cfg(entorno, tmp_path)  # config
    orden = []  # llamadas

    def preparar(c):
        """Doble de preparar."""
        orden.append("preparar")  # registro
        return {"filas": {}}  # resultado falso

    def comprobar(c, rutas):
        """Doble del pre-chequeo."""
        orden.append(("comprobar", rutas.resultados.name))  # registro

    def buscar(c, **k):
        """Doble de la búsqueda: escribe un mlp_final.yaml válido."""
        orden.append("buscar")  # registro
        busqueda = {"tipo": "optuna", "ensayo": 0, "n_ensayos": 1, "tiempo_s": 1.0,
                    "evidencia": {"sha256_particion": "a", "sha256_preprocesador": "b"}, "fecha": "2026-10-27"}
        Path(c["hpo"]["destino"]).parent.mkdir(parents=True, exist_ok=True)  # carpeta
        Path(c["hpo"]["destino"]).write_text(hpo.texto_mlp_final(MLP_RAPIDO, busqueda), encoding="utf-8")  # YAML

    def correr(c, **k):
        """Doble de correr_pd."""
        orden.append(("pd", Path(c["salida"]).name, c["busqueda"]["tipo"]))  # registro
        return {}  # resultado falso

    def comparar(c, modos):
        """Doble de la comparación."""
        orden.append(("comparar", tuple(modos)))  # registro
        return {}  # resultado falso

    sens.correr_sensibilidad(cfg, preparar_fn=preparar, correr_pd_fn=correr, buscar_fn=buscar,
                             comprobar_fn=comprobar, comparar_fn=comparar)  # orquestación
    assert orden == ["preparar", ("comprobar", "reusar"), ("comprobar", "optuna"), ("pd", "reusar", "sin_busqueda"),
                     "buscar", ("pd", "optuna", "optuna"), ("comparar", ("reusar", "optuna"))]  # orden
    orden.clear()  # de nuevo con un pre-chequeo que falla en optuna

    def comprobar_falla(c, rutas):
        """Pre-chequeo que falla en optuna."""
        orden.append(("comprobar", rutas.resultados.name))  # registro
        if rutas.resultados.name == "optuna":  # salidas previas ajenas
            raise ValueError("pred_test.parquet tiene otros ids")  # como _verificar_salidas

    with pytest.raises(ValueError):  # se propaga
        sens.correr_sensibilidad(cfg, preparar_fn=preparar, correr_pd_fn=correr, buscar_fn=buscar,
                                 comprobar_fn=comprobar_falla, comparar_fn=comparar)  # orquestación
    assert orden == ["preparar", ("comprobar", "reusar"), ("comprobar", "optuna")]  # nada se entrenó


# %% [markdown]
# ## 6c. Solo un modo
#
# Con `modos: [reusar]` no hay búsqueda y la comparación recibe solo ese modo.

# %%
def test_06c_solo_reusar(entorno, tmp_path):
    """modos [reusar]: sin buscar; comparar con ('reusar',)."""
    cfg = _cfg(entorno, tmp_path, modos=["reusar"])  # un modo
    vistos = []  # llamadas

    def no_buscar(c, **k):
        """La búsqueda no debe correr."""
        raise AssertionError("no debía buscar")  # falla si se llama

    sens.correr_sensibilidad(cfg, preparar_fn=lambda c: {"filas": {}}, correr_pd_fn=lambda c, **k: {},
                             buscar_fn=no_buscar, comprobar_fn=lambda c, r: None,
                             comparar_fn=lambda c, m: vistos.append(tuple(m)))  # orquestación
    assert vistos == [("reusar",)]  # solo reusar


# %% [markdown]
# ## 7. `comparacion` (pura)
#
# Insumos a mano: Δ con signo, `calibracion_global` (media de la PD y tasa de `y`), desv 0 con n = 1, solo los modos
# presentes, `misma_particion` false si una huella difiere y error si falta un modelo o una métrica.

# %%
def _entrada(medias: dict, desv=0.0, n=1, huella="h", n_param=10, configs=13, tiempo=1.0) -> dict:
    """Entrada de summary.json con las métricas de test pedidas."""
    metricas = {m: {"media": v, "desv": desv, "mejor": v, "peor": v, "n": n} for m, v in medias.items()}  # resumen
    return {"n_parametros": n_param, "n_configuraciones_busqueda": configs, "tiempo_total_s": tiempo,
            "metricas": metricas, "evidencia": {"sha256_particion": huella, "sha256_preprocesador": "p"}}  # entrada


def _fuente(auc_log, auc_mlp, pd_log, pd_mlp, y, huella="h", n_mlp=3) -> dict:
    """{summary, pred_test} con AUC dados (las otras métricas fijas) y PD de test por fila."""
    base = {"gini_test": 0.3, "ks_test": 0.2, "brier_test": 0.18}  # resto de métricas
    summary = {"logistica": _entrada({"auc_test": auc_log, **base}, huella=huella),  # n = 1
               "mlp": _entrada({"auc_test": auc_mlp, **base}, desv=0.01, n=n_mlp, huella=huella, configs=None)}
    pred = pd.DataFrame({"id": [f"i{k}" for k in range(len(y))], "y": y, "pred_logistica": pd_log,
                         "pred_mlp": pd_mlp})  # pred_test
    return {"summary": summary, "pred_test": pred}  # fuente


def test_07_comparacion_a_mano():
    """Δ con signo, calibración a mano, desv 0 con n = 1, solo modos presentes y misma_particion."""
    principal = _fuente(0.67, 0.672, [0.1, 0.2, 0.3, 0.2], [0.2, 0.2, 0.2, 0.2], [0, 1, 0, 0])  # principal
    reusar = _fuente(0.70, 0.668, [0.3, 0.3, 0.3, 0.3], [0.1, 0.5, 0.1, 0.1], [0, 1, 0, 0])  # reusar
    comp = sens.comparacion(principal, {"reusar": reusar})  # solo un modo
    assert comp["modos"] == ["reusar"]  # solo los corridos
    auc = comp["metricas"]["logistica"]["auc_test"]  # AUC de la logística
    assert auc["principal"]["media"] == 0.67 and auc["reusar"]["delta"] == pytest.approx(0.70 - 0.67)  # Δ > 0
    assert comp["metricas"]["mlp"]["auc_test"]["reusar"]["delta"] == pytest.approx(0.668 - 0.672)  # Δ < 0
    assert auc["reusar"]["desv"] == 0.0 and auc["reusar"]["n"] == 1  # n = 1 → 0
    assert set(comp["metricas"]["mlp"]) == set(METRICAS_TEST)  # las 4 métricas de test
    cal = comp["calibracion_global"]  # calibración
    assert cal["logistica"]["principal"] == pytest.approx({"pd_media_test": 0.2, "tasa_test": 0.25})  # a mano
    assert cal["mlp"]["reusar"] == pytest.approx({"pd_media_test": 0.2, "tasa_test": 0.25})  # a mano
    assert cal["logistica"]["reusar"]["pd_media_test"] == pytest.approx(0.3)  # a mano
    pres = comp["presupuesto"]["mlp"]["reusar"]  # presupuesto
    assert pres == {"n_parametros": 10, "n_configuraciones_busqueda": None, "tiempo_total_s": 1.0}  # núcleo §6
    assert comp["misma_particion"] is True  # misma huella
    otra = _fuente(0.70, 0.668, [0.3] * 4, [0.1] * 4, [0, 1, 0, 0], huella="otra")  # otra partición
    assert sens.comparacion(principal, {"reusar": reusar, "optuna": otra})["misma_particion"] is False  # difiere


# %% [markdown]
# ## 7b. `comparacion`: rechazos
#
# Falta un modelo, una métrica, una columna `pred_*` o un modo desconocido → `ValueError`; las entradas no se mutan.

# %%
@pytest.mark.parametrize("romper", [  # defectos
    lambda f: f["summary"].pop("mlp"),  # falta un modelo
    lambda f: f["summary"]["logistica"]["metricas"].pop("ks_test"),  # falta una métrica
    lambda f: f.update(pred_test=f["pred_test"].drop(columns=["pred_mlp"])),  # falta la PD del MLP
    lambda f: f["summary"]["mlp"].pop("evidencia"),  # falta la evidencia
], ids=["modelo", "metrica", "pred", "evidencia"])
def test_07b_comparacion_rechaza(romper):
    """Cada defecto (en el principal o en un modo) → ValueError."""
    for donde in ("principal", "modo"):  # en las dos fuentes
        principal = _fuente(0.67, 0.672, [0.1] * 4, [0.2] * 4, [0, 1, 0, 0])  # sana
        modo = _fuente(0.70, 0.668, [0.3] * 4, [0.1] * 4, [0, 1, 0, 0])  # sana
        romper(principal if donde == "principal" else modo)  # defecto
        with pytest.raises(ValueError):  # rechazo
            sens.comparacion(principal, {"reusar": modo})  # comparación
    sana = _fuente(0.67, 0.672, [0.1] * 4, [0.2] * 4, [0, 1, 0, 0])  # sana
    with pytest.raises(ValueError):  # modo desconocido
        sens.comparacion(sana, {"pd": sana})  # comparación
    with pytest.raises(ValueError):  # sin modos no hay comparación
        sens.comparacion(sana, {})  # vacía
    copia = copy.deepcopy(sana["summary"])  # para ver que no se muta
    sens.comparacion(sana, {"reusar": sana})  # comparación sana
    assert sana["summary"] == copia  # sin mutar


# %% [markdown]
# ## 7c. `escribir_comparacion` con lectura inyectable
#
# Lee el `summary.json` del principal y el `pred_test.parquet` de su misma carpeta, y los de cada modo, y escribe
# `{salida}/comparacion.json`.

# %%
def test_07c_escribir_comparacion(entorno, tmp_path):
    """Lee las rutas esperadas y escribe comparacion.json en salida."""
    cfg = _cfg(entorno, tmp_path, modos=["reusar"])  # un modo
    principal = Path(cfg["sensibilidad"]["principal"])  # summary del principal
    leidos, escritos = [], {}  # registros
    fuentes = {principal: _fuente(0.67, 0.672, [0.1] * 4, [0.2] * 4, [0, 1, 0, 0])}  # principal
    r = sens.rutas_sensibilidad(cfg, "reusar").resultados  # carpeta del modo
    fuentes[r / ex.ARCHIVO_SUMMARY] = _fuente(0.7, 0.67, [0.3] * 4, [0.1] * 4, [0, 1, 0, 0])  # reusar

    def leer(ruta):
        """Lector falso por ruta."""
        ruta = Path(ruta)  # normaliza
        leidos.append(ruta)  # registro
        clave = ruta if ruta.name == ex.ARCHIVO_SUMMARY else ruta.parent / ex.ARCHIVO_SUMMARY  # fuente
        return fuentes[clave]["summary" if ruta.suffix == ".json" else "pred_test"]  # objeto

    comp = sens.escribir_comparacion(cfg, ["reusar"], leer=leer, escribir=lambda p, o: escritos.update({p: o}))
    assert set(leidos) == {principal, principal.parent / ex.ARCHIVO_PRED_TEST, r / ex.ARCHIVO_SUMMARY,
                           r / ex.ARCHIVO_PRED_TEST}  # cuatro lecturas
    assert escritos == {Path(cfg["sensibilidad"]["salida"]) / sens.ARCHIVO_COMPARACION: comp}  # una escritura


# %% [markdown]
# ## 9. Config real (alarma)
#
# `sensibilidad.yaml` es completa (sin `hereda`); su `mlp` es el de `mlp_final.yaml`, su `logistica` la de `base.yaml`
# y su `hpo` el espacio y los ensayos de `hpo.yaml`; las salidas están en las carpetas de B y los insumos son los de A.

# %%
def test_09_config_real_alarma():
    """Alarma si pd-hpo, base.yaml o hpo.yaml cambian; rutas de B y de A."""
    crudo = yaml.safe_load(RUTA_CFG.read_text(encoding="utf-8"))  # YAML sin resolver
    assert "hereda" not in crudo  # completa
    cfg = load_config(RUTA_CFG)  # como el CLI
    sens.validar_cfg_sensibilidad(cfg)  # válida
    assert cfg["mlp"] == load_config(RUTA_FINAL_REAL)["mlp"]  # misma arquitectura que el principal
    assert cfg["logistica"] == load_config(RUTA_BASE)["logistica"]  # misma rejilla de C
    h = load_config(RUTA_HPO)["hpo"]  # búsqueda del principal
    assert cfg["hpo"] == {"n_ensayos": h["n_ensayos"], "espacio": h["espacio"]}  # mismo presupuesto y espacio
    assert cfg["semillas"]["n"] == 30 and cfg["datos"]["muestra"] is None  # corrida reportada
    assert cfg["busqueda"] == {"tipo": ex.TIPO_SIN_BUSQUEDA}  # reusar
    assert cfg["sensibilidad"]["modos"] == ["reusar", "optuna"]  # los dos modos
    s = cfg["sensibilidad"]  # rutas
    assert Path(s["salida"]) == Path("results/data/pd_sensibilidad")  # carpeta de B
    assert Path(s["modelos"]) == Path("models/pd_sensibilidad")  # carpeta de B
    assert Path(s["union"]).parent == Path(s["salida"])  # unión en salida
    assert Path(s["preprocesador"]).parent == Path(s["modelos"])  # preprocesador en modelos
    assert Path(s["principal"]) == ex.RUTAS_PD.resultados / ex.ARCHIVO_SUMMARY  # summary del principal
    datos_a = load_config(RUTA_DATOS_A)  # config de A
    carpeta_a = Path(datos_a["salida"]["carpeta"])  # data/processed
    assert Path(s["loans_pd"]) == ex.RUTAS_PD.parquet  # loans_pd de A
    assert Path(s["features"]) == Path(datos_a["features"])  # 15 variables de A
    assert Path(s["adicionales"]) == Path(datos_a["sensibilidad"]["features"])  # adicionales de A
    assert Path(s["datos_lc"]) == carpeta_a / datos_a["sensibilidad"]["archivo"]  # sensibilidad_lc de A
    assert Path(s["config_datos"]) == Path("configs/datos.yaml")  # percentiles de A


# %% [markdown]
# ## 10. Contra el esquema real
#
# La unión de los parquets reales tiene 237 695 filas y la misma `sha256_particion` que `results/data/pd/summary.json`
# (la comparación es justa: misma partición). Se omite si faltan los datos.

# %%
def test_10_union_real():
    """Unión real: 237 695 filas, misma huella de partición que el principal y 17 variables válidas."""
    cfg = load_config(RUTA_CFG)  # config real
    s = cfg["sensibilidad"]  # rutas
    faltan = [p for p in (s["loans_pd"], s["datos_lc"], s["principal"]) if not (ROOT / p).is_file()]  # insumos
    if faltan:  # clon sin datos
        pytest.skip(f"faltan {faltan}")  # se omite
    loans = pd.read_parquet(ROOT / s["loans_pd"])  # A
    lc = pd.read_parquet(ROOT / s["datos_lc"])  # A
    adicionales = load_config(ROOT / s["adicionales"])["adicionales"]  # A
    union = sens.unir_sensibilidad(loans, lc, [a["nombre"] for a in adicionales])  # unión
    assert len(union) == 237695  # filas de loans_pd (A)
    huella = ex.huella_particion(union["id"], union["split"])  # partición de la unión
    principal = load_json(ROOT / s["principal"])  # summary del principal
    assert huella == principal["logistica"]["evidencia"]["sha256_particion"]  # misma partición
    assert huella == principal["mlp"]["evidencia"]["sha256_particion"]  # ídem
    feats = sens.features_sensibilidad(load_config(ROOT / s["features"])["features"], adicionales)  # 17
    assert len(feats) == 17  # 15 + 2


# %% [markdown]
# ## 11. CLI: config inválida
#
# `run.py pd-sensibilidad` imprime la config resuelta y, con un modo desconocido, sale con código 2 y **una** línea en
# stderr, sin leer datos.

# %%
def test_11_cli_config_invalida(capsys, monkeypatch):
    """Código 2, una línea con el prefijo y el tipo; la config impresa."""
    monkeypatch.chdir(ROOT)  # rutas relativas
    assert cli.SUBCOMANDOS["pd-sensibilidad"].manejador is cli._pd_sensibilidad  # conectado
    codigo = cli.main(["pd-sensibilidad", "--config", str(RUTA_CFG), "--set", "sensibilidad.modos=[otro]"])  # CLI
    salida = capsys.readouterr()  # consola
    assert codigo == ex.CODIGO_ERROR  # 2
    assert cli.CABECERA_CONFIG in salida.out  # config resuelta impresa
    lineas = salida.err.strip().splitlines()  # stderr
    assert len(lineas) == 1 and lineas[0].startswith(sens.PREFIJO_ERROR + "ValueError: ")  # una línea


# %% [markdown]
# ## 11b. `manejador`: errores esperados y bugs
#
# Los errores esperados salen con código 2; un `TypeError` (bug) se propaga con su traza.

# %%
def test_11b_manejador_errores(monkeypatch, capsys):
    """FileNotFoundError → 2; TypeError → se propaga."""
    def lanzar(exc):
        """correr_sensibilidad falso que lanza ``exc``."""
        def f(cfg, **k):
            """Lanza."""
            raise exc  # error
        return f  # doble

    monkeypatch.setattr(sens, "correr_sensibilidad", lanzar(FileNotFoundError("no existe x.parquet\nlínea 2")))
    assert sens.manejador({}) == ex.CODIGO_ERROR  # esperado
    assert capsys.readouterr().err.count("\n") == 1  # una línea
    monkeypatch.setattr(sens, "correr_sensibilidad", lanzar(TypeError("bug")))  # bug
    with pytest.raises(TypeError):  # con traza
        sens.manejador({})  # se propaga


# %% [markdown]
# ## 13. Lenta: `run.py pd-sensibilidad` de punta a punta con los datos reales
#
# Muestra de 5 000 filas, 2 semillas, 2 ensayos y todas las salidas en `tmp_path`: escribe la unión, el preprocesador,
# los dos modos y `comparacion.json`; nada versionado cambia y `results/data/pd_sensibilidad/` no se crea.

# %%
@pytest.mark.lento
def test_13_cli_de_punta_a_punta(tmp_path, monkeypatch, capsys):
    """CLI real con muestra, en tmp_path; el principal y lo versionado intactos."""
    monkeypatch.chdir(ROOT)  # rutas relativas a la raíz
    cfg = load_config(RUTA_CFG)  # config real
    s = cfg["sensibilidad"]  # rutas
    if not all((ROOT / p).is_file() for p in (s["loans_pd"], s["datos_lc"], s["principal"])):  # clon sin datos
        pytest.skip("faltan los datos reales")  # se omite
    antes = _huellas_principal()  # huellas del principal
    existia = (ROOT / s["salida"]).exists(), (ROOT / s["modelos"]).exists()  # carpetas reales de sensibilidad
    sets = [f"sensibilidad.{k}={(tmp_path / v).as_posix()}" for k, v in (  # rutas temporales
        ("union", "res/union.parquet"), ("preprocesador", "models/pre.json"), ("salida", "res"),
        ("modelos", "models"))]
    sets += ["datos.muestra=5000", "semillas.n=2", "hpo.n_ensayos=2"]  # demo
    argv = ["pd-sensibilidad", "--config", str(RUTA_CFG)] + [x for s_ in sets for x in ("--set", s_)]  # CLI
    assert cli.main(argv) == 0  # éxito
    salida = capsys.readouterr().out  # consola
    assert "Tiempo de ejecución" in salida and "reusar" in salida and "optuna" in salida  # informe
    comp = load_json(tmp_path / "res" / sens.ARCHIVO_COMPARACION)  # comparación
    assert comp["modos"] == ["reusar", "optuna"]  # dos modos
    assert comp["misma_particion"] is False  # la muestra no es la partición completa del principal
    for modo in sens.MODOS:  # cada modo
        summary = load_json(tmp_path / "res" / modo / ex.ARCHIVO_SUMMARY)  # summary del modo
        assert summary["mlp"]["metricas"]["auc_test"]["n"] == 2  # dos semillas
        assert (tmp_path / "models" / modo / ex.ARCHIVO_MODELO_MLP).is_file()  # red guardada
    assert load_json(tmp_path / "res" / "optuna" / hpo.ARCHIVO_HPO)["n_ensayos"] == 2  # dos ensayos
    assert _huellas_principal() == antes  # el principal intacto
    assert ((ROOT / s["salida"]).exists(), (ROOT / s["modelos"]).exists()) == existia  # nada nuevo en el repo


# %% [markdown]
# ## 13b. `--set` cambia la config
#
# Las claves de la config real admiten los `--set` de la demo (claves existentes).

# %%
def test_13b_set_de_la_demo():
    """Los --set de la demo existen en la config real."""
    cfg = apply_overrides(load_config(RUTA_CFG), ["datos.muestra=5000", "semillas.n=2", "hpo.n_ensayos=2",
                                                  "sensibilidad.modos=[reusar]"])  # como el CLI
    assert cfg["datos"]["muestra"] == 5000 and cfg["sensibilidad"]["modos"] == ["reusar"]  # aplicados
    sens.validar_cfg_sensibilidad(cfg)  # válida
