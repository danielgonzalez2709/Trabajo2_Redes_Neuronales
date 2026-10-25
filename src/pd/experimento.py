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
# # Experimento de la PD: datos, métricas y resultados (Parte B, núcleo §5.4 y §6)
#
# Orquesta un modelo de PD de punta a punta: carga la partición y el preprocesador **ya ajustado por A** (sin
# reajustarlo: núcleo §5.2 y §5.7), entrena, mide en validación y prueba con las métricas comunes y escribe los
# resultados del núcleo §5.4 con la evidencia de comparación justa (núcleo §6). B3 conecta la logística; B5 añade
# el MLP reutilizando las mismas piezas (`medir`, `resumir`, `actualizar_summary`, `actualizar_pred_test`).
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable
# sin efectos al importar.

# %%
"""Experimento de la PD: matrices por split, métricas, resúmenes y archivos de resultados (núcleo §5.4)."""

import contextlib  # borrar el temporal si la escritura falla
import hashlib  # SHA-256 de la evidencia de comparación justa
import os  # os.replace para la escritura atómica del parquet
import tempfile  # temporal en la misma carpeta del destino
from dataclasses import dataclass  # rutas inyectables e inmutables
from pathlib import Path  # rutas

import numpy as np  # medias, desviaciones y alineación
import pandas as pd  # parquet de datos y de predicciones

from src.common.metricas import auc, brier, gini, ks  # métricas comunes (B1)
from src.common.preprocesamiento import Preprocesador  # preprocesador común (núcleo §5.2)
from src.common.results import load_json, save_json  # JSON sin NaN y escritura atómica (núcleo §5.4)
from src.common.seeds import validar_entero  # semilla entera ≥ 0
from src.pd.logistica import LogisticaPD, _bloque_logistica, esta_en_borde, logloss  # modelo de referencia

__all__ = [  # interfaz pública acordada (tarea B3); lo demás es interno
    "RutasPD", "RUTAS_PD", "huella_particion", "huella_archivo", "cargar_matrices",  # datos y evidencia
    "METRICAS", "SENTIDO", "evaluar", "medir", "resumir",  # métricas
    "actualizar_summary", "actualizar_pred_test", "escribir_semilla",  # archivos de resultados
    "correr_logistica",  # experimento de la logística
]

COLUMNA_ID = "id"  # identificador del crédito (núcleo §5.1)
COLUMNA_SPLIT = "split"  # partición (núcleo §5.1)
COLUMNA_OBJETIVO = "target"  # objetivo 0/1 (núcleo §5.1)
COLUMNA_Y = "y"  # columna del objetivo en pred_test.parquet (núcleo §5.4)
SPLITS = ("train", "val", "test")  # valores de split (núcleo §5.1)
SPLITS_MEDIDOS = ("val", "test")  # val elige, test se mide una vez
CODIFICACION = "utf-8"  # texto de la huella de la partición
CARPETA_SEMILLAS = "semillas"  # results/data/pd/semillas/ (núcleo §5.4)
ARCHIVO_SUMMARY = "summary.json"  # resumen por modelo (núcleo §5.4)
ARCHIVO_PRED_TEST = "pred_test.parquet"  # predicciones de test (núcleo §5.4)
MODELO_LOGISTICA = "logistica"  # nombre del modelo en los resultados
ARCHIVO_BUSQUEDA_LOGISTICA = "busqueda_logistica.json"  # tabla de la búsqueda de C
ARCHIVO_MODELO_LOGISTICA = "pd_logistica.joblib"  # modelo guardado (núcleo §5.4)
CRITERIO_LOGISTICA = "logloss_val"  # criterio de la búsqueda de C
DEVICE_LOGISTICA = "cpu"  # sklearn entrena en CPU
PREFIJO_PRED = "pred_"  # columna de cada modelo en pred_test.parquet
PERMISOS_ARCHIVO = 0o644  # rw-r--r--: los mismos que deja save_json en los JSON (src/common/results.py)


# %% [markdown]
# ## Rutas
#
# Las rutas se reciben como un objeto (inversión de dependencias): las pruebas apuntan a una carpeta temporal y el
# proyecto usa `RUTAS_PD` (núcleo §5.1 y §5.4).

# %%
@dataclass(frozen=True)
class RutasPD:
    """Rutas de entrada (parquet y preprocesador) y de salida (resultados y modelos) del experimento de PD."""

    parquet: Path  # data/processed/loans_pd.parquet
    preprocesador: Path  # models/preprocesador_pd.json
    resultados: Path  # results/data/pd
    modelos: Path  # models


RUTAS_PD = RutasPD(  # rutas del proyecto
    Path("data/processed/loans_pd.parquet"),  # datos de A (núcleo §5.1)
    Path("models/preprocesador_pd.json"),  # preprocesador de A (núcleo §5.2)
    Path("results/data/pd"),  # resultados de la PD (núcleo §5.4)
    Path("models"),  # modelos guardados (núcleo §5.4)
)


# %% [markdown]
# ## Huella de la partición
#
# $$h_{\text{part}} = \mathrm{SHA256}\Big(\mathrm{UTF8}\big(\textstyle\bigoplus_{i} \texttt{f"\{id\_i\},\{split\_i\}\textbackslash n"}\big)\Big)$$
#
# en hexadecimal, con las filas en el orden del parquet ($\oplus$ = concatenación). Dos modelos con la misma huella
# usaron exactamente la misma partición (núcleo §6, "mismos datos").

# %%
def huella_particion(ids, splits) -> str:
    """SHA-256 (hex) del texto UTF-8 de las líneas f"{id},{split}\\n" en orden."""
    texto = "".join(f"{i},{s}\n" for i, s in zip(ids, splits, strict=True))  # una línea por crédito
    return hashlib.sha256(texto.encode(CODIFICACION)).hexdigest()  # 64 dígitos hexadecimales


# %% [markdown]
# ## Huella de un archivo
#
# $h_{\text{pre}} = \mathrm{SHA256}(\text{bytes del archivo})$ en hexadecimal: la del JSON del preprocesador
# demuestra que todos los modelos usaron el mismo preprocesamiento (núcleo §6).

# %%
def huella_archivo(ruta) -> str:
    """SHA-256 (hex) de los bytes del archivo ``ruta``."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()  # 64 dígitos hexadecimales


# %% [markdown]
# ## Evidencia de comparación justa
#
# `{sha256_particion, sha256_preprocesador}`: va en la semilla, en la búsqueda y en la entrada del summary. La huella
# de la partición se calcula sobre el **mismo** DataFrame del que salen las matrices (sin releer el parquet), así
# que describe exactamente los datos con que se entrenó.

# %%
def _evidencia(df: pd.DataFrame, ruta_preprocesador) -> dict:
    """Huellas de la partición de ``df`` y del archivo del preprocesador."""
    return {  # evidencia del núcleo §6
        "sha256_particion": huella_particion(df[COLUMNA_ID], df[COLUMNA_SPLIT]),  # orden del parquet
        "sha256_preprocesador": huella_archivo(ruta_preprocesador),  # bytes del JSON
    }


# %% [markdown]
# ## Matrices por split
#
# $X_s = \text{pre.transform}(\text{filas de } s)$ con el preprocesador **guardado** (`from_dict`, sin `fit`): fue
# ajustado solo con train por A. $y_s$ = `target` y los ids en el orden del parquet.
#
# `_cargar_datos` lee el parquet **una vez** y devuelve el DataFrame junto con las matrices: el experimento usa el
# DataFrame para la huella de la partición. `cargar_matrices` conserva su firma pública y solo devuelve las matrices.

# %%
def _matrices(df: pd.DataFrame, pre: Preprocesador) -> dict:
    """{split: (X float32, y, ids)} de ``df`` transformado con ``pre`` (sin fit)."""
    matrices = {}  # resultado por split
    for split in SPLITS:  # train, val, test
        parte = df[df[COLUMNA_SPLIT] == split]  # filas del split, en el orden del parquet
        matrices[split] = (  # tripleta del split
            pre.transform(parte),  # X float32 (núcleo §5.2)
            parte[COLUMNA_OBJETIVO].to_numpy(dtype=np.int64),  # y 0/1
            parte[COLUMNA_ID].to_numpy(dtype=object),  # ids tal cual (str en el parquet de A)
        )
    return matrices  # tres splits


def _cargar_datos(rutas: RutasPD) -> tuple[pd.DataFrame, dict]:
    """(DataFrame del parquet, matrices por split) con una sola lectura del parquet."""
    df = pd.read_parquet(rutas.parquet)  # datos de A
    pre = Preprocesador.from_dict(load_json(rutas.preprocesador))  # sin reajustar
    return df, _matrices(df, pre)  # el mismo df sirve para la huella


def cargar_matrices(rutas: RutasPD) -> dict:
    """{split: (X float32, y, ids)} para train, val y test con el preprocesador guardado."""
    return _cargar_datos(rutas)[1]  # solo las matrices


# %% [markdown]
# ## Métricas
#
# AUC, Gini $= 2\,\mathrm{AUC} - 1$, KS $= \max_t |F_1(t) - F_0(t)|$ y Brier $= \frac1n\sum(p_i - y_i)^2$ de
# `src/common/metricas.py` (sus definiciones están allí, Parte B §3.4), más la log-loss. Sentido: mayor es mejor en
# AUC, Gini y KS; menor es mejor en Brier y log-loss.

# %%
METRICAS = {"auc": auc, "gini": gini, "ks": ks, "brier": brier, "logloss": logloss}  # nombre → función (y, p)
SENTIDO = {"auc": "max", "gini": "max", "ks": "max", "brier": "min", "logloss": "min"}  # nombre → mejor


def evaluar(y, p, metricas=METRICAS) -> dict:
    """{nombre: valor} de cada métrica de ``metricas`` sobre (y, p)."""
    return {nombre: float(f(y, p)) for nombre, f in metricas.items()}  # float nativo (va al JSON)


# %% [markdown]
# ## Métricas por split
#
# `{f"{métrica}_{split}"}` con las PD del modelo en cada split medido (por defecto val y test) y las métricas de
# `metricas` (por defecto `METRICAS`; B5 puede inyectar otras).

# %%
def medir(modelo, datos: dict, splits=SPLITS_MEDIDOS, metricas=METRICAS) -> dict:
    """{f"{m}_{split}": valor} con ``modelo.predict_proba`` en cada split."""
    resultado = {}  # métricas con nombre
    for split in splits:  # val y test
        X, y, _ = datos[split]  # matrices del split
        for nombre, valor in evaluar(y, modelo.predict_proba(X), metricas).items():  # métricas del split
            resultado[f"{nombre}_{split}"] = valor  # p. ej. auc_test
    return resultado  # 5 métricas × splits


# %% [markdown]
# ## Resumen entre corridas
#
# Para cada métrica con valores $x_1, \dots, x_n$ (una por semilla):
#
# $$\bar x = \frac1n\sum_i x_i, \qquad s = \sqrt{\frac{\sum_i (x_i - \bar x)^2}{n - 1}} \quad (\text{desviación
# muestral}, \texttt{ddof} = 1).$$
#
# Con $n = 1$, $s$ no está definida y se escribe $0$ por **convención declarada** (la logística corre una vez).
# `mejor` y `peor` siguen el sentido de la métrica (`SENTIDO`, buscado por el nombre sin el sufijo `_{split}`).

# %%
def _desviacion_muestral(x: np.ndarray) -> float:
    """s con ddof = 1; 0 por convención si n = 1."""
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0  # n = 1: no definida → 0


def _sentido_de(nombre: str, sentido: dict) -> str:
    """Sentido de ``nombre`` buscando su base sin el sufijo _{split} (p. ej. auc_test → auc)."""
    base = nombre.rsplit("_", 1)[0]  # quita el sufijo del split
    if base not in sentido:  # métrica sin sentido declarado
        raise ValueError(f"La métrica {nombre!r} no tiene sentido declarado (base {base!r})")  # error
    return sentido[base]  # "max" o "min"


def resumir(corridas: list[dict], sentido=SENTIDO) -> dict:
    """{nombre: {media, desv, mejor, peor, n}} de las métricas de ``corridas``."""
    if not corridas:  # sin corridas no hay resumen
        raise ValueError("resumir necesita al menos una corrida")  # vacío
    resumen = {}  # resultado por métrica
    for nombre in corridas[0]:  # mismas métricas en todas las corridas
        x = np.array([c[nombre] for c in corridas], dtype=np.float64)  # un valor por corrida
        mayor_mejor = _sentido_de(nombre, sentido) == "max"  # auc, gini, ks
        resumen[nombre] = {  # resumen de la métrica
            "media": float(np.mean(x)),  # x̄
            "desv": _desviacion_muestral(x),  # s (ddof = 1)
            "mejor": float(x.max() if mayor_mejor else x.min()),  # según el sentido
            "peor": float(x.min() if mayor_mejor else x.max()),  # el contrario
            "n": int(x.size),  # número de corridas
        }
    return resumen  # una entrada por métrica


# %% [markdown]
# ## `summary.json`
#
# Lee el archivo si existe y reemplaza **solo** la entrada del modelo: las de otros modelos (p. ej. `mlp`, B5) se
# conservan. Una raíz que no sea diccionario es un archivo ajeno → `ValueError` sin tocarlo.

# %%
def actualizar_summary(ruta, modelo: str, entrada: dict) -> None:
    """Reemplaza la entrada ``modelo`` de summary.json conservando las demás."""
    ruta = Path(ruta)  # acepta texto o Path
    summary = load_json(ruta) if ruta.exists() else {}  # vacío si es la primera vez
    if not isinstance(summary, dict):  # raíz inesperada
        raise ValueError(f"{ruta} debe tener un diccionario en la raíz; tiene {type(summary).__name__}")  # error
    summary[modelo] = entrada  # solo esta entrada
    save_json(ruta, summary)  # escritura atómica, sin NaN


# %% [markdown]
# ## Escritura atómica de un parquet
#
# Se escribe en un temporal de la misma carpeta y se mueve con `os.replace`: el destino nunca queda a medias.
# `mkstemp` crea el temporal con permisos 0o600 y `os.replace` los conserva, así que al final se fijan en 0o644
# (`PERMISOS_ARCHIVO`), como hace `save_json` con los JSON: el resultado debe quedar legible (en Windows es inocuo).

# %%
def _escribir_parquet_atomico(ruta: Path, df: pd.DataFrame) -> None:
    """Escribe ``df`` en ``ruta`` vía un temporal de la misma carpeta y ``os.replace``."""
    ruta.parent.mkdir(parents=True, exist_ok=True)  # crea la carpeta si falta
    fd, temporal = tempfile.mkstemp(prefix=f".{ruta.name}.", suffix=".tmp", dir=ruta.parent)  # misma carpeta
    os.close(fd)  # pandas abre el archivo por su cuenta
    try:  # si algo falla, se borra el temporal
        df.to_parquet(temporal, index=False)  # escribe el temporal
        os.replace(temporal, ruta)  # movimiento atómico
        os.chmod(ruta, PERMISOS_ARCHIVO)  # el temporal nació en 0o600: se deja legible como los JSON
    except BaseException:  # cualquier falla, incluida una interrupción
        with contextlib.suppress(FileNotFoundError):  # el temporal pudo no existir
            os.remove(temporal)  # limpia
        raise  # propaga la falla


# %% [markdown]
# ## `pred_test.parquet`
#
# Agrega (o reemplaza) la columna del modelo **alineando por id**. Si el archivo ya existe debe tener el mismo
# conjunto de ids y la misma $y$ para cada id; si no → `ValueError` sin tocarlo. Las columnas de otros modelos se
# conservan.

# %%
def _tabla_nueva(ids, y, columna: str, pred) -> pd.DataFrame:
    """DataFrame (id, y, columna) validado: mismas longitudes e ids únicos."""
    tabla = pd.DataFrame({COLUMNA_ID: list(ids), COLUMNA_Y: np.asarray(y),  # id e y
                          columna: np.asarray(pred, dtype=np.float64)})  # predicción del modelo
    if tabla[COLUMNA_ID].duplicated().any():  # un id repetido haría ambigua la alineación
        raise ValueError("pred_test: hay ids repetidos")  # error
    return tabla  # tabla del modelo


def _combinar(previo: pd.DataFrame, nuevo: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Copia de ``previo`` con ``columna`` de ``nuevo`` alineada por id; ValueError si ids o y no coinciden."""
    if len(previo) != len(nuevo) or set(previo[COLUMNA_ID]) != set(nuevo[COLUMNA_ID]):  # otro conjunto de ids
        raise ValueError("pred_test: el archivo existente tiene otro conjunto de ids")  # error
    alineado = nuevo.set_index(COLUMNA_ID).loc[previo[COLUMNA_ID].to_numpy()]  # orden del archivo existente
    if not np.array_equal(alineado[COLUMNA_Y].to_numpy(), previo[COLUMNA_Y].to_numpy()):  # misma y por id
        raise ValueError("pred_test: la y de algún id no coincide con la del archivo existente")  # error
    combinado = previo.copy()  # conserva las columnas ajenas
    combinado[columna] = alineado[columna].to_numpy()  # agrega o reemplaza la del modelo
    return combinado  # tabla final


def actualizar_pred_test(ruta, ids, y, columna: str, pred) -> None:
    """Agrega o reemplaza ``columna`` en pred_test.parquet alineando por id (escritura atómica)."""
    ruta = Path(ruta)  # acepta texto o Path
    nuevo = _tabla_nueva(ids, y, columna, pred)  # tabla del modelo
    tabla = _combinar(pd.read_parquet(ruta), nuevo, columna) if ruta.exists() else nuevo  # combina si existe
    _escribir_parquet_atomico(ruta, tabla)  # nunca queda a medias


# %% [markdown]
# ## Archivo de una semilla
#
# `{resultados}/semillas/{modelo}_seed{seed}.json` (núcleo §5.4). La semilla es un entero $\ge 0$ (regla única de
# `src/common/seeds.py`); `_semilla_valida` convierte su `TypeError` en `ValueError` para que el experimento falle
# siempre con el mismo tipo de error ante una semilla inválida.

# %%
def _semilla_valida(seed) -> int:
    """``seeds.validar_entero(seed, minimo=0)`` con su TypeError convertido en ValueError."""
    try:  # regla única de enteros del proyecto
        return validar_entero(seed, "seed", minimo=0)  # ValueError si es < 0
    except TypeError as exc:  # texto, flotante, bool…
        raise ValueError(str(exc)) from exc  # mismo mensaje, ValueError


def escribir_semilla(rutas: RutasPD, modelo: str, seed: int, registro: dict) -> Path:
    """Escribe el registro de una corrida y devuelve su ruta."""
    seed = _semilla_valida(seed)  # entero ≥ 0 (ValueError si no)
    ruta = rutas.resultados / CARPETA_SEMILLAS / f"{modelo}_seed{seed}.json"  # nombre del núcleo §5.4
    return save_json(ruta, registro)  # escritura atómica, sin NaN


# %% [markdown]
# ## Registros de la logística
#
# - **semilla:** `config` = bloque validado + `C_elegido`; `tiempo_s` = ajuste del $C$ elegido; `epocas` =
#   **iteraciones de lbfgs** del $C$ elegido (la logística no tiene épocas de minilotes; es el análogo más cercano);
#   `device` = `cpu`.
# - **búsqueda:** criterio, filas, $C$ elegido, si está en el borde y `tiempo_total_s` (todo el `fit`).
# - **summary:** `n_configuraciones_busqueda` = tamaño de la rejilla y `tiempo_total_s` = todo el `fit`
#   (presupuesto de la familia, núcleo §6); métricas resumidas de una sola corrida (desv = 0).

# %%
def _fila_elegida(filas: list[dict], C: float) -> dict:
    """Fila de la búsqueda cuyo C es el elegido."""
    return next(f for f in filas if f["C"] == C)  # la rejilla no tiene duplicados


def _semilla_logistica(modelo, cfg_log: dict, metricas: dict, seed: int, evidencia: dict) -> dict:
    """Registro de la corrida de la logística (núcleo §5.4 + evidencia)."""
    fila = _fila_elegida(modelo.busqueda, modelo.C_elegido)  # ajuste del C elegido
    return {  # claves del núcleo §5.4
        "modelo": MODELO_LOGISTICA,  # nombre
        "seed": seed,  # semilla (lbfgs la ignora)
        "config": {**cfg_log, "C_elegido": modelo.C_elegido},  # bloque validado + C
        "metricas": metricas,  # {m}_{split}
        "n_parametros": modelo.n_parametros(),  # d + 1
        "tiempo_s": fila["tiempo_s"],  # ajuste del C elegido
        "device": DEVICE_LOGISTICA,  # CPU
        "epocas": fila["n_iter"],  # iteraciones de lbfgs
        "evidencia": evidencia,  # huellas
    }


def _busqueda_logistica(modelo, cfg_log: dict, evidencia: dict) -> dict:
    """Contenido de busqueda_logistica.json."""
    return {  # tabla de la búsqueda
        "criterio": CRITERIO_LOGISTICA,  # log-loss de val
        "filas": modelo.busqueda,  # una por C
        "C_elegido": modelo.C_elegido,  # C elegido
        "en_borde": esta_en_borde(cfg_log["rejilla_C"], modelo.C_elegido),  # ¿ampliar la rejilla?
        "tiempo_total_s": modelo.tiempo_total_s,  # todo el fit
        "evidencia": evidencia,  # huellas
    }


def _summary_logistica(modelo, cfg_log: dict, metricas: dict, evidencia: dict) -> dict:
    """Entrada "logistica" de summary.json."""
    return {  # presupuesto y métricas
        "n_parametros": modelo.n_parametros(),  # complejidad (núcleo §6)
        "n_configuraciones_busqueda": len(cfg_log["rejilla_C"]),  # presupuesto de búsqueda
        "tiempo_total_s": modelo.tiempo_total_s,  # presupuesto de tiempo
        "metricas": resumir([metricas]),  # una corrida
        "evidencia": evidencia,  # huellas
    }


# %% [markdown]
# ## Escrituras comunes a todos los modelos de PD
#
# Semilla (`semillas/{modelo}_seed{seed}.json`), entrada del summary (conservando las ajenas) y columna
# `pred_{modelo}` de `pred_test.parquet` (alineada por id). B5 la reutiliza para el MLP; los archivos propios de la
# logística (`busqueda_logistica.json` y el joblib) se escriben aparte.

# %%
def _escribir_resultados(rutas: RutasPD, modelo: str, seed: int, registro: dict, entrada: dict, ids_te, y_te,
                         p_te) -> None:
    """Escribe la semilla, la entrada del summary y la columna de pred_test del modelo."""
    escribir_semilla(rutas, modelo, seed, registro)  # semillas/{modelo}_seed{seed}.json
    actualizar_summary(rutas.resultados / ARCHIVO_SUMMARY, modelo, entrada)  # conserva las entradas ajenas
    actualizar_pred_test(rutas.resultados / ARCHIVO_PRED_TEST, ids_te, y_te, PREFIJO_PRED + modelo, p_te)  # por id


# %% [markdown]
# ## Experimento de la logística
#
# Valida la semilla y la config **antes** de leer datos o entrenar; carga las matrices, entrena `LogisticaPD`
# (busca $C$ con val), mide val y test una vez y escribe los 5 archivos: semilla, búsqueda, summary,
# `pred_test.parquet` y `pd_logistica.joblib`. Devuelve la entrada del summary.

# %%
def correr_logistica(cfg: dict, rutas: RutasPD = RUTAS_PD, modelo_cls=LogisticaPD, seed: int = 0) -> dict:
    """Entrena y evalúa la logística, escribe sus resultados y devuelve su entrada de summary.json."""
    seed = _semilla_valida(seed)  # primero: una semilla inválida no debe costar un entrenamiento
    cfg_log = _bloque_logistica(cfg)  # bloque validado (va a la semilla); ValueError si falta
    df, datos = _cargar_datos(rutas)  # una sola lectura del parquet
    evidencia = _evidencia(df, rutas.preprocesador)  # huellas del mismo df y del preprocesador
    X_tr, y_tr, _ = datos["train"]  # entrenamiento
    X_val, y_val, _ = datos["val"]  # validación (elige C)
    X_te, y_te, ids_te = datos["test"]  # prueba (se mide una vez)
    modelo = modelo_cls().fit(X_tr, y_tr, X_val, y_val, cfg, seed)  # búsqueda + modelo final
    metricas = medir(modelo, datos)  # val y test
    registro = _semilla_logistica(modelo, cfg_log, metricas, seed, evidencia)  # archivo de la semilla
    entrada = _summary_logistica(modelo, cfg_log, metricas, evidencia)  # entrada del summary
    _escribir_resultados(rutas, MODELO_LOGISTICA, seed, registro, entrada, ids_te, y_te,
                         modelo.predict_proba(X_te))  # escrituras comunes (B5 las reutiliza)
    save_json(rutas.resultados / ARCHIVO_BUSQUEDA_LOGISTICA, _busqueda_logistica(modelo, cfg_log, evidencia))  # tabla
    rutas.modelos.mkdir(parents=True, exist_ok=True)  # carpeta de modelos
    modelo.guardar(rutas.modelos / ARCHIVO_MODELO_LOGISTICA)  # joblib
    return entrada  # entrada "logistica" del summary
