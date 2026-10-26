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
import copy  # copias de los bloques validados (B5)
import functools  # lectores por defecto con la muestra de la config (B6)
import hashlib  # SHA-256 de la evidencia de comparación justa
import os  # os.replace para la escritura atómica del parquet
import re  # nombre de los archivos de semilla del MLP (B5)
import sys  # stderr de los errores de run.py pd (B6)
import tempfile  # temporal en la misma carpeta del destino
import time  # reloj por defecto del fit y del subcomando (B5)
from dataclasses import dataclass  # rutas inyectables e inmutables
from pathlib import Path  # rutas

import numpy as np  # medias, desviaciones y alineación
import pandas as pd  # parquet de datos y de predicciones
import torch  # versión e hilos registrados en los JSON del MLP (B5)

from src.common.config import load_config  # relee mlp_final.yaml (B5)
from src.common.metricas import auc, brier, gini, ks  # métricas comunes (B1)
from src.common.preprocesamiento import Preprocesador  # preprocesador común (núcleo §5.2)
from src.common.results import load_json, save_json  # JSON sin NaN y escritura atómica (núcleo §5.4)
from src.common.seeds import make_rng, seeds_for_runs, validar_entero  # muestreo, semillas y enteros de la config
from src.common.torch_utils import resolver_device  # device efectivo que se registra (B5)
from src.pd.logistica import LogisticaPD, bloque_logistica, esta_en_borde, logloss  # modelo de referencia
from src.pd.mlp import CLAVES_MLP, MLPPD, validar_cfg_mlp  # MLP de PD y reglas del bloque mlp (B2)

__all__ = [  # interfaz pública acordada (tareas B3 y B4); lo demás es interno
    "RutasPD", "RUTAS_PD", "huella_particion", "huella_archivo", "cargar_matrices",  # datos y evidencia
    "cargar_datos", "evidencia",  # públicas desde B4: la búsqueda Optuna (src/pd/hpo.py) las reutiliza
    "METRICAS", "SENTIDO", "evaluar", "medir", "resumir",  # métricas
    "actualizar_summary", "actualizar_pred_test", "escribir_semilla",  # archivos de resultados
    "correr_logistica",  # experimento de la logística
    "MODELO_MLP", "ARCHIVO_MODELO_MLP", "RUTA_MLP_FINAL", "TIPO_OPTUNA", "CLAVES_BUSQUEDA",  # B5: nombres
    "CRITERIO_SEMILLA_MEDIANA", "AVISO_SUMMARY_VIEJO",  # B5: textos de las reglas
    "validar_cfg_mlp_final", "claves_distintas", "leer_mlp_final", "procedencia", "presupuesto_busqueda",  # B5
    "borrar_semillas_mlp", "entrenar_semilla", "evidencia_rapida", "calibracion_global",  # B5: piezas
    "registro_semilla_mlp", "semilla_mediana", "entrada_summary_mlp", "correr_mlp",  # B5: MLP
    "rutas_de", "leer_test_rapido", "comprobar_antes", "correr_pd", "manejador",  # B5: run.py pd
    "TIPO_SIN_BUSQUEDA",  # B5 ronda 2: config fijada a mano
    "validar_datos", "muestra_de", "muestrear", "filas_muestra",  # B6: datos.muestra
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
CLAVES_DATOS = ("muestra",)  # claves exactas del bloque datos (B6, núcleo §4.1)
MINIMO_MUESTRA = len(SPLITS)  # datos.muestra ≥ 3: al menos una fila por split (como src/lgd/experimento.py)
RUN_ID_MUESTRA = 0  # run_id del generador del muestreo: make_rng(base_seed, 0) (como src/lgd/experimento.py)
CODIGO_ERROR = 2  # código de salida de run.py pd ante un error esperado (el mismo de argparse en cli.main)
PREFIJO_ERROR = "run.py pd: error: "  # inicio de la línea de error en stderr (como argparse)
ERRORES_ESPERADOS = (ValueError, RuntimeError, FloatingPointError, FileNotFoundError)  # config, datos, fit, archivo
ERRORES_DE_PROGRAMACION = (NotImplementedError, RecursionError)  # subclases de RuntimeError que son bugs: con traza


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
# que describe exactamente los datos con que se entrenó. Es pública desde B4: la búsqueda Optuna la reutiliza.

# %%
def evidencia(df: pd.DataFrame, ruta_preprocesador) -> dict:
    """Huellas de la partición de ``df`` y del archivo del preprocesador."""
    return {  # evidencia del núcleo §6
        "sha256_particion": huella_particion(df[COLUMNA_ID], df[COLUMNA_SPLIT]),  # orden del parquet
        "sha256_preprocesador": huella_archivo(ruta_preprocesador),  # bytes del JSON
    }


# %% [markdown]
# ## B6: validación de `datos.muestra`
#
# `datos` (núcleo §4.1) es un mapeo con **exactamente** la clave `muestra`: null (todos los datos, la corrida reportada)
# o un entero $\ge 3$ (una fila por split como mínimo). bool, flotantes (también `20000.0`), texto, 0, 2 o negativos →
# `ValueError`. Sin bloque `datos` vale como null (decisión de Andrés, 26-oct: las configs de B3 y B5 no cambian). Se
# valida **antes** de leer datos.
#
# TODO(refactor-post-B5): en el refactor posterior a B5, `pd-hpo` debe rechazar `datos.muestra` (hoy la ignora y busca
# con todos los datos).

# %%
def validar_datos(cfg: dict) -> int | None:
    """datos.muestra validado: sin bloque datos o null → None, entero ≥ 3 → int; otra cosa → ValueError."""
    if not isinstance(cfg, dict):  # p. ej. run.py pd sin config
        raise ValueError(f"La config de run.py pd debe ser un mapeo; se recibió {type(cfg).__name__}")  # tipo
    if "datos" not in cfg:  # configs de B3/B5 sin el bloque
        return None  # todos los datos
    datos = cfg["datos"]  # bloque datos
    if not isinstance(datos, dict) or set(datos) != set(CLAVES_DATOS):  # p. ej. datos: 5 o una clave de más
        raise ValueError(f"datos debe ser un mapeo con exactamente la clave muestra; se recibió {datos!r}")  # forma
    if datos["muestra"] is None:  # null
        return None  # todos los datos
    return _entero_cfg(datos["muestra"], "datos.muestra", minimo=MINIMO_MUESTRA)  # entero ≥ 3


# %% [markdown]
# ## B6: argumentos del muestreo
#
# `{muestra, base_seed}` para `muestrear` (y para los lectores que la reciben). La semilla del sorteo es `base_seed` de
# la config, y solo se lee y valida (entero $\ge 0$) si hay muestra: sin muestra no hay sorteo y queda en `None`, así
# que la logística de B3 sigue corriendo con configs sin `base_seed`.

# %%
def muestra_de(cfg: dict) -> dict:
    """{"muestra", "base_seed"} para muestrear; base_seed se valida solo si hay muestra (si no, None)."""
    muestra = validar_datos(cfg)  # null o entero ≥ 3
    if muestra is None:  # sin muestra no hay sorteo
        return {"muestra": None, "base_seed": None}  # base_seed no se usa
    if "base_seed" not in cfg:  # la semilla del sorteo es obligatoria con muestra
        raise ValueError("datos.muestra necesita base_seed (semilla del muestreo) en la config")  # clave
    return {"muestra": muestra, "base_seed": _entero_cfg(cfg["base_seed"], "base_seed", minimo=0)}  # validados


# %% [markdown]
# ## B6: reparto de la muestra entre splits
#
# Con $N_s$ filas en el split $s$, $N = \sum_s N_s$ y una muestra de $n$ filas, la cuota de cada split es
#
# $$\text{cuota}_s = \frac{n\,N_s}{N};$$
#
# cada split recibe $\lfloor \text{cuota}_s \rfloor$ y las $n - \sum_s \lfloor \text{cuota}_s \rfloor$ filas que faltan van a
# los splits con los **restos mayores** $\text{cuota}_s - \lfloor \text{cuota}_s \rfloor$ (empate → orden de `SPLITS`).
# Así $\sum_s n_s = n$ y las proporciones de la partición se conservan.
#
# Copia de `_repartir` de `src/lgd/experimento.py` (Parte D), sin importarla para no acoplar B a D.
# TODO(refactor-post-B5): unificar con la de la LGD en un módulo común.

# %%
def _repartir(n_total: int, tamanos: dict[str, int]) -> dict[str, int]:
    """Reparto proporcional de n_total entre splits por el mayor resto (empate → orden de SPLITS)."""
    total = sum(tamanos.values())  # N
    cuotas = {s: n_total * tamanos[s] / total for s in tamanos}  # n·N_s/N
    n = {s: int(np.floor(c)) for s, c in cuotas.items()}  # ⌊cuota_s⌋
    faltan = n_total - sum(n.values())  # filas por repartir
    for s in sorted(tamanos, key=lambda t: (-(cuotas[t] - n[t]), SPLITS.index(t)))[:faltan]:  # restos mayores
        n[s] += 1  # una fila más
    return n  # n_s por split


# %% [markdown]
# ## B6: muestra estratificada por split
#
# Sin muestra, o con $n \ge$ filas de `df`, devuelve **el mismo** `df` (sin copia ni `reset_index`: todo queda como en
# B5). Si no, reparte $n$ con `_repartir`, sortea sin reemplazo dentro de cada split con `make_rng(base_seed, 0)` y
# devuelve las filas en el **orden del parquet** (posiciones ordenadas) con el índice $0, \dots, n-1$. Solo depende de
# la columna `split` y del orden de las filas: el DataFrame completo y una lectura de `id, split` dan los mismos ids. Un
# split que queda con 0 filas → `ValueError` que lo nombra; un valor de `split` fuera de `SPLITS` → `ValueError` que lo
# nombra, antes del reparto. Con muestra, `base_seed` es obligatorio (None → `ValueError`).
#
# Copia de `_muestrear` de `src/lgd/experimento.py` (Parte D). TODO(refactor-post-B5): unificar con la de la LGD.

# %%
def muestrear(df: pd.DataFrame, muestra: int | None, base_seed: int | None) -> pd.DataFrame:
    """Filas de df sin reemplazo por split con make_rng(base_seed, 0), en el orden del parquet; null o ≥ len → df."""
    if muestra is None:  # todos los datos
        return df  # el mismo objeto
    muestra = _entero_cfg(muestra, "datos.muestra", minimo=MINIMO_MUESTRA)  # entero ≥ 3
    if muestra >= len(df):  # nada que recortar
        return df  # el mismo objeto
    base_seed = _entero_cfg(base_seed, "base_seed", minimo=0)  # semilla del sorteo (None → ValueError)
    splits = df[COLUMNA_SPLIT].to_numpy()  # split de cada fila, en orden
    ajenos = sorted(set(splits) - set(SPLITS))  # valores de split desconocidos
    if ajenos:  # el reparto solo conoce train, val y test
        raise ValueError(f"datos.muestra: la columna split tiene valores fuera de {list(SPLITS)}: {ajenos}")  # nombra
    n = _repartir(muestra, {s: int((splits == s).sum()) for s in SPLITS})  # filas por split
    vacios = [s for s in SPLITS if n[s] == 0]  # splits sin filas
    if vacios:  # una métrica de ese split no existiría
        raise ValueError(f"datos.muestra={muestra} deja vacío el split {vacios} (reparto {n}); usa una muestra "
                         "mayor")  # nombra el split
    rng = make_rng(base_seed, RUN_ID_MUESTRA)  # generador del sorteo
    elegidos = [rng.choice(np.flatnonzero(splits == s), size=n[s], replace=False) for s in SPLITS]  # sin reemplazo
    return df.iloc[np.sort(np.concatenate(elegidos))].reset_index(drop=True)  # orden del parquet


# %% [markdown]
# ## Matrices por split
#
# $X_s = \text{pre.transform}(\text{filas de } s)$ con el preprocesador **guardado** (`from_dict`, sin `fit`): fue
# ajustado solo con train por A. $y_s$ = `target` y los ids en el orden del parquet.
#
# `cargar_datos` (pública desde B4, la reutiliza la búsqueda Optuna) lee el parquet **una vez** y devuelve el
# DataFrame junto con las matrices: el experimento usa el DataFrame para la huella de la partición.
# `cargar_matrices` conserva su firma pública y solo devuelve las matrices.
#
# B6: con `muestra`, el DataFrame se recorta con `muestrear` **antes** de transformar (y por tanto antes de la huella):
# las matrices y la evidencia describen las mismas filas. El preprocesador no se reajusta nunca. Sin muestra devuelve el
# DataFrame leído tal cual (lo de B3–B5).

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


def cargar_datos(
    rutas: RutasPD,  # entradas y salidas del experimento
    muestra: int | None = None,  # datos.muestra (None: todos los datos)
    base_seed: int | None = None,  # semilla del sorteo (obligatoria con muestra)
) -> tuple[pd.DataFrame, dict]:
    """(DataFrame del parquet o de su muestra, matrices por split) con una sola lectura del parquet."""
    df = muestrear(pd.read_parquet(rutas.parquet), muestra, base_seed)  # datos de A (o su muestra, B6)
    pre = Preprocesador.from_dict(load_json(rutas.preprocesador))  # sin reajustar
    return df, _matrices(df, pre)  # el mismo df sirve para la huella


def cargar_matrices(rutas: RutasPD) -> dict:
    """{split: (X float32, y, ids)} para train, val y test con el preprocesador guardado."""
    return cargar_datos(rutas)[1]  # solo las matrices


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
#
# TODO(refactor-post-B5): es la tercera copia del envoltorio TypeError → ValueError de `validar_entero` (las otras
# dos son `_entero_config` de `src/pd/mlp.py` y de `src/pd/logistica.py`); al cerrar B5 se unifican en un módulo
# común.

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


def _semilla_logistica(modelo, cfg_log: dict, metricas: dict, seed: int, huellas: dict) -> dict:
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
        "evidencia": huellas,  # huellas de la partición y del preprocesador
    }


def _busqueda_logistica(modelo, cfg_log: dict, huellas: dict) -> dict:
    """Contenido de busqueda_logistica.json."""
    return {  # tabla de la búsqueda
        "criterio": CRITERIO_LOGISTICA,  # log-loss de val
        "filas": modelo.busqueda,  # una por C
        "C_elegido": modelo.C_elegido,  # C elegido
        "en_borde": esta_en_borde(cfg_log["rejilla_C"], modelo.C_elegido),  # ¿ampliar la rejilla?
        "tiempo_total_s": modelo.tiempo_total_s,  # todo el fit
        "evidencia": huellas,  # huellas de la partición y del preprocesador
    }


def _summary_logistica(modelo, cfg_log: dict, metricas: dict, huellas: dict) -> dict:
    """Entrada "logistica" de summary.json."""
    return {  # presupuesto y métricas
        "n_parametros": modelo.n_parametros(),  # complejidad (núcleo §6)
        "n_configuraciones_busqueda": len(cfg_log["rejilla_C"]),  # presupuesto de búsqueda
        "tiempo_total_s": modelo.tiempo_total_s,  # presupuesto de tiempo
        "metricas": resumir([metricas]),  # una corrida
        "evidencia": huellas,  # huellas de la partición y del preprocesador
    }


# %% [markdown]
# ## Escrituras de un modelo con una sola corrida
#
# Semilla (`semillas/{modelo}_seed{seed}.json`), entrada del summary (conservando las ajenas) y columna
# `pred_{modelo}` de `pred_test.parquet` (alineada por id): la composición para un modelo de **una** corrida, como
# la logística. B5 no la reutiliza tal cual (el MLP escribe 30 semillas, pero un solo summary y una sola columna de
# `pred_test`, la de la semilla mediana): usa `escribir_semilla`, `actualizar_summary` y `actualizar_pred_test` por
# separado. Los archivos propios de la logística (`busqueda_logistica.json` y el joblib) se escriben aparte.

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
# Valida la semilla, la config y `datos.muestra` (B6) **antes** de leer datos o entrenar; carga las matrices (de la
# muestra, si la hay), entrena `LogisticaPD`
# (busca $C$ con val), mide val y test una vez y escribe los 5 archivos: semilla, búsqueda, summary,
# `pred_test.parquet` y `pd_logistica.joblib`. Devuelve la entrada del summary.

# %%
def correr_logistica(cfg: dict, rutas: RutasPD = RUTAS_PD, modelo_cls=LogisticaPD, seed: int = 0) -> dict:
    """Entrena y evalúa la logística, escribe sus resultados y devuelve su entrada de summary.json."""
    seed = _semilla_valida(seed)  # primero: una semilla inválida no debe costar un entrenamiento
    cfg_log = bloque_logistica(cfg)  # bloque validado (va a la semilla); ValueError si falta
    muestra = muestra_de(cfg)  # datos.muestra y su semilla validados antes de leer (B6)
    df, datos = cargar_datos(rutas, **muestra)  # una sola lectura del parquet
    huellas = evidencia(df, rutas.preprocesador)  # huellas del mismo df y del preprocesador
    X_tr, y_tr, _ = datos["train"]  # entrenamiento
    X_val, y_val, _ = datos["val"]  # validación (elige C)
    X_te, y_te, ids_te = datos["test"]  # prueba (se mide una vez)
    modelo = modelo_cls().fit(X_tr, y_tr, X_val, y_val, cfg, seed)  # búsqueda + modelo final
    metricas = medir(modelo, datos)  # val y test
    registro = _semilla_logistica(modelo, cfg_log, metricas, seed, huellas)  # archivo de la semilla
    entrada = _summary_logistica(modelo, cfg_log, metricas, huellas)  # entrada del summary
    _escribir_resultados(rutas, MODELO_LOGISTICA, seed, registro, entrada, ids_te, y_te,
                         modelo.predict_proba(X_te))  # escrituras comunes (B5 las reutiliza)
    save_json(rutas.resultados / ARCHIVO_BUSQUEDA_LOGISTICA, _busqueda_logistica(modelo, cfg_log, huellas))  # tabla
    rutas.modelos.mkdir(parents=True, exist_ok=True)  # carpeta de modelos
    modelo.guardar(rutas.modelos / ARCHIVO_MODELO_LOGISTICA)  # joblib
    return entrada  # entrada "logistica" del summary


# %% [markdown]
# # B5: 30 semillas del MLP final y `run.py pd` (Parte B §3.5, Spec v0.16)
#
# `python run.py pd --config configs/pd/base.yaml` corre la logística (arriba) y después el MLP con el bloque `mlp`
# resuelto (herencia de `mlp_final.yaml` y luego `--set`) en `semillas.n` semillas efectivas $s_k$ = `base_seed` + k,
# $k = 0, \dots, n-1$ (`seeds_for_runs`). Cada semilla entrena un `MLPPD` nuevo y escribe
# `semillas/mlp_seed{s}.json`; al final se escriben la entrada `mlp` de `summary.json`, la columna `pred_mlp` de
# `pred_test.parquet` (de la semilla mediana) y `models/pd_mlp.pt` (la red de la semilla mediana, sin reentrenar).
#
# Constantes: nombres de archivo del núcleo §5.4, la ruta de `mlp_final.yaml` (la reescribe `run.py pd-hpo`), las claves
# del bloque `busqueda` que escribe `hpo.armar_busqueda` y el texto de las reglas. `TIPO_OPTUNA` repite
# `hpo.TIPO_BUSQUEDA` sin importar `src.pd.hpo` (que importa este módulo: el import sería circular); una prueba
# comprueba que son iguales.

# %%
MODELO_MLP = "mlp"  # nombre del modelo en los resultados (núcleo §5.4)
ARCHIVO_MODELO_MLP = "pd_mlp.pt"  # red de la semilla mediana (núcleo §5.4)
RUTA_MLP_FINAL = Path("configs/pd/mlp_final.yaml")  # YAML que reescribe pd-hpo (relativo a la raíz)
TIPO_OPTUNA = "optuna"  # = hpo.TIPO_BUSQUEDA (constante local: evita el import circular hpo → experimento)
TIPO_SIN_BUSQUEDA = "sin_busqueda"  # busqueda: {tipo: sin_busqueda} = config fijada a mano (B-pd §3.5)
CLAVES_BUSQUEDA = ("tipo", "ensayo", "n_ensayos", "tiempo_s", "evidencia", "fecha")  # las de hpo.armar_busqueda
PATRON_SEMILLA_MLP = re.compile(r"mlp_seed\d+\.json", re.ASCII)  # mlp_seed<entero>.json (con fullmatch)
CRITERIO_SEMILLA_MEDIANA = ("orden ascendente por (logloss_val, seed); posición (n - 1) // 2 "
                            "(n par: mediana inferior; empate: semilla menor)")  # regla de B-pd §3.5
AVISO_SUMMARY_VIEJO = ("summary.json conserva la entrada mlp anterior, que ya no corresponde a semillas/; "
                       "vuelve a correr run.py pd")  # texto exacto (Spec v0.16)
CLAVES_CFG_MLP_FINAL = ("semillas", "base_seed", "mlp", "busqueda")  # bloques que necesita correr_mlp


# %% [markdown]
# ## Validación: enteros de la config
#
# `seeds.validar_entero` (int de Python o de numpy, nunca bool) con su `TypeError` convertido en `ValueError`: toda
# falla de la config es `ValueError`.
#
# TODO(refactor-post-B5): es otra copia del envoltorio TypeError → ValueError (`_semilla_valida` arriba, `_entero` de
# `hpo.py` y `_entero_config` de `mlp.py` y `logistica.py`); se unifica en el refactor posterior a B5.

# %%
def _entero_cfg(valor, nombre: str, minimo: int) -> int:
    """``seeds.validar_entero`` con su TypeError convertido en ValueError."""
    try:  # regla única de enteros del proyecto
        return validar_entero(valor, nombre, minimo=minimo)  # ValueError si es < minimo
    except TypeError as exc:  # flotante, bool, texto…
        raise ValueError(str(exc)) from exc  # mismo mensaje, ValueError


# %% [markdown]
# ## Validación: bloque `semillas`
#
# `semillas` es un mapeo con **exactamente** la clave `n`, entero $\ge 1$ (B-pd §3.5).

# %%
def _n_semillas(cfg: dict) -> int:
    """``semillas.n`` como entero ≥ 1 (``semillas`` debe ser un mapeo con exactamente la clave ``n``)."""
    semillas = cfg["semillas"]  # bloque semillas
    if not isinstance(semillas, dict) or set(semillas) != {"n"}:  # p. ej. semillas: 30 o una clave de más
        raise ValueError(f"semillas debe ser un mapeo con exactamente la clave n; se recibió {semillas!r}")  # forma
    return _entero_cfg(semillas["n"], "semillas.n", minimo=1)  # al menos una semilla


# %% [markdown]
# ## Validación: bloque `busqueda`
#
# Dos formas válidas (B-pd §3.5): el bloque de `pd-hpo`, con **exactamente** las claves de `hpo.armar_busqueda`
# (`CLAVES_BUSQUEDA`) y `tipo: optuna`, o exactamente `{tipo: sin_busqueda}` (config fijada a mano). Cualquier otro
# tipo o claves de más o de menos → `ValueError`. Devuelve una copia.

# %%
def _validar_busqueda(busqueda) -> dict:
    """Copia de ``busqueda`` si es el bloque de pd-hpo (optuna) o {tipo: sin_busqueda}; si no, ValueError."""
    if not isinstance(busqueda, dict):  # p. ej. busqueda: optuna
        raise ValueError(f"El bloque busqueda debe ser un mapeo; se recibió {busqueda!r}")  # tipo
    tipo = busqueda.get("tipo")  # procedencia declarada
    if tipo == TIPO_SIN_BUSQUEDA:  # config fijada a mano
        if set(busqueda) != {"tipo"}:  # sin_busqueda no lleva más claves
            raise ValueError(f"busqueda con tipo {TIPO_SIN_BUSQUEDA!r} no admite otras claves: {busqueda!r}")  # extra
        return copy.deepcopy(busqueda)  # copia
    if tipo != TIPO_OPTUNA:  # tipo desconocido o ausente
        raise ValueError(f"busqueda.tipo debe ser {TIPO_OPTUNA!r} o {TIPO_SIN_BUSQUEDA!r}; se recibió {tipo!r}")  # tipo
    faltan, sobran = set(CLAVES_BUSQUEDA) - set(busqueda), set(busqueda) - set(CLAVES_BUSQUEDA)  # diferencias
    if faltan or sobran:  # el bloque debe ser exacto
        raise ValueError(f"El bloque busqueda es inválido: faltan {sorted(faltan)}, sobran {sorted(sobran)}")  # cuáles
    return copy.deepcopy(busqueda)  # copia: no comparte la evidencia con la config


# %% [markdown]
# ## Validación de la config del MLP
#
# Antes de leer datos: los bloques `semillas`, `base_seed`, `mlp` y `busqueda` existen; `semillas.n` entero $\ge 1$,
# `base_seed` entero $\ge 0$, `mlp` pasa `validar_cfg_mlp` (B2) y `busqueda` tiene sus 6 claves. Toda falla →
# `ValueError`. Devuelve copias: la config del llamador no se muta.

# %%
def validar_cfg_mlp_final(cfg: dict) -> tuple[int, int, dict, dict]:
    """(n, base_seed, mlp validado, busqueda validada) antes de leer datos; cualquier falla → ValueError."""
    if not isinstance(cfg, dict):  # sin config no hay corrida
        raise ValueError(f"La config de run.py pd debe ser un mapeo; se recibió {type(cfg).__name__}")  # tipo
    faltan = [c for c in CLAVES_CFG_MLP_FINAL if c not in cfg]  # bloques necesarios
    if faltan:  # p. ej. una config sin semillas
        raise ValueError(f"La config de run.py pd no tiene {faltan}")  # cuáles
    n = _n_semillas(cfg)  # número de semillas
    base_seed = _entero_cfg(cfg["base_seed"], "base_seed", minimo=0)  # semilla base
    if not isinstance(cfg["mlp"], dict):  # validar_cfg_mlp indexa el bloque
        raise ValueError(f"El bloque mlp debe ser un mapeo; se recibió {cfg['mlp']!r}")  # tipo
    return n, base_seed, validar_cfg_mlp(cfg["mlp"]), _validar_busqueda(cfg["busqueda"])  # todo validado


# %% [markdown]
# ## Procedencia: claves que difieren
#
# Claves de `CLAVES_MLP` (en su orden) cuyo valor en el bloque de la corrida difiere del de la búsqueda (pura).

# %%
def claves_distintas(mlp: dict, mlp_busqueda: dict) -> list[str]:
    """Claves de CLAVES_MLP cuyo valor difiere entre ``mlp`` y ``mlp_busqueda``."""
    return [k for k in CLAVES_MLP if mlp[k] != mlp_busqueda[k]]  # en el orden de CLAVES_MLP


# %% [markdown]
# ## Procedencia: `mlp_final.yaml`
#
# Relee el YAML que generó `run.py pd-hpo` con `load_config` y valida sus bloques `mlp` y `busqueda` (mismas reglas).
# Si el archivo no existe → `ValueError` ("corre pd-hpo"), antes de leer datos.

# %%
def leer_mlp_final(ruta) -> tuple[dict, dict]:
    """(mlp validado, busqueda validada) de ``ruta`` (mlp_final.yaml generado por pd-hpo)."""
    if not Path(ruta).is_file():  # pd-hpo nunca corrió (o se borró el archivo)
        raise ValueError(f"No existe {Path(ruta).as_posix()}: corre pd-hpo (python run.py pd-hpo)")  # E1
    final = load_config(ruta)  # mismo cargador que el CLI
    faltan = [c for c in ("mlp", "busqueda") if c not in final]  # bloques que escribe pd-hpo
    if faltan:  # archivo ajeno o editado a mano
        raise ValueError(f"{ruta} no tiene {faltan}; repite pd-hpo")  # cuáles
    if not isinstance(final["mlp"], dict):  # validar_cfg_mlp indexa el bloque
        raise ValueError(f"El bloque mlp de {ruta} debe ser un mapeo")  # tipo
    return validar_cfg_mlp(final["mlp"]), _validar_busqueda(final["busqueda"])  # validados


# %% [markdown]
# ## Procedencia: ¿es la config de la búsqueda?
#
# `config_de_busqueda` = (el `mlp` validado de la corrida es igual al de `mlp_final.yaml`) ∧ (el `busqueda` también)
# ∧ (`busqueda.tipo` es `optuna`): con `sin_busqueda` es false aunque todo coincida. Devuelve además las claves que
# difieren (con `"busqueda"` al final si difiere ese bloque) para el aviso.

# %%
def procedencia(mlp: dict, busqueda: dict, ruta_mlp_final) -> tuple[bool, list[str]]:
    """(config_de_busqueda, claves que difieren) frente a ``ruta_mlp_final``."""
    mlp_final, busqueda_final = leer_mlp_final(ruta_mlp_final)  # lo que eligió pd-hpo
    difieren = claves_distintas(mlp, mlp_final)  # claves del bloque mlp
    if busqueda != busqueda_final:  # otra procedencia
        difieren.append("busqueda")  # se nombra el bloque
    return not difieren and busqueda["tipo"] == TIPO_OPTUNA, difieren  # true solo si nada difiere y hubo búsqueda


# %% [markdown]
# ## Presupuesto de la búsqueda
#
# Con `config_de_busqueda` true (solo posible con `tipo: optuna`, ver `procedencia`), la búsqueda debe ser con los
# **mismos datos** (`busqueda.evidencia` = huellas de esta corrida); si no → `ValueError`. Entonces el presupuesto del
# MLP es `n_configuraciones_busqueda = busqueda.n_ensayos` y `tiempo_busqueda_s = busqueda.tiempo_s`. Con false, los tres
# campos quedan en `None` (el presupuesto de la búsqueda no corresponde a esta config).

# %%
def presupuesto_busqueda(busqueda: dict, de_busqueda: bool, huellas: dict) -> dict:
    """{n_configuraciones_busqueda, tiempo_busqueda_s, busqueda}; ValueError si la búsqueda no es coherente."""
    if not de_busqueda:  # otra config: sin presupuesto de búsqueda
        return {"n_configuraciones_busqueda": None, "tiempo_busqueda_s": None, "busqueda": None}  # null
    if busqueda["evidencia"] != huellas:  # otra partición u otro preprocesador
        raise ValueError("busqueda.evidencia no coincide con la de esta corrida: la búsqueda se hizo con otros "
                         "datos; repite pd-hpo")  # texto de B-pd §3.5
    return {"n_configuraciones_busqueda": busqueda["n_ensayos"], "tiempo_busqueda_s": busqueda["tiempo_s"],
            "busqueda": busqueda}  # presupuesto de la búsqueda


# %% [markdown]
# ## Aviso de procedencia
#
# Si la config no es la de la búsqueda, la consola dice por qué: las claves que difieren (p. ej.
# `--set mlp.capas=[8]`) o que el bloque es `sin_busqueda` (config fijada a mano).

# %%
def _avisar_procedencia(de_busqueda: bool, difieren: list[str], tipo: str) -> None:
    """Imprime por qué la config no es la de la búsqueda (claves que difieren o sin_busqueda)."""
    if de_busqueda:  # config de la búsqueda
        return  # nada que avisar
    motivo = f"difieren: {', '.join(difieren)}" if difieren else f"busqueda.tipo: {tipo}"  # por qué
    print(f"Aviso: la config del MLP no es la de la búsqueda ({motivo}); "
          "n_configuraciones_busqueda, tiempo_busqueda_s y busqueda quedan en null")  # B-pd §3.5


# %% [markdown]
# ## Borrado de las semillas anteriores
#
# Antes de entrenar se borran **solo** los archivos cuyo nombre completo es `mlp_seed<entero>.json` (`fullmatch`):
# las semillas de la logística, otros nombres parecidos y las carpetas quedan. Carpeta inexistente → 0.

# %%
def borrar_semillas_mlp(carpeta) -> int:
    """Borra los archivos mlp_seed<entero>.json de ``carpeta`` y devuelve cuántos."""
    carpeta = Path(carpeta)  # acepta texto o Path
    if not carpeta.is_dir():  # primera corrida
        return 0  # nada que borrar
    viejos = [p for p in carpeta.iterdir() if p.is_file() and PATRON_SEMILLA_MLP.fullmatch(p.name)]  # solo MLP
    for p in viejos:  # uno por uno
        p.unlink()  # borra
    return len(viejos)  # cuántos


# %% [markdown]
# ## Salidas previas incompatibles
#
# Falla **antes del primer fit** (no se pierden 30 entrenamientos) si `summary.json` existe con una raíz que no es
# diccionario o si `pred_test.parquet` existe con otro conjunto de ids o con otra $y$: las mismas reglas de
# `actualizar_summary` y `actualizar_pred_test` (se reutilizan `_tabla_nueva` y `_combinar`), sin escribir nada.

# %%
def _verificar_salidas(rutas: RutasPD, ids_test, y_test) -> None:
    """ValueError si summary.json o pred_test.parquet existentes no admitirían la entrada del MLP."""
    ruta_summary = rutas.resultados / ARCHIVO_SUMMARY  # summary previo
    if ruta_summary.exists() and not isinstance(load_json(ruta_summary), dict):  # archivo ajeno
        raise ValueError(f"{ruta_summary} debe tener un diccionario en la raíz")  # como actualizar_summary
    ruta_pred = rutas.resultados / ARCHIVO_PRED_TEST  # pred_test previo
    if ruta_pred.exists():  # se combinará por id
        columna = PREFIJO_PRED + MODELO_MLP  # pred_mlp
        nuevo = _tabla_nueva(ids_test, y_test, columna, np.zeros(len(ids_test)))  # PD provisional (no se escribe)
        _combinar(pd.read_parquet(ruta_pred), nuevo, columna)  # ValueError si ids o y no coinciden


# %% [markdown]
# ## Una semilla: entrenamiento cronometrado
#
# Un `modelo_cls()` **nuevo** por semilla, entrenado con `{"mlp": cfg_mlp}` y la semilla $s$; $t_s$ = `reloj()` después
# − `reloj()` antes del `fit` (solo el fit: ni la carga ni la medición). Si el fit diverge (`FloatingPointError`, B2)
# el error sale del **mismo tipo** con la semilla y el aviso de que `summary.json` quedó viejo.

# %%
def entrenar_semilla(modelo_cls, datos: dict, cfg_mlp: dict, seed: int, reloj) -> tuple:
    """(modelo entrenado, tiempo del fit en s) para la semilla ``seed``."""
    X_tr, y_tr, _ = datos["train"]  # entrenamiento
    X_val, y_val, _ = datos["val"]  # validación (parada temprana)
    modelo = modelo_cls()  # modelo nuevo por semilla
    inicio = reloj()  # comienzo del fit
    try:  # solo la divergencia se reetiqueta
        modelo.fit(X_tr, y_tr, X_val, y_val, {"mlp": cfg_mlp}, seed=seed)  # entrena (no muta cfg_mlp)
    except FloatingPointError as exc:  # pérdida no finita
        raise FloatingPointError(f"semilla {seed}: {exc}. {AVISO_SUMMARY_VIEJO}") from exc  # mismo tipo
    return modelo, reloj() - inicio  # tiempo de solo el fit


# %% [markdown]
# ## Evidencia rápida
#
# Para el pre-chequeo de `correr_pd`, antes de la logística: la misma `evidencia` leyendo del parquet **solo** las
# columnas `id` y `split` (mismo orden de filas). Una prueba exige que coincida con la del DataFrame completo. B6: con
# muestra, la huella es la de las filas muestreadas (`muestrear` solo mira `split` y el orden).

# %%
def evidencia_rapida(ruta_parquet, ruta_preprocesador, muestra: int | None = None,
                     base_seed: int | None = None) -> dict:
    """``evidencia`` del parquet (o de su muestra) leyendo solo id y split."""
    df = pd.read_parquet(ruta_parquet, columns=[COLUMNA_ID, COLUMNA_SPLIT])  # dos columnas
    df = muestrear(df, muestra, base_seed)  # las mismas filas que cargar_datos (B6)
    return evidencia(df, ruta_preprocesador)  # misma función que correr_mlp


# %% [markdown]
# ## Calibración global
#
# Para cada split $s \in \{\text{val}, \text{test}\}$ con $n_s$ créditos:
#
# $$\bar p_s = \frac1{n_s}\sum_{i \in s} p_i, \qquad \bar y_s = \frac1{n_s}\sum_{i \in s} y_i,$$
#
# la PD media del modelo y la tasa de malos observada (calibración en el agregado).

# %%
def calibracion_global(modelo, datos: dict) -> dict:
    """{pd_media_val, tasa_val, pd_media_test, tasa_test}."""
    resultado = {}  # en el orden del núcleo §5.4
    for split in SPLITS_MEDIDOS:  # val y test
        X, y, _ = datos[split]  # matrices del split
        resultado[f"pd_media_{split}"] = float(np.mean(modelo.predict_proba(X)))  # p̄
        resultado[f"tasa_{split}"] = float(np.mean(y))  # ȳ
    return resultado  # cuatro valores


# %% [markdown]
# ## Registro de una semilla
#
# `semillas/mlp_seed{s}.json`: las claves del núcleo §5.4 (`epocas` = épocas corridas) + la evidencia + los extras de
# PD mlp (v0.16): `mejor_epoca`, `mejor_perdida_val`, `historial`, `hilos_torch`, `version_torch`,
# `calibracion_global` y `n` (filas por split).

# %%
def registro_semilla_mlp(modelo, cfg_mlp: dict, metricas: dict, seed: int, tiempo_s: float, device: str,
                         huellas: dict, datos: dict) -> dict:
    """Contenido de semillas/mlp_seed{seed}.json."""
    resultado = modelo.resultado  # ResultadoEntrenamiento del fit
    return {  # núcleo §5.4 (v0.16)
        "modelo": MODELO_MLP,  # nombre
        "seed": seed,  # semilla efectiva
        "config": cfg_mlp,  # bloque mlp validado
        "metricas": metricas,  # {m}_{split}
        "n_parametros": modelo.n_parametros(),  # parámetros entrenables
        "tiempo_s": tiempo_s,  # solo el fit
        "device": device,  # dispositivo efectivo
        "epocas": resultado.epocas_corridas,  # épocas entrenadas
        "evidencia": huellas,  # huellas de la partición y del preprocesador
        "mejor_epoca": resultado.mejor_epoca,  # época de los pesos restaurados
        "mejor_perdida_val": resultado.mejor_perdida_val,  # BCE de val de esa época
        "historial": copy.deepcopy(resultado.historial),  # [{epoca, perdida_train, perdida_val}]
        "hilos_torch": torch.get_num_threads(),  # hilos de CPU (determinismo)
        "version_torch": str(torch.__version__),  # versión de PyTorch
        "calibracion_global": calibracion_global(modelo, datos),  # p̄ y ȳ de val y test
        "n": {split: int(len(datos[split][1])) for split in SPLITS},  # filas por split
    }


# %% [markdown]
# ## Semilla mediana
#
# Con las corridas ordenadas de menor a mayor por la clave $(\text{logloss\_val}, s)$, la mediana es la de la posición
#
# $$k^{*} = \left\lfloor \frac{n - 1}{2} \right\rfloor \quad (\text{desde } 0),$$
#
# así que con $n$ par es la mediana inferior y un empate se resuelve por la semilla menor. Decide validación, como la
# búsqueda (B-pd §3.3): test nunca decide.

# %%
def semilla_mediana(corridas: list[dict]) -> int:
    """Semilla de la posición (n − 1) // 2 al ordenar por (logloss_val, seed)."""
    if not corridas:  # sin corridas no hay mediana
        raise ValueError("semilla_mediana necesita al menos una corrida")  # vacío
    ordenadas = sorted(corridas, key=lambda c: (c["metricas"]["logloss_val"], c["seed"]))  # val y desempate
    return ordenadas[(len(ordenadas) - 1) // 2]["seed"]  # mediana inferior


# %% [markdown]
# ## Entrada `mlp` de `summary.json`
#
# Presupuesto (núcleo §6) con $t_s$ el tiempo del fit de la semilla $s$:
#
# $$\texttt{tiempo\_semillas\_s} = \sum_{s} t_s, \qquad \texttt{tiempo\_total\_s} = \begin{cases}
# \texttt{tiempo\_busqueda\_s} + \sum_s t_s & \text{si config\_de\_busqueda} \\ \sum_s t_s & \text{si no.}
# \end{cases}$$
#
# Métricas resumidas con `resumir` (media, desv muestral, mejor, peor, n); `n_parametros`, `hilos_torch` y
# `version_torch` de la semilla mediana (son iguales en todas).

# %%
def entrada_summary_mlp(corridas: list[dict], cfg_mlp: dict, presupuesto: dict, de_busqueda: bool, seed_med: int,
                        huellas: dict) -> dict:
    """Entrada "mlp" de summary.json (núcleo §5.4, v0.16)."""
    mediana = next(c for c in corridas if c["seed"] == seed_med)  # registro de la semilla mediana
    tiempo_semillas = sum((c["tiempo_s"] for c in corridas), 0.0)  # Σ de los fit
    tiempo_total = presupuesto["tiempo_busqueda_s"] + tiempo_semillas if de_busqueda else tiempo_semillas  # total
    return {  # en el orden del núcleo §5.4
        "n_parametros": mediana["n_parametros"],  # complejidad
        "n_configuraciones_busqueda": presupuesto["n_configuraciones_busqueda"],  # ensayos de la búsqueda o null
        "tiempo_total_s": tiempo_total,  # presupuesto de tiempo
        "tiempo_busqueda_s": presupuesto["tiempo_busqueda_s"],  # de mlp_final.yaml o null
        "tiempo_semillas_s": tiempo_semillas,  # Σ de los fit
        "metricas": resumir([c["metricas"] for c in corridas]),  # media, desv, mejor, peor, n
        "evidencia": huellas,  # huellas de la partición y del preprocesador
        "semillas": [c["seed"] for c in corridas],  # semillas efectivas
        "semilla_mediana": seed_med,  # la de pd_mlp.pt y pred_mlp
        "criterio_semilla_mediana": CRITERIO_SEMILLA_MEDIANA,  # regla
        "config": cfg_mlp,  # bloque mlp validado
        "config_de_busqueda": de_busqueda,  # ¿es la config de la búsqueda?
        "busqueda": presupuesto["busqueda"],  # bloque de mlp_final.yaml o null
        "hilos_torch": mediana["hilos_torch"],  # hilos de CPU
        "version_torch": mediana["version_torch"],  # versión de PyTorch
    }


# %% [markdown]
# ## Todas las semillas
#
# Por cada semilla efectiva: entrena (cronometrado), mide val y test, arma el registro y lo escribe. Los modelos
# quedan en memoria (`{seed: modelo}`) para guardar el de la mediana sin reentrenar.

# %%
def _correr_semillas(modelo_cls, datos: dict, cfg_mlp: dict, semillas: list[int], reloj, rutas: RutasPD,
                     device: str, huellas: dict) -> tuple[list[dict], dict]:
    """(registros en orden de semilla, {seed: modelo entrenado})."""
    corridas, modelos = [], {}  # registros y modelos
    for seed in semillas:  # base_seed + 0…n−1
        modelo, tiempo_s = entrenar_semilla(modelo_cls, datos, cfg_mlp, seed, reloj)  # fit cronometrado
        registro = registro_semilla_mlp(modelo, cfg_mlp, medir(modelo, datos), seed, tiempo_s, device, huellas,
                                        datos)  # contenido del archivo
        escribir_semilla(rutas, MODELO_MLP, seed, registro)  # semillas/mlp_seed{seed}.json
        corridas.append(registro)  # en memoria
        modelos[seed] = modelo  # para la mediana
    return corridas, modelos  # todo lo entrenado


# %% [markdown]
# ## Escrituras finales del MLP
#
# Entrada `mlp` de `summary.json` (conserva la de la logística), `pred_mlp` de `pred_test.parquet` (PD de test de la
# red mediana, alineada por id) y `models/pd_mlp.pt` (la misma red, sin reentrenar).

# %%
def _escribir_mlp(rutas: RutasPD, entrada: dict, modelo, datos: dict) -> None:
    """Escribe la entrada del summary, la columna pred_mlp y pd_mlp.pt."""
    X_te, y_te, ids_te = datos["test"]  # prueba
    actualizar_summary(rutas.resultados / ARCHIVO_SUMMARY, MODELO_MLP, entrada)  # conserva las ajenas
    actualizar_pred_test(rutas.resultados / ARCHIVO_PRED_TEST, ids_te, y_te, PREFIJO_PRED + MODELO_MLP,
                         modelo.predict_proba(X_te))  # alineada por id
    rutas.modelos.mkdir(parents=True, exist_ok=True)  # carpeta de modelos
    modelo.guardar(rutas.modelos / ARCHIVO_MODELO_MLP)  # red de la semilla mediana


# %% [markdown]
# ## Experimento del MLP
#
# Valida (antes de leer datos), carga una vez, compara con la búsqueda (antes de borrar y de entrenar), verifica las
# salidas previas, borra las semillas viejas, entrena las $n$ semillas, elige la mediana y escribe. `cargar` y
# `ruta_mlp_final` se resuelven **al llamar** (`cargar_datos`, `RUTA_MLP_FINAL`), como en `hpo.py`. La lectura de
# `mlp_final.yaml` va antes de leer datos (si falta → `ValueError`). Devuelve la entrada `mlp` del summary.
#
# B6: `datos.muestra` se valida al inicio; el `cargar` por defecto es `cargar_datos` con la muestra de la config. Un
# `cargar` inyectado conserva su firma `cargar(rutas)` y es responsable de su muestra.

# %%
def correr_mlp(cfg: dict, rutas: RutasPD = RUTAS_PD, modelo_cls=MLPPD, cargar=None, reloj=time.perf_counter,
               ruta_mlp_final=None) -> dict:
    """30 semillas (semillas.n) del MLP final: semillas/, entrada mlp, pred_mlp y pd_mlp.pt."""
    muestra = muestra_de(cfg)  # datos.muestra validado antes de leer (B6)
    n, base_seed, cfg_mlp, busqueda = validar_cfg_mlp_final(cfg)  # antes de leer datos
    cargar = functools.partial(cargar_datos, **muestra) if cargar is None else cargar  # al llamar
    ruta_mlp_final = RUTA_MLP_FINAL if ruta_mlp_final is None else ruta_mlp_final  # al llamar
    de_busqueda, difieren = procedencia(cfg_mlp, busqueda, ruta_mlp_final)  # antes de leer datos (E1)
    df, datos = cargar(rutas)  # una sola lectura
    huellas = evidencia(df, rutas.preprocesador)  # huellas del mismo df
    presupuesto = presupuesto_busqueda(busqueda, de_busqueda, huellas)  # ValueError antes de borrar
    _avisar_procedencia(de_busqueda, difieren, busqueda["tipo"])  # aviso si no es la de la búsqueda
    _verificar_salidas(rutas, datos["test"][2], datos["test"][1])  # antes del 1.er fit
    borrados = borrar_semillas_mlp(rutas.resultados / CARPETA_SEMILLAS)  # semillas viejas del MLP
    print(f"Semillas anteriores del MLP borradas: {borrados}")  # consola
    device = str(resolver_device(cfg_mlp["device"]))  # dispositivo efectivo
    corridas, modelos = _correr_semillas(modelo_cls, datos, cfg_mlp, seeds_for_runs(base_seed, n), reloj, rutas,
                                         device, huellas)  # n semillas
    seed_med = semilla_mediana(corridas)  # decide val
    entrada = entrada_summary_mlp(corridas, cfg_mlp, presupuesto, de_busqueda, seed_med, huellas)  # summary
    _escribir_mlp(rutas, entrada, modelos[seed_med], datos)  # summary, pred_mlp y pd_mlp.pt
    return entrada  # entrada "mlp" del summary


# %% [markdown]
# ## Rutas de `run.py pd`
#
# Entradas de `RUTAS_PD` (leídas al llamar) y salidas de la config: `resultados = salida` y `modelos = modelos`
# (carpetas, texto no vacío; si no → `ValueError`).

# %%
def rutas_de(cfg: dict) -> RutasPD:
    """RutasPD con el parquet y el preprocesador de RUTAS_PD y las carpetas salida y modelos de la config."""
    for clave in ("salida", "modelos"):  # carpetas de salida
        valor = cfg.get(clave) if isinstance(cfg, dict) else None  # sin la clave → None
        if not isinstance(valor, str) or not valor.strip():  # '' o un número no son rutas
            raise ValueError(f"{clave} debe ser una ruta (texto no vacío); se recibió {valor!r}")  # tipo
    return RutasPD(RUTAS_PD.parquet, RUTAS_PD.preprocesador, Path(cfg["salida"]), Path(cfg["modelos"]))  # rutas


# %% [markdown]
# ## Ids e $y$ de test sin preprocesar
#
# Para el pre-chequeo de las salidas previas: del parquet se leen solo `id`, `split` y `target`; las filas de test
# salen en el orden del parquet y con los mismos tipos que `cargar_datos` (ids `object`, $y$ `int64`). B6: con muestra,
# las de test de la muestra.

# %%
def leer_test_rapido(
    ruta_parquet,  # data/processed/loans_pd.parquet
    muestra: int | None = None,  # datos.muestra (None: todos los datos)
    base_seed: int | None = None,  # semilla del sorteo (obligatoria con muestra)
) -> tuple[np.ndarray, np.ndarray]:
    """(ids, y) de test (del parquet o de su muestra) leyendo solo id, split y target."""
    df = pd.read_parquet(ruta_parquet, columns=[COLUMNA_ID, COLUMNA_SPLIT, COLUMNA_OBJETIVO])  # tres columnas
    df = muestrear(df, muestra, base_seed)  # las mismas filas que cargar_datos (B6)
    parte = df[df[COLUMNA_SPLIT] == "test"]  # filas de test, en el orden del parquet
    return parte[COLUMNA_ID].to_numpy(dtype=object), parte[COLUMNA_OBJETIVO].to_numpy(dtype=np.int64)  # como _matrices


# %% [markdown]
# ## Pre-chequeo de `run.py pd`
#
# Antes de la logística, sin preprocesar ni entrenar: las mismas validaciones de `correr_mlp`, el bloque `logistica`
# (`bloque_logistica`), la existencia de `mlp_final.yaml`, las salidas previas (`_verificar_salidas` con los ids e $y$
# de test de `leer_test`, por defecto `leer_test_rapido`) y, si la config es la de la búsqueda, la evidencia de la
# búsqueda frente a la de los datos leyendo solo `id, split` (`leer_particion`, por defecto `evidencia_rapida`).
# Así una búsqueda hecha con otros datos o una salida ajena no cuestan la corrida de la logística.
#
# B6: primero `datos.muestra`; los lectores por defecto reciben la muestra de la config (`functools.partial`), así que
# la huella y el test del pre-chequeo son los de las filas que se entrenarán. Los inyectados conservan su firma.

# %%
def comprobar_antes(cfg: dict, rutas: RutasPD, ruta_mlp_final=None, leer_particion=None, leer_test=None) -> None:
    """ValueError si la config, mlp_final.yaml o las salidas previas no sirven, o si la búsqueda usó otros datos."""
    muestra = muestra_de(cfg)  # datos.muestra validado antes de todo (B6)
    _, _, cfg_mlp, busqueda = validar_cfg_mlp_final(cfg)  # mismas reglas que correr_mlp
    bloque_logistica(cfg)  # bloque logistica validado (E4)
    ruta_mlp_final = RUTA_MLP_FINAL if ruta_mlp_final is None else ruta_mlp_final  # al llamar
    leer_particion = (functools.partial(evidencia_rapida, **muestra) if leer_particion is None
                      else leer_particion)  # al llamar, con la muestra
    leer_test = functools.partial(leer_test_rapido, **muestra) if leer_test is None else leer_test  # ídem
    de_busqueda, _ = procedencia(cfg_mlp, busqueda, ruta_mlp_final)  # ¿config de la búsqueda? (E1 si falta)
    ids_test, y_test = leer_test(rutas.parquet)  # solo id, split y target
    _verificar_salidas(rutas, ids_test, y_test)  # summary y pred_test previos (E4)
    huellas = leer_particion(rutas.parquet, rutas.preprocesador)  # solo id y split
    presupuesto_busqueda(busqueda, de_busqueda, huellas)  # ValueError si los datos no son los de la búsqueda


# %% [markdown]
# ## B6: filas por split de la corrida
#
# Cuenta las filas por split que usará la corrida leyendo del parquet solo `id` y `split` y aplicando la misma
# `muestrear` que `cargar_datos` (sin muestra: todas las filas).

# %%
def filas_muestra(ruta_parquet, muestra: int | None, base_seed: int | None) -> dict[str, int]:
    """Filas por split que usará la corrida (lee solo id y split)."""
    df = pd.read_parquet(ruta_parquet, columns=[COLUMNA_ID, COLUMNA_SPLIT])  # dos columnas
    splits = muestrear(df, muestra, base_seed)[COLUMNA_SPLIT]  # split de las filas de la corrida
    return {s: int((splits == s).sum()) for s in SPLITS}  # en el orden de SPLITS


# %% [markdown]
# ## B6: aviso de la muestra
#
# Con muestra, la consola dice cuántas filas de cada split entran a la corrida, p. ej.
# `Muestra: datos.muestra=50000 → train 19005 · val 15397 · test 15598`; sin muestra no imprime nada. `contar` se
# inyecta (por defecto `filas_muestra`).

# %%
def _avisar_muestra(cfg: dict, rutas: RutasPD, contar) -> None:
    """Imprime 'Muestra: datos.muestra=N → train a · val b · test c' si hay muestra; si no, nada."""
    muestra = muestra_de(cfg)  # {muestra, base_seed} validados
    if muestra["muestra"] is None:  # todos los datos
        return  # nada que avisar
    filas = contar(rutas.parquet, **muestra)  # filas por split de la muestra
    detalle = " · ".join(f"{s} {filas[s]}" for s in SPLITS)  # train a · val b · test c
    print(f"Muestra: datos.muestra={muestra['muestra']} → {detalle}")  # consola


# %% [markdown]
# ## `run.py pd`: composición
#
# Config que no es un mapeo → `ValueError` (antes de `rutas_de`); luego rutas de la config → pre-chequeo → aviso de la
# muestra (B6) → logística (semilla `base_seed`) → MLP. Las piezas se inyectan (también el contador de filas).

# %%
def correr_pd(cfg: dict, correr_log=correr_logistica, correr_red=correr_mlp, comprobar=comprobar_antes,
              contar=filas_muestra) -> dict:
    """{"logistica", "mlp", "rutas"}: pre-chequeo, aviso de la muestra, logística y MLP con las mismas rutas."""
    if not isinstance(cfg, dict):  # p. ej. None si el subcomando no recibió config
        raise ValueError(f"la config de pd debe ser un mapeo; se recibió {type(cfg).__name__}")  # tipo
    rutas = rutas_de(cfg)  # salidas de la config
    comprobar(cfg, rutas)  # antes de la logística
    _avisar_muestra(cfg, rutas, contar)  # filas por split si hay muestra (B6)
    entrada_log = correr_log(cfg, rutas, seed=cfg["base_seed"])  # logística (B3)
    entrada_mlp = correr_red(cfg, rutas)  # MLP (B5)
    return {"logistica": entrada_log, "mlp": entrada_mlp, "rutas": rutas}  # en ese orden


# %% [markdown]
# ## Línea de un modelo en la consola
#
# AUC, KS y Brier de test (media ± desv con varias corridas; valor único con una), parámetros, configuraciones de la
# búsqueda y tiempo total (núcleo §6).

# %%
def _linea_modelo(nombre: str, entrada: dict) -> str:
    """Texto de una línea con las métricas de test y el presupuesto de ``entrada``."""
    partes = []  # piezas de la línea
    for metrica, etiqueta in (("auc_test", "AUC"), ("ks_test", "KS"), ("brier_test", "Brier")):  # de test
        m = entrada["metricas"][metrica]  # resumen
        texto = f"{m['media']:.4f} ± {m['desv']:.4f}" if m["n"] > 1 else f"{m['media']:.4f}"  # media ± desv
        partes.append(f"{etiqueta} test {texto}")  # pieza
    configs = entrada["n_configuraciones_busqueda"]  # puede ser null (MLP fuera de la búsqueda)
    partes.append(f"{entrada['n_parametros']} parámetros")  # complejidad
    partes.append(f"{configs} configuraciones" if configs is not None else "sin presupuesto de búsqueda")  # búsqueda
    partes.append(f"{entrada['tiempo_total_s']:.1f} s")  # tiempo
    return f"{nombre}: " + " · ".join(partes)  # una línea


# %% [markdown]
# ## `run.py pd`: corrida e informe
#
# Corre todo e imprime una línea por modelo, la semilla mediana y su criterio, si la config es la de la búsqueda,
# dónde quedaron los archivos y el tiempo total (núcleo §4.1). La config resuelta ya la imprime el CLI.

# %%
def _correr_e_informar(cfg: dict | None) -> int:
    """Corre todo e imprime métricas, mediana, procedencia, rutas y tiempo; devuelve 0."""
    inicio = time.perf_counter()  # tiempo total del subcomando
    res = correr_pd(cfg)  # corrida completa (valida la config)
    mlp, rutas = res["mlp"], res["rutas"]  # atajos
    print(_linea_modelo(MODELO_LOGISTICA, res["logistica"]))  # logística
    print(_linea_modelo(MODELO_MLP, mlp))  # MLP
    print(f"Semilla mediana: {mlp['semilla_mediana']} ({mlp['criterio_semilla_mediana']})")  # pd_mlp.pt
    print(f"Config de la búsqueda: {'sí' if mlp['config_de_busqueda'] else 'no'}")  # procedencia
    print(f"Resultados: {rutas.resultados.as_posix()} · Modelos: {rutas.modelos.as_posix()}")  # archivos
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")  # núcleo §4.1
    return 0  # éxito


# %% [markdown]
# ## B6: mensaje de error en una línea
#
# El texto de la excepción con sus saltos de línea cambiados por espacios: el error de `run.py pd` ocupa una sola
# línea de stderr.

# %%
def _una_linea(exc: BaseException) -> str:
    """Texto de ``exc`` en una sola línea (saltos de línea → espacios)."""
    return " ".join(str(exc).splitlines())  # sin saltos de línea


# %% [markdown]
# ## `run.py pd`
#
# Llama a `_correr_e_informar`. Los errores **esperados** (config inválida, salidas previas incompatibles, búsqueda con
# otros datos: `ValueError`; fallas del fit: `RuntimeError` y `FloatingPointError`; un archivo de entrada que no
# existe: `FileNotFoundError`) salen como una línea `run.py pd: error: <Tipo>: <mensaje>` en stderr y código 2, como los errores de argumentos del CLI (argparse), sin traza
# (B6: en la demo en vivo se lee el mensaje). Cualquier otra excepción (p. ej. `TypeError`, un error de programación)
# se propaga con su traza; también las demás `OSError` (p. ej. `PermissionError`).
#
# `NotImplementedError` y `RecursionError` son subclases de `RuntimeError` pero indican un bug, no una falla esperada:
# se relanzan **antes** del `except` de los errores esperados (`ERRORES_DE_PROGRAMACION`, decisión de Andrés, 26-oct).

# %%
def manejador(cfg: dict | None) -> int:
    """`run.py pd`: _correr_e_informar; ValueError/RuntimeError/FloatingPointError → mensaje en stderr y código 2."""
    try:  # solo los errores esperados se informan sin traza
        return _correr_e_informar(cfg)  # corrida completa (0 si termina)
    except ERRORES_DE_PROGRAMACION:  # bugs que heredan de RuntimeError
        raise  # se propagan con su traza
    except ERRORES_ESPERADOS as exc:  # config, datos o fit
        print(f"{PREFIJO_ERROR}{type(exc).__name__}: {_una_linea(exc)}", file=sys.stderr)  # una línea con el tipo
        return CODIGO_ERROR  # como argparse
