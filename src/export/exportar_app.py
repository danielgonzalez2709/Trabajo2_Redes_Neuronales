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
# # Exportador de la app: `pd_model_{modelo}.json` y `casos_paridad_{modelo}.json` (Parte C §3.5, tareas C2 y C5)
#
# **Los dos modelos de PD van a la app** (núcleo §5.5, Spec v0.17). Una sola corrida de `python run.py export` escribe
# en `app/public/model/` **dos JSON por modelo**, con `{modelo}` ∈ `export.modelos` (`logistica` y `mlp`). La app web
# (Parte E) los lee para calcular en el navegador la PD, el puntaje y el percentil de una persona con cualquiera de los
# dos modelos (enunciado, Entregable 4; R-APP-1 a R-APP-4):
#
# - `pd_model_{modelo}.json`: formulario (`entradas`), preprocesamiento (`Preprocesador.to_dict()`), pesos de la red,
#   parámetros de la scorecard, percentiles del puntaje en test y métricas de test. `entradas`, `preprocesamiento` y
#   `scorecard` son iguales en los dos modelos; `red`, `poblacion` y `metricas` son de cada uno;
# - `casos_paridad_{modelo}.json`: 1 000 filas de test con sus entradas **crudas** y la PD y el puntaje calculados en
#   Python; la prueba de paridad de la app exige, para **cada** modelo, $|PD_{TS} - PD_{Py}| < 10^{-6}$ y
#   $|score_{TS} - score_{Py}| < 10^{-3}$.
#
# **De dónde sale cada cosa:** los modelos de PD de la Parte B (`models/pd_logistica.joblib`, B3; `models/pd_mlp.pt`,
# el MLP de la semilla mediana, B5), el preprocesador común de la Parte A (`models/preprocesador_pd.json`), la scorecard
# de C1 (`src/scorecard/scorecard.py`) y la config `configs/scorecard/export.yaml` (comando `python run.py export`).
#
# **Fuga de información (núcleo §5.7):** los estadísticos del formulario (recorte, mediana, moda) salen **solo de
# train** (el preprocesador ya se ajustó solo con train); de test solo salen los percentiles del puntaje (población de
# referencia) y las métricas.
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable sin
# efectos al importar.

# %%
"""Exportador de la app: pd_model_{modelo}.json y casos_paridad_{modelo}.json de los dos modelos de PD según el
contrato del núcleo §5.5 (Parte C, tareas C2 y C5)."""

import copy  # copias de la config para no mutar la recibida
import math  # floor, ceil, isfinite
import subprocess  # git rev-parse HEAD (local, sin red)
from datetime import datetime, timezone  # sello de tiempo ISO 8601 del export
from decimal import Decimal  # decimales del paso (0.1 → 1 decimal)
from numbers import Real  # comprobación de que un valor es numérico
from pathlib import Path  # rutas de entrada y salida

import numpy as np  # cálculo vectorizado en float64
import pandas as pd  # datos procesados y entradas crudas
import torch  # lectura del .pt del MLP
from scipy.special import expit  # sigmoide estable
from torch import nn  # tipo nn.Linear para recorrer las capas

from src.common.config import load_config  # features.yaml
from src.common.metricas import auc, brier, ks  # métricas comunes (B1)
from src.common.preprocesamiento import VERSION as VERSION_PREPROCESADOR  # versión de to_dict() que sabe leer la receta
from src.common.preprocesamiento import Preprocesador  # camino de Python (métricas y percentiles)
from src.common.results import load_json, save_json  # JSON UTF-8 sin NaN/inf (núcleo §5.4)
from src.common.seeds import make_rng, validar_entero  # semillas y regla única de enteros
from src.pd.logistica import LogisticaPD  # modelo de referencia (B3)
from src.pd.mlp import MLPPD, construir_red, validar_cfg_mlp  # MLP de PD (B2/B5)
from src.scorecard.scorecard import bloque_scorecard, parametros_scorecard, score_desde_logit  # scorecard de C1

__all__ = [  # interfaz pública acordada (tarea C2); lo demás es interno
    "VERSION_CONTRATO", "MODELOS", "RUTA_CONFIG", "TOLERANCIA_PARIDAD",  # constantes del contrato
    "PLANTILLA_MODELO", "PLANTILLA_PARIDAD",  # nombres de archivo por modelo
    "archivos_modelo",  # nombres de los dos archivos de un modelo
    "validar_cfg_export", "entradas_app",  # config y formulario
    "capas_logistica", "capas_mlp",  # pesos de la red
    "percentiles_score", "metricas_test", "commit_actual",  # población, métricas y procedencia
    "construir_pd_model",  # pd_model_{modelo}.json en memoria
    "preprocesar_desde_json", "forward_desde_json", "inferir_desde_json",  # receta de la app
    "elegir_casos", "construir_casos_paridad",  # casos de paridad
    "exportar",  # escribe los dos archivos de cada modelo
]


# %% [markdown]
# ## Constantes del contrato
#
# Todas salen del núcleo §5.5 (o de la tarea C2); lo demás (rutas, pasos, número de casos, semilla) sale de la config.

# %%
VERSION_CONTRATO = 1  # "version_contrato" de pd_model_{modelo}.json y casos_paridad_{modelo}.json (núcleo §5.5)
MODELOS = ("logistica", "mlp")  # "modelo": "mlp | logistica" (núcleo §5.5)
RUTA_CONFIG = Path("configs/scorecard/export.yaml")  # config de `python run.py export` (relativa a la raíz)
TOLERANCIA_PARIDAD = {"pd": 1e-6, "score": 1e-3}  # prueba de paridad en CI (núcleo §5.5)
PLANTILLA_MODELO = "pd_model_{modelo}.json"  # archivo del modelo, uno por modelo (núcleo §5.5, v0.17)
PLANTILLA_PARIDAD = "casos_paridad_{modelo}.json"  # archivo de casos, uno por modelo (núcleo §5.5, v0.17)

PERCENTILES = np.arange(101)  # p0 … p100: 101 valores (núcleo §5.5)
REFERENCIA_POBLACION = "test"  # "referencia" de la población (núcleo §5.5)
SALIDA_RED = "sigmoid"  # "salida" de la red (núcleo §5.5)
ACTIVACION_LINEAL = "identidad"  # activación de la última capa y de la logística (núcleo §5.5)
PENDIENTE_LEAKY = 0.01  # negative_slope por defecto de torch.nn.LeakyReLU, la que usa construir_red de src/pd/mlp.py
SPLIT_TRAIN, SPLIT_TEST = "train", "test"  # valores de la columna split (núcleo §5.1)

CLAVES_EXPORT = ("modelos", "rutas", "destino", "n_casos_paridad", "n_forzados_por_grupo", "base_seed",
                 "pasos")  # bloque export
CLAVES_RUTAS = ("logistica", "mlp", "preprocesador", "datos", "features")  # sub-bloque export.rutas

TOL_ENTERO = 1e-9  # tolerancia relativa para tratar un cociente x/paso como entero (0.1 no es exacto en binario)
COMANDO_COMMIT = ("git", "rev-parse", "HEAD")  # procedencia del export (proceso local, sin red)
COMMIT_DESCONOCIDO = "desconocido"  # si git no está o no es un repositorio
RAIZ_REPO = Path(__file__).resolve().parents[2]  # carpeta donde se pregunta a git


# %% [markdown]
# ## Nombres de los archivos de cada modelo
#
# Cada modelo $m \in$ `MODELOS` tiene sus dos archivos: `pd_model_{m}.json` y `casos_paridad_{m}.json` (núcleo §5.5,
# v0.17). Un modelo fuera de `MODELOS` no tiene nombre: `ValueError`.

# %%
def _validar_modelo(modelo, nombre: str = "modelo") -> None:
    """ValueError si ``modelo`` no es uno de ``MODELOS``.

    Parámetros:
        modelo: valor a revisar.
        nombre: cómo se llama el valor en el mensaje (p. ej. "export.modelos[1]").
    """
    if not isinstance(modelo, str) or modelo not in MODELOS:  # p. ej. "svm", None o True
        raise ValueError(f"{nombre} debe ser uno de {MODELOS}; se recibió {modelo!r}")  # opción


def archivos_modelo(modelo: str) -> tuple[str, str]:
    """Nombres de los dos archivos de ``modelo``.

    Parámetros:
        modelo: "logistica" o "mlp".

    Retorna:
        tuple[str, str]: (nombre de pd_model, nombre de casos_paridad).
    """
    _validar_modelo(modelo)  # solo los modelos del contrato
    return PLANTILLA_MODELO.format(modelo=modelo), PLANTILLA_PARIDAD.format(modelo=modelo)  # pd_model y casos


# %% [markdown]
# ## Validación de tipos
#
# Mismas reglas que el resto del proyecto: un entero es `int` (o entero de numpy) y nunca `bool` ni `1000.0`
# (`seeds.validar_entero`, cuyo `TypeError` se convierte en `ValueError`); un real es finito y no es `bool`.
#
# TODO(refactor-post-B5): `_es_real` y `_entero_config` duplican los privados de `src/pd/` y de C1 (no se importa nada
# privado de otro módulo); se unifican en el mismo refactor.

# %%
def _entero_config(valor, nombre: str, minimo: int) -> int:
    """``seeds.validar_entero`` con su TypeError convertido en ValueError.

    Parámetros:
        valor: valor de la config.
        nombre: nombre de la clave, para el mensaje.
        minimo: menor valor admitido.

    Retorna:
        int: el valor como int nativo.
    """
    try:  # regla única de enteros del proyecto
        return validar_entero(valor, nombre, minimo=minimo)  # ValueError si es < minimo
    except TypeError as exc:  # flotante, bool, texto…
        raise ValueError(str(exc)) from exc  # toda falla de la config es ValueError


def _es_real(valor) -> bool:
    """True si ``valor`` es un número real finito (los bool no cuentan).

    Parámetros:
        valor: cualquier objeto.

    Retorna:
        bool: si es un real finito que no es bool.
    """
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, Real):  # bool es subclase de int: se excluye
        return False  # texto, None, bool…
    return math.isfinite(valor)  # NaN y ±inf tampoco


def _claves_exactas(bloque, claves, nombre: str) -> None:
    """ValueError si ``bloque`` no es un diccionario con exactamente ``claves``.

    Parámetros:
        bloque: diccionario a revisar.
        claves: claves esperadas.
        nombre: nombre del bloque, para el mensaje.
    """
    if not isinstance(bloque, dict):  # p. ej. una lista pasada por error
        raise ValueError(f"{nombre} debe ser un diccionario; se recibió {type(bloque).__name__}")  # tipo
    faltan = set(claves) - set(bloque)  # sin valor por defecto
    sobran = set(bloque) - set(claves)  # una errata se ignoraría en silencio
    if faltan or sobran:  # el bloque debe ser exacto
        raise ValueError(f"{nombre} inválido: faltan {sorted(map(str, faltan))}, sobran {sorted(map(str, sobran))}")


# %% [markdown]
# ## Validación del bloque `export`
#
# | Clave | Regla |
# |---|---|
# | (bloque) | `modelos`, `rutas`, `destino`, `n_casos_paridad`, `n_forzados_por_grupo`, `base_seed`, `pasos` |
# | `modelos` | lista no vacía, sin repetidos, cada valor `logistica` o `mlp` (orden de escritura) |
# | `rutas` | exactamente `logistica`, `mlp`, `preprocesador`, `datos`, `features`; cada una texto no vacío |
# | `destino` | texto no vacío (carpeta de salida) |
# | `n_casos_paridad`, `n_forzados_por_grupo` | enteros $\ge 1$ |
# | `base_seed` | entero $\ge 0$ |
# | `pasos` | exactamente las variables **numéricas** con `en_app: true`; cada paso real finito $> 0$ |
#
# Las rutas son relativas al directorio de trabajo (`run.py` se corre desde la raíz) o absolutas. Devuelve una copia
# profunda; nunca muta la config recibida.

# %%
def _texto_no_vacio(valor, nombre: str) -> None:
    """ValueError si ``valor`` no es un texto no vacío."""
    if not isinstance(valor, str) or not valor.strip():  # None, número o ""
        raise ValueError(f"{nombre} debe ser un texto no vacío; se recibió {valor!r}")  # tipo


def validar_cfg_export(cfg_export, features) -> dict:
    """Valida el bloque ``export`` (tabla de arriba); cualquier falla → ValueError.

    Parámetros:
        cfg_export: bloque ``export`` de la config.
        features: lista de variables de ``configs/features.yaml`` (clave ``features``).

    Retorna:
        dict: copia profunda del bloque validado.
    """
    _claves_exactas(cfg_export, CLAVES_EXPORT, "El bloque export")  # primero las claves
    modelos = cfg_export["modelos"]  # modelos que se exportan, en orden
    if not isinstance(modelos, list) or not modelos:  # texto, None o lista vacía
        raise ValueError(f"export.modelos debe ser una lista no vacía; se recibió {modelos!r}")  # tipo
    for i, modelo in enumerate(modelos):  # cada valor por separado
        _validar_modelo(modelo, f"export.modelos[{i}]")  # "svm", True…; el mensaje nombra la posición
    if len(set(modelos)) != len(modelos):  # el mismo archivo se escribiría dos veces
        raise ValueError(f"export.modelos tiene repetidos: {modelos!r}")  # config
    _claves_exactas(cfg_export["rutas"], CLAVES_RUTAS, "El bloque export.rutas")  # cinco rutas exactas
    for clave, ruta in cfg_export["rutas"].items():  # cada ruta es un texto
        _texto_no_vacio(ruta, f"export.rutas.{clave}")  # ruta vacía o no textual
    _texto_no_vacio(cfg_export["destino"], "export.destino")  # carpeta de salida
    _entero_config(cfg_export["n_casos_paridad"], "export.n_casos_paridad", minimo=1)  # al menos un caso
    _entero_config(cfg_export["n_forzados_por_grupo"], "export.n_forzados_por_grupo", minimo=1)  # al menos uno
    _entero_config(cfg_export["base_seed"], "export.base_seed", minimo=0)  # semilla no negativa
    numericas = [f["nombre"] for f in features if f["en_app"] and f["tipo"] == "num"]  # las que llevan paso
    _claves_exactas(cfg_export["pasos"], numericas, "El bloque export.pasos")  # un paso por numérica del formulario
    for nombre, paso in cfg_export["pasos"].items():  # cada paso por separado
        if not _es_real(paso) or paso <= 0:  # bool, texto, NaN, ±inf, 0 o negativo
            raise ValueError(f"export.pasos.{nombre} debe ser un real finito > 0; se recibió {paso!r}")  # rango
    return copy.deepcopy(cfg_export)  # copia: no se muta la entrada


# %% [markdown]
# ## Regla del formulario: rejilla del paso (decisión de Juan José, 25-oct)
#
# Cada numérica se ofrece con un paso $h$ (config `export.pasos`) en la **rejilla anclada en 0** $\{k h : k \in
# \mathbb{Z}\}$, con redondeo **hacia afuera**:
#
# $$\text{min} = \lfloor r_{\inf}/h \rfloor\,h, \qquad \text{max} = \lceil r_{\sup}/h \rceil\,h, \qquad
# \text{valor\_defecto} = \operatorname{clip}\big(\operatorname{round}(\text{mediana}/h)\,h,\ \text{min},\
# \text{max}\big),$$
#
# donde $[r_{\inf}, r_{\sup}]$ es el recorte de train del preprocesador y la mediana es la de train.
#
# **Por qué hacia afuera:** el rango del formulario cubre todo el recorte y se pasa en menos de un paso. El tramo extra
# queda **fuera** del recorte, donde el preprocesador recorta y el modelo es **plano**; así no se pierde ningún tramo
# donde el modelo cambia (redondear hacia adentro sí lo perdería).
#
# **Artefactos de coma flotante:** $0{,}6346/0{,}1$ y $97{,}6/0{,}1 = 975{,}9999999999999$ no son exactos en binario.
# Si el cociente es entero salvo una tolerancia relativa `TOL_ENTERO`, se toma ese entero (no se baja ni se sube un
# paso de más); el resultado se redondea a los decimales del paso ($0{,}6000000000000001 \to 0{,}6$). Un empate de la
# mediana en $k h + h/2$ (con la misma tolerancia: $0{,}15/0{,}1 = 1{,}4999999999999998$ cuenta como $1{,}5$) se
# redondea hacia arriba ($0{,}15 \to 0{,}2$).

# %%
def _decimales(paso) -> int:
    """Decimales del paso tal como se escribió (0.1 → 1, 25 → 0, 1e-05 → 5).

    Parámetros:
        paso: real > 0.

    Retorna:
        int: número de decimales (≥ 0).
    """
    return max(0, -Decimal(str(paso)).as_tuple().exponent)  # exponente decimal de su representación más corta


def _multiplo(x: float, paso, modo: str):
    """Múltiplo de ``paso`` según ``modo`` ("abajo", "arriba" o "cercano"), redondeado a los decimales del paso.

    Parámetros:
        x: valor real.
        paso: real > 0.
        modo: "abajo" (floor), "arriba" (ceil) o "cercano" (empate hacia arriba).

    Retorna:
        int o float: k · paso (int si el paso es entero).
    """
    q = x / paso  # posición en la rejilla
    k = round(q)  # entero más cercano
    if abs(q - k) <= TOL_ENTERO * max(1.0, abs(q)):  # ya es múltiplo salvo coma flotante
        n = k  # no se baja ni se sube un paso de más
    elif modo == "abajo":  # min del formulario
        n = math.floor(q)  # hacia afuera por abajo
    elif modo == "arriba":  # max del formulario
        n = math.ceil(q)  # hacia afuera por arriba
    elif abs(q - math.floor(q) - 0.5) <= TOL_ENTERO * max(1.0, abs(q)):  # empate k + 1/2 salvo coma flotante
        n = math.floor(q) + 1  # empate hacia arriba
    else:  # valor por defecto sin empate
        n = math.floor(q + 0.5)  # múltiplo más cercano
    return round(n * paso, _decimales(paso))  # sin artefactos (0.6000000000000001 → 0.6)


def _entrada_num(f: dict, v: dict, paso) -> dict:
    """Entrada numérica del formulario (regla de arriba).

    Parámetros:
        f: variable de features.yaml.
        v: variable del preprocesador (to_dict()).
        paso: paso de la config.

    Retorna:
        dict: {nombre, tipo, etiqueta_es, ayuda_es, min, max, paso, valor_defecto}.
    """
    lo, hi = v["recorte"]  # recorte de train
    minimo, maximo = _multiplo(lo, paso, "abajo"), _multiplo(hi, paso, "arriba")  # hacia afuera
    defecto = min(max(_multiplo(v["mediana"], paso, "cercano"), minimo), maximo)  # mediana de train, acotada
    return {"nombre": f["nombre"], "tipo": "num", "etiqueta_es": f["etiqueta_es"], "ayuda_es": f["ayuda_es"],  # textos
            "min": minimo, "max": maximo, "paso": paso, "valor_defecto": defecto}  # rejilla


# %% [markdown]
# ## Limpieza de valores crudos (núcleo §5.2)
#
# Reglas comunes al formulario, a la receta de la app y a la elección de casos:
#
# - **ausente** (`_es_vacio`): `None`, NaN o `pd.NA` (en TypeScript, `null` o `undefined`). Un **texto** nunca es
#   ausente: `"nan"` es un valor real;
# - **categórica**: ausente → `vacio`; si no, el valor se convierte a texto (36 → `"36"`), se le quitan los espacios
#   al inicio y al final (`limpieza: "strip"`) y el texto vacío → `vacio` (`"DESCONOCIDO"`). Que el valor esté o no
#   en las categorías de train (→ `OTRA`) se decide después;
# - **numérica**: ausente → NaN (luego se imputa la mediana); un texto se lee como número tras quitarle los espacios
#   (`" 10000 "` → 10000) y el texto vacío o solo espacios → NaN; un texto que no es número → `ValueError`.
#
# Para un valor categórico $x$: `vacio` si $x$ es ausente; si no, $t = $ `str(x).strip()` y el resultado es $t$, o
# `vacio` si $t = $ `""`. Es la regla que se porta a TypeScript; aquí `_textos` y `_limpiar_serie` la aplican a una
# columna entera de forma vectorizada (los 74 154 créditos de test). Solo se conoce `limpieza: "strip"`
# (`_validar_limpieza`, una vez por llamada pública).

# %%
def _es_vacio(x):
    """True donde ``x`` es ausente: None, NaN o pd.NA (un texto nunca lo es).

    Parámetros:
        x: escalar o array de objetos.

    Retorna:
        bool o np.ndarray de bool: ``pd.isna`` elemento a elemento.
    """
    return pd.isna(x)  # None, NaN, pd.NA y NaT; "nan" (texto) no


def _validar_limpieza(limpieza: dict) -> None:
    """ValueError si el bloque ``categoricas`` declara una limpieza distinta de "strip"."""
    if limpieza["limpieza"] != "strip":  # la receta solo conoce strip
        raise ValueError(f"Limpieza de categóricas desconocida: {limpieza['limpieza']!r}")  # contrato §5.2


def _textos(serie: pd.Series) -> tuple[np.ndarray, pd.Series]:
    """(máscara de vacíos, textos sin espacios) de una columna; vacío = ausente o texto vacío tras strip."""
    valores = serie.to_numpy(dtype=object)  # objetos de Python (NaN de pandas incluidos)
    ausente = np.asarray(_es_vacio(valores), dtype=bool)  # None, NaN, pd.NA
    textos = pd.Series(valores, dtype=object).where(~ausente, "").astype(str).str.strip()  # str(x).strip()
    return ausente | (textos == "").to_numpy(), textos  # vacío y textos


def _limpiar_serie(serie: pd.Series, limpieza: dict) -> pd.Series:
    """Limpieza de categóricas (regla de arriba) aplicada a toda una columna, vectorizada.

    Parámetros:
        serie: valores crudos.
        limpieza: bloque ``categoricas`` de to_dict() (ya validado).

    Retorna:
        pd.Series: textos limpios (object), con el índice de ``serie``.
    """
    vacio, textos = _textos(serie)  # ausente, "" o solo espacios; textos sin espacios
    limpio = np.where(vacio, limpieza["vacio"], textos.to_numpy(dtype=object))  # vacío → DESCONOCIDO
    return pd.Series(limpio, index=serie.index, dtype=object)  # textos limpios


def _a_float64(serie: pd.Series) -> np.ndarray:
    """Columna numérica cruda → float64 con NaN en los vacíos (ausente, "" o solo espacios).

    Parámetros:
        serie: valores crudos.

    Retorna:
        np.ndarray: float64 (n,); ValueError si un texto no es un número.
    """
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):  # ya numérica
        return serie.to_numpy(dtype=np.float64, na_value=np.nan)  # NaN = vacío
    vacio, textos = _textos(serie)  # columna de objetos (p. ej. desde list[dict])
    try:  # " 10000 " → 10000; "abc" → error
        numeros = pd.to_numeric(textos.where(~vacio, "0"), errors="raise")  # "0" provisional: los vacíos van a NaN abajo
    except (TypeError, ValueError) as exc:  # texto que no es número
        raise ValueError(f"{serie.name}: hay un valor que no es un número") from exc  # dato
    return np.where(vacio, np.nan, numeros.to_numpy(dtype=np.float64))  # vacíos → NaN


# %% [markdown]
# ## Entradas categóricas del formulario
#
# - `categorias`: las del preprocesador **sin** `OTRA` (es interna: la app no la ofrece), en el **orden** de
#   `categorias_es` de `features.yaml` (orden pensado para el usuario, no alfabético);
# - `etiquetas_categorias`: `categorias_es` tal cual (núcleo §5.5, v0.12);
# - `valor_defecto`: la categoría **más frecuente en train** tras la limpieza del núcleo §5.2 (strip, vacío →
#   `DESCONOCIDO`); con empate, la primera de ellas en el orden de `categorias_es` (determinista).
#
# Si el conjunto de `categorias_es` no es exactamente el del preprocesador menos `OTRA`, la app ofrecería una categoría
# que el modelo no conoce o escondería una que sí: `ValueError`.

# %%
def _entrada_cat(f: dict, v: dict, df_train: pd.DataFrame, limpieza: dict) -> dict:
    """Entrada categórica del formulario (reglas de arriba).

    Parámetros:
        f: variable de features.yaml.
        v: variable del preprocesador (to_dict()).
        df_train: filas de train (crudas).
        limpieza: bloque ``categoricas`` de to_dict().

    Retorna:
        dict: {nombre, tipo, etiqueta_es, ayuda_es, categorias, valor_defecto, etiquetas_categorias}.
    """
    nombre, etiquetas = f["nombre"], f.get("categorias_es") or {}  # traducciones de la app
    del_pre = {c for c in v["categorias"] if c != limpieza["no_vista"]}  # las de train, sin OTRA
    if set(etiquetas) != del_pre:  # la app y el modelo deben ofrecer lo mismo
        raise ValueError(f"{nombre}: categorias_es {sorted(etiquetas)} no coincide con el preprocesador sin "
                         f"{limpieza['no_vista']} {sorted(del_pre)}")  # incoherencia de features.yaml
    conteos = _limpiar_serie(df_train[nombre], limpieza).value_counts()  # frecuencias en train
    empatadas = [c for c in etiquetas if conteos.get(c, 0) == conteos.max()]  # las más frecuentes
    if not empatadas:  # train sin ninguna categoría del formulario
        raise ValueError(f"{nombre}: ninguna categoría de categorias_es aparece en train")  # dato
    moda = empatadas[0]  # empate → la primera en el orden de categorias_es (determinista)
    return {"nombre": nombre, "tipo": "cat", "etiqueta_es": f["etiqueta_es"], "ayuda_es": f["ayuda_es"],  # textos
            "categorias": list(etiquetas), "valor_defecto": str(moda), "etiquetas_categorias": dict(etiquetas)}


# %% [markdown]
# ## Formulario completo (`entradas`)
#
# Solo las variables con `en_app: true`, en el **orden** de `features.yaml` (orden del formulario, núcleo §5.5).
# Los estadísticos vienen del preprocesador (ajustado solo con train) y de `df_train`: nada de test (núcleo §5.7).

# %%
def _variables_por_nombre(pre_json: dict) -> dict:
    """{nombre: variable} del to_dict() del preprocesador."""
    return {v["nombre"]: v for v in pre_json["variables"]}  # acceso por nombre


def entradas_app(features, pre_json: dict, df_train: pd.DataFrame, pasos: dict) -> list[dict]:
    """Entradas del formulario de la app (núcleo §5.5).

    Parámetros:
        features: lista de variables de features.yaml.
        pre_json: Preprocesador.to_dict().
        df_train: filas de train, crudas.
        pasos: {numérica en_app: paso}.

    Retorna:
        list[dict]: una entrada por variable en_app, en el orden de features.yaml.
    """
    _validar_limpieza(pre_json["categoricas"])  # una vez por llamada
    variables = _variables_por_nombre(pre_json)  # estadísticos de train
    entradas = []  # en el orden de features.yaml
    for f in (f for f in features if f["en_app"]):  # solo las del formulario
        if f["nombre"] not in variables:  # una variable de la app que el modelo no usa
            raise ValueError(f"{f['nombre']} está en la app pero no en el preprocesador")  # incoherencia
        v = variables[f["nombre"]]  # su bloque del preprocesador
        if f["tipo"] == "num":  # numérica: rejilla del paso
            if f["nombre"] not in pasos:  # sin paso no hay rejilla
                raise ValueError(f"Falta el paso de {f['nombre']} en export.pasos")  # config incompleta
            entradas.append(_entrada_num(f, v, pasos[f["nombre"]]))  # min, max, defecto
        else:  # categórica
            entradas.append(_entrada_cat(f, v, df_train, pre_json["categoricas"]))  # categorías y moda
    return entradas  # lista del formulario


# %% [markdown]
# ## Pesos de la red (`red.capas`)
#
# Cada capa es $z = W h + b$ con $W$ de forma **salida × entrada** (igual que `torch.nn.Linear.weight`) y una
# activación. **Logística:** una sola capa $1 \times d$ con $W = w^\top$, $b$ el intercepto y activación `identidad`
# (núcleo §5.5). **MLP:** una capa por `nn.Linear`, en orden; las ocultas con la activación de la config y la última
# con `identidad` (la red devuelve el logit, núcleo §5.3). El `Dropout` no se exporta: en inferencia es la identidad.
#
# `capas_mlp` lee el `.pt` con `torch.load(weights_only=True)` (formato de `MLPPD.guardar`: `{state_dict, config,
# n_entradas}`), reconstruye la red con `construir_red` de `src/pd/mlp.py` y carga los pesos; devuelve también
# `MLPPD.cargar(ruta)` para predecir por el camino de Python.

# %%
def capas_logistica(modelo: LogisticaPD) -> list[dict]:
    """Una capa ``identidad`` con los coeficientes de la logística.

    Parámetros:
        modelo: LogisticaPD entrenada o cargada.

    Retorna:
        list[dict]: [{"W": [coef] (1 × d), "b": [intercepto], "activacion": "identidad"}].
    """
    w, b = modelo.coeficientes()  # copias en float64
    return [{"W": [w.tolist()], "b": [b], "activacion": ACTIVACION_LINEAL}]  # salida × entrada = 1 × d


def capas_mlp(ruta) -> tuple[list[dict], MLPPD]:
    """Capas del MLP guardado en ``ruta`` y el modelo cargado.

    Parámetros:
        ruta: archivo .pt de MLPPD.guardar.

    Retorna:
        tuple[list[dict], MLPPD]: capas del JSON y el modelo para predecir.
    """
    paquete = torch.load(Path(ruta), map_location="cpu", weights_only=True)  # solo tensores y tipos simples
    cfg_mlp = validar_cfg_mlp(paquete["config"])  # bloque mlp de entrenamiento
    red = construir_red(int(paquete["n_entradas"]), cfg_mlp["capas"], cfg_mlp["activacion"], cfg_mlp["dropout"])  # pila
    red.load_state_dict(paquete["state_dict"])  # falla si la arquitectura no coincide
    lineales = [m for m in red if isinstance(m, nn.Linear)]  # una por capa, en orden (Dropout se omite)
    capas = []  # capas del JSON
    for i, lineal in enumerate(lineales):  # oculta, oculta, …, salida
        activacion = ACTIVACION_LINEAL if i == len(lineales) - 1 else cfg_mlp["activacion"]  # la salida es el logit
        capas.append({"W": lineal.weight.detach().double().numpy().tolist(),  # salida × entrada
                      "b": lineal.bias.detach().double().numpy().tolist(),  # un sesgo por salida
                      "activacion": activacion})  # nombre de la config
    return capas, MLPPD.cargar(ruta)  # el modelo predice por el camino de Python


# %% [markdown]
# ## Población, métricas y procedencia
#
# - `percentiles_score`: los 101 percentiles $p_0, \dots, p_{100}$ del puntaje con interpolación **lineal**
#   (`np.percentile`, método por defecto); la app ubica a la persona entre ellos (R-APP-3).
# - `metricas_test`: AUC, KS y Brier de `src/common/metricas.py` (las mismas de la Parte B).
# - `commit_actual`: el `HEAD` de git **al exportar**. El commit que contendrá el JSON es posterior (aún no existe al
#   exportar), así que `commit` identifica el código que lo generó. `generado` (fecha y hora UTC) cambia en cada
#   export, de modo que dos exports idénticos difieren en ese campo.

# %%
def percentiles_score(score) -> list[float]:
    """Percentiles 0…100 (lineales) del puntaje.

    Parámetros:
        score: puntajes (array no vacío y finito).

    Retorna:
        list[float]: 101 valores no decrecientes.
    """
    s = np.asarray(score, dtype=np.float64)  # float64
    if s.size == 0 or not np.all(np.isfinite(s)):  # sin población o con NaN
        raise ValueError("score debe ser un array no vacío y finito")  # dato
    return [float(x) for x in np.percentile(s, PERCENTILES)]  # interpolación lineal (por defecto)


def metricas_test(y, p) -> dict:
    """AUC, KS y Brier de test con las métricas comunes.

    Parámetros:
        y: objetivo 0/1.
        p: PD.

    Retorna:
        dict: {"auc_test", "ks_test", "brier_test"} en float.
    """
    return {"auc_test": float(auc(y, p)), "ks_test": float(ks(y, p)), "brier_test": float(brier(y, p))}  # B1


def commit_actual() -> str:
    """SHA de ``git rev-parse HEAD`` o "desconocido".

    Retorna:
        str: sha de 40 caracteres o "desconocido".
    """
    try:  # git puede no estar instalado
        r = subprocess.run(COMANDO_COMMIT, cwd=RAIZ_REPO, capture_output=True, text=True, check=False)  # local
    except OSError:  # sin git
        return COMMIT_DESCONOCIDO  # procedencia desconocida
    sha = r.stdout.strip()  # sin salto de línea
    return sha if r.returncode == 0 and sha else COMMIT_DESCONOCIDO  # fuera de un repositorio


# %% [markdown]
# ## Receta de la app: preprocesamiento desde el JSON
#
# **Guía para portar a TypeScript.** Lee **solo** el diccionario `preprocesamiento` (= `Preprocesador.to_dict()`); no
# usa la clase `Preprocesador`. Para cada variable de `variables`, en orden:
#
# - **categórica** (bloque `categoricas`, reglas de la celda "Limpieza de valores crudos"): ausente → `vacio`; si no,
#   a texto y sin espacios al inicio y al final (`" 36 months"` → `"36 months"`), texto vacío → `vacio`
#   (`"DESCONOCIDO"`); si el resultado no está en `categorias` → `no_vista` (`"OTRA"`); one-hot en el orden de
#   `categorias`;
# - **numérica**: vacío (ausente, `""` o solo espacios) → NaN; después, en el orden de `orden_numericas`:
#   `imputar_mediana` (NaN → `mediana`), `recortar` a `recorte`, `log1p_si_aplica` ($\ln(1 + x)$ si `log1p`),
#   `estandarizar` $(x - \text{media})/\text{desv}$.
#
# Los nombres de las columnas generadas (`nombre` en las numéricas, `nombre=categoría` en las categóricas) deben ser
# **exactamente** `columnas_salida`, en su orden; si no, `ValueError`. **Todo en float64.** El preprocesador de Python
# entrega float32 (núcleo §5.2), así que el modelo de Python y la receta difieren en el redondeo de float32 (del orden
# de $10^{-7}$ en $X$ y $10^{-8}$ en la PD), muy por debajo de la tolerancia de paridad.
#
# **Trampas al portar a TypeScript** (los casos de paridad salen de datos reales y **no** las detectan; las cubren las
# pruebas `test_07d2` y `test_07d3`):
#
# 1. **Numérica vacía:** `null`, `undefined`, `""` y `"   "` → mediana. Nunca `Number("")` ni `Number("   ")`, que en
#    JavaScript valen **0**; un texto con espacios alrededor de un número (`" 10000 "`) es ese número.
# 2. **Categórica ausente:** solo `null`/`undefined` (y el texto vacío o de solo espacios) son vacío. El texto `"nan"`
#    es un valor real (→ `OTRA`). No usar `String(null)` ni `String(undefined)`: darían `"null"`/`"undefined"` → `OTRA`
#    en vez de `DESCONOCIDO`.
# 3. **Paso a paso:** vacío → `DESCONOCIDO` y, **después**, si `DESCONOCIDO` no se vio en train (p. ej. en `purpose`)
#    → `OTRA`. No saltar el segundo paso.
# 4. **No textos:** un valor que no es texto en una categórica se convierte a texto antes de comparar (36 → `"36"` →
#    `OTRA`).
# 5. **Sigmoide y puntaje:** PD $= 1/(1 + e^{-z})$ con `Math.exp` en float64; el puntaje se calcula **desde el logit**
#    $z$ (score = offset − factor · z), **no** desde la PD: con PD cerca de 0 o 1, volver al logit desde la PD pierde
#    precisión.
# 6. **Espacios:** el `strip()` de Python quita los espacios Unicode según `str.isspace` (comprobado: quita
#    U+00A0, U+3000 y los separadores `\x1c`–`\x1f`, pero **no** U+FEFF); el `trim()` de JavaScript sigue la definición
#    de espacio de ECMAScript, que incluye U+FEFF y no incluye `\x1c`–`\x1f` [VERIFICAR en la app]. Con los datos
#    actuales no aparece ninguno de esos caracteres; si apareciera, la app debe imitar `str.isspace`.

# %%
def _imputar(x: np.ndarray, v: dict) -> np.ndarray:
    """Vacío (NaN) → mediana de train."""
    return np.where(np.isnan(x), v["mediana"], x)  # imputación por la mediana


def _recortar(x: np.ndarray, v: dict) -> np.ndarray:
    """Recorte a los percentiles de train."""
    return np.clip(x, v["recorte"][0], v["recorte"][1])  # [p_inf, p_sup]


def _log1p_si_aplica(x: np.ndarray, v: dict) -> np.ndarray:
    """ln(1 + x) si la variable lo declara."""
    return np.log1p(x) if v["log1p"] else x  # transformacion: log1p


def _estandarizar(x: np.ndarray, v: dict) -> np.ndarray:
    """(x − media) / desv de train."""
    return (x - v["media"]) / v["desv"]  # estandarización


OPERACIONES_NUMERICAS = {"imputar_mediana": _imputar, "recortar": _recortar, "log1p_si_aplica": _log1p_si_aplica,
                         "estandarizar": _estandarizar}  # nombre de orden_numericas → operación


def _como_dataframe(entradas) -> pd.DataFrame:
    """DataFrame o lista de diccionarios de entradas crudas → DataFrame."""
    if isinstance(entradas, pd.DataFrame):  # ya es tabla
        return entradas  # sin copiar
    if not isinstance(entradas, list) or not all(isinstance(e, dict) for e in entradas):  # otro tipo
        raise ValueError("entradas debe ser un DataFrame o una lista de diccionarios")  # tipo
    return pd.DataFrame(entradas)  # una fila por diccionario (claves faltantes → NaN; se comprueban abajo)


def _columna(df: pd.DataFrame, entradas, nombre: str) -> pd.Series:
    """Columna ``nombre``; ValueError si falta (en la tabla o en algún diccionario)."""
    falta = nombre not in df.columns or (isinstance(entradas, list) and any(nombre not in e for e in entradas))  # ausente
    if falta:  # la receta no inventa valores
        raise ValueError(f"Falta la entrada {nombre!r}")  # entrada incompleta
    return df[nombre]  # valores crudos


def _numerica(crudo: pd.Series, v: dict, orden: list) -> np.ndarray:
    """Una numérica: vacío → NaN y las operaciones de ``orden`` (orden_numericas)."""
    x = _a_float64(crudo)  # vacíos → NaN
    for operacion in orden:  # orden declarado en to_dict()
        if operacion not in OPERACIONES_NUMERICAS:  # operación que la receta no conoce
            raise ValueError(f"Operación numérica desconocida: {operacion!r}")  # contrato §5.2
        x = OPERACIONES_NUMERICAS[operacion](x, v)  # siguiente paso
    return x[:, None]  # una columna


def _categorica(crudo: pd.Series, v: dict, limpieza: dict) -> np.ndarray:
    """Una categórica: limpieza, OTRA si no se vio en train y one-hot en el orden de ``categorias``."""
    cats = _limpiar_serie(crudo, limpieza).to_numpy(dtype=object)  # vacío → DESCONOCIDO
    cats = np.where(np.isin(cats, v["categorias"]), cats, limpieza["no_vista"])  # no vista → OTRA
    return (cats[:, None] == np.array(v["categorias"], dtype=object)[None, :]).astype(np.float64)  # one-hot


def preprocesar_desde_json(pre_json: dict, entradas) -> np.ndarray:
    """Receta de la app: entradas crudas → matriz preprocesada (float64), leyendo solo ``pre_json``.

    Parámetros:
        pre_json: Preprocesador.to_dict().
        entradas: DataFrame o list[dict] de entradas crudas.

    Retorna:
        np.ndarray: float64 (n, len(columnas_salida)).
    """
    if pre_json.get("version") != VERSION_PREPROCESADOR:  # otra versión del contrato §5.2
        raise ValueError(f"Versión de preprocesador {pre_json.get('version')!r} no soportada")  # contrato
    limpieza = pre_json["categoricas"]  # strip, vacío y no vista
    _validar_limpieza(limpieza)  # una vez por llamada
    df = _como_dataframe(entradas)  # tabla de entradas
    columnas, nombres = [], []  # bloques de columnas y sus nombres, en el orden de variables
    for v in pre_json["variables"]:  # orden de to_dict()
        crudo = _columna(df, entradas, v["nombre"])  # valores crudos
        if v["tipo"] == "num":  # numérica
            columnas.append(_numerica(crudo, v, pre_json["orden_numericas"]))  # una columna
            nombres.append(v["nombre"])  # nombre de la columna
        else:  # categórica
            columnas.append(_categorica(crudo, v, limpieza))  # una columna por categoría
            nombres.extend(f"{v['nombre']}={c}" for c in v["categorias"])  # nombre=categoría
    if nombres != list(pre_json["columnas_salida"]):  # to_dict() incoherente (nombres u orden)
        raise ValueError(f"Las columnas generadas no coinciden con columnas_salida: {nombres} != "
                         f"{pre_json['columnas_salida']}")  # contrato
    return np.hstack(columnas).astype(np.float64) if columnas else np.zeros((len(df), 0))  # (n, d) en float64


# %% [markdown]
# ## Receta de la app: red, PD y puntaje
#
# Para cada capa, $h \leftarrow \varphi(W h + b)$ con $W$ salida × entrada (en filas: $H \leftarrow
# \varphi(H W^\top + b)$). Activaciones: `relu` $\max(0, a)$; `leaky_relu` $a$ si $a \ge 0$ y $0{,}01\,a$ si $a < 0$
# (`negative_slope` por defecto de `torch.nn.LeakyReLU`, la de `construir_red`); `tanh`; `identidad`. La última capa
# tiene una salida: el **logit** $z$. Después:
#
# $$PD = \sigma(z) = \frac{1}{1 + e^{-z}} \ (\text{salida } \texttt{sigmoid}, \text{ estable con } \texttt{expit}),
# \qquad \text{score} = \text{offset} - \text{factor}\cdot z$$
#
# (`score_desde_logit` de C1 con el bloque `scorecard` del JSON). Todo en float64.

# %%
ACTIVACIONES_JSON = {  # nombre del JSON → función elemento a elemento (float64)
    "relu": lambda a: np.maximum(a, 0.0),  # max(0, a)
    "leaky_relu": lambda a: np.where(a >= 0, a, PENDIENTE_LEAKY * a),  # pendiente 0.01 si a < 0
    "tanh": np.tanh,  # tangente hiperbólica
    ACTIVACION_LINEAL: lambda a: a,  # identidad
}


def forward_desde_json(red_json: dict, X) -> np.ndarray:
    """Logit de cada fila con las capas del JSON.

    Parámetros:
        red_json: bloque ``red`` ({capas, salida}).
        X: matriz preprocesada (n, d).

    Retorna:
        np.ndarray: logit float64 (n,).
    """
    h = np.asarray(X, dtype=np.float64)  # entrada en float64
    for i, capa in enumerate(red_json["capas"]):  # en orden
        W = np.asarray(capa["W"], dtype=np.float64)  # salida × entrada
        b = np.asarray(capa["b"], dtype=np.float64)  # un sesgo por salida
        if W.ndim != 2 or W.shape[1] != h.shape[1] or b.shape != (W.shape[0],):  # dimensiones encadenadas
            raise ValueError(f"Capa {i}: W {W.shape} y b {b.shape} no encajan con una entrada de {h.shape[1]}")
        if capa["activacion"] not in ACTIVACIONES_JSON:  # activación desconocida
            raise ValueError(f"Capa {i}: activación desconocida {capa['activacion']!r}")  # contrato
        h = ACTIVACIONES_JSON[capa["activacion"]](h @ W.T + b)  # φ(W h + b)
    if h.shape[1] != 1:  # la red debe terminar en el logit
        raise ValueError(f"La última capa debe tener una salida; tiene {h.shape[1]}")  # contrato
    return h[:, 0]  # (n,)


def inferir_desde_json(modelo_json: dict, entradas) -> tuple[np.ndarray, np.ndarray]:
    """PD y puntaje desde entradas crudas leyendo solo ``modelo_json`` (la receta de la app).

    Parámetros:
        modelo_json: pd_model_{modelo}.json (dict).
        entradas: DataFrame o list[dict] de entradas crudas.

    Retorna:
        tuple[np.ndarray, np.ndarray]: (PD, puntaje), float64 (n,).
    """
    if modelo_json["red"]["salida"] != SALIDA_RED:  # la receta solo conoce la sigmoide
        raise ValueError(f"Salida de la red desconocida: {modelo_json['red']['salida']!r}")  # contrato
    X = preprocesar_desde_json(modelo_json["preprocesamiento"], entradas)  # float64
    z = forward_desde_json(modelo_json["red"], X)  # logit
    return expit(z), np.asarray(score_desde_logit(z, modelo_json["scorecard"]), dtype=np.float64)  # PD y puntaje


# %% [markdown]
# ## Casos de paridad: elección de filas
#
# 1 000 filas de **test** (núcleo §5.5). Para que la prueba de paridad recorra los caminos difíciles de la receta, se
# fuerzan primero hasta `n_forzados_por_grupo` filas (al azar dentro del grupo) de cada grupo no vacío, en el orden
# de `variables`:
#
# - numéricas: vacías, por debajo de `recorte[0]` y por encima de `recorte[1]`;
# - categóricas: vacías (NA o solo espacios) y no vistas en train (→ `OTRA`).
#
# El resto se completa al azar sin repetir hasta `n`. Todo con `make_rng(seed, 0)`: misma semilla → mismas filas.
# Si los forzados ya superan `n`, o el test tiene menos de `n` filas, `ValueError`.

# %%
def _grupos(df_test: pd.DataFrame, pre_json: dict) -> list[np.ndarray]:
    """Posiciones (no vacías) de cada grupo forzado."""
    limpieza = pre_json["categoricas"]  # reglas de limpieza (validadas en elegir_casos)
    grupos = []  # en el orden de variables
    for v in pre_json["variables"]:  # cada variable
        crudo = df_test[v["nombre"]]  # valores crudos
        if v["tipo"] == "num":  # numérica
            x = _a_float64(crudo)  # NaN = vacío (misma regla que la receta)
            mascaras = [np.isnan(x), x < v["recorte"][0], x > v["recorte"][1]]  # vacías, bajo, alto
        else:  # categórica
            vacio, _ = _textos(crudo)  # ausente, "" o solo espacios
            limpio = _limpiar_serie(crudo, limpieza).to_numpy()  # vacío → DESCONOCIDO
            mascaras = [vacio, ~np.isin(limpio, v["categorias"])]  # vacías y no vistas
        grupos.extend(np.flatnonzero(m) for m in mascaras if m.any())  # solo los grupos con filas
    return grupos  # posiciones por grupo


def elegir_casos(df_test: pd.DataFrame, pre_json: dict, n: int, n_por_grupo: int, seed: int) -> pd.DataFrame:
    """Filas de test para los casos de paridad (reglas de arriba).

    Parámetros:
        df_test: filas de test crudas (con ``id``).
        pre_json: Preprocesador.to_dict().
        n: número de casos.
        n_por_grupo: máximo de filas forzadas por grupo.
        seed: semilla (make_rng(seed, 0)).

    Retorna:
        pd.DataFrame: n filas de df_test, sin repetir, forzadas primero.
    """
    n = _entero_config(n, "n", minimo=1)  # al menos un caso
    n_por_grupo = _entero_config(n_por_grupo, "n_por_grupo", minimo=1)  # al menos uno por grupo
    if len(df_test) < n:  # no hay filas suficientes
        raise ValueError(f"El test tiene {len(df_test)} filas; se piden {n} casos")  # dato
    _validar_limpieza(pre_json["categoricas"])  # una vez por llamada
    rng = make_rng(seed, 0)  # regla única de semillas
    elegidas = []  # posiciones en orden de elección
    for posiciones in _grupos(df_test, pre_json):  # cada grupo forzado
        libres = np.setdiff1d(posiciones, elegidas)  # sin repetir filas de otros grupos
        if libres.size == 0:  # ya cubierto por otro grupo
            continue  # siguiente grupo
        k = min(n_por_grupo, libres.size)  # hasta n_por_grupo
        tomadas = rng.choice(libres, size=k, replace=False)  # al azar dentro del grupo
        elegidas.extend(int(p) for p in tomadas)  # en orden de elección (libres ya excluye las elegidas)
    if len(elegidas) > n:  # los forzados no caben
        raise ValueError(f"Hay {len(elegidas)} filas forzadas y solo {n} casos")  # config incoherente
    resto = np.setdiff1d(np.arange(len(df_test)), elegidas)  # filas no elegidas
    relleno = rng.choice(resto, size=n - len(elegidas), replace=False)  # al azar sin repetir
    return df_test.iloc[elegidas + [int(p) for p in relleno]]  # forzadas primero


# %% [markdown]
# ## Casos de paridad: contenido
#
# Cada caso guarda el `id`, las entradas **crudas** de las variables del formulario (exactamente las `entradas` del
# modelo; NaN → `null`; textos **sin limpiar**, p. ej. `" 36 months"`, para que la app pruebe también la limpieza) y la
# PD y el puntaje de la receta (`inferir_desde_json`) calculados sobre esas mismas entradas. La tolerancia es la del
# núcleo §5.5.

# %%
def _crudo(x):
    """Valor crudo → tipo nativo de JSON (NaN/None/NA → None)."""
    if _es_vacio(x):  # None, NaN, pd.NA (un texto nunca)
        return None  # null en el JSON
    return x.item() if isinstance(x, np.generic) else x  # numpy → nativo; texto tal cual


def construir_casos_paridad(modelo_json: dict, df_casos: pd.DataFrame) -> dict:
    """Contenido de casos_paridad_{modelo}.json (el modelo es el de ``modelo_json``).

    Parámetros:
        modelo_json: pd_model_{modelo}.json (dict) del mismo modelo.
        df_casos: filas elegidas (elegir_casos).

    Retorna:
        dict: {version_contrato, modelo, tolerancia, casos: [{id, entradas, pd, score}]}.
    """
    nombres = [e["nombre"] for e in modelo_json["entradas"]]  # variables del formulario
    entradas = [{nombre: _crudo(fila[nombre]) for nombre in nombres}  # crudas, una por caso
                for fila in df_casos[nombres].to_dict(orient="records")]  # filas como diccionarios
    p, score = inferir_desde_json(modelo_json, entradas)  # la receta sobre lo mismo que verá la app
    casos = [{"id": str(i), "entradas": e, "pd": float(pi), "score": float(si)}  # una fila del archivo
             for i, e, pi, si in zip(df_casos["id"], entradas, p, score)]  # en el orden elegido
    return {"version_contrato": VERSION_CONTRATO, "modelo": modelo_json["modelo"],  # cabecera
            "tolerancia": dict(TOLERANCIA_PARIDAD), "casos": casos}  # tolerancia del núcleo §5.5


# %% [markdown]
# ## `pd_model_{modelo}.json` completo
#
# La carga se parte en dos para no repetir trabajo al exportar los dos modelos:
#
# 1. **Lo común** (`_cargar`), una sola vez: valida la config (`export` y `scorecard`) y carga `features.yaml`, el
#    preprocesador, los datos, el formulario (`entradas`, con train), los parámetros de la scorecard de C1 y
#    $X_{test}$ = `pre.transform(test)` (float32, el camino de Python de la Parte B). Es lo que el núcleo §5.5 pide
#    **igual** en los dos archivos (`entradas`, `preprocesamiento`, `scorecard`).
# 2. **Lo de cada modelo** (`_cargar_modelo` y `_pd_model`): el modelo de `export.rutas.{modelo}`, sus capas y, sobre
#    el mismo $X_{test}$, **una sola pasada** de la red: logit $z$ = `predict_logit`, PD $= \sigma(z)$ = `expit(z)` y
#    puntaje = `score_desde_logit(z)` → sus percentiles y sus métricas. $\sigma(z)$ es lo mismo que calculan los dos
#    `predict_proba`: el MLP aplica `expit` al logit, y en la logística `LogisticRegression.predict_proba` de
#    scikit-learn (1.9.1, el fijado), con dos clases, es `expit(decision_function(X))` (`_predict_proba_lr`,
#    comprobado en el código fuente instalado). Así `metricas` coincide con la Parte B: la logística con
#    `results/data/pd/summary.json` y el MLP (semilla mediana, `models/pd_mlp.pt`) con su
#    `results/data/pd/semillas/mlp_seed{s}.json`. En el MLP la AUC puede diferir en unos pocos pares (malo, bueno) cuyo
#    orden cambia por el redondeo de float32 (medido el 26-oct: $3{,}8 \times 10^{-9}$, cuando un par vale
#    $1/(n_{malos}\, n_{buenos}) \approx 9{,}4 \times 10^{-10}$); por eso la prueba admite $10^{-8}$ en esa métrica.

# %%
def _cargar(cfg: dict) -> dict:
    """Config validada y todo lo común a los dos modelos.

    Parámetros:
        cfg: config completa (bloques scorecard y export).

    Retorna:
        dict: contexto {exp, features, params, pre, pre_json, entradas, train, test, X_te}.
    """
    if not isinstance(cfg, dict) or "export" not in cfg:  # sin bloque export
        raise ValueError("La config debe ser un diccionario con el bloque 'export'")  # config
    rutas = cfg["export"].get("rutas", {}) if isinstance(cfg["export"], dict) else {}  # para leer features
    if not isinstance(rutas, dict) or "features" not in rutas:  # sin ruta de features no hay validación
        raise ValueError("Falta export.rutas.features")  # config
    features = load_config(rutas["features"])["features"]  # variables del modelo y de la app
    exp = validar_cfg_export(cfg["export"], features)  # bloque validado
    params = parametros_scorecard(bloque_scorecard(cfg))  # base, odds_base, pdo, factor, offset
    pre = Preprocesador.from_dict(load_json(exp["rutas"]["preprocesador"]))  # ajustado solo con train (A)
    pre_json = pre.to_dict()  # tal cual lo usará la app
    datos = pd.read_parquet(exp["rutas"]["datos"])  # loans_pd.parquet
    train, test = datos[datos["split"] == SPLIT_TRAIN], datos[datos["split"] == SPLIT_TEST]  # particiones
    return {"exp": exp, "features": features, "params": params, "pre": pre, "pre_json": pre_json,  # config y receta
            "entradas": entradas_app(features, pre_json, train, exp["pasos"]),  # formulario: solo train
            "train": train, "test": test, "X_te": pre.transform(test)}  # X de test (float32), una sola vez


def _cargar_modelo(ctx: dict, modelo: str) -> tuple:
    """(modelo de Python, capas del JSON) de ``modelo`` desde ``export.rutas``.

    Parámetros:
        ctx: contexto de _cargar.
        modelo: "logistica" o "mlp".

    Retorna:
        tuple: (LogisticaPD o MLPPD, list[dict] de capas).
    """
    _validar_modelo(modelo)  # protege a exportar, cuyos modelos vienen de la config (construir_pd_model ya validó)
    if modelo == "logistica":  # modelo de referencia
        modelo_py = LogisticaPD.cargar(ctx["exp"]["rutas"]["logistica"])  # B3
        return modelo_py, capas_logistica(modelo_py)  # una capa identidad
    capas, modelo_py = capas_mlp(ctx["exp"]["rutas"]["mlp"])  # B5 (semilla mediana)
    return modelo_py, capas  # una capa por nn.Linear


def _pd_model(ctx: dict, modelo: str) -> dict:
    """pd_model_{modelo}.json a partir del contexto de _cargar.

    Parámetros:
        ctx: contexto de _cargar.
        modelo: "logistica" o "mlp".

    Retorna:
        dict: el contenido de pd_model_{modelo}.json.
    """
    modelo_py, capas = _cargar_modelo(ctx, modelo)  # solo el modelo cambia
    if len(capas[0]["W"][0]) != len(ctx["pre_json"]["columnas_salida"]):  # modelo y preprocesador incoherentes
        raise ValueError("La primera capa no recibe las columnas del preprocesador")  # dato
    logit = modelo_py.predict_logit(ctx["X_te"])  # una sola pasada sobre X de test
    score = score_desde_logit(logit, ctx["params"])  # puntaje de cada crédito de test
    p = expit(logit)  # PD de test = σ(logit), igual que predict_proba
    return {
        "version_contrato": VERSION_CONTRATO,  # núcleo §5.5
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # ISO 8601, cambia en cada export
        "commit": commit_actual(),  # HEAD al exportar
        "modelo": modelo,  # logistica | mlp
        "entradas": copy.deepcopy(ctx["entradas"]),  # común: formulario con train
        "preprocesamiento": copy.deepcopy(ctx["pre_json"]),  # común: Preprocesador.to_dict()
        "red": {"capas": capas, "salida": SALIDA_RED},  # pesos de este modelo y sigmoide
        "scorecard": dict(ctx["params"]),  # común: C1
        "poblacion": {"percentiles_score": percentiles_score(score), "referencia": REFERENCIA_POBLACION},  # test
        "metricas": metricas_test(ctx["test"]["target"].to_numpy(), p),  # test, de este modelo
    }


def construir_pd_model(cfg: dict, modelo: str) -> dict:
    """pd_model_{modelo}.json en memoria (núcleo §5.5).

    No exige que ``modelo`` esté en ``export.modelos``; solo usa ``export.rutas.{modelo}``.

    Parámetros:
        cfg: config completa (bloques scorecard y export).
        modelo: "logistica" o "mlp" (ValueError si no).

    Retorna:
        dict: el contenido de pd_model_{modelo}.json.
    """
    _validar_modelo(modelo)  # protege al llamador directo: falla antes de cargar datos y modelos
    return _pd_model(_cargar(cfg), modelo)  # carga y construye


# %% [markdown]
# ## Escritura de los cuatro archivos
#
# **Se valida y construye todo antes de escribir:** primero se arman EN MEMORIA los pares (pd_model, casos) de todos
# los modelos de `export.modelos`; solo si todos se construyeron se escriben. Así, si falla un modelo (p. ej.
# `export.rutas.mlp` apunta a un `.pt` que no existe), el destino no queda con los archivos nuevos de un modelo
# mezclados con los viejos del otro. Después, para cada modelo, en ese orden, `save_json` (UTF-8, sangría 2, sin
# NaN/inf, escritura atómica de cada archivo; núcleo §5.4) escribe en `export.destino` primero `pd_model_{modelo}.json`
# y luego `casos_paridad_{modelo}.json`.
#
# **La muestra de casos es la misma para los dos modelos:** `elegir_casos` (con `export.n_casos_paridad`,
# `export.n_forzados_por_grupo` y `export.base_seed`) se llama **una sola vez** y el mismo `df_casos` alimenta los dos
# `construir_casos_paridad`. La app compara los dos modelos con las mismas entradas (R-APP-2): con la misma muestra, la
# prueba de paridad de cada modelo recorre exactamente las mismas personas y los mismos caminos difíciles de la receta
# (vacíos, recortes, categorías no vistas), y un caso se puede mirar con los dos modelos a la vez.

# %%
def exportar(cfg: dict) -> list[Path]:
    """Escribe pd_model_{modelo}.json y casos_paridad_{modelo}.json de cada modelo en ``cfg["export"]["destino"]``.

    Parámetros:
        cfg: config completa (bloques scorecard y export).

    Retorna:
        list[Path]: rutas escritas, en el orden de export.modelos y, por modelo, pd_model y luego casos.
    """
    ctx = _cargar(cfg)  # una sola carga de lo común
    exp = ctx["exp"]  # bloque validado
    df_casos = elegir_casos(ctx["test"], ctx["pre_json"], exp["n_casos_paridad"], exp["n_forzados_por_grupo"],
                            exp["base_seed"])  # una sola muestra de test para los dos modelos
    pares = {}  # {modelo: (pd_model, casos)}, en el orden de export.modelos
    for modelo in exp["modelos"]:  # primero se construye todo en memoria
        modelo_json = _pd_model(ctx, modelo)  # pd_model de este modelo (falla aquí si su archivo no existe)
        pares[modelo] = (modelo_json, construir_casos_paridad(modelo_json, df_casos))  # misma muestra para todos
    destino = Path(exp["destino"])  # carpeta de la app
    rutas = []  # en el orden de export.modelos
    for modelo, (modelo_json, casos) in pares.items():  # solo se escribe si todos se construyeron
        nombre_modelo, nombre_casos = archivos_modelo(modelo)  # pd_model_{modelo}.json, casos_paridad_{modelo}.json
        rutas.append(Path(save_json(destino / nombre_modelo, modelo_json)))  # escritura atómica
        rutas.append(Path(save_json(destino / nombre_casos, casos)))  # escritura atómica
    return rutas  # dos rutas por modelo
