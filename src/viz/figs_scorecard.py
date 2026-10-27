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
# # Figuras de la scorecard (Parte C, tarea C7; blog 07)
#
# Solo leen `results/data/scorecard/` (C3 y C4) y, para la dependencia SHAP, los valores crudos de
# `data/processed/loans_pd.parquet` (A): no entrenan ni recalculan nada del modelo (núcleo §3 "Arquitectura": los
# modelos no dibujan; §7.1: `python run.py figures` las reconstruye desde `results/`).
#
# | Archivo | Qué muestra |
# |---|---|
# | `sc-transformacion.png` | puntaje frente a PD con PDO 20 (el de `scorecard.json`) y PDO 40, con el ancla y p = ½ |
# | `sc-puntos-variable.png` | puntos de la logística (curva) y del MLP (media por tramo) frente al valor crudo |
# | `sc-importancia.png` | media \|φ\| del MLP y caída de AUC por permutación de los dos modelos, mismo orden |
# | `sc-dependencia-shap.png` | φ frente al valor crudo de las 4 numéricas con mayor media \|φ\| |
# | `sc-bandas.png` | histograma del puntaje de test y tasa de malos y PD media por banda de los dos modelos |
#
# Organización: funciones **puras** de datos (no leen archivos ni dibujan), una función de **dibujo** por figura y
# piezas de **lectura y escritura** aparte. Las fórmulas son las de C1/C3 (`score_desde_pd`, `parametros_scorecard`,
# `puntos_valor_crudo`): aquí no se duplica ninguna.

# %%
"""Figuras de la Parte C (scorecard; blog 07) desde results/data/scorecard/: transformación, puntos, SHAP y bandas."""

from __future__ import annotations  # anotaciones como texto (dict | None en 3.12 sin costo)

from pathlib import Path  # rutas

import matplotlib  # backend sin pantalla

matplotlib.use("Agg")  # dibuja en archivos, sin ventana (CI y run.py figures)
import matplotlib.pyplot as plt  # noqa: E402  figuras y ejes
import numpy as np  # noqa: E402  rejillas y arreglos
import pandas as pd  # noqa: E402  loans_pd.parquet

from src.common.results import load_json  # noqa: E402  lectura de JSON
from src.scorecard.puntos import puntos_valor_crudo  # noqa: E402  puntos de una numérica (C3)
from src.scorecard.scorecard import CLAVES_PARAMETROS, parametros_scorecard, score_desde_pd  # noqa: E402  C1
from src.viz import paleta as p  # noqa: E402  color por entidad
from src.viz.style import aplicar_estilo  # noqa: E402  estilo común de matplotlib

SALIDA = Path("blog/docs/figures")  # figuras versionadas del blog (test_no_network la redirige)
ENTRADA = Path("results/data/scorecard")  # resultados de C3 y C4 (núcleo §5.4)
RUTA_LOANS = Path("data/processed/loans_pd.parquet")  # valores crudos por id (A, núcleo §5.1)
MODELOS = ("logistica", "mlp")  # modelos de la PD con scorecard
ARCHIVO_SC = {"logistica": "scorecard.json", "mlp": "scorecard_mlp.json"}  # scorecard por modelo
ARCHIVO_SHAP = "shap_mlp.json"  # SHAP y permutación (C4)
ARCHIVOS = ("sc-transformacion.png", "sc-puntos-variable.png", "sc-importancia.png",  # salida, en orden
            "sc-dependencia-shap.png", "sc-bandas.png")  # 5 figuras
COMANDO_C4 = "C4: python run.py scorecard --config configs/scorecard/mlp.yaml"  # escribe scorecard_mlp y shap_mlp
COMO_GENERAR = {  # insumo → tarea y comando que lo genera (mensaje de FileNotFoundError)
    "scorecard.json": "C3: python run.py scorecard",  # scorecard de la logística
    "scorecard_mlp.json": COMANDO_C4,  # scorecard del MLP
    "shap_mlp.json": COMANDO_C4,  # SHAP y permutación
    "loans_pd.parquet": "A: python scripts/preparar_datos.py",  # crudo → data/processed (núcleo §4.1)
}
CLAVES_DATOS = frozenset({"scorecard_logistica", "scorecard_mlp", "shap", "loans"})  # insumos de generar(datos=)
COLUMNA_ID = "id"  # identificador del crédito (núcleo §5.1)
TIPO_NUM = "num"  # tipo de una numérica en los JSON de C
SPLIT_TEST = "test"  # partición de sc-bandas, de la permutación y de los explicados SHAP (C4)
VARIABLES_PUNTOS = ("annual_inc", "revol_util", "inq_last_6mths")  # paneles de sc-puntos-variable.png
ETIQUETA_X_PUNTOS = "Valor"  # eje x de sc-puntos-variable.png (la variable va en el título del panel)
N_DEPENDENCIA = 4  # paneles de sc-dependencia-shap.png: las 4 numéricas con mayor media |φ|
PDO_ALTERNATIVO = 40.0  # segunda curva de sc-transformacion.png (mismo base y odds_base)
PD_ANCLA_MITAD = 0.5  # p = ½: odds 1:1, logit 0 → puntaje = offset
# Solo visuales (no afectan a ningún resultado): rango y resolución de las rejillas y transparencia de la dispersión.
PD_MIN, PD_MAX, N_PD = 0.005, 0.8, 400  # rejilla de PD (log) de sc-transformacion.png
N_REJILLA = 400  # puntos de la curva de la logística por variable
MARGEN_REJILLA = 0.15  # fracción del rango p1–p99 que la rejilla pasa del p99 (muestra lo plano tras el recorte)
ALFA_DISPERSION = 0.25  # transparencia de los puntos de la dependencia SHAP


# %% [markdown]
# ## Parámetros de cada curva de la transformación
#
# $\text{score} = \text{offset} - \text{factor}\cdot\operatorname{logit}(p)$ con $\text{factor} = \text{PDO}/\ln 2$ y
# $\text{offset} = \text{base} - \text{factor}\cdot\ln(\text{odds}_{\text{base}})$ (C §3.1). La primera curva usa los
# cinco parámetros tal como están en `scorecard.json`; la segunda, `parametros_scorecard` con el mismo `base` y
# `odds_base` y PDO 40. Ambas pasan por el ancla $p = 1/(1 + \text{odds}_{\text{base}}) \to \text{base}$; en
# $p = ½$ valen su `offset`. Con PDO 40 la pendiente frente a $\ln o$ se duplica.

# %%
def _params(sc: dict) -> dict:
    """Los cinco parámetros de la scorecard (CLAVES_PARAMETROS) tomados del JSON."""
    return {k: sc[k] for k in CLAVES_PARAMETROS}  # base, odds_base, pdo, factor, offset


def datos_transformacion(sc: dict, pdo_alternativo: float = PDO_ALTERNATIVO) -> dict:
    """Rejilla de PD, una curva de puntaje por PDO (la del JSON y la alternativa), el ancla y p = ½."""
    pd_rejilla = np.geomspace(PD_MIN, PD_MAX, N_PD)  # PD en escala logarítmica
    base = {k: sc[k] for k in ("base", "odds_base")}  # comunes a las dos curvas
    juegos = [_params(sc), parametros_scorecard({**base, "pdo": pdo_alternativo})]  # PDO del JSON y alternativo
    curvas = [{"pdo": float(q["pdo"]), "params": q, "score": score_desde_pd(pd_rejilla, q)} for q in juegos]  # C1
    return {"pd": pd_rejilla, "curvas": curvas,  # rejilla y curvas
            "ancla": {"pd": 1.0 / (1.0 + float(sc["odds_base"])), "score": float(sc["base"])},  # odds_base → base
            "mitad": {"pd": PD_ANCLA_MITAD, "scores": [float(q["offset"]) for q in juegos]}}  # p = ½ → offset


# %% [markdown]
# ## Puntos por variable
#
# Logística (C §3.2): $\text{puntos}(v) = -\text{factor}\,\beta\,(t(v) - \mu)/\sigma$ con $t$ = imputar → recortar →
# `log1p` → estandarizar; se evalúa con `puntos_valor_crudo` de C3 usando `transformacion` (las mismas cinco claves
# del preprocesador) y `puntos_por_unidad_estandarizada` $= -\text{factor}\,\beta$. La rejilla va del p1 de la tabla
# al p99 más `MARGEN_REJILLA` del rango, para ver lo plano después del recorte.
#
# MLP (C §3.2, C4): media de $-\text{factor}\,\varphi$ por tramo; solo los tramos interiores (`desde` y `hasta`
# definidos) con $n > 0$; las colas abiertas no tienen extremo que dibujar.

# %%
def _entrada(sc: dict, nombre: str) -> dict:
    """Entrada de `puntos_por_variable` de `nombre`; ValueError si no está."""
    for e in sc["puntos_por_variable"]:  # 15 entradas
        if e["nombre"] == nombre:  # encontrada
            return e  # su entrada
    raise ValueError(f"{nombre} no está en puntos_por_variable de la scorecard de {sc.get('modelo')}")  # dato


def curva_puntos_logistica(entrada: dict, valores) -> np.ndarray:
    """Puntos de la logística de una numérica en `valores` crudos (puntos_valor_crudo de C3)."""
    return puntos_valor_crudo(valores, entrada["transformacion"], entrada["puntos_por_unidad_estandarizada"])  # C3


def tramos_interiores(tramos: list[dict]) -> list[dict]:
    """Tramos del MLP con los dos extremos definidos y n > 0, en su orden."""
    return [t for t in tramos if t["desde"] is not None and t["hasta"] is not None and t["n"] > 0]  # interiores


def datos_puntos_variable(sc_log: dict, sc_mlp: dict, variables=VARIABLES_PUNTOS) -> list[dict]:
    """Por variable: etiqueta, rejilla, puntos de la logística, tabla y tramos interiores del MLP."""
    salida = []  # una entrada por panel
    for nombre in variables:  # en el orden de los paneles
        e_log, e_mlp = _entrada(sc_log, nombre), _entrada(sc_mlp, nombre)  # entradas de los dos modelos
        valores = [t["valor"] for t in e_log["tabla"]]  # percentiles de train
        lo, hi = min(valores), max(valores)  # p1 y p99
        x = np.linspace(lo, hi + MARGEN_REJILLA * (hi - lo), N_REJILLA)  # rejilla con margen a la derecha
        salida.append({"nombre": nombre, "etiqueta": e_log["etiqueta_es"], "x": x,  # identidad y rejilla
                       "puntos_logistica": curva_puntos_logistica(e_log, x),  # curva
                       "tabla_valor": valores, "tabla_puntos": [t["puntos"] for t in e_log["tabla"]],  # tabla
                       "tramos_mlp": tramos_interiores(e_mlp["tramos"])})  # medias por tramo
    return salida  # listo para dibujar


# %% [markdown]
# ## Importancia
#
# Orden de las variables por `puesto` SHAP (1 = mayor media $|\varphi|$). La caída de AUC por permutación de cada
# modelo se alinea **por nombre** a ese orden; si las variables no coinciden, el JSON no es coherente.

# %%
def datos_importancia(shap: dict) -> dict:
    """Nombres por puesto SHAP, su media |φ| y la caída de AUC (media y desv) de cada modelo alineada por nombre."""
    imp = sorted(shap["importancia_global"], key=lambda r: r["puesto"])  # puesto 1 primero
    nombres = [r["nombre"] for r in imp]  # orden común de los dos paneles
    caida = {}  # por modelo
    for m in MODELOS:  # logística y MLP
        por_nombre = {v["nombre"]: v for v in shap["permutacion"][m]["variables"]}  # índice por nombre
        if set(por_nombre) != set(nombres):  # variables distintas
            raise ValueError(f"permutacion.{m} no tiene las mismas variables que importancia_global")  # incoherente
        caida[m] = {"media": [por_nombre[n]["caida_auc_media"] for n in nombres],  # Δ AUC media
                    "desv": [por_nombre[n]["caida_auc_desv"] for n in nombres]}  # su desviación
    return {"nombres": nombres, "media_abs_phi": [r["media_abs_phi"] for r in imp], "caida": caida}  # paneles


# %% [markdown]
# ## Dependencia SHAP
#
# Las `N_DEPENDENCIA` numéricas (tipo `num` en `direccion`) con mayor media $|\varphi|$ según `importancia_global`.
# Para cada una: la columna de `explicados.phi` del jugador (orden de `metodo.jugadores`) frente al valor crudo del
# crédito, unido por `id` como texto con `loans_pd.parquet`. Los crudos vacíos (NaN) se excluyen. Un id explicado
# que no está en el parquet (o un id repetido en él) es un error.

# %%
def datos_dependencia(shap: dict, loans: pd.DataFrame, n: int = N_DEPENDENCIA) -> list[dict]:
    """Por cada una de las n numéricas con mayor media |φ|: valores crudos, φ y ρ de Spearman de `direccion`."""
    numericas = [r for r in sorted(shap["importancia_global"], key=lambda r: r["puesto"])  # por puesto
                 if shap["direccion"][r["nombre"]]["tipo"] == TIPO_NUM][:n]  # solo numéricas, las n primeras
    ids = [str(i) for i in shap["explicados"]["ids"]]  # créditos explicados
    tabla = loans.assign(**{COLUMNA_ID: loans[COLUMNA_ID].astype(str)})  # id como texto
    if not tabla[COLUMNA_ID].is_unique:  # la unión sería ambigua
        raise ValueError("loans_pd.parquet tiene ids repetidos")  # dato
    tabla = tabla.set_index(COLUMNA_ID)  # índice por id
    faltan = [i for i in ids if i not in tabla.index]  # explicados sin crudo
    if faltan:  # no se puede dibujar su valor
        raise ValueError(f"{len(faltan)} ids de shap_mlp.json no están en loans_pd.parquet: {faltan[:5]}")  # claro
    phi = np.asarray(shap["explicados"]["phi"], dtype=float)  # n_explicados × jugadores
    jugadores = list(shap["metodo"]["jugadores"])  # orden de las columnas
    if phi.ndim != 2 or phi.shape[0] != len(ids) or phi.shape[1] != len(jugadores):  # matriz incoherente
        raise ValueError(f"{ARCHIVO_SHAP}: explicados.phi tiene forma {phi.shape}; se esperaba "
                         f"({len(ids)}, {len(jugadores)}) (ids × jugadores)")  # nombra el archivo
    ausentes = [r["nombre"] for r in numericas if r["nombre"] not in jugadores]  # sin columna en phi
    if ausentes:  # no hay φ que dibujar
        raise ValueError(f"{ARCHIVO_SHAP}: {ausentes} no están en metodo.jugadores")  # nombra el archivo
    salida = []  # una entrada por panel
    for r in numericas:  # en orden de importancia
        nombre = r["nombre"]  # variable
        x = tabla.loc[ids, nombre].to_numpy(dtype=float)  # crudos en el orden de los explicados
        f = phi[:, jugadores.index(nombre)]  # φ de la variable
        ok = ~np.isnan(x)  # sin crudos vacíos
        salida.append({"nombre": nombre, "valores": x[ok].tolist(), "phi": f[ok].tolist(),  # puntos
                       "spearman": shap["direccion"][nombre]["spearman"]})  # ρ de C4
    return salida  # listo para dibujar


# %% [markdown]
# ## Bandas
#
# Histograma y bandas del puntaje de test de cada modelo, tal cual de `distribucion_scores` (C3/C4): las bandas son
# los deciles del puntaje de **train** de cada modelo (0-based en el JSON; la figura las numera 1–10, 1 = menor
# puntaje = mayor riesgo).

# %%
def datos_bandas(sc_log: dict, sc_mlp: dict) -> dict:
    """Bordes y conteos del histograma, y banda, tasa de malos y PD media por banda, de test, por modelo."""
    salida = {}  # por modelo
    for m, sc in zip(MODELOS, (sc_log, sc_mlp)):  # logística y MLP
        t = sc["distribucion_scores"][SPLIT_TEST]  # bloque de test
        salida[m] = {"bordes": t["histograma"]["bordes"], "conteos": t["histograma"]["conteos"],  # histograma
                     "banda": [b["banda"] for b in t["bandas"]],  # índice de banda
                     "tasa_malos": [b["tasa_malos"] for b in t["bandas"]],  # observada
                     "pd_media": [b["pd_media"] for b in t["bandas"]]}  # predicha
    return salida  # listo para dibujar


# %% [markdown]
# ## Figura: transformación

# %%
def figura_transformacion(d: dict, destino: Path) -> Path:
    """sc-transformacion.png: puntaje frente a PD con dos PDO, el ancla y p = ½."""
    fig, ax = plt.subplots(figsize=(8, 4.6))  # un solo eje
    colores = (p.CATEGORICA[0], p.CATEGORICA[2])  # una por PDO
    for c, color, s_mitad in zip(d["curvas"], colores, d["mitad"]["scores"]):  # cada PDO
        ax.plot(d["pd"], c["score"], color=color, label=f"PDO = {c['pdo']:g}")  # curva
        ax.scatter([d["mitad"]["pd"]], [s_mitad], color=color, zorder=3,  # p = ½
                   label=f"PD = {d['mitad']['pd']:g} → offset = {s_mitad:.2f}")  # valor
    a = d["ancla"]  # ancla común
    ax.scatter([a["pd"]], [a["score"]], s=90, facecolors="none", edgecolors=p.TINTA, linewidths=1.5, zorder=4,  # anillo
               label=f"Ancla: PD = {a['pd']:.4f} → {a['score']:g} puntos")  # odds_base → base
    ax.set_xscale("log")  # PD en escala logarítmica
    ax.set_xlabel("PD (escala logarítmica)")  # eje x
    ax.set_ylabel("Puntaje")  # eje y
    ax.set_title("Transformación de la PD en puntaje")  # título
    ax.legend(loc="upper right", fontsize=8)  # leyenda
    return _guardar(fig, destino / "sc-transformacion.png")  # escribe


# %% [markdown]
# ## Figura: puntos por variable

# %%
def figura_puntos_variable(paneles: list[dict], destino: Path) -> Path:
    """sc-puntos-variable.png: puntos de la logística (curva) y del MLP (media por tramo) por variable."""
    fig, ejes = plt.subplots(1, len(paneles), figsize=(4.2 * len(paneles), 4.2))  # un panel por variable
    for ax, d in zip(np.atleast_1d(ejes), paneles):  # cada variable
        ax.plot(d["x"], d["puntos_logistica"], color=p.COLOR_MODELO["logistica"],  # curva
                label=p.NOMBRE_MODELO["logistica"])  # entidad
        for i, t in enumerate(d["tramos_mlp"]):  # segmentos horizontales del MLP
            ax.hlines(t["puntos_medios"], t["desde"], t["hasta"], color=p.COLOR_MODELO["mlp"], linewidth=3,  # tramo
                      label=f"{p.NOMBRE_MODELO['mlp']} (media por tramo)" if i == 0 else None)  # una vez
        ax.axhline(0, color=p.TINTA_TENUE, linewidth=1)  # referencia
        ax.set_xlabel(ETIQUETA_X_PUNTOS)  # eje x corto: la variable va en el título
        ax.set_title(d["etiqueta"], fontsize=10)  # panel
        ax.legend(loc="best", fontsize=8)  # leyenda
    np.atleast_1d(ejes)[0].set_ylabel("Puntos")  # eje y
    fig.suptitle("Puntos por variable: logística frente a MLP")  # título general
    fig.tight_layout()  # sin solapes
    return _guardar(fig, destino / "sc-puntos-variable.png")  # escribe


# %% [markdown]
# ## Figura: importancia

# %%
def figura_importancia(d: dict, etiquetas: dict, destino: Path) -> Path:
    """sc-importancia.png: media |φ| del MLP y caída de AUC por permutación de los dos modelos."""
    nombres = d["nombres"]  # orden común
    y = np.arange(len(nombres))  # posiciones
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)  # dos paneles
    ax1.barh(y, d["media_abs_phi"], color=p.COLOR_MODELO["mlp"], label=p.NOMBRE_MODELO["mlp"])  # SHAP
    ax1.set_yticks(y, [etiquetas[n] for n in nombres])  # etiquetas en español
    ax1.invert_yaxis()  # puesto 1 arriba
    ax1.set_xlabel("Media |φ| (logit)")  # eje x
    ax1.set_title("SHAP del MLP")  # panel
    ax1.legend(loc="lower right", fontsize=8)  # leyenda
    alto = 0.4  # alto de cada barra del panel agrupado
    for k, m in enumerate(MODELOS):  # barras agrupadas
        c = d["caida"][m]  # media y desv
        ax2.barh(y + (k - 0.5) * alto, c["media"], height=alto, xerr=c["desv"], color=p.COLOR_MODELO[m],  # barra
                 error_kw={"ecolor": p.TINTA_SECUNDARIA, "elinewidth": 1}, label=p.NOMBRE_MODELO[m])  # entidad
    ax2.axvline(0, color=p.TINTA_TENUE, linewidth=1)  # referencia
    ax2.set_xlabel("Caída de AUC al permutar (media ± desv)")  # eje x
    ax2.set_title(f"Importancia por permutación en {p.NOMBRE_SPLIT[SPLIT_TEST]}")  # panel
    ax2.legend(loc="lower right", fontsize=8)  # leyenda
    fig.suptitle("Importancia de las variables (orden por SHAP del MLP)")  # título general
    fig.tight_layout()  # sin solapes
    return _guardar(fig, destino / "sc-importancia.png")  # escribe


# %% [markdown]
# ## Figura: dependencia SHAP

# %%
def figura_dependencia(paneles: list[dict], etiquetas: dict, destino: Path) -> Path:
    """sc-dependencia-shap.png: φ frente al valor crudo de las numéricas más importantes."""
    fig, ejes = plt.subplots(1, len(paneles), figsize=(4.2 * len(paneles), 4.2), sharey=True)  # un panel por variable
    for ax, d in zip(np.atleast_1d(ejes), paneles):  # cada variable
        ax.scatter(d["valores"], d["phi"], s=10, alpha=ALFA_DISPERSION, color=p.COLOR_MODELO["mlp"],  # puntos
                   label=f"{p.NOMBRE_MODELO['mlp']} ({len(d['phi'])} créditos)")  # entidad
        ax.axhline(0, color=p.TINTA_TENUE, linewidth=1, label="φ = 0")  # referencia
        ax.set_xlabel(etiquetas[d["nombre"]])  # variable en español
        ax.set_title(f"{etiquetas[d['nombre']]}\nρ de Spearman = {d['spearman']:.3f}", fontsize=10)  # panel
        ax.legend(loc="best", fontsize=8)  # leyenda
    np.atleast_1d(ejes)[0].set_ylabel("φ (contribución al logit de la PD)")  # eje y
    fig.suptitle(f"Dependencia SHAP del MLP en {p.NOMBRE_SPLIT[SPLIT_TEST]}")  # título general
    fig.tight_layout()  # sin solapes
    return _guardar(fig, destino / "sc-dependencia-shap.png")  # escribe


# %% [markdown]
# ## Figura: bandas

# %%
def figura_bandas(d: dict, destino: Path) -> Path:
    """sc-bandas.png: histograma del puntaje de test y tasa de malos y PD media por banda, por modelo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))  # dos paneles
    for m in MODELOS:  # logística y MLP
        b = d[m]  # datos del modelo
        ax1.stairs(b["conteos"], b["bordes"], color=p.COLOR_MODELO[m], linewidth=p.ANCHO_LINEA,  # escalón propio
                   label=p.NOMBRE_MODELO[m])  # entidad
        x = np.asarray(b["banda"]) + 1  # bandas 1–10
        ax2.plot(x, b["tasa_malos"], marker="o", color=p.COLOR_MODELO[m],  # observada
                 label=f"{p.NOMBRE_MODELO[m]}: tasa de malos")  # entidad
        ax2.plot(x, b["pd_media"], linestyle="--", color=p.COLOR_MODELO[m],  # predicha
                 label=f"{p.NOMBRE_MODELO[m]}: PD media")  # entidad
    ax1.set_xlabel("Puntaje")  # eje x
    ax1.set_ylabel("Créditos")  # eje y
    ax1.set_title(f"Distribución del puntaje en {p.NOMBRE_SPLIT[SPLIT_TEST]}")  # panel
    ax1.legend(loc="upper right", fontsize=8)  # leyenda
    ax2.set_xticks(np.asarray(d[MODELOS[0]]["banda"]) + 1)  # una marca por banda
    ax2.set_xlabel(f"Banda de puntaje (deciles de {p.NOMBRE_SPLIT['train']}; 1 = mayor riesgo)")  # eje x
    ax2.set_ylabel("Tasa de malos / PD media")  # eje y
    ax2.set_title(f"Tasa de malos por banda en {p.NOMBRE_SPLIT[SPLIT_TEST]}")  # panel
    ax2.legend(loc="upper right", fontsize=8)  # leyenda
    fig.tight_layout()  # sin solapes
    return _guardar(fig, destino / "sc-bandas.png")  # escribe


# %% [markdown]
# ## Lectura y escritura
#
# `_exigir` falla con el comando que genera el insumo que falta. Del parquet solo se leen `id` y las numéricas de
# `direccion`.

# %%
def _exigir(ruta: Path, insumo: str) -> Path:
    """Devuelve `ruta` si existe; si no, FileNotFoundError con la tarea y el comando que la genera."""
    if not ruta.exists():  # falta el insumo
        raise FileNotFoundError(f"falta {ruta.as_posix()}; genéralo con {COMO_GENERAR[insumo]}")  # cómo arreglarlo
    return ruta  # existe


def insumos(raiz: Path) -> dict:
    """`datos` para generar() desde el repositorio: las dos scorecards, shap_mlp.json y loans_pd (id + numéricas)."""
    base = Path(raiz) / ENTRADA  # results/data/scorecard
    sc = {m: load_json(_exigir(base / ARCHIVO_SC[m], ARCHIVO_SC[m])) for m in MODELOS}  # scorecards
    shap = load_json(_exigir(base / ARCHIVO_SHAP, ARCHIVO_SHAP))  # SHAP y permutación
    numericas = [n for n, v in shap["direccion"].items() if v["tipo"] == TIPO_NUM]  # columnas necesarias
    loans = pd.read_parquet(_exigir(Path(raiz) / RUTA_LOANS, RUTA_LOANS.name), columns=[COLUMNA_ID, *numericas])  # A
    return {"scorecard_logistica": sc["logistica"], "scorecard_mlp": sc["mlp"], "shap": shap, "loans": loans}  # ok


def _guardar(fig, ruta: Path) -> Path:
    """Escribe la figura en `ruta` (crea la carpeta) y la cierra."""
    ruta.parent.mkdir(parents=True, exist_ok=True)  # carpeta destino
    fig.savefig(ruta)  # PNG con el estilo común
    plt.close(fig)  # libera memoria
    return ruta  # ruta escrita


# %% [markdown]
# ## Orquestación
#
# `generar` aplica el estilo, resuelve `destino or raiz / SALIDA` **al llamarse** (así `test_no_network` lo
# redirige), lee los insumos (o usa `datos`), calcula **todos** los datos puros antes de dibujar (un insumo
# incoherente falla sin dejar figuras a medias) y dibuja en el orden de `ARCHIVOS`.

# %%
def _validar_datos(datos: dict) -> None:
    """ValueError si `datos` no tiene exactamente las claves de CLAVES_DATOS."""
    faltan, sobran = CLAVES_DATOS - set(datos), set(datos) - CLAVES_DATOS  # diferencias de claves
    if faltan or sobran:  # claves de menos o de más
        raise ValueError(f"`datos` debe tener las claves {sorted(CLAVES_DATOS)}; faltan {sorted(faltan)}, "
                         f"sobran {sorted(sobran)}")  # nombra ambas


def generar(raiz: Path = Path("."), destino: Path | None = None, datos: dict | None = None) -> list[Path]:
    """Regenera las figuras de la Parte C desde results/data/scorecard/ y devuelve sus rutas (orden de ARCHIVOS)."""
    aplicar_estilo()  # estilo común
    destino = Path(destino) if destino is not None else Path(raiz) / SALIDA  # SALIDA leído al llamar
    datos = datos if datos is not None else insumos(Path(raiz))  # lectura o inyección
    _validar_datos(datos)  # claves exactas
    sc_log, sc_mlp, shap = datos["scorecard_logistica"], datos["scorecard_mlp"], datos["shap"]  # insumos
    etiquetas = {e["nombre"]: e["etiqueta_es"] for e in sc_log["puntos_por_variable"]}  # nombres en español
    trans = datos_transformacion(sc_log)  # curvas PD → puntaje
    puntos = datos_puntos_variable(sc_log, sc_mlp)  # paneles de puntos
    imp = datos_importancia(shap)  # SHAP y permutación
    dep = datos_dependencia(shap, datos["loans"])  # dependencia SHAP
    bandas = datos_bandas(sc_log, sc_mlp)  # histograma y bandas
    return [figura_transformacion(trans, destino), figura_puntos_variable(puntos, destino),  # transformación y puntos
            figura_importancia(imp, etiquetas, destino), figura_dependencia(dep, etiquetas, destino),  # SHAP
            figura_bandas(bandas, destino)]  # bandas
