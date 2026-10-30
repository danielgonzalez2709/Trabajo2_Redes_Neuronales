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
# # Figuras de B11: de la PD a la decisión (Parte B, B-pd §3.9; blog 06 §6.8)
#
# Solo leen `results/data/pd_umbral/umbral.json` (o el resultado en memoria): no entrenan ni recalculan nada.
#
# | Archivo | Qué muestra |
# |---|---|
# | `pd-umbral-sensibilidad.png` | TPR, TNR y tasa de rechazo contra el umbral de PD, dos modelos, val y test |
# | `pd-umbral-costo.png` | cuota óptima de val y rechazo de prueba con el umbral de val contra $L/G$; $t$ vs $p^\*$ |
#
# Organización como `figs_pd.py`: funciones **puras** de datos, una de **dibujo** por figura y `generar(raiz,
# destino, datos)` con `SALIDA` leído al llamarse (redirigible). Conectado a `run.py figures` (Spec v0.30, Daniel):
# antes se llamaba `umbral_pd.py`, sin el prefijo `figs_`, para no tocar `cli.py`, que es compartido.

# %%
"""Figuras de B11 (blog 06 §6.8) desde results/data/pd_umbral/umbral.json: sensibilidad al umbral y costo L/G."""

from __future__ import annotations  # anotaciones como texto

import math  # NaN de los umbrales nulos
from pathlib import Path  # rutas

import matplotlib  # backend sin pantalla

matplotlib.use("Agg")  # dibuja en archivos, sin ventana
import matplotlib.pyplot as plt  # noqa: E402  figuras y ejes

from src.common.results import load_json  # noqa: E402  lectura sin NaN
from src.pd.tablas_umbral import ARCHIVO_UMBRAL, COMANDO_UMBRAL, MODELOS, bloque  # noqa: E402  contrato de umbral.json
from src.viz import paleta as p  # noqa: E402  color por entidad
from src.viz.style import aplicar_estilo  # noqa: E402  estilo común

# %% [markdown]
# ## Constantes
#
# Rutas, nombres de los PNG, splits en orden fijo y los estilos de trazo por métrica (solo visuales). El nombre de
# `umbral.json`, el orden de los modelos, el comando y `bloque` se importan de `src/pd/tablas_umbral.py` (un solo
# lugar para el contrato).

# %%
SALIDA = Path("blog/docs/figures")  # figuras versionadas del blog (redirigible en las pruebas)
ENTRADA = Path("results/data/pd_umbral")  # resultados de B11
ARCHIVOS = ("pd-umbral-sensibilidad.png", "pd-umbral-costo.png")  # salida, en orden
SPLITS = ("val", "test")  # paneles de la sensibilidad
CURVAS = ("tpr", "tnr", "tasa_rechazo")  # métricas de la sensibilidad
ETIQUETA_CURVA = {"tpr": "TPR (malos rechazados)", "tnr": "TNR (buenos aprobados)",  # leyendas
                  "tasa_rechazo": "Tasa de rechazo"}  # rechazo
TRAZO_CURVA = {"tpr": "-", "tnr": "--", "tasa_rechazo": ":"}  # estilo por métrica (solo visual)


# %% [markdown]
# ## Comprobación de largos
#
# Cada curva debe tener tantos puntos como su eje; si no → `ValueError` (un JSON incoherente).

# %%
def _mismo_largo(eje: list, curva: list, nombre: str) -> list:
    """``curva`` si tiene el largo de ``eje``; si no, ValueError."""
    if len(curva) != len(eje):  # incoherente
        raise ValueError(f"{nombre}: {len(curva)} puntos para un eje de {len(eje)}")  # largo
    return list(curva)  # copia


# %% [markdown]
# ## Datos de la sensibilidad al umbral (pura)
#
# Para cada umbral $t$ de la rejilla: $\mathrm{TPR}(t)$, $\mathrm{TNR}(t)$ y la tasa de rechazo de la política
# $p \ge t$, por modelo y split; más el umbral inicial (0,5) para marcarlo.

# %%
def datos_sensibilidad(umbral: dict) -> dict:
    """{umbrales, umbral_inicial, modelos: {m: {split: {curva: lista}}}}."""
    s = bloque(umbral, "sensibilidad_umbral")  # bloque
    eje = list(s["umbrales"])  # umbrales
    modelos = {m: {sp: {c: _mismo_largo(eje, s["modelos"][m][sp][c], f"{m}.{sp}.{c}") for c in CURVAS}  # curvas
                   for sp in SPLITS} for m in MODELOS}  # por modelo y split
    return {"umbrales": eje, "umbral_inicial": bloque(umbral, "matriz_inicial")["umbral"],  # eje y 0,5
            "modelos": modelos}  # curvas


# %% [markdown]
# ## Datos del costo (pura)
#
# Por razón $L/G$: cuota óptima $r^\*$ de val, tasas de rechazo de prueba por cuota y por PD absoluta, el umbral de
# val $t_{\text{val}}$ (`null` → NaN, cuota 0) y $p^\* = 1/(1 + L/G)$.

# %%
def datos_costo(umbral: dict) -> dict:
    """{razones, p_estrella, modelos: {m: {cuota, rechazo_cuota, rechazo_umbral, umbral_val}}}."""
    c = bloque(umbral, "curva_costo")  # bloque
    eje = list(c["razones"])  # razones L/G
    modelos = {}  # por modelo
    for m in MODELOS:  # logística y MLP
        d = c["modelos"][m]  # listas del modelo
        modelos[m] = {"cuota": _mismo_largo(eje, d["cuota"], f"{m}.cuota"),  # r*
                      "rechazo_cuota": _mismo_largo(eje, d["tasa_rechazo_test_cuota"], f"{m}.rechazo_cuota"),  # test
                      "rechazo_umbral": _mismo_largo(eje, d["tasa_rechazo_test_umbral"],  # PD absoluta en test
                                                     f"{m}.rechazo_umbral"),  # nombre
                      "umbral_val": [math.nan if t is None else t  # null → NaN (hueco en la curva)
                                     for t in _mismo_largo(eje, d["umbral_val"], f"{m}.umbral_val")]}  # t_val
    return {"razones": eje, "p_estrella": _mismo_largo(eje, c["p_estrella"], "p_estrella"),  # eje y p*
            "modelos": modelos}  # curvas


# %% [markdown]
# ## Figura: sensibilidad al umbral
#
# Un panel por split (val, test); color = modelo, trazo = métrica; línea vertical en el umbral inicial.

# %%
def figura_sensibilidad(d: dict, destino: Path) -> Path:
    """pd-umbral-sensibilidad.png."""
    fig, ejes = plt.subplots(1, len(SPLITS), figsize=(11, 4.6), sharey=True)  # val y test
    for ax, sp in zip(ejes, SPLITS):  # cada split
        for m in MODELOS:  # cada modelo
            for c in CURVAS:  # cada métrica
                ax.plot(d["umbrales"], d["modelos"][m][sp][c], color=p.COLOR_MODELO[m],  # color del modelo
                        linestyle=TRAZO_CURVA[c], label=f"{p.NOMBRE_MODELO[m]}: {ETIQUETA_CURVA[c]}")  # trazo
        ax.axvline(d["umbral_inicial"], color=p.TINTA_TENUE, linewidth=1)  # umbral inicial
        ax.set_xlim(0, 1)  # umbrales en [0, 1]
        ax.set_ylim(0, 1)  # tasas en [0, 1]
        ax.set_xlabel("Umbral de PD t (se rechaza si PD ≥ t)")  # eje x
        ax.set_title(p.NOMBRE_SPLIT[sp])  # Validación / Prueba
    ejes[0].set_ylabel("Tasa")  # eje y
    ejes[-1].legend(loc="center right", fontsize=7)  # leyenda
    return _guardar(fig, destino / ARCHIVOS[0])  # escribe


# %% [markdown]
# ## Figura: costo $L/G$
#
# Izquierda: cuota óptima de val (que es también la tasa de rechazo de prueba por cuota, salvo el techo de $k$) y la
# tasa de rechazo de prueba con el umbral de val (PD absoluta) contra $L/G$ en escala logarítmica. Derecha: el
# umbral de val $t_{\text{val}}$ de cada modelo frente a la referencia $p^\*$.

# %%
def figura_costo(d: dict, destino: Path) -> Path:
    """pd-umbral-costo.png."""
    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.6))  # dos paneles
    for m in MODELOS:  # cada modelo
        dm, color, nombre = d["modelos"][m], p.COLOR_MODELO[m], p.NOMBRE_MODELO[m]  # atajos
        izq.plot(d["razones"], dm["cuota"], color=color,  # r* (= rechazo de prueba por cuota, salvo el techo de k)
                 label=f"{nombre}: cuota de val")  # leyenda (la nota del eje dice que es el rechazo por cuota)
        izq.plot(d["razones"], dm["rechazo_umbral"], color=color, linestyle="--",  # PD absoluta en test
                 label=f"{nombre}: umbral de val en prueba")  # contraste
        der.plot(d["razones"], dm["umbral_val"], color=color, label=f"{nombre}: umbral de val")  # t_val
    der.plot(d["razones"], d["p_estrella"], color=p.TINTA_TENUE, linestyle=":", label="p* = 1/(1 + L/G)")  # teoría
    for ax, lugar in ((izq, "upper left"), (der, "upper right")):  # los dos ejes y dónde no tapan las curvas
        ax.set_xscale("log")  # L/G logarítmica
        ax.set_xlabel("Razón de costos L/G")  # eje x
        ax.legend(loc=lugar, fontsize=7)  # leyenda
    izq.set_ylim(0, 1)  # cuotas y tasas en [0, 1]
    izq.set_ylabel("Tasa de rechazo en prueba")  # eje y (cuota de val = rechazo por cuota)
    izq.set_title("Cuota que minimiza el costo")  # título
    der.set_ylabel("Umbral de PD")  # eje y
    der.set_title("Umbral observado frente a p*")  # título
    return _guardar(fig, destino / ARCHIVOS[1])  # escribe


# %% [markdown]
# ## Escritura
#
# Escribe una figura (crea la carpeta) y la cierra.

# %%
def _guardar(fig, ruta: Path) -> Path:
    """Escribe la figura en ``ruta`` y la cierra."""
    ruta.parent.mkdir(parents=True, exist_ok=True)  # carpeta destino
    fig.savefig(ruta)  # PNG con el estilo común
    plt.close(fig)  # libera memoria
    return ruta  # ruta escrita


# %% [markdown]
# ## Orquestación
#
# `generar` aplica el estilo, resuelve `destino or raiz / SALIDA` **al llamarse**, lee `umbral.json` (o usa `datos`),
# calcula **todos** los datos puros antes de dibujar (una entrada incoherente falla sin dejar figuras a medias) y
# dibuja. Sin `umbral.json` → `FileNotFoundError` con el comando que lo genera.

# %%
def generar(raiz: Path = Path("."), destino: Path | None = None, datos: dict | None = None) -> list[Path]:
    """Dibuja las dos figuras de B11 y devuelve sus rutas (en el orden de ARCHIVOS)."""
    aplicar_estilo()  # estilo común
    destino = Path(destino) if destino is not None else Path(raiz) / SALIDA  # SALIDA leído al llamar
    if datos is None:  # lectura del JSON
        ruta = Path(raiz) / ENTRADA / ARCHIVO_UMBRAL  # umbral.json
        if not ruta.is_file():  # no generado
            raise FileNotFoundError(f"No existe {ruta}; genéralo con {COMANDO_UMBRAL}")  # cómo
        datos = load_json(ruta)  # resultado
    sens, costo = datos_sensibilidad(datos), datos_costo(datos)  # todo antes de dibujar
    return [figura_sensibilidad(sens, destino), figura_costo(costo, destino)]  # las dos
