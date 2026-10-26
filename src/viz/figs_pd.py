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
# # Figuras de la PD (Parte B, tarea B7; blog 06)
#
# Solo leen `results/data/pd/` (B3–B5): no entrenan ni recalculan nada del modelo (núcleo §3 "Arquitectura": los
# modelos no dibujan; §7.1: `python run.py figures` las reconstruye desde `results/`).
#
# | Archivo | Qué muestra |
# |---|---|
# | `pd-roc.png` | ROC de prueba de la logística y el MLP (semilla mediana), con AUC y Gini |
# | `pd-ks.png` | acumuladas F₁ (malos) y F₀ (buenos) de la PD por modelo y la separación máxima (KS) |
# | `pd-calibracion.png` + `.json` | PD media frente a tasa observada en 10 bins por cuantiles (B-pd §3.4) |
# | `pd-semillas.png` | AUC, KS y Brier de prueba de las semillas del MLP frente a la logística |
# | `pd-aprendizaje.png` | BCE de entrenamiento y validación por época de la semilla mediana |
# | `pd-busqueda-mlp.png` | log-loss de validación por ensayo de Optuna y el mejor hasta cada ensayo |
# | `pd-busqueda-logistica.png` | log-loss y AUC de validación contra C en la rejilla de la logística |
#
# Organización: funciones **puras** de datos (no leen archivos ni dibujan), una función de **dibujo** por figura
# (recibe lo de la pura y un destino) y piezas de **lectura y escritura** aparte. `generar(..., datos=)` acepta los
# insumos en memoria (el notebook 02 dibuja así la corrida demo, que no tiene `hpo_ensayos.json`).

# %%
"""Figuras de la Parte B (PD; blog 06) desde results/data/pd/: ROC, KS, calibración, semillas y búsquedas."""

from __future__ import annotations  # anotaciones como texto (dict | None en 3.12 sin costo)

from pathlib import Path  # rutas

import matplotlib  # backend sin pantalla

matplotlib.use("Agg")  # dibuja en archivos, sin ventana (CI y run.py figures)
import matplotlib.pyplot as plt  # noqa: E402  figuras y ejes
import numpy as np  # noqa: E402  arreglos, mínimo acumulado y ordenamientos
import pandas as pd  # noqa: E402  pred_test.parquet

from src.common.metricas import auc, curva_calibracion, curva_ks, curva_roc, gini  # noqa: E402  métricas de B1
from src.common.results import load_json, save_json  # noqa: E402  JSON sin NaN y escritura atómica
from src.viz import paleta as p  # noqa: E402  color por entidad
from src.viz.style import aplicar_estilo  # noqa: E402  estilo común de matplotlib

SALIDA = Path("blog/docs/figures")  # figuras versionadas del blog (test_no_network la redirige)
ENTRADA = Path("results/data/pd")  # resultados de B3–B5 (núcleo §5.4)
N_BINS_CALIBRACION = 10  # B-pd §3.4: curva de calibración con 10 bins por cuantiles
SPLIT_CALIBRACION = "test"  # pred_test.parquet es la partición de prueba (núcleo §5.4)
MODELOS = ("logistica", "mlp")  # modelos de la PD, en el orden de las columnas pred_<modelo> (núcleo §5.4)
COLUMNA_Y = "y"  # desenlace en pred_test.parquet (núcleo §5.4)
METRICAS_SEMILLAS = ("auc_test", "ks_test", "brier_test")  # paneles de pd-semillas.png (B-pd §4, fila B7)
ESTADO_COMPLETO = "COMPLETE"  # estado de Optuna de un ensayo terminado (hpo_ensayos.json, B4)
CLAVES_DATOS = frozenset({"resumen", "pred", "semillas", "hpo", "busqueda"})  # insumos de generar(datos=)
ARCHIVO_CALIBRACION = "pd-calibracion.json"  # puntos de la calibración para el video (escena 7)
FIGURA_BUSQUEDA_MLP = "pd-busqueda-mlp.png"  # única figura opcional (sin hpo_ensayos.json no se dibuja)
ARCHIVOS = ("pd-roc.png", "pd-ks.png", "pd-calibracion.png", ARCHIVO_CALIBRACION, "pd-semillas.png",  # salida
            "pd-aprendizaje.png", FIGURA_BUSQUEDA_MLP, "pd-busqueda-logistica.png")  # 7 figuras + el JSON
COMANDO_PD = "B3/B5: python run.py pd --config configs/pd/base.yaml"  # escribe summary, pred_test, semillas y C
COMO_GENERAR = {  # insumo → tarea y comando que lo genera (mensaje de FileNotFoundError)
    "summary.json": COMANDO_PD,  # resumen por modelo
    "pred_test.parquet": COMANDO_PD,  # PD de test de los dos modelos
    "busqueda_logistica.json": COMANDO_PD,  # rejilla de C
    "semillas": COMANDO_PD,  # mlp_seed*.json
    "hpo_ensayos.json": "B4: python run.py pd-hpo --config configs/pd/hpo.yaml",  # búsqueda Optuna (opcional)
}
# Solo visuales (no afectan a ninguna métrica): media anchura de los desplazamientos horizontales fijos de los puntos
# de pd-semillas.png y paso al que se redondea hacia arriba el límite de los ejes de la calibración.
DESPLAZAMIENTO_PUNTOS = 0.18  # en unidades del eje x de la caja (ancho de la caja: 0,5)
PASO_EJE_CALIBRACION = 0.1  # ejes [0, L] con L = máximo de los puntos redondeado hacia arriba a 0,1


# %% [markdown]
# ## Datos de la ROC
#
# Para cada umbral $t$ (de $+\infty$ al mínimo de $p$), $\mathrm{TPR}(t) = VP(t)/P$ y $\mathrm{FPR}(t) = FP(t)/N$:
# la curva va de $(0, 0)$ a $(1, 1)$. $\mathrm{AUC} = \int \mathrm{TPR}\, d(\mathrm{FPR})$ (trapecio) y
# $\mathrm{Gini} = 2\,\mathrm{AUC} - 1$. Todo sale de `src/common/metricas.py` (B1), sin recalcular nada aquí.

# %%
def datos_roc(y, prob) -> dict:
    """Puntos (FPR, TPR) de la ROC con su AUC y su Gini (metricas de B1)."""
    roc = curva_roc(y, prob)  # puntos de (0, 0) con umbral +inf a (1, 1)
    return {"fpr": roc.fpr, "tpr": roc.tpr, "auc": auc(y, prob), "gini": gini(y, prob)}  # curva y resúmenes


# %% [markdown]
# ## Datos del KS
#
# $F_1(t)$ y $F_0(t)$ son las acumuladas empíricas de la PD en malos y en buenos, evaluadas en cada valor único de
# $p$. $\mathrm{KS} = \max_t |F_1(t) - F_0(t)|$ y el umbral del KS es $t^\* = \arg\max_t |F_1(t) - F_0(t)|$ (el
# primero si hay empate, como `np.argmax`). El máximo se toma sobre la misma curva que usa `metricas.ks`.

# %%
def datos_ks(y, prob) -> dict:
    """Acumuladas F₁ (malos) y F₀ (buenos), el KS y el umbral donde se alcanza."""
    c = curva_ks(y, prob)  # F₁(t) y F₀(t) en los valores únicos de p (B1)
    separacion = np.abs(np.asarray(c.f_malos) - np.asarray(c.f_buenos))  # |F₁(t) − F₀(t)|
    i = int(np.argmax(separacion))  # t* = argmax (primero si hay empate)
    return {"umbrales": c.umbrales, "f_malos": c.f_malos, "f_buenos": c.f_buenos,  # curvas
            "ks": float(separacion[i]), "umbral_ks": float(c.umbrales[i])}  # máximo y su umbral


# %% [markdown]
# ## Datos de la calibración (y el JSON del video)
#
# Con los bordes $b_k = Q_p(k/B)$, $B = 10$ (B-pd §3.4), cada bin no vacío da
# $\bar p_k = \frac{1}{n_k}\sum_{i \in k} p_i$ (PD media) y $\bar y_k = \frac{1}{n_k}\sum_{i \in k} y_i$ (tasa
# observada); los bins vacíos se omiten (`metricas.curva_calibracion`). Se añaden las medias globales
# $\bar p = \frac{1}{n}\sum_i p_i$ y $\bar y = \frac{1}{n}\sum_i y_i$. Todo en `float`/`int` nativos para que
# `pd-calibracion.json` se escriba sin tipos de numpy ni NaN.

# %%
def datos_calibracion(pred: pd.DataFrame, semilla_mediana: int, n_bins: int = N_BINS_CALIBRACION) -> dict:
    """Puntos de calibración por bin y medias globales de los dos modelos en prueba (pd-calibracion.json)."""
    faltan = [c for c in (COLUMNA_Y, *(f"pred_{m}" for m in MODELOS)) if c not in pred.columns]  # columnas ausentes
    if faltan:  # sin alguna columna no hay curva de ese modelo
        raise ValueError(f"pred_test no tiene las columnas {faltan}; regenera con {COMANDO_PD}")  # error claro
    y = pred[COLUMNA_Y].to_numpy()  # desenlace 0/1
    modelos = {}  # bloque por modelo
    for m in MODELOS:  # logística y MLP
        prob = pred[f"pred_{m}"].to_numpy(dtype=float)  # PD del modelo
        c = curva_calibracion(y, prob, n_bins=n_bins)  # bins por cuantiles (B1)
        modelos[m] = {"pd_media": [float(v) for v in c.pd_media],  # \bar p_k
                      "tasa_observada": [float(v) for v in c.tasa_observada],  # \bar y_k
                      "n": [int(v) for v in c.n_por_bin],  # n_k
                      "pd_media_global": float(prob.mean()),  # \bar p
                      "tasa_global": float(y.mean())}  # \bar y
    return {"split": SPLIT_CALIBRACION, "n_bins": int(n_bins), "semilla_mediana": int(semilla_mediana),  # cabecera
            "modelos": modelos}  # puntos


# %% [markdown]
# ## Semilla mediana
#
# `summary.json` (B5) declara la semilla mediana del MLP (orden por (log-loss de validación, semilla), posición
# $\lfloor (n-1)/2 \rfloor$). Aquí solo se busca su registro; si no está entre las semillas leídas, las figuras no
# corresponden al `summary.json` y se detiene. Una semilla repetida (archivos copiados) también se rechaza.

# %%
def _registro_mediana(semillas: list[dict], resumen: dict) -> dict:
    """Registro mlp_seed*.json de la semilla mediana de summary.json; ValueError si falta o hay semillas repetidas."""
    seeds = [r["seed"] for r in semillas]  # semillas leídas
    if len(set(seeds)) != len(seeds):  # dos archivos con la misma semilla
        raise ValueError(f"Semillas repetidas entre los mlp_seed*.json: {sorted(seeds)}")  # no se dibuja dos veces
    mediana = resumen["mlp"]["semilla_mediana"]  # declarada por B5
    for r in semillas:  # búsqueda lineal (30 registros)
        if r["seed"] == mediana:  # encontrada
            return r  # su registro
    raise ValueError(f"La semilla mediana ({mediana}) de summary.json no está entre las semillas leídas {seeds}.")


# %% [markdown]
# ## Métricas por semilla
#
# Para cada métrica (AUC, KS y Brier de prueba): los valores de las semillas ordenados por semilla, el par
# (semilla mediana, su valor) y la referencia de la logística (`summary.logistica.metricas.<métrica>.media`, n = 1).

# %%
def metricas_por_semilla(semillas: list[dict], resumen: dict, nombres=METRICAS_SEMILLAS) -> dict:
    """{métrica: {seeds, valores, mediana: (seed, valor), logistica}} ordenado por semilla."""
    mediana = _registro_mediana(semillas, resumen)  # valida mediana y duplicados
    ordenadas = sorted(semillas, key=lambda r: r["seed"])  # por semilla
    salida = {}  # resultado por métrica
    for nombre in nombres:  # en el orden pedido
        salida[nombre] = {"seeds": [r["seed"] for r in ordenadas],  # eje de semillas
                          "valores": [r["metricas"][nombre] for r in ordenadas],  # valor por semilla
                          "mediana": (mediana["seed"], mediana["metricas"][nombre]),  # semilla mediana
                          "logistica": resumen["logistica"]["metricas"][nombre]["media"]}  # referencia
    return salida  # listo para dibujar


# %% [markdown]
# ## Curva de aprendizaje
#
# Historial de la semilla mediana: BCE media de entrenamiento y de validación por época (desde 1), su
# `mejor_epoca` (la de menor pérdida de validación, cuyos pesos se restauran) y la `paciencia` de su config: el
# entrenamiento se detiene tras `paciencia` épocas sin mejora.

# %%
def datos_aprendizaje(semillas: list[dict], resumen: dict) -> dict:
    """Épocas, pérdidas de train y val, mejor época y paciencia de la semilla mediana."""
    r = _registro_mediana(semillas, resumen)  # registro de la mediana
    historial = r["historial"]  # una entrada por época
    return {"seed": r["seed"], "epocas": [h["epoca"] for h in historial],  # semilla y épocas
            "perdida_train": [h["perdida_train"] for h in historial],  # BCE de entrenamiento
            "perdida_val": [h["perdida_val"] for h in historial],  # BCE de validación
            "mejor_epoca": r["mejor_epoca"], "paciencia": r["config"]["paciencia"]}  # parada temprana


# %% [markdown]
# ## Trayectoria de la búsqueda del MLP
#
# Solo los ensayos `COMPLETE`, por número. El mejor hasta el ensayo $j$ es el mínimo acumulado
# $m_j = \min_{i \le j} \ell_i$ (`np.minimum.accumulate`), con $\ell_i$ la log-loss de validación. Se exige que el
# ganador declarado (`mejor.numero`) sea un ensayo completo con el mínimo $\min_i \ell_i$; si no, el JSON no es
# coherente. `n_startup_trials` marca el fin de la fase aleatoria del TPE.

# %%
def trayectoria_busqueda(hpo: dict) -> dict:
    """Log-loss de val por ensayo COMPLETE, su mínimo acumulado, el ganador y el fin de la fase aleatoria."""
    completos = sorted((e for e in hpo["ensayos"] if e["estado"] == ESTADO_COMPLETO), key=lambda e: e["numero"])  # ok
    if not completos:  # nada que dibujar
        raise ValueError(f"hpo_ensayos.json no tiene ensayos {ESTADO_COMPLETO}")  # error claro
    perdidas = np.array([e["logloss_val"] for e in completos], dtype=float)  # ℓ_i
    por_numero = {e["numero"]: e["logloss_val"] for e in completos}  # para verificar el ganador
    mejor = hpo["mejor"]["numero"]  # ganador declarado
    if mejor not in por_numero or por_numero[mejor] != perdidas.min():  # no es completo o no es el mínimo
        raise ValueError(f"El mejor ensayo declarado ({mejor}) no es un {ESTADO_COMPLETO} con la log-loss mínima")
    return {"numeros": [e["numero"] for e in completos], "logloss_val": perdidas.tolist(),  # puntos
            "mejor_hasta_ahora": np.minimum.accumulate(perdidas).tolist(),  # m_j
            "mejor_numero": mejor, "n_startup": hpo["muestreador"]["n_startup_trials"]}  # ganador y fase aleatoria


# %% [markdown]
# ## Búsqueda de C de la logística
#
# Las filas de `busqueda_logistica.json` (B3) ordenadas por $C$ ascendente; el $C$ elegido (mínima log-loss de
# validación, B-pd §3.1) debe ser uno de la rejilla.

# %%
def datos_busqueda_c(busqueda: dict) -> dict:
    """C ascendente con su log-loss y AUC de validación, el C elegido y si quedó en el borde."""
    filas = sorted(busqueda["filas"], key=lambda f: f["C"])  # por C
    rejilla = [f["C"] for f in filas]  # valores probados
    if busqueda["C_elegido"] not in rejilla:  # elegido que no se probó
        raise ValueError(f"C_elegido = {busqueda['C_elegido']} no está en la rejilla {rejilla}")  # incoherente
    return {"C": rejilla, "logloss_val": [f["logloss_val"] for f in filas],  # criterio
            "auc_val": [f["auc_val"] for f in filas],  # discriminación
            "C_elegido": busqueda["C_elegido"], "en_borde": busqueda["en_borde"]}  # elección


# %% [markdown]
# ## Figura: ROC
#
# Las dos curvas en un mismo eje con la diagonal del azar (AUC = 0,5); AUC y Gini en la leyenda.

# %%
def figura_roc(rocs: dict, destino: Path) -> Path:
    """pd-roc.png: ROC de prueba de los dos modelos y la diagonal."""
    fig, ax = plt.subplots(figsize=(5.8, 5.4))  # un solo eje
    for m in MODELOS:  # color por entidad
        d = rocs[m]  # datos del modelo
        ax.plot(d["fpr"], d["tpr"], color=p.COLOR_MODELO[m],  # curva
                label=f"{p.NOMBRE_MODELO[m]}: AUC = {d['auc']:.3f}, Gini = {d['gini']:.3f}")  # resumen
    ax.plot([0, 1], [0, 1], color=p.TINTA_TENUE, linestyle="--", linewidth=1, label="Azar (AUC = 0.5)")  # diagonal
    ax.set_xlim(0, 1)  # FPR en [0, 1]
    ax.set_ylim(0, 1)  # TPR en [0, 1]
    ax.set_aspect("equal")  # cuadrado
    ax.set_xlabel("Tasa de falsos positivos (buenos señalados)")  # FPR
    ax.set_ylabel("Tasa de verdaderos positivos (malos detectados)")  # TPR
    ax.set_title("Curva ROC en Prueba")  # test
    ax.legend(loc="lower right", fontsize=8)  # leyenda
    return _guardar(fig, destino / "pd-roc.png")  # escribe


# %% [markdown]
# ## Figura: KS
#
# Un panel por modelo: $F_1$ (malos) y $F_0$ (buenos) contra la PD, y el segmento vertical de la separación
# máxima en $t^\*$.

# %%
def figura_ks(kss: dict, destino: Path) -> Path:
    """pd-ks.png: acumuladas de la PD en malos y buenos y el KS, un panel por modelo."""
    fig, ejes = plt.subplots(1, len(MODELOS), figsize=(11, 4.2), sharey=True)  # un panel por modelo
    for ax, m in zip(ejes, MODELOS):  # cada modelo
        d = kss[m]  # datos del modelo
        ax.step(d["umbrales"], d["f_malos"], where="post", color=p.COLOR_TASA, label="Malos: F₁(t)")  # F₁
        ax.step(d["umbrales"], d["f_buenos"], where="post", color=p.TINTA_SECUNDARIA, label="Buenos: F₀(t)")  # F₀
        i = int(np.flatnonzero(np.asarray(d["umbrales"]) == d["umbral_ks"])[0])  # índice de t*
        ax.vlines(d["umbral_ks"], d["f_malos"][i], d["f_buenos"][i], color=p.COLOR_MODELO[m], linewidth=3,  # KS
                  label=f"KS = {d['ks']:.3f} en PD = {d['umbral_ks']:.3f}")  # valor y umbral
        ax.set_title(p.NOMBRE_MODELO[m])  # entidad
        ax.set_xlabel("PD predicha (t)")  # eje x
        ax.legend(loc="lower right", fontsize=8)  # leyenda
    ejes[0].set_ylabel("Fracción acumulada con PD ≤ t")  # eje y compartido
    fig.suptitle("Separación de malos y buenos (KS) en Prueba")  # título general
    fig.tight_layout()  # sin solapes
    return _guardar(fig, destino / "pd-ks.png")  # escribe


# %% [markdown]
# ## Figura: calibración
#
# $\bar y_k$ contra $\bar p_k$ para los dos modelos con la diagonal $\bar y = \bar p$ (calibración perfecta). Los
# ejes van de 0 al máximo de los puntos redondeado hacia arriba a 0,1, para que los puntos no queden aplastados.

# %%
def figura_calibracion(calib: dict, destino: Path) -> Path:
    """pd-calibracion.png: PD media frente a tasa observada por bin, con la diagonal."""
    bloques = calib["modelos"]  # puntos por modelo
    maximo = max(max(b["pd_media"] + b["tasa_observada"]) for b in bloques.values())  # punto más alto
    limite = np.ceil(maximo / PASO_EJE_CALIBRACION) * PASO_EJE_CALIBRACION  # redondeo hacia arriba (visual)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))  # un solo eje
    for m, marcador in zip(MODELOS, ("o", "s")):  # color y marcador por entidad
        b = bloques[m]  # datos del modelo
        ax.plot(b["pd_media"], b["tasa_observada"], marker=marcador, markersize=5, color=p.COLOR_MODELO[m],  # curva
                label=(f"{p.NOMBRE_MODELO[m]}: PD media {b['pd_media_global']:.3f}, "  # media global de la PD
                       f"tasa {b['tasa_global']:.3f}"))  # tasa global
    ax.plot([0, limite], [0, limite], color=p.TINTA_TENUE, linestyle="--", linewidth=1,  # diagonal
            label="Calibración perfecta (y = x)")  # referencia
    ax.set_xlim(0, limite)  # mismo rango en x
    ax.set_ylim(0, limite)  # y en y
    ax.set_aspect("equal")  # cuadrado: la diagonal a 45°
    ax.set_xlabel("PD media del bin")  # \bar p_k
    ax.set_ylabel("Tasa observada de malos del bin")  # \bar y_k
    ax.set_title(f"Calibración en Prueba ({calib['n_bins']} bins por cuantiles)")  # B-pd §3.4
    ax.legend(loc="lower right", fontsize=8)  # leyenda bajo la diagonal: los puntos quedan por encima
    return _guardar(fig, destino / "pd-calibracion.png")  # escribe


# %% [markdown]
# ## Figura: semillas
#
# Tres paneles (un eje y por métrica): caja de las semillas del MLP, sus puntos con un desplazamiento horizontal
# **fijo por índice** (sin azar: la figura es idéntica en cada corrida), la semilla mediana marcada y la logística
# como línea horizontal.

# %%
_TITULOS_METRICA = {"auc_test": "AUC", "ks_test": "KS", "brier_test": "Brier (menor es mejor)"}  # paneles


def figura_semillas(por_semilla: dict, destino: Path) -> Path:
    """pd-semillas.png: distribución entre semillas del MLP de cada métrica de prueba frente a la logística."""
    fig, ejes = plt.subplots(1, len(por_semilla), figsize=(12, 4.2))  # un panel por métrica
    for ax, (nombre, d) in zip(np.atleast_1d(ejes), por_semilla.items()):  # cada métrica
        n = len(d["valores"])  # número de semillas
        x = 1 + np.linspace(-DESPLAZAMIENTO_PUNTOS, DESPLAZAMIENTO_PUNTOS, n)  # desplazamiento fijo por índice
        ax.boxplot(d["valores"], positions=[1], widths=0.5, showfliers=False,  # caja sin atípicos (ya son puntos)
                   medianprops={"color": p.COLOR_MODELO["mlp"]}, boxprops={"color": p.TINTA_SECUNDARIA},  # colores
                   whiskerprops={"color": p.TINTA_SECUNDARIA}, capprops={"color": p.TINTA_SECUNDARIA})  # bigotes
        ax.scatter(x, d["valores"], s=18, color=p.COLOR_MODELO["mlp"], zorder=3,  # puntos
                   label=f"{p.NOMBRE_MODELO['mlp']} ({n} semillas)")  # entidad
        i = d["seeds"].index(d["mediana"][0])  # posición de la mediana
        ax.scatter([x[i]], [d["mediana"][1]], s=110, facecolors="none", edgecolors=p.TINTA, linewidths=1.5,  # anillo
                   zorder=4, label=f"Semilla mediana ({d['mediana'][0]})")  # semilla de B5
        ax.axhline(d["logistica"], color=p.COLOR_MODELO["logistica"], linewidth=1.5,  # referencia
                   label=p.NOMBRE_MODELO["logistica"])  # entidad (su valor va en el título del panel)
        ax.set_xticks([])  # una sola caja: sin marcas en x
        ax.set_title(f"{_TITULOS_METRICA.get(nombre, nombre)}\nlogística = {d['logistica']:.4f}")  # métrica y valor
    manijas, etiquetas = np.atleast_1d(ejes)[0].get_legend_handles_labels()  # iguales en todos los paneles
    fig.legend(manijas, etiquetas, loc="lower center", ncol=len(etiquetas), bbox_to_anchor=(0.5, -0.06))  # abajo
    fig.suptitle("Variabilidad entre semillas del MLP en Prueba")  # título general
    fig.tight_layout()  # sin solapes
    return _guardar(fig, destino / "pd-semillas.png")  # escribe


# %% [markdown]
# ## Figura: aprendizaje
#
# BCE de entrenamiento y validación por época, la línea en la mejor época (pesos restaurados) y la franja de las
# épocas de paciencia que siguieron sin mejora hasta la parada.

# %%
def figura_aprendizaje(apr: dict, destino: Path) -> Path:
    """pd-aprendizaje.png: pérdidas por época de la semilla mediana con la parada temprana."""
    fig, ax = plt.subplots(figsize=(8, 4.2))  # un solo eje
    ax.plot(apr["epocas"], apr["perdida_train"], marker="o", markersize=4, color=p.COLOR_SPLIT["train"],  # train
            label=p.NOMBRE_SPLIT["train"])  # entidad
    ax.plot(apr["epocas"], apr["perdida_val"], marker="o", markersize=4, color=p.COLOR_SPLIT["val"],  # val
            label=p.NOMBRE_SPLIT["val"])  # entidad
    fin = min(apr["mejor_epoca"] + apr["paciencia"], apr["epocas"][-1])  # última época de la paciencia
    ax.axvspan(apr["mejor_epoca"], fin, color=p.CUADRICULA, alpha=0.6,  # franja
               label=f"Paciencia ({apr['paciencia']} épocas sin mejora)")  # regla de parada
    ax.axvline(apr["mejor_epoca"], color=p.TINTA, linestyle="--", linewidth=1.2,  # mejor época
               label=f"Mejor época ({apr['mejor_epoca']}): pesos restaurados")  # explicación
    ax.set_xlabel("Época")  # eje x
    ax.set_ylabel("Pérdida BCE media")  # eje y
    ax.set_title(f"Curva de aprendizaje del MLP (semilla mediana {apr['seed']})")  # título
    ax.legend(loc="best", fontsize=8)  # leyenda
    return _guardar(fig, destino / "pd-aprendizaje.png")  # escribe


# %% [markdown]
# ## Figura: búsqueda del MLP
#
# Un punto por ensayo completo, el escalón del mínimo acumulado $m_j$, el ganador marcado, una línea vertical al
# final de los `n_startup_trials` ensayos aleatorios del TPE y la log-loss de validación de la logística como
# referencia.

# %%
def figura_busqueda_mlp(tray: dict, logloss_logistica: float, destino: Path) -> Path:
    """pd-busqueda-mlp.png: log-loss de validación por ensayo de Optuna y el mejor hasta cada ensayo."""
    fig, ax = plt.subplots(figsize=(9, 4.2))  # un solo eje
    ax.scatter(tray["numeros"], tray["logloss_val"], s=22, color=p.COLOR_MODELO["mlp"], zorder=3,  # ensayos
               label=f"Ensayo del {p.NOMBRE_MODELO['mlp']}")  # entidad
    ax.step(tray["numeros"], tray["mejor_hasta_ahora"], where="post", color=p.TINTA_SECUNDARIA, linewidth=1.5,  # m_j
            label="Mejor hasta ese ensayo")  # mínimo acumulado
    i = tray["numeros"].index(tray["mejor_numero"])  # posición del ganador
    ax.scatter([tray["mejor_numero"]], [tray["logloss_val"][i]], s=130, facecolors="none", edgecolors=p.TINTA,  # anillo
               linewidths=1.5, zorder=4, label=f"Elegido: ensayo {tray['mejor_numero']} ({tray['logloss_val'][i]:.4f})")
    ax.axvline(tray["n_startup"] - 0.5, color=p.TINTA_TENUE, linestyle=":", linewidth=1.5,  # fin de la fase aleatoria
               label=f"Fin de los {tray['n_startup']} ensayos aleatorios del TPE")  # n_startup_trials
    ax.axhline(logloss_logistica, color=p.COLOR_MODELO["logistica"], linewidth=1.5,  # referencia
               label=f"{p.NOMBRE_MODELO['logistica']} = {logloss_logistica:.4f}")  # valor
    ax.set_xlabel("Número de ensayo")  # eje x
    ax.set_ylabel("Log-loss de Validación")  # eje y
    ax.set_title("Búsqueda TPE del MLP")  # título
    ax.legend(loc="upper right", fontsize=8)  # leyenda
    return _guardar(fig, destino / FIGURA_BUSQUEDA_MLP)  # escribe


# %% [markdown]
# ## Figura: búsqueda de C
#
# Dos paneles con $C$ en escala logarítmica: log-loss de validación (el criterio) y AUC de validación, con el $C$
# elegido marcado.

# %%
def figura_busqueda_logistica(bc: dict, destino: Path) -> Path:
    """pd-busqueda-logistica.png: log-loss y AUC de validación contra C, con el C elegido."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))  # dos paneles, un eje y cada uno
    for ax, clave, nombre in ((ax1, "logloss_val", "Log-loss de Validación (criterio)"),  # criterio
                              (ax2, "auc_val", "AUC de Validación")):  # discriminación
        ax.plot(bc["C"], bc[clave], marker="o", markersize=5, color=p.COLOR_MODELO["logistica"],  # curva
                label=p.NOMBRE_MODELO["logistica"])  # entidad
        ax.axvline(bc["C_elegido"], color=p.TINTA, linestyle="--", linewidth=1.2,  # elegido
                   label=f"C elegido = {bc['C_elegido']:g}")  # valor
        ax.set_xscale("log")  # rejilla logarítmica
        ax.set_xlabel("C (inverso de la regularización L2)")  # eje x
        ax.set_title(nombre)  # panel
        ax.legend(loc="best", fontsize=8)  # leyenda
    borde = " (en el borde de la rejilla)" if bc["en_borde"] else ""  # aviso si quedó en un extremo
    fig.suptitle(f"Búsqueda de C de la regresión logística ({len(bc['C'])} valores){borde}")  # título general
    fig.tight_layout()  # sin solapes
    return _guardar(fig, destino / "pd-busqueda-logistica.png")  # escribe


# %% [markdown]
# ## Lectura de insumos
#
# `_exigir` falla con el comando que genera el insumo que falta. `insumos_de_corrida` lee una carpeta de
# `run.py pd` (la real o la demo): `summary.json`, `pred_test.parquet`, `busqueda_logistica.json` y
# `semillas/mlp_seed*.json` (ordenadas por semilla; los `logistica_seed*.json` no entran), y `hpo_ensayos.json` si
# existe (la demo no lo tiene: queda `None`).

# %%
def _exigir(ruta: Path, insumo: str) -> Path:
    """Devuelve `ruta` si existe; si no, FileNotFoundError con la tarea y el comando que la genera."""
    if not ruta.exists():  # falta el insumo
        raise FileNotFoundError(f"falta {ruta.as_posix()}; genéralo con {COMO_GENERAR[insumo]}")  # cómo arreglarlo
    return ruta  # existe


# %%
def insumos_de_corrida(carpeta: Path) -> dict:
    """`datos` para generar() desde una carpeta de run.py pd: {resumen, pred, semillas, hpo | None, busqueda}."""
    carpeta = Path(carpeta)  # admite str
    resumen = load_json(_exigir(carpeta / "summary.json", "summary.json"))  # resumen por modelo
    pred = pd.read_parquet(_exigir(carpeta / "pred_test.parquet", "pred_test.parquet"))  # PD de test
    busqueda = load_json(_exigir(carpeta / "busqueda_logistica.json", "busqueda_logistica.json"))  # rejilla de C
    ruta_hpo = carpeta / "hpo_ensayos.json"  # opcional
    hpo = load_json(ruta_hpo) if ruta_hpo.exists() else None  # None en la demo
    sem = carpeta / "semillas"  # carpeta de semillas
    semillas = [load_json(r) for r in sem.glob("mlp_seed*.json")] if sem.is_dir() else []  # solo las del MLP
    semillas.sort(key=lambda r: r["seed"])  # por semilla (el glob ordena como texto)
    return {"resumen": resumen, "pred": pred, "semillas": semillas, "hpo": hpo, "busqueda": busqueda}  # insumos


# %% [markdown]
# ## Escritura
#
# `_guardar` escribe y cierra una figura; `_escribir_json` usa `results.save_json` (UTF-8, sangría 2, sin NaN,
# escritura atómica).

# %%
def _guardar(fig, ruta: Path) -> Path:
    """Escribe la figura en `ruta` (crea la carpeta) y la cierra."""
    ruta.parent.mkdir(parents=True, exist_ok=True)  # carpeta destino
    fig.savefig(ruta)  # PNG con el estilo común
    plt.close(fig)  # libera memoria
    return ruta  # ruta escrita


# %%
def _escribir_json(datos: dict, ruta: Path) -> Path:
    """Escribe `datos` como JSON con save_json (sin NaN; escritura atómica)."""
    return save_json(ruta, datos)  # UTF-8, sangría 2


# %% [markdown]
# ## Orquestación
#
# `generar` aplica el estilo, resuelve `destino or raiz / SALIDA` **al llamarse** (así `test_no_network` lo
# redirige), lee los insumos (o usa `datos`), calcula **todos** los datos puros antes de dibujar (un insumo
# incoherente falla sin dejar figuras a medias) y dibuja. Si `hpo` es `None` no dibuja `pd-busqueda-mlp.png`.

# %%
def _validar_datos(datos: dict) -> None:
    """ValueError si `datos` no tiene exactamente las claves de CLAVES_DATOS o no hay semillas."""
    faltan, sobran = CLAVES_DATOS - set(datos), set(datos) - CLAVES_DATOS  # diferencias de claves
    if faltan or sobran:  # claves de menos o de más (p. ej. un error de tipeo)
        raise ValueError(f"`datos` debe tener las claves {sorted(CLAVES_DATOS)}; faltan {sorted(faltan)}, "
                         f"sobran {sorted(sobran)}")  # nombra ambas
    if not datos["semillas"]:  # sin mlp_seed*.json
        raise ValueError(f"No hay semillas del MLP (mlp_seed*.json); genéralas con {COMO_GENERAR['semillas']}")


# %%
def generar(raiz: Path = Path("."), destino: Path | None = None, datos: dict | None = None) -> list[Path]:
    """Regenera las figuras de la Parte B y pd-calibracion.json desde results/data/pd/ y devuelve sus rutas."""
    aplicar_estilo()  # estilo común
    destino = Path(destino) if destino is not None else Path(raiz) / SALIDA  # SALIDA leído al llamar
    datos = datos if datos is not None else insumos_de_corrida(Path(raiz) / ENTRADA)  # lectura o inyección
    _validar_datos(datos)  # claves exactas y semillas
    resumen, pred, semillas, hpo = datos["resumen"], datos["pred"], datos["semillas"], datos["hpo"]  # insumos
    y = pred[COLUMNA_Y].to_numpy()  # desenlace de test
    calib = datos_calibracion(pred, resumen["mlp"]["semilla_mediana"])  # valida columnas primero
    rocs = {m: datos_roc(y, pred[f"pred_{m}"].to_numpy()) for m in MODELOS}  # ROC por modelo
    kss = {m: datos_ks(y, pred[f"pred_{m}"].to_numpy()) for m in MODELOS}  # KS por modelo
    por_semilla = metricas_por_semilla(semillas, resumen)  # valida mediana y duplicados
    apr = datos_aprendizaje(semillas, resumen)  # semilla mediana
    tray = trayectoria_busqueda(hpo) if hpo is not None else None  # opcional
    bc = datos_busqueda_c(datos["busqueda"])  # rejilla de C
    rutas = [figura_roc(rocs, destino), figura_ks(kss, destino), figura_calibracion(calib, destino),  # discriminación
             _escribir_json(calib, destino / ARCHIVO_CALIBRACION), figura_semillas(por_semilla, destino),  # JSON
             figura_aprendizaje(apr, destino)]  # aprendizaje
    if tray is not None:  # solo con hpo_ensayos.json
        logloss_log = resumen["logistica"]["metricas"]["logloss_val"]["media"]  # referencia de la logística
        rutas.append(figura_busqueda_mlp(tray, logloss_log, destino))  # búsqueda Optuna
    rutas.append(figura_busqueda_logistica(bc, destino))  # búsqueda de C
    return rutas  # en el orden de ARCHIVOS
