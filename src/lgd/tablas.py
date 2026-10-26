"""Tablas y utilidades del notebook ``04_lgd`` (tarea D8; núcleo §5.8: cero lógica en el notebook).

Solo leen los JSON que ya produjeron D1/D1b/D4 (``resumen_objetivo.json``, ``descriptivo.json``, ``summary.json``) y los
ordenan en tablas; no entrenan ni recalculan métricas. ``redirigir_salidas`` apunta la salida de la demo a una carpeta
temporal para que el notebook no escriba en ``results/`` ni en ``models/``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from src.lgd.experimento import METRICAS, MODELOS, SPLITS

NOMBRE_METRICA = {"mae_val": "MAE val", "rmse_val": "RMSE val", "mae_test": "MAE test", "rmse_test": "RMSE test"}
DECIMALES = 4  # solo presentación de la tabla compacta
SIN_DATO = "—"


def redirigir_salidas(cfg: dict, carpeta: str | Path) -> dict:
    """Copia profunda de ``cfg`` con ``salida`` y ``modelos.*`` dentro de ``carpeta`` (mismo nombre de archivo).

    ``entrenar_semillas`` usa ``raiz / cfg["salida"]``; con una ruta absoluta el resultado es ``carpeta``.
    """
    carpeta = Path(carpeta)
    nueva = copy.deepcopy(cfg)
    nueva["salida"] = str(carpeta)
    nueva["modelos"] = {k: str(carpeta / Path(v).name) for k, v in cfg["modelos"].items()}
    return nueva


# ----------------------------------------------------------------------------------------- resultados (summary.json)
def tabla_metricas(resumen: dict) -> pd.DataFrame:
    """Lineal, MLP y referencias constantes de ``summary.json`` (por modelo, como el de B: ``resumen[modelo]``):
    MAE/RMSE (media y desv), semillas, parámetros, configuraciones probadas y tiempo. Las constantes no tienen desv
    (0) ni tiempo (NaN); ``configuraciones`` del MLP es null si su config no es la de la búsqueda (núcleo §5.4)."""
    filas = {}
    for nombre in MODELOS:
        m = resumen[nombre]
        fila = {}
        for k in METRICAS:
            fila[NOMBRE_METRICA[k]] = m["metricas"][k]["media"]
            fila[f"{NOMBRE_METRICA[k]} (desv)"] = m["metricas"][k]["desv"]
        fila.update({"semillas": m["metricas"]["mae_test"]["n"], "parámetros": m["n_parametros"],
                     "configuraciones": m["n_configuraciones_busqueda"], "tiempo (s)": m["tiempo_total_s"]})
        filas[nombre] = fila
    for nombre, ref in resumen["referencias"].items():
        fila = {}
        for k in METRICAS:
            fila[NOMBRE_METRICA[k]] = ref["metricas"][k]
            fila[f"{NOMBRE_METRICA[k]} (desv)"] = 0.0
        fila.update({"semillas": 1, "parámetros": ref["n_parametros"],
                     "configuraciones": ref["n_configuraciones_busqueda"], "tiempo (s)": float("nan")})
        filas[nombre] = fila
    return pd.DataFrame.from_dict(filas, orient="index")


def tabla_metricas_compacta(resumen: dict, decimales: int = DECIMALES) -> pd.DataFrame:
    """La misma tabla con cada métrica en una columna ``"media ± desv"`` (solo la media si hay una corrida), para
    proyectar."""
    t = tabla_metricas(resumen)
    out = pd.DataFrame(index=t.index)
    for k in METRICAS:
        col = NOMBRE_METRICA[k]
        out[col] = [f"{m:.{decimales}f} ± {d:.{decimales}f}" if n > 1 else f"{m:.{decimales}f}"
                    for m, d, n in zip(t[col], t[f"{col} (desv)"], t["semillas"])]
    for col in ("semillas", "parámetros"):
        out[col] = t[col]
    out["configuraciones"] = [SIN_DATO if pd.isna(v) else int(v) for v in t["configuraciones"]]
    out["tiempo (s)"] = [SIN_DATO if pd.isna(v) else f"{v:.1f}" for v in t["tiempo (s)"]]
    return out


def tabla_referencias(resumen: dict) -> pd.DataFrame:
    """Valor de cada predictor constante de referencia (``summary.json`` → ``referencias[*].valor``, calculado en
    train)."""
    return pd.DataFrame({"valor (de train)": {n: r["valor"] for n, r in resumen["referencias"].items()}})


def comparacion(resumen: dict, metrica: str = "mae_test") -> dict:
    """``metrica`` (media) de cada modelo y referencia, diferencias MLP − otro (``mlp_menos``) y el de menor valor."""
    if metrica not in METRICAS:
        raise ValueError(f"métrica desconocida {metrica!r}; opciones: {', '.join(METRICAS)}")
    valores = {n: resumen[n]["metricas"][metrica]["media"] for n in MODELOS}
    valores.update({n: r["metricas"][metrica] for n, r in resumen["referencias"].items()})
    return {"metrica": metrica, "valores": valores, "desv_mlp": resumen["mlp"]["metricas"][metrica]["desv"],
            "mejor": min(valores, key=valores.get),
            "mlp_menos": {n: valores["mlp"] - v for n, v in valores.items() if n != "mlp"}}


def describir_mlp(mlp: dict) -> str:
    """Una línea con la procedencia, la arquitectura y el presupuesto de la entrada ``mlp`` de ``summary.json``.

    Procedencia: ``busqueda.origen`` si ``config_de_busqueda``; si no, lo dice (el bloque ``busqueda`` es null).
    """
    if mlp["config_de_busqueda"]:
        origen = mlp["busqueda"].get("origen", mlp["busqueda"]["tipo"])
    else:
        origen = "fuera de la búsqueda (config_de_busqueda: false)"
    t_busqueda = mlp["tiempo_busqueda_s"]
    busqueda = SIN_DATO if t_busqueda is None else f"{t_busqueda:.1f} s"
    return (f"Configuración del MLP: {origen} · capas {mlp['config']['capas']} · pérdida {mlp['config']['perdida']} · "
            f"búsqueda {busqueda} · semillas {mlp['tiempo_semillas_s']:.1f} s")


# ------------------------------------------------------------------------- LGD observada (descriptivo["observado"])
def _fila_grupo(g: dict) -> dict:
    return {"créditos": g["n"], "LGD media": g["lgd"]["media"], "LGD mediana": g["lgd"]["mediana"],
            "fracción LGD = 1": g["fraccion_lgd_1"], "fracción LGD = 0": g["fraccion_lgd_0"],
            "EAD mediana": g["ead"]["mediana"]}


def tabla_por_split(obs: dict) -> pd.DataFrame:
    """LGD observada por partición (``descriptivo.json``, clave ``observado``)."""
    return pd.DataFrame.from_dict({s: _fila_grupo(obs["por_split"][s]) for s in SPLITS if s in obs["por_split"]},
                                  orient="index")


def tabla_por_anio(obs: dict) -> pd.DataFrame:
    """LGD observada por año de originación, con la partición a la que pertenece cada año."""
    filas = {a: {"split": ", ".join(obs["split_por_anio"][a]), **_fila_grupo(g)} for a, g in obs["por_anio"].items()}
    return pd.DataFrame.from_dict(filas, orient="index")


def tabla_por_plazo(obs: dict) -> pd.DataFrame:
    """EAD y LGD observadas por plazo (``term`` tal como viene en los datos)."""
    return pd.DataFrame.from_dict({p: _fila_grupo(g) for p, g in obs["por_plazo"].items()}, orient="index")


def tabla_poblacion(resumen_objetivo: dict) -> pd.DataFrame:
    """Población de la LGD, exclusiones y masa en 0 y 1 (``resumen_objetivo.json``) en una tabla de una columna."""
    r = resumen_objetivo
    filas = {"créditos en loans_pd": r["n_total"], "población LGD": r["n_poblacion"]}
    filas.update({f"población: {e}": n for e, n in r["poblacion_por_estado"].items()})
    filas.update({f"excluidos: {e}": n for e, n in r["excluidos_por_estado"].items()})
    filas.update({f"fuera de la población: {e}": n for e, n in r["otros_estados"].items()})
    filas.update({"EAD ≤ 0 (excluidos)": r["n_ead_no_positiva"], "LGD = 0": r["n_lgd_igual_0"],
                  "LGD = 1": r["n_lgd_igual_1"], "LGD < 0 antes del recorte": r["n_lgd_sin_recorte_menor_0"],
                  "LGD > 1 antes del recorte": r["n_lgd_sin_recorte_mayor_1"]})
    return pd.DataFrame({"valor": filas})


# ------------------------------------------------------ después del preprocesamiento (descriptivo["preprocesado"])
def tabla_efecto_transformaciones(pre: dict) -> pd.DataFrame:
    """Fracción de filas imputadas / recortadas (numéricas) o desconocidas / llevadas a OTRA (categóricas) por variable;
    columnas (split, efecto). NaN donde el efecto no aplica a la variable."""
    efecto = pre["efecto_transformaciones"]
    partes = {s: pd.DataFrame.from_dict(efecto[s], orient="index") for s in efecto}
    return pd.concat(partes, axis=1)


def tabla_psi(pre: dict, por: str = "test") -> pd.DataFrame:
    """PSI de cada variable transformada frente a train, por split; de mayor a menor PSI del split ``por``."""
    t = pd.DataFrame(pre["psi_vs_train"])
    if por not in t.columns:
        raise ValueError(f"split {por!r} sin PSI; opciones: {', '.join(t.columns)}")
    return t.sort_values(por, ascending=False)


def tabla_relacion_objetivo(pre: dict) -> pd.DataFrame:
    """Relación de cada variable numérica transformada con la LGD en train (LGD media por grupo; ρ de Spearman)."""
    filas = {v: {"ρ (Spearman por grupos)": r["spearman_grupos"], "n_grupos": r["n_grupos"], "monótona": r["monotona"],
                 "LGD media grupo bajo": r["tasa_grupo_bajo"], "LGD media grupo alto": r["tasa_grupo_alto"]}
             for v, r in pre["relacion_con_objetivo_train"].items()}
    return pd.DataFrame.from_dict(filas, orient="index")
