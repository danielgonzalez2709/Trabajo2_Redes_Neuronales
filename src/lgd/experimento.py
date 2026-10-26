"""Experimento de la LGD: búsqueda Optuna, 30 semillas, ``summary.json`` y predicciones (tarea D4; Parte D §3.3–§3.4;
núcleo §5.4, §6).

    python run.py lgd-hpo --config configs/lgd/hpo.yaml    # búsqueda → hpo_ensayos.json + reescribe mlp_final.yaml
    python run.py lgd     --config configs/lgd/base.yaml   # lineal 1 vez + MLP con 30 semillas
    python run.py lgd     --config configs/lgd/demo.yaml   # DEMO (< 2 min CPU; escribe en results/data/lgd_demo/)

Datos: ``loans_lgd.parquet`` (D2) y ``preprocesador_lgd.json`` (D2) **tal cual**: aquí nunca se llama ``fit`` del
preprocesador ([DECIDIDO, Camila, 24-oct]: lo comparten lineal y MLP y se aplica también al test de ``loans_pd``).
``datos.muestra`` (opcional, para la demo): n total de créditos, repartido entre train/val/test en proporción a su
tamaño (redondeo por restos mayores) y elegido con ``base_seed``; si deja un split vacío → ``ValueError``.

Configuración final del MLP: ``configs/lgd/base.yaml`` declara ``hereda: mlp_final.yaml``, que trae el bloque ``mlp`` y
su procedencia (bloque ``busqueda``). ``run.py lgd-hpo`` **reescribe** ese archivo (ruta ``hpo.destino``; la demo usa
otra ruta). ``run.py lgd`` entrena con la config tal como la resolvió el cargador (herencia y luego ``--set``): no
fusiona nada después, así que ``--set mlp.*`` siempre gana y la config impresa es la que se entrena.
De la procedencia salen ``config_de_busqueda`` y el presupuesto de la búsqueda (semántica de B, B-pd §3.5): true si
``tipo: optuna`` y los bloques ``mlp`` y ``busqueda`` de la corrida son los de ``mlp_final.yaml`` (``hpo.destino``)
releído; entonces ``n_configuraciones_busqueda = busqueda.n_ensayos`` y ``tiempo_busqueda_s = busqueda.tiempo_s``. Si
es false (``sin_busqueda``, ``--set mlp.*`` u otro bloque ``busqueda``), los dos y ``busqueda`` quedan en null y
``tiempo_total_s`` = ``tiempo_semillas_s`` (núcleo §5.4). Con ``optuna`` y sin ese archivo → ``ValueError``.
El bloque ``busqueda`` sigue el formato de B (``configs/pd/mlp_final.yaml``): ``evidencia`` =
``{sha256_particion, sha256_preprocesador}`` de la búsqueda (``null`` con ``tipo: sin_busqueda``). Con ``tipo: optuna``,
si falta ``busqueda.evidencia`` (formato anterior a D9) o si alguna de sus huellas o la ``muestra`` de la búsqueda no
coinciden con la corrida actual → ``ValueError`` (p. ej. búsqueda con muestra y corrida final sin muestra).

Búsqueda (``buscar``):
- Optuna ``TPESampler(seed=base_seed)``, ``hpo.n_ensayos`` ensayos, espacio ``hpo.espacio`` (el de B §3.3): número de
  capas entero en [1, 4]; neuronas elegidas **por capa** de la lista; dropout uniforme; ``l2`` y ``lr``
  log-uniformes; activación, optimizador y batch categóricos. Con ``sgd`` el momento es ``mlp.momento`` (no se busca).
  Las demás claves de ``mlp`` (``max_epocas``, ``paciencia``, ``device``, ``perdida``) se toman de la config.
- Cada ensayo entrena el MLP con la semilla fija ``base_seed`` (la misma en todos los ensayos).
- Criterio de selección (``CRITERIO_SELECCION``) **[PROPUESTO por el orquestador]**: la pérdida de validación del
  propio MLP (la media de ``mlp.perdida`` en val en la mejor época, ``min(historial_["val"])``); el MAE de val va al
  lado en cada ensayo. Un ensayo que diverge (``RuntimeError`` del MLP) queda con estado ``FAIL``; si todos fallan,
  ``RuntimeError`` y no se reescribe nada.
- Nota sobre ``l2``: en D es ``(l2/2)·Σ‖W‖²`` explícita sin sesgos (gradiente ``l2·W`` ≡ ``weight_decay = l2``), la
  misma convención que B (Parte B §3.2, Spec v0.13): mismo rango y mismo significado.

Tiempo de cómputo (cifra de presupuesto del núcleo §6), **misma definición en búsqueda y semillas**: suma del tiempo de
reloj de cada llamada a ``fit`` (una por ensayo o por semilla, incluidos los ensayos fallidos), sin la carga de datos ni la
escritura de resultados. ``tiempo_total_s`` del MLP = búsqueda + semillas (solo semillas si ``config_de_busqueda`` es
false).

Semillas (``entrenar_semillas``):
- Lineal 1 vez (``seed = base_seed``; MCO es determinista) y MLP con ``base_seed + k``, k = 0…``semillas.n``−1; un JSON
  por (modelo, semilla) en ``{salida}/semillas/{modelo}_seed{k}.json``.
- ``semillas/mlp_seed{k}.json`` lleva además, como B (núcleo §5.4, v0.16): ``mejor_epoca``, ``mejor_perdida_val``
  (pérdida de val de esa época), ``historial [{epoca, perdida_train, perdida_val}]`` (épocas desde 1),
  ``hilos_torch`` y ``version_torch``; ``n`` y ``muestra`` en las dos (lineal y MLP).
- ``summary.json`` **por modelo**, como el de B (tarea D10): raíz ``{lineal, mlp, referencias}``. Por modelo,
  ``n_parametros``, ``n_configuraciones_busqueda``, ``tiempo_total_s``, ``metricas`` (MAE y RMSE en val y test:
  ``{media, desv, mejor, peor, n}``; ``desv`` muestral, ddof = 1, y 0 si n = 1; mejor = mínimo) y ``evidencia``.
  ``lineal`` añade ``fuera_de_rango`` y ``nota_pred``. ``mlp`` añade los campos de la entrada ``mlp`` de B:
  ``tiempo_busqueda_s``, ``tiempo_semillas_s`` (Σ de los fit), ``semillas``, ``semilla_mediana``,
  ``criterio_semilla_mediana``, ``config`` (bloque ``mlp``), ``config_de_busqueda``, ``busqueda`` (bloque de
  ``mlp_final.yaml`` o null), ``hilos_torch``, ``version_torch`` (de la semilla mediana), y ``base_seed``, ``n``
  (créditos por split) y ``muestra``.
- ``referencias`` (en la raíz, no es un modelo; [DECIDIDO, Camila, 25-oct], D §3.4): predictor constante con la
  media y con la mediana de la LGD de **train** (sin fuga), con su valor y MAE/RMSE en val y test; ``n_parametros`` 1 y
  ``n_configuraciones_busqueda`` 1. No van en ``pred_test``/``pred_pd_test``.
- La lineal no se recorta: ``pred_lineal`` puede salir de [0, 1], también en ``pred_pd_test.parquet`` (y
  ``src/el/perdida_esperada.py`` rechaza LGD fuera de [0, 1]). ``fuera_de_rango`` cuenta cuántas veces en val, test
  (LGD) y ``pd_test`` (todo el test de ``loans_pd``).
- ``pred_test.parquet`` (test de la LGD: ``id, y, pred_lineal, pred_mlp``) y ``pred_pd_test.parquet`` (todo el test de
  ``loans_pd``: ``id, pred_lineal, pred_mlp``) con la semilla del MLP de **MAE de test mediano** (``_semilla_mediana``);
  esa semilla se guarda en ``mlp.semilla_mediana``.
- ``evidencia`` = ``{sha256_particion, sha256_preprocesador}`` (núcleo §5.4 y §6) en cada ``semillas/*.json``, en cada
  modelo de ``summary.json`` y en ``hpo_ensayos.json``: huella de las líneas ``"{id},{split}\\n"`` del mismo DataFrame
  de ``loans_lgd`` del que salen las matrices (tras ``datos.muestra``, en su orden) y de los bytes del archivo del
  preprocesador; se definen en ``src/lgd/procedencia.py`` (igual que ``src/pd/experimento.py``).
"""

from __future__ import annotations

import copy
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json, save_json
from src.common.seeds import make_rng, seeds_for_runs, validar_entero
from src.lgd.lineal import LinealLGD
from src.lgd.metricas import mae, rmse
from src.lgd.mlp import MLPLGD
from src.lgd.preparar import rechazar_posteriores
from src.lgd.procedencia import evidencia

SPLITS = ("train", "val", "test")
MODELOS = ("lineal", "mlp")  # claves de summary.json además de "referencias" (núcleo §5.4, como B)
NOTA_PRED_LINEAL = "pred_lineal no se recorta: puede salir de [0, 1] (ver lineal.fuera_de_rango)"
METRICAS = ("mae_val", "rmse_val", "mae_test", "rmse_test")
ARCHIVO_HPO = "hpo_ensayos.json"
CRITERIO_SELECCION = ("mínima pérdida de validación del propio MLP (media de mlp.perdida en val, mejor época); "
                      "MAE de val al lado [PROPUESTO por el orquestador]")
CRITERIO_MEDIANA = "MAE de test mediano: posición (n−1)//2 al ordenar por (mae_test, seed)"
TIPOS_BUSQUEDA = ("sin_busqueda", "optuna")
CABECERA_MLP_FINAL = """# Generado por `python run.py lgd-hpo` (src/lgd/experimento.py). No se edita a mano.
# Configuración final del MLP de LGD (Parte D §3.3) y su procedencia; la hereda configs/lgd/base.yaml.
"""


# ------------------------------------------------------------------------------------------------------------ datos
def _repartir(n_total: int, tamanos: dict[str, int]) -> dict[str, int]:
    """Reparto proporcional de ``n_total`` entre splits (restos mayores; empate → orden de SPLITS)."""
    total = sum(tamanos.values())
    cuotas = {s: n_total * tamanos[s] / total for s in tamanos}
    n = {s: int(np.floor(c)) for s, c in cuotas.items()}
    faltan = n_total - sum(n.values())
    for s in sorted(tamanos, key=lambda s: (-(cuotas[s] - n[s]), SPLITS.index(s)))[:faltan]:
        n[s] += 1
    return n


def _muestrear(df: pd.DataFrame, muestra, base_seed: int) -> pd.DataFrame:
    if muestra is None:
        return df
    muestra = validar_entero(muestra, "datos.muestra", minimo=len(SPLITS))
    if muestra >= len(df):
        return df
    n = _repartir(muestra, {s: int((df["split"] == s).sum()) for s in SPLITS})
    vacios = [s for s in SPLITS if n[s] == 0]
    if vacios:
        raise ValueError(f"datos.muestra={muestra} deja vacío el split {vacios} (reparto {n}); usa una muestra mayor")
    rng = make_rng(base_seed, 0)
    elegidos = [np.sort(rng.choice(np.flatnonzero(df["split"].to_numpy() == s), size=n[s], replace=False))
                for s in SPLITS]
    return df.iloc[np.sort(np.concatenate(elegidos))].reset_index(drop=True)


def cargar_datos(cfg: dict, raiz: Path = Path(".")) -> dict:
    """``X``/``y``/``ids`` por split de la LGD, ``X`` e ``ids`` del test de ``loans_pd`` y la ``evidencia``.

    Usa el preprocesador guardado (``Preprocesador.from_dict``): nunca lo reajusta.
    """
    raiz = Path(raiz)
    datos = cfg["datos"]
    ruta_pre = raiz / datos["salida"]["preprocesador"]
    pre_dict = load_json(ruta_pre)
    pre = Preprocesador.from_dict(pre_dict)
    nombres = [v["nombre"] for v in pre.variables]
    rechazar_posteriores(nombres)

    lgd = pd.read_parquet(raiz / datos["salida"]["loans_lgd"])
    lgd = _muestrear(lgd, datos["muestra"], validar_entero(cfg["base_seed"], "base_seed"))
    ev = evidencia(lgd, ruta_pre)  # el mismo df del que salen las matrices
    X, y, ids = {}, {}, {}
    for s in SPLITS:
        g = lgd[lgd["split"] == s]
        if g.empty:
            raise ValueError(f"loans_lgd no tiene filas en el split {s!r}")
        X[s] = pre.transform(g[nombres])
        y[s] = g["lgd"].to_numpy(dtype=np.float64)
        ids[s] = g["id"].astype(str).to_numpy()

    pd_test = pd.read_parquet(raiz / datos["loans_pd"], columns=["id", "split", *nombres])
    pd_test = pd_test[pd_test["split"] == "test"]
    return {"X": X, "y": y, "ids": ids, "X_pd_test": pre.transform(pd_test[nombres]),
            "ids_pd_test": pd_test["id"].astype(str).to_numpy(),
            "evidencia": ev, "n": {s: int(len(y[s])) for s in SPLITS}, "muestra": datos["muestra"]}


def _metricas(modelo, d: dict) -> dict:
    p_val, p_test = modelo.predict(d["X"]["val"]), modelo.predict(d["X"]["test"])
    return {"mae_val": mae(d["y"]["val"], p_val), "rmse_val": rmse(d["y"]["val"], p_val),
            "mae_test": mae(d["y"]["test"], p_test), "rmse_test": rmse(d["y"]["test"], p_test)}


def _fit_cronometrado(modelo, d: dict, cfg_modelo: dict | None, seed: int):
    """``fit`` con train y val; devuelve (modelo, segundos de reloj de la llamada). Definición única de tiempo."""
    t0 = time.perf_counter()
    try:
        modelo.fit(d["X"]["train"], d["y"]["train"], d["X"]["val"], d["y"]["val"], cfg_modelo, seed)
    except RuntimeError as exc:
        exc.tiempo_s = time.perf_counter() - t0
        raise
    return modelo, time.perf_counter() - t0


# ---------------------------------------------------------------------------------------------------------- mlp_final
def escribir_mlp_final(ruta: Path, mlp: dict, busqueda: dict) -> Path:
    """Escribe ``mlp_final.yaml`` (cabecera + bloques ``mlp`` y ``busqueda``)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    texto = CABECERA_MLP_FINAL + yaml.safe_dump({"mlp": mlp, "busqueda": busqueda}, sort_keys=False,
                                                allow_unicode=True, default_flow_style=False)
    temporal = ruta.with_name(f".{ruta.name}.tmp")
    temporal.write_text(texto, encoding="utf-8")
    temporal.replace(ruta)
    return ruta


def _validar_formato_evidencia(ev_busqueda, ev_actual: dict) -> None:
    """``busqueda.evidencia`` debe ser un mapeo con las claves de la evidencia de esta corrida (formato de B)."""
    if not isinstance(ev_busqueda, dict) or set(ev_busqueda) != set(ev_actual):
        raise ValueError(f"busqueda.evidencia debe ser {{{', '.join(ev_actual)}}} (formato de B; se recibió "
                         f"{ev_busqueda!r}): repite lgd-hpo para regenerar mlp_final.yaml")


def _validar_mismos_datos(b: dict, d: dict) -> None:
    """``ValueError`` si alguna huella de ``busqueda.evidencia`` o la ``muestra`` de la búsqueda no es la de esta
    corrida (lista plana ``(nombre, registrado, actual)``)."""
    _validar_formato_evidencia(b.get("evidencia"), d["evidencia"])
    comparaciones = [(f"evidencia.{k}", b["evidencia"][k], v) for k, v in d["evidencia"].items()]
    comparaciones.append(("muestra", b["muestra"], d["muestra"]))
    for nombre, registrado, actual in comparaciones:
        if registrado != actual:
            raise ValueError(f"busqueda.{nombre} = {registrado!r} no coincide con esta corrida ({actual!r}): la "
                             "búsqueda se hizo con otros datos; repite lgd-hpo con la misma partición, preprocesador y "
                             "muestra")


def _leer_procedencia(cfg: dict, d: dict, raiz: Path) -> dict:
    """``{config_de_busqueda, n_configuraciones_busqueda, tiempo_busqueda_s, busqueda}`` (núcleo §5.4; semántica de
    B, B-pd §3.5).

    ``config_de_busqueda`` es true si ``busqueda.tipo`` es ``optuna`` y los bloques ``mlp`` y ``busqueda`` de la
    corrida son iguales a los de ``mlp_final.yaml`` (``hpo.destino``) releído; entonces el presupuesto es
    ``busqueda.n_ensayos`` y ``busqueda.tiempo_s``. Si es false, esos tres campos son ``None``. Con ``optuna`` se
    exige además (siempre, como en D9) que la búsqueda sea de los mismos datos y que exista ``mlp_final.yaml``.
    """
    b = cfg["busqueda"]
    nulos = {"config_de_busqueda": False, "n_configuraciones_busqueda": None, "tiempo_busqueda_s": None,
             "busqueda": None}
    if b["tipo"] not in TIPOS_BUSQUEDA:
        raise ValueError(f"busqueda.tipo debe ser uno de {list(TIPOS_BUSQUEDA)}; se recibió {b['tipo']!r}")
    if b["tipo"] == "sin_busqueda":
        return nulos
    _validar_mismos_datos(b, d)
    ruta_final = Path(raiz) / cfg["hpo"]["destino"]
    if not ruta_final.is_file():
        raise ValueError(f"No existe {ruta_final.as_posix()} (hpo.destino): corre lgd-hpo (python run.py lgd-hpo)")
    final = yaml.safe_load(ruta_final.read_text(encoding="utf-8"))
    if cfg["mlp"] != final.get("mlp") or b != final.get("busqueda"):
        return nulos
    return {"config_de_busqueda": True,
            "n_configuraciones_busqueda": validar_entero(b["n_ensayos"], "busqueda.n_ensayos", minimo=1),
            "tiempo_busqueda_s": float(b["tiempo_s"]), "busqueda": copy.deepcopy(b)}


# ---------------------------------------------------------------------------------------------------------- búsqueda
def _sugerir(trial, espacio: dict) -> dict:
    n_capas = trial.suggest_int("n_capas", *espacio["n_capas"])
    capas = [trial.suggest_categorical(f"neuronas_capa{i + 1}", espacio["neuronas"]) for i in range(n_capas)]
    return {
        "capas": capas,
        "activacion": trial.suggest_categorical("activacion", espacio["activacion"]),
        "dropout": trial.suggest_float("dropout", *espacio["dropout"]),
        "l2": trial.suggest_float("l2", *espacio["l2"], log=True),
        "optimizador": trial.suggest_categorical("optimizador", espacio["optimizador"]),
        "lr": trial.suggest_float("lr", *espacio["lr"], log=True),
        "batch": trial.suggest_categorical("batch", espacio["batch"]),
    }


def _resumen_hpo(cfg: dict, d: dict, registros: list[dict], mejor: dict) -> dict:
    base_seed = cfg["base_seed"]
    return {"n_ensayos": len(registros), "muestreador": "TPESampler", "semilla_muestreador": base_seed,
            "semilla_ensayos": base_seed, "criterio": CRITERIO_SELECCION, "espacio": cfg["hpo"]["espacio"],
            "mlp_base": cfg["mlp"], "tiempo_total_s": sum(r["tiempo_s"] for r in registros),
            "n": d["n"], "muestra": d["muestra"], "evidencia": d["evidencia"],
            "mejor": {"numero": mejor["numero"], "valor_val": mejor["valor_val"], "mae_val": mejor["mae_val"],
                      "parametros": mejor["parametros"]},
            "ensayos": registros}


def buscar(cfg: dict, raiz: Path = Path(".")) -> dict:
    """Búsqueda Optuna TPE del MLP: escribe ``{salida}/hpo_ensayos.json`` y reescribe ``hpo.destino``."""
    import optuna

    raiz = Path(raiz)
    d = cargar_datos(cfg, raiz)
    base_seed = validar_entero(cfg["base_seed"], "base_seed")
    n_ensayos = validar_entero(cfg["hpo"]["n_ensayos"], "hpo.n_ensayos", minimo=1)
    registros: list[dict] = []

    def objetivo(trial) -> float:
        params = _sugerir(trial, cfg["hpo"]["espacio"])
        registro = {"numero": trial.number, "parametros": params, "valor_val": None, "mae_val": None,
                    "tiempo_s": None, "estado": "FAIL", "epocas": None}
        registros.append(registro)
        try:
            m, t = _fit_cronometrado(MLPLGD(), d, {"mlp": {**cfg["mlp"], **params}}, base_seed)
        except RuntimeError as exc:
            registro["tiempo_s"] = getattr(exc, "tiempo_s", 0.0)
            registro["error"] = str(exc)
            raise
        valor = float(min(m.historial_["val"]))
        registro.update(valor_val=valor, mae_val=mae(d["y"]["val"], m.predict(d["X"]["val"])), tiempo_s=t,
                        estado="COMPLETE", epocas=m.epocas_, mejor_epoca=m.mejor_epoca_, device=m.device_)
        return valor

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=base_seed))
    estudio.optimize(objetivo, n_trials=n_ensayos, catch=(RuntimeError,))

    completos = [r for r in registros if r["estado"] == "COMPLETE"]
    if not completos:
        raise RuntimeError(f"ningún ensayo de la búsqueda terminó ({len(registros)} divergieron); no se escribió nada")
    mejor = min(completos, key=lambda r: (r["valor_val"], r["numero"]))
    hpo = _resumen_hpo(cfg, d, registros, mejor)
    save_json(raiz / cfg["salida"] / ARCHIVO_HPO, hpo)
    busqueda = {"tipo": "optuna", "origen": f"lgd-hpo: mejor de {len(registros)} ensayos ({CRITERIO_SELECCION})",
                "ensayo": mejor["numero"], "n_ensayos": len(registros), "tiempo_s": hpo["tiempo_total_s"],
                "evidencia": d["evidencia"], "muestra": d["muestra"],
                "fecha": dt.datetime.now().isoformat(timespec="seconds")}
    escribir_mlp_final(raiz / cfg["hpo"]["destino"], {**cfg["mlp"], **mejor["parametros"]}, busqueda)
    return hpo


# ---------------------------------------------------------------------------------------------------------- semillas
def _resumir(valores: list[float]) -> dict:
    v = np.asarray(valores, dtype=np.float64)
    return {"media": float(v.mean()), "desv": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "mejor": float(v.min()), "peor": float(v.max()), "n": int(v.size)}


def _registro(*, modelo: str, seed: int, config: dict, metricas: dict, n_parametros: int, tiempo_s: float,
              device: str, epocas: int | None, d: dict, **extra) -> dict:
    return {"modelo": modelo, "seed": seed, "config": config, "metricas": metricas, "n_parametros": n_parametros,
            "tiempo_s": tiempo_s, "device": device, "epocas": epocas, "evidencia": d["evidencia"], "n": d["n"],
            "muestra": d["muestra"], **extra}


def _entrenar_lineal(d: dict, seed: int, carpeta: Path) -> dict:
    lineal, t = _fit_cronometrado(LinealLGD(), d, None, seed)
    fuera = {"val": lineal.contar_fuera_de_rango(d["X"]["val"]), "test": lineal.contar_fuera_de_rango(d["X"]["test"]),
             "pd_test": lineal.contar_fuera_de_rango(d["X_pd_test"])}
    reg = _registro(modelo="lineal", seed=seed, config={"modelo": "LinearRegression"}, metricas=_metricas(lineal, d),
                    n_parametros=lineal.n_parametros(), tiempo_s=t, device="cpu", epocas=None, d=d,
                    fuera_de_rango=fuera)
    save_json(carpeta / f"lineal_seed{seed}.json", reg)
    return {"modelo": lineal, "registro": reg}


def _historial(m: MLPLGD) -> list[dict]:
    """``[{epoca, perdida_train, perdida_val}]`` (épocas desde 1), el formato de B, desde ``m.historial_``."""
    return [{"epoca": e, "perdida_train": tr, "perdida_val": va}
            for e, (tr, va) in enumerate(zip(m.historial_["train"], m.historial_["val"], strict=True), start=1)]


def _entrenar_mlp(cfg: dict, d: dict, semillas: list[int], carpeta: Path) -> list[dict]:
    corridas = []
    for seed in semillas:
        m, t = _fit_cronometrado(MLPLGD(), d, {"mlp": cfg["mlp"]}, seed)
        reg = _registro(modelo="mlp", seed=seed, config={"mlp": copy.deepcopy(m.cfg_mlp_)}, metricas=_metricas(m, d),
                        n_parametros=m.n_parametros(), tiempo_s=t, device=m.device_, epocas=m.epocas_, d=d,
                        mejor_epoca=m.mejor_epoca_, mejor_perdida_val=m.historial_["val"][m.mejor_epoca_ - 1],
                        historial=_historial(m), hilos_torch=torch.get_num_threads(),
                        version_torch=str(torch.__version__))
        save_json(carpeta / f"mlp_seed{seed}.json", reg)
        corridas.append({"seed": seed, "mae_test": reg["metricas"]["mae_test"], "modelo": m, "registro": reg})
    return corridas


def _referencias_constantes(y_train, y_val, y_test) -> dict:
    """Predictor constante con la media y con la mediana de ``y_train`` (solo train), evaluado en val y test."""
    y_train = np.asarray(y_train, dtype=np.float64)
    referencias = {}
    for nombre, valor in (("constante_media", float(np.mean(y_train))),
                          ("constante_mediana", float(np.median(y_train)))):
        metricas = {}
        for split, y in (("val", y_val), ("test", y_test)):
            y = np.asarray(y, dtype=np.float64)
            p = np.full(y.shape, valor)
            metricas[f"mae_{split}"] = mae(y, p)
            metricas[f"rmse_{split}"] = rmse(y, p)
        referencias[nombre] = {"valor": valor, "metricas": {k: metricas[k] for k in METRICAS},
                               "n_parametros": 1, "n_configuraciones_busqueda": 1}
    return referencias


def _semilla_mediana(corridas: list[dict]) -> int:
    """Semilla de MAE de test mediano: posición (n−1)//2 al ordenar por (mae_test, seed) (mediana inferior si n es par;
    empate → semilla menor)."""
    orden = sorted(corridas, key=lambda c: (c["mae_test"], c["seed"]))
    return orden[(len(orden) - 1) // 2]["seed"]


def _entrada_lineal(d: dict, lineal: dict) -> dict:
    """Entrada ``lineal`` de summary.json: campos comunes del núcleo §5.4 + ``fuera_de_rango`` y ``nota_pred``."""
    reg = lineal["registro"]
    return {"n_parametros": reg["n_parametros"], "n_configuraciones_busqueda": 1, "tiempo_total_s": reg["tiempo_s"],
            "metricas": {k: _resumir([reg["metricas"][k]]) for k in METRICAS}, "evidencia": d["evidencia"],
            "fuera_de_rango": reg["fuera_de_rango"], "nota_pred": NOTA_PRED_LINEAL}


def _entrada_mlp(cfg: dict, d: dict, corridas: list[dict], seed_mediana: int, procedencia: dict) -> dict:
    """Entrada ``mlp`` de summary.json con los campos de la de B (núcleo §5.4, v0.16) + ``base_seed``, ``n`` y
    ``muestra``. ``tiempo_total_s`` = búsqueda + semillas si ``config_de_busqueda``; si no, solo las semillas."""
    regs = [c["registro"] for c in corridas]
    mediana = next(c["registro"] for c in corridas if c["seed"] == seed_mediana)
    t_semillas = sum((r["tiempo_s"] for r in regs), 0.0)
    t_busqueda = procedencia["tiempo_busqueda_s"]
    return {"n_parametros": mediana["n_parametros"],
            "n_configuraciones_busqueda": procedencia["n_configuraciones_busqueda"],
            "tiempo_total_s": t_busqueda + t_semillas if procedencia["config_de_busqueda"] else t_semillas,
            "tiempo_busqueda_s": t_busqueda, "tiempo_semillas_s": t_semillas,
            "metricas": {k: _resumir([r["metricas"][k] for r in regs]) for k in METRICAS},
            "evidencia": d["evidencia"], "semillas": [r["seed"] for r in regs], "semilla_mediana": seed_mediana,
            "criterio_semilla_mediana": CRITERIO_MEDIANA, "config": copy.deepcopy(mediana["config"]["mlp"]),
            "config_de_busqueda": procedencia["config_de_busqueda"], "busqueda": procedencia["busqueda"],
            "hilos_torch": mediana["hilos_torch"], "version_torch": mediana["version_torch"],
            "base_seed": cfg["base_seed"], "n": d["n"], "muestra": d["muestra"]}


def _armar_resumen(cfg: dict, d: dict, lineal: dict, corridas: list[dict], seed_mediana: int,
                   procedencia: dict) -> dict:
    """summary.json por modelo (``lineal``, ``mlp``), como el de B, + ``referencias`` (D §3.4)."""
    return {"lineal": _entrada_lineal(d, lineal),
            "mlp": _entrada_mlp(cfg, d, corridas, seed_mediana, procedencia),
            "referencias": _referencias_constantes(d["y"]["train"], d["y"]["val"], d["y"]["test"])}


def _escribir_predicciones(d: dict, lineal, mlp, salida: Path) -> None:
    pd.DataFrame({"id": d["ids"]["test"], "y": d["y"]["test"], "pred_lineal": lineal.predict(d["X"]["test"]),
                  "pred_mlp": mlp.predict(d["X"]["test"])}).to_parquet(salida / "pred_test.parquet", index=False)
    pd.DataFrame({"id": d["ids_pd_test"], "pred_lineal": lineal.predict(d["X_pd_test"]),
                  "pred_mlp": mlp.predict(d["X_pd_test"])}).to_parquet(salida / "pred_pd_test.parquet", index=False)


def entrenar_semillas(cfg: dict, raiz: Path = Path(".")) -> dict:
    """Lineal 1 vez + MLP con ``semillas.n`` semillas, con la config tal cual; escribe semillas/, summary.json,
    predicciones y modelos."""
    raiz = Path(raiz)
    d = cargar_datos(cfg, raiz)
    procedencia = _leer_procedencia(cfg, d, raiz)
    base_seed = validar_entero(cfg["base_seed"], "base_seed")
    semillas = seeds_for_runs(base_seed, validar_entero(cfg["semillas"]["n"], "semillas.n", minimo=1))
    salida = raiz / cfg["salida"]
    lineal = _entrenar_lineal(d, base_seed, salida / "semillas")
    corridas = _entrenar_mlp(cfg, d, semillas, salida / "semillas")
    seed_mediana = _semilla_mediana(corridas)
    mlp_mediano = next(c["modelo"] for c in corridas if c["seed"] == seed_mediana)
    resumen = _armar_resumen(cfg, d, lineal, corridas, seed_mediana, procedencia)
    save_json(salida / "summary.json", resumen)
    _escribir_predicciones(d, lineal["modelo"], mlp_mediano, salida)
    lineal["modelo"].guardar(raiz / cfg["modelos"]["lineal"])
    mlp_mediano.guardar(raiz / cfg["modelos"]["mlp"])
    return resumen


# -------------------------------------------------------------------------------------------------------- manejadores
def manejador(cfg: dict | None) -> int:
    """``run.py lgd`` (la CLI ya imprimió la config resuelta, que es la que se entrena)."""
    inicio = time.perf_counter()
    r = entrenar_semillas(cfg)
    print(f"Configuración del MLP: {cfg['busqueda']['origen']}")
    for nombre in MODELOS:
        m = r[nombre]
        met = m["metricas"]
        print(f"{nombre}: MAE test {met['mae_test']['media']:.4f} ± {met['mae_test']['desv']:.4f} · "
              f"RMSE test {met['rmse_test']['media']:.4f} · n_parametros {m['n_parametros']} · "
              f"configuraciones {m['n_configuraciones_busqueda']} · tiempo {m['tiempo_total_s']:.1f} s")
    if not r["mlp"]["config_de_busqueda"]:
        print("Aviso: la config del MLP no es la de la búsqueda (sin_busqueda, o mlp/busqueda distintos de "
              "mlp_final.yaml): n_configuraciones_busqueda, tiempo_busqueda_s y busqueda quedan en null")
    for nombre, ref in r["referencias"].items():
        met = ref["metricas"]
        print(f"{nombre} (= {ref['valor']:.4f}, de train): MAE test {met['mae_test']:.4f} · "
              f"RMSE test {met['rmse_test']:.4f} · MAE val {met['mae_val']:.4f}")
    fr = r["lineal"]["fuera_de_rango"]
    print(f"Lineal fuera de [0, 1]: test LGD {fr['test']['total']} de {fr['test']['n']} · "
          f"test de loans_pd {fr['pd_test']['total']} de {fr['pd_test']['n']}")
    print(f"Resultados en {cfg['salida']}")
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0


def manejador_hpo(cfg: dict | None) -> int:
    """``run.py lgd-hpo``."""
    inicio = time.perf_counter()
    h = buscar(cfg)
    n_ok = sum(e["estado"] == "COMPLETE" for e in h["ensayos"])
    print(f"Ensayos: {len(h['ensayos'])} ({n_ok} completos) · mejor #{h['mejor']['numero']}: "
          f"pérdida val {h['mejor']['valor_val']:.4f}, MAE val {h['mejor']['mae_val']:.4f}")
    print(f"Resultados en {cfg['salida']}; configuración final en {cfg['hpo']['destino']}")
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0
