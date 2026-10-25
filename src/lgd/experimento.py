"""Experimento de la LGD: búsqueda Optuna, 30 semillas, ``summary.json`` y predicciones (tarea D4; Parte D §3.3–§3.4;
núcleo §5.4, §6).

    python run.py lgd-hpo --config configs/lgd/hpo.yaml    # búsqueda → hpo_ensayos.json + mejor_config.json
    python run.py lgd     --config configs/lgd/base.yaml   # lineal 1 vez + MLP con 30 semillas
    python run.py lgd     --config configs/lgd/demo.yaml   # DEMO (< 2 min CPU; escribe en results/data/lgd_demo/)

Datos: ``loans_lgd.parquet`` (D2) y ``preprocesador_lgd.json`` (D2) **tal cual**: aquí nunca se llama ``fit`` del
preprocesador ([DECIDIDO, Camila, 24-oct]: lo comparten lineal y MLP y se aplica también al test de ``loans_pd``).
``datos.muestra`` (opcional, para la demo): n total de créditos, repartido entre train/val/test en proporción a su
tamaño (redondeo por restos mayores) y elegido con ``base_seed``.

Búsqueda (``buscar``):
- Optuna ``TPESampler(seed=base_seed)``, ``hpo.n_ensayos`` ensayos, espacio ``hpo.espacio`` (el de B §3.3): número de
  capas entero en [1, 4]; neuronas elegidas **por capa** de la lista; dropout uniforme; ``l2`` y ``lr``
  log-uniformes; activación, optimizador y batch categóricos. Con ``sgd`` el momento es ``mlp.momento`` (no se busca).
  Las demás claves de ``mlp`` (``max_epocas``, ``paciencia``, ``device``, ``perdida``) se toman de la config.
- Cada ensayo entrena el MLP con la semilla fija ``base_seed`` (la misma en todos los ensayos).
- Criterio de selección **[PROPUESTO por el orquestador]**: la pérdida de validación del propio MLP (la BCE fraccional
  media en val de la mejor época, ``min(historial_["val"])``; si ``mlp.perdida = mse``, el MSE); se guarda el MAE de
  val al lado en cada ensayo. Un ensayo que diverge (``RuntimeError`` del MLP) queda con estado ``FAIL``.
- Nota sobre ``l2``: en D es ``l2·Σ‖W‖²`` explícita sin sesgos (= ``weight_decay = 2·l2`` acoplado); mismo rango
  numérico que B, significado distinto por un factor 2. Pendiente coordinar con B.
- Escribe ``{salida}/hpo_ensayos.json`` y ``{salida}/mejor_config.json`` (``{"mlp": {...}, ...}``). ``base.yaml`` no se
  edita a mano: con ``hpo.usar_mejor = true``, ``entrenar_semillas`` lee ese archivo si existe y sus claves reemplazan
  las de ``mlp`` (también las que se pasen con ``--set mlp.*``; para usar las de la config, ``--set hpo.usar_mejor=false``).
  Si sus hashes de partición o preprocesador no coinciden con los datos actuales → ``ValueError``.

Semillas (``entrenar_semillas``):
- Lineal 1 vez (``seed = base_seed``; MCO es determinista) y MLP con ``base_seed + k``, k = 0…``semillas.n``−1; un JSON
  por (modelo, semilla) en ``{salida}/semillas/{modelo}_seed{k}.json``.
- ``summary.json``: ``{"modelos": {"lineal": {...}, "mlp": {...}}, ...}`` con, por modelo, ``n_parametros``,
  ``n_configuraciones_busqueda`` (lineal 1; MLP = ensayos de ``hpo_ensayos.json`` si se usó la mejor configuración, si no
  1), ``tiempo_total_s`` (MLP: tiempo de la búsqueda + de todas las semillas) y ``metricas`` (MAE y RMSE en val y test:
  ``{media, desv, mejor, peor, n}``; ``desv`` muestral, ddof = 1, y 0 si n = 1; mejor = mínimo). La lineal agrega
  ``fuera_de_rango`` en val y test (sus predicciones no se recortan).
- ``pred_test.parquet`` (test de la LGD: ``id, y, pred_lineal, pred_mlp``) y ``pred_pd_test.parquet`` (todo el test de
  ``loans_pd``: ``id, pred_lineal, pred_mlp``) con la semilla del MLP de **MAE de test mediano** (mediana inferior si n
  es par: posición (n−1)//2 al ordenar por (MAE de test, semilla)); esa semilla se guarda en ``modelos.mlp``.
- Cada JSON lleva ``hash_particion`` (SHA-256 de las parejas ``id,split`` de ``loans_lgd`` completo, en su orden) y
  ``hash_preprocesador`` (SHA-256 del JSON canónico del preprocesador).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json, save_json
from src.common.seeds import make_rng, seeds_for_runs, validar_entero
from src.lgd.lineal import LinealLGD
from src.lgd.metricas import mae, rmse
from src.lgd.mlp import MLPLGD
from src.lgd.preparar import _rechazar_posteriores

SPLITS = ("train", "val", "test")
METRICAS = ("mae_val", "rmse_val", "mae_test", "rmse_test")
CLAVES_BUSCADAS = ("capas", "activacion", "dropout", "l2", "optimizador", "lr", "batch")
ARCHIVO_HPO = "hpo_ensayos.json"
ARCHIVO_MEJOR = "mejor_config.json"


# ------------------------------------------------------------------------------------------------------------ datos
def _sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def hash_particion(df: pd.DataFrame) -> str:
    """SHA-256 de las líneas ``id,split`` en el orden del archivo."""
    return _sha256_texto("\n".join(df["id"].astype(str) + "," + df["split"].astype(str)))


def hash_preprocesador(pre_dict: dict) -> str:
    """SHA-256 del JSON canónico (claves ordenadas, sin espacios) del preprocesador."""
    return _sha256_texto(json.dumps(pre_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


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
    rng = make_rng(base_seed, 0)
    n = _repartir(muestra, {s: int((df["split"] == s).sum()) for s in SPLITS})
    elegidos = [np.sort(rng.choice(np.flatnonzero(df["split"].to_numpy() == s), size=n[s], replace=False))
                for s in SPLITS]
    return df.iloc[np.sort(np.concatenate(elegidos))].reset_index(drop=True)


def cargar_datos(cfg: dict, raiz: Path = Path(".")) -> dict:
    """``X``/``y``/``ids`` por split de la LGD, ``X`` e ``ids`` del test de ``loans_pd`` y los hashes.

    Usa el preprocesador guardado (``Preprocesador.from_dict``): nunca lo reajusta.
    """
    raiz = Path(raiz)
    datos = cfg["datos"]
    pre_dict = load_json(raiz / datos["salida"]["preprocesador"])
    pre = Preprocesador.from_dict(pre_dict)
    _rechazar_posteriores([v["nombre"] for v in pre.variables])
    nombres = [v["nombre"] for v in pre.variables]

    lgd = pd.read_parquet(raiz / datos["salida"]["loans_lgd"])
    h_part = hash_particion(lgd)
    lgd = _muestrear(lgd, datos.get("muestra"), validar_entero(cfg["base_seed"], "base_seed"))
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
            "hash_particion": h_part, "hash_preprocesador": hash_preprocesador(pre_dict),
            "n": {s: int(len(y[s])) for s in SPLITS}, "muestra": datos.get("muestra")}


def _hashes(d: dict) -> dict:
    return {"hash_particion": d["hash_particion"], "hash_preprocesador": d["hash_preprocesador"]}


def _metricas(modelo, d: dict) -> dict:
    p_val, p_test = modelo.predict(d["X"]["val"]), modelo.predict(d["X"]["test"])
    return {"mae_val": mae(d["y"]["val"], p_val), "rmse_val": rmse(d["y"]["val"], p_val),
            "mae_test": mae(d["y"]["test"], p_test), "rmse_test": rmse(d["y"]["test"], p_test)}


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


def buscar(cfg: dict, raiz: Path = Path(".")) -> dict:
    """Búsqueda Optuna TPE del MLP; escribe ``hpo_ensayos.json`` y ``mejor_config.json`` en ``salida``."""
    import optuna

    raiz = Path(raiz)
    inicio = time.perf_counter()
    d = cargar_datos(cfg, raiz)
    base_seed = validar_entero(cfg["base_seed"], "base_seed")
    n_ensayos = validar_entero(cfg["hpo"]["n_ensayos"], "hpo.n_ensayos", minimo=1)
    espacio = cfg["hpo"]["espacio"]
    registros: list[dict] = []

    def objetivo(trial) -> float:
        params = _sugerir(trial, espacio)
        cfg_t = {"mlp": {**cfg["mlp"], **params}}
        t0 = time.perf_counter()
        registro = {"numero": trial.number, "parametros": params, "valor_val": None, "mae_val": None,
                    "tiempo_s": None, "estado": "FAIL", "epocas": None}
        registros.append(registro)
        try:
            m = MLPLGD().fit(d["X"]["train"], d["y"]["train"], d["X"]["val"], d["y"]["val"], cfg_t, base_seed)
        finally:
            registro["tiempo_s"] = time.perf_counter() - t0
        valor = float(min(m.historial_["val"]))
        registro.update(valor_val=valor, mae_val=mae(d["y"]["val"], m.predict(d["X"]["val"])),
                        estado="COMPLETE", epocas=m.epocas_, mejor_epoca=m.mejor_epoca_, device=m.device_)
        return valor

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=base_seed))
    estudio.optimize(objetivo, n_trials=n_ensayos, catch=(RuntimeError,))

    completos = [r for r in registros if r["estado"] == "COMPLETE"]
    if not completos:
        raise RuntimeError("ningún ensayo de la búsqueda terminó (todos divergieron)")
    mejor = min(completos, key=lambda r: (r["valor_val"], r["numero"]))
    tiempo_total = time.perf_counter() - inicio
    criterio = (f"mínima pérdida de validación del MLP ({cfg['mlp']['perdida']}, media en val, mejor época); "
                "MAE de val al lado [PROPUESTO por el orquestador]")
    hpo = {"n_ensayos": n_ensayos, "muestreador": "TPESampler", "semilla_muestreador": base_seed,
           "semilla_ensayos": base_seed, "criterio": criterio, "espacio": espacio,
           "mlp_fijo": {k: v for k, v in cfg["mlp"].items() if k not in CLAVES_BUSCADAS},
           "tiempo_total_s": tiempo_total, "n": d["n"], "muestra": d["muestra"], **_hashes(d),
           "mejor": {"numero": mejor["numero"], "valor_val": mejor["valor_val"], "mae_val": mejor["mae_val"]},
           "ensayos": registros}
    salida = raiz / cfg["salida"]
    save_json(salida / ARCHIVO_HPO, hpo)
    save_json(salida / ARCHIVO_MEJOR, {"mlp": mejor["parametros"], "ensayo": mejor["numero"],
                                       "valor_val": mejor["valor_val"], "mae_val": mejor["mae_val"],
                                       "criterio": criterio, **_hashes(d)})
    return hpo


# ---------------------------------------------------------------------------------------------------------- semillas
def _resolver_mlp(cfg: dict, raiz: Path, d: dict) -> tuple[dict, int, float]:
    """(bloque mlp final, n_configuraciones_busqueda, tiempo de la búsqueda)."""
    salida = raiz / cfg["salida"]
    ruta = salida / ARCHIVO_MEJOR
    if not (cfg["hpo"]["usar_mejor"] and ruta.is_file()):
        if cfg["hpo"]["usar_mejor"]:
            print(f"Aviso: no existe {ruta.as_posix()} (sin búsqueda): se usa el bloque mlp de la config "
                  "y n_configuraciones_busqueda = 1.")
        return dict(cfg["mlp"]), 1, 0.0
    mejor = load_json(ruta)
    for clave in ("hash_particion", "hash_preprocesador"):
        if mejor.get(clave) != d[clave]:
            raise ValueError(f"{ruta.as_posix()}: {clave} no coincide con los datos actuales; la búsqueda se hizo "
                             "con otra partición o preprocesador (repite lgd-hpo)")
    hpo = load_json(salida / ARCHIVO_HPO)
    print(f"MLP con la mejor configuración de la búsqueda ({ruta.as_posix()}, ensayo {mejor['ensayo']}).")
    return {**cfg["mlp"], **mejor["mlp"]}, len(hpo["ensayos"]), float(hpo["tiempo_total_s"])


def _resumir(valores: list[float]) -> dict:
    v = np.asarray(valores, dtype=np.float64)
    return {"media": float(v.mean()), "desv": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "mejor": float(v.min()), "peor": float(v.max()), "n": int(v.size)}


def _registro(modelo: str, seed: int, cfg_modelo: dict, metricas: dict, m, tiempo_s: float, device: str,
              epocas, d: dict) -> dict:
    return {"modelo": modelo, "seed": seed, "config": cfg_modelo, "metricas": metricas,
            "n_parametros": m.n_parametros(), "tiempo_s": tiempo_s, "device": device, "epocas": epocas,
            "n": d["n"], "muestra": d["muestra"], **_hashes(d)}


def entrenar_semillas(cfg: dict, raiz: Path = Path(".")) -> dict:
    """Lineal 1 vez + MLP con ``semillas.n`` semillas; escribe semillas/, summary.json, predicciones y modelos."""
    raiz = Path(raiz)
    d = cargar_datos(cfg, raiz)
    base_seed = validar_entero(cfg["base_seed"], "base_seed")
    n_semillas = validar_entero(cfg["semillas"]["n"], "semillas.n", minimo=1)
    salida = raiz / cfg["salida"]
    carpeta_semillas = salida / "semillas"
    mlp_cfg, n_conf_mlp, t_busqueda = _resolver_mlp(cfg, raiz, d)

    # Lineal (referencia)
    t0 = time.perf_counter()
    lineal = LinealLGD().fit(d["X"]["train"], d["y"]["train"])
    t_lineal = time.perf_counter() - t0
    met_lineal = _metricas(lineal, d)
    fuera = {s: lineal.contar_fuera_de_rango(d["X"][s]) for s in ("val", "test")}
    reg = _registro("lineal", base_seed, {"modelo": "LinearRegression"}, met_lineal, lineal, t_lineal, "cpu", None, d)
    reg["fuera_de_rango"] = fuera
    save_json(carpeta_semillas / f"lineal_seed{base_seed}.json", reg)

    # MLP: una corrida por semilla
    corridas = []
    for seed in seeds_for_runs(base_seed, n_semillas):
        m = MLPLGD().fit(d["X"]["train"], d["y"]["train"], d["X"]["val"], d["y"]["val"], {"mlp": mlp_cfg}, seed)
        met = _metricas(m, d)
        save_json(carpeta_semillas / f"mlp_seed{seed}.json",
                  _registro("mlp", seed, {"mlp": copy.deepcopy(m.cfg_mlp_)}, met, m, m.tiempo_s_, m.device_,
                            m.epocas_, d) | {"mejor_epoca": m.mejor_epoca_})
        corridas.append((met, seed, m))

    orden = sorted(corridas, key=lambda c: (c[0]["mae_test"], c[1]))
    _, seed_mediana, mlp_mediano = orden[(len(orden) - 1) // 2]

    resumen = {
        "modelos": {
            "lineal": {"n_parametros": lineal.n_parametros(), "n_configuraciones_busqueda": 1,
                       "tiempo_total_s": t_lineal, "metricas": {k: _resumir([met_lineal[k]]) for k in METRICAS},
                       "fuera_de_rango": fuera},
            "mlp": {"n_parametros": mlp_mediano.n_parametros(), "n_configuraciones_busqueda": n_conf_mlp,
                    "tiempo_total_s": t_busqueda + sum(c[2].tiempo_s_ for c in corridas),
                    "tiempo_busqueda_s": t_busqueda,
                    "metricas": {k: _resumir([c[0][k] for c in corridas]) for k in METRICAS},
                    "semillas": [c[1] for c in corridas], "semilla_mediana": seed_mediana,
                    "config": {"mlp": copy.deepcopy(mlp_mediano.cfg_mlp_)}},
        },
        "base_seed": base_seed, "n": d["n"], "muestra": d["muestra"],
        "criterio_semilla_mediana": "MAE de test mediano (mediana inferior si n es par)", **_hashes(d),
    }
    save_json(salida / "summary.json", resumen)

    pd.DataFrame({"id": d["ids"]["test"], "y": d["y"]["test"],
                  "pred_lineal": lineal.predict(d["X"]["test"]),
                  "pred_mlp": mlp_mediano.predict(d["X"]["test"])}).to_parquet(salida / "pred_test.parquet",
                                                                               index=False)
    pd.DataFrame({"id": d["ids_pd_test"], "pred_lineal": lineal.predict(d["X_pd_test"]),
                  "pred_mlp": mlp_mediano.predict(d["X_pd_test"])}).to_parquet(salida / "pred_pd_test.parquet",
                                                                               index=False)
    lineal.guardar(raiz / cfg["modelos"]["lineal"])
    mlp_mediano.guardar(raiz / cfg["modelos"]["mlp"])
    return resumen


# -------------------------------------------------------------------------------------------------------- manejadores
def manejador(cfg: dict | None) -> int:
    """``run.py lgd`` (la CLI ya imprimió la config resuelta)."""
    inicio = time.perf_counter()
    r = entrenar_semillas(cfg)
    for nombre, m in r["modelos"].items():
        met = m["metricas"]
        print(f"{nombre}: MAE test {met['mae_test']['media']:.4f} ± {met['mae_test']['desv']:.4f} · "
              f"RMSE test {met['rmse_test']['media']:.4f} · n_parametros {m['n_parametros']} · "
              f"configuraciones {m['n_configuraciones_busqueda']} · tiempo {m['tiempo_total_s']:.1f} s")
    fr = r["modelos"]["lineal"]["fuera_de_rango"]["test"]
    print(f"Lineal fuera de [0, 1] en test: {fr['total']} de {fr['n']}")
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
    print(f"Resultados en {cfg['salida']}")
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0
