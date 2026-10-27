# %% [markdown]
# # Tablas y textos del notebook `02_pd` (B9; B-pd §3.7; núcleo §5.8: cero lógica en el notebook)
#
# Funciones **puras**: reciben la config o los JSON ya cargados (resultados de B3–B5, `pd-calibracion.json` de B7 y
# los resúmenes de la Parte A) y devuelven un `DataFrame`, un dict o un texto. No leen archivos, no entrenan y no
# imprimen; el notebook las llama y muestra lo que devuelven.

# %%
from __future__ import annotations  # anotaciones como texto

import copy  # copias profundas de la config
import shlex  # comando de terminal citado como en la shell
from pathlib import Path  # carpeta de la demo

import pandas as pd  # tablas
import yaml  # valores del --set en YAML de flujo, como config_notebook

from src.pd.experimento import SENTIDO, SPLITS, SPLITS_MEDIDOS  # sentido de cada métrica y splits (B5)
from src.pd.hpo import CLAVES_ESPACIO, ESTADO_COMPLETO, RANGOS  # espacio de la búsqueda y su escala (B4)

CLAVES_FIJADAS = ("salida", "modelos", "busqueda")  # las fija el notebook o la demo: no van en los moldes
GRUPOS_MOLDES = {"comunes": ("base_seed", "datos", "semillas"), "logistica": ("logistica",),  # bloque → molde
                 "mlp": ("mlp",)}  # B-pd §3.7
SUBCARPETA_RESULTADOS = "resultados"  # salida de la demo dentro de la carpeta temporal
SUBCARPETA_MODELOS = "modelos"  # modelos de la demo dentro de la carpeta temporal
COMANDO_PD = ("python", "run.py", "pd")  # prefijo del comando de terminal (núcleo §4.1)
MODELOS = ("logistica", "mlp")  # modelos de la PD, en el orden de summary.json
ETIQUETA_METRICA = {"auc": "AUC", "gini": "Gini", "ks": "KS", "brier": "Brier", "logloss": "log-loss"}  # títulos
COLUMNAS_PRESUPUESTO = ("semillas", "parámetros", "configuraciones", "tiempo (s)")  # al final de tabla_metricas
COLUMNAS_METRICAS = tuple(c for s in SPLITS_MEDIDOS for m in ETIQUETA_METRICA  # orden fijo de tabla_metricas
                          for c in (f"{ETIQUETA_METRICA[m]} {s}", f"{ETIQUETA_METRICA[m]} {s} (desv)")) + \
    COLUMNAS_PRESUPUESTO  # métricas (split por fuera, como summary.json) y presupuesto
VACIO = ""  # celda sin dato (p. ej. configuraciones de un MLP fijado a mano): no NaN
SI, NO = "sí", "no"  # booleanos en texto
EMPATE = "empate"  # comparación sin ganador
FUERA = "fuera"  # año de originación fuera del modelado (posterior a test_hasta)
ESCALA_OPCIONES = "opciones"  # escala de una lista de valores del espacio (no un rango)
PARAMETROS_ENSAYO = ("capas", "activacion", "dropout", "l2", "optimizador", "lr", "batch")  # buscados (§3.3)
COLUMNAS_ENSAYO = {"logloss_val": "log-loss val", "auc_val": "AUC val", "epocas": "épocas",  # métricas del ensayo
                   "mejor_epoca": "mejor época", "n_parametros": "parámetros"}  # → título
K_ENSAYOS = 5  # mejores ensayos que muestra la sección 3.4 por defecto


# %% [markdown]
# ## Moldes de la config de la demo
#
# Las hojas de la config con clave con punto (`"mlp.capas"`), como en `SET` y `--set`, repartidas en tres moldes
# (claves comunes, logística y MLP). Una clave que no esté clasificada es un error: así un cambio en `demo.yaml`
# obliga a actualizar los moldes del notebook (y su prueba lo detecta).

# %%
def _hojas(bloque, prefijo: str) -> dict:
    """{clave.con.punto: valor} de las hojas de ``bloque`` (los dict se recorren; lo demás es hoja)."""
    if not isinstance(bloque, dict):  # hoja (número, lista, texto o null)
        return {prefijo: copy.deepcopy(bloque)}  # copia: el molde no comparte listas con la config
    return {k: v for clave, sub in bloque.items() for k, v in _hojas(sub, f"{prefijo}.{clave}").items()}  # baja


# %%
def moldes_de(cfg: dict) -> dict:
    """{"comunes", "logistica", "mlp"}: {clave con punto: valor} de la config, sin CLAVES_FIJADAS."""
    grupo_de = {bloque: grupo for grupo, bloques in GRUPOS_MOLDES.items() for bloque in bloques}  # bloque → molde
    sueltas = [k for k in cfg if k not in grupo_de and k not in CLAVES_FIJADAS]  # sin clasificar
    if sueltas:  # p. ej. una clave nueva en demo.yaml
        raise ValueError(f"claves de la config sin molde: {sueltas}; agrégalas a GRUPOS_MOLDES y al notebook")  # qué
    moldes = {grupo: {} for grupo in GRUPOS_MOLDES}  # vacíos, en orden
    for clave, valor in cfg.items():  # bloques de la config, en su orden
        if clave in grupo_de:  # no fijada
            moldes[grupo_de[clave]].update(_hojas(valor, clave))  # hojas del bloque
    return moldes  # tres moldes


# %% [markdown]
# ## Comando de terminal equivalente
#
# `python run.py pd --config … --set clave=valor …` con cada valor en YAML de flujo (el mismo texto que
# `config_notebook` pasa a `apply_overrides`) y cada argumento citado con `shlex.quote` (`[64, 32]` lleva un espacio).

# %%
def _valor_set(valor) -> str:
    """Texto YAML de flujo de ``valor`` (como src/common/notebooks.config_notebook)."""
    return yaml.safe_dump(valor, default_flow_style=True).strip().removesuffix("...").strip()  # p. ej. [64, 32]


# %%
def comando_terminal(config: str, cambios: dict) -> str:
    """Texto del comando de run.py pd con ``config`` y un --set por cada entrada de ``cambios``."""
    partes = [*COMANDO_PD, "--config", str(config)]  # base
    for clave, valor in cambios.items():  # en el orden del SET
        partes += ["--set", f"{clave}={_valor_set(valor)}"]  # un --set por clave
    return " ".join(shlex.quote(p) for p in partes)  # citado para pegar en la terminal


# %% [markdown]
# ## Salidas de la demo en la carpeta temporal
#
# Copia profunda con `salida = carpeta/resultados` y `modelos = carpeta/modelos`, **aunque `SET` las haya cambiado**:
# la demo del notebook nunca escribe en `results/` ni en `models/`.

# %%
def redirigir_salidas(cfg: dict, carpeta) -> dict:
    """Copia de ``cfg`` con salida y modelos dentro de ``carpeta`` (texto, como exige rutas_de)."""
    nueva = copy.deepcopy(cfg)  # el original no cambia
    nueva["salida"] = str(Path(carpeta) / SUBCARPETA_RESULTADOS)  # resultados de la demo
    nueva["modelos"] = str(Path(carpeta) / SUBCARPETA_MODELOS)  # modelos de la demo
    return nueva  # config redirigida


# %% [markdown]
# ## Métricas de una corrida (demo o reportada)
#
# Logística y MLP con media y desviación de las 10 métricas `{auc,gini,ks,brier,logloss}_{val,test}` de
# `summary.json` (o del resultado de `correr_pd`, que trae además `rutas`), semillas, parámetros, configuraciones de la
# búsqueda (null → vacío) y tiempo total.

# %%
def tabla_metricas(resumen: dict) -> pd.DataFrame:
    """Filas logistica y mlp; columnas COLUMNAS_METRICAS."""
    filas = {}  # modelo → fila
    for modelo in MODELOS:  # solo los modelos (no 'rutas')
        entrada = resumen[modelo]  # bloque del modelo
        fila = {}  # columnas de la fila
        for s in SPLITS_MEDIDOS:  # val y test
            for m, etiqueta in ETIQUETA_METRICA.items():  # cada métrica
                bloque = entrada["metricas"][f"{m}_{s}"]  # {media, desv, …}
                fila[f"{etiqueta} {s}"] = bloque["media"]  # media
                fila[f"{etiqueta} {s} (desv)"] = bloque["desv"]  # desviación muestral (n = 1 → 0)
        configs = entrada["n_configuraciones_busqueda"]  # null si la config no es la de la búsqueda
        fila.update({"semillas": entrada["metricas"]["auc_test"]["n"], "parámetros": entrada["n_parametros"],  # n
                     "configuraciones": VACIO if configs is None else configs,  # presupuesto
                     "tiempo (s)": entrada["tiempo_total_s"]})  # tiempo
        filas[modelo] = fila  # fila lista
    return pd.DataFrame.from_dict(filas, orient="index")[list(COLUMNAS_METRICAS)]  # orden fijo


# %%
def describir_mlp(entrada: dict) -> str:
    """Una línea con capas, activación, parámetros, semilla mediana y si es la config de la búsqueda."""
    cfg = entrada["config"]  # bloque mlp de la corrida
    return (f"MLP: capas {cfg['capas']} · activación {cfg['activacion']} · {entrada['n_parametros']} parámetros · "
            f"semilla mediana {entrada['semilla_mediana']} de {len(entrada['semillas'])} · "
            f"config de la búsqueda: {SI if entrada['config_de_busqueda'] else NO}")  # una línea


# %%
def filas_por_particion(semillas: list[dict]) -> dict:
    """{train, val, test}: filas que usó la corrida (iguales en todas las semillas del MLP)."""
    if not semillas:  # sin registros
        raise ValueError("no hay semillas del MLP (mlp_seed*.json) de las que leer las filas")  # vacío
    distintas = {tuple(r["n"][s] for s in SPLITS) for r in semillas}  # una por muestra usada
    if len(distintas) != 1:  # semillas de corridas distintas
        raise ValueError(f"las semillas no usaron las mismas filas por split: {sorted(distintas)}")  # incoherente
    return dict(zip(SPLITS, distintas.pop()))  # en el orden de SPLITS


# %% [markdown]
# ## Datos que consume la PD (Parte A)
#
# Partición temporal y tasa de malos por split, la tabla por año de originación (la partición de cada año sale de los
# cortes; los años posteriores a `test_hasta` quedan fuera), las variables de `configs/features.yaml` con su cita del
# diccionario y el PSI frente a train de `post_preprocesamiento.json`.

# %%
def tabla_particion(resumen_datos: dict) -> pd.DataFrame:
    """Filas train, val, test: fecha de corte, n y tasa de malos."""
    return pd.DataFrame({s: {"hasta": resumen_datos["particion"][f"{s}_hasta"],  # corte (inclusive)
                             "n": resumen_datos["n_por_split"][s],  # filas modeladas
                             "tasa de malos": resumen_datos["tasa_malos_por_split"][s]}  # tasa
                         for s in SPLITS}).T  # una fila por split


# %%
def split_de_anio(anio: int, particion: dict) -> str:
    """Partición del año de originación según los cortes (o FUERA si es posterior a test_hasta)."""
    for s in SPLITS:  # train, val, test en orden
        if anio <= pd.Timestamp(particion[f"{s}_hasta"]).year:  # dentro del corte
            return s  # primer split que lo contiene
    return FUERA  # posterior a test


# %%
def tabla_por_anio(resumen_datos: dict) -> pd.DataFrame:
    """Por año (entero, en orden numérico): partición, créditos, fracción sin resolver y tasa de malos."""
    n = {int(a): v for a, v in resumen_datos["n_por_anio"].items()}  # años como texto → entero
    na = {int(a): v for a, v in resumen_datos["fraccion_na_por_anio"].items()}  # ídem
    tasa = {int(a): v for a, v in resumen_datos["tasa_malos_por_anio"].items()}  # ídem
    sobran = sorted((set(na) | set(tasa)) - set(n))  # años sin conteo
    if sobran:  # resumen incoherente
        raise ValueError(f"años con tasa o fracción NA pero sin n_por_anio: {sobran}")  # cuáles
    anios = sorted(n)  # orden numérico (no de texto)
    return pd.DataFrame({"partición": [split_de_anio(a, resumen_datos["particion"]) for a in anios],  # split
                         "créditos": [n[a] for a in anios],  # todos los del año
                         "fracción sin resolver (NA)": [na.get(a, float("nan")) for a in anios],  # objetivo NA
                         "tasa de malos (resueltos)": [tasa.get(a, float("nan")) for a in anios]},  # entre resueltos
                        index=pd.Index(anios, name="año"))  # índice entero


# %%
def tabla_variables(features_cfg: dict) -> pd.DataFrame:
    """Las variables de features.yaml: tipo, etiqueta, transformación y cita literal del diccionario."""
    posteriores = [f["nombre"] for f in features_cfg["features"] if f["disponible_en_originacion"] is not True]  # §5.7
    if posteriores:  # fuga de información
        raise ValueError(f"variables no disponibles en la originación: {posteriores}")  # núcleo §5.7
    return pd.DataFrame({f["nombre"]: {"tipo": f["tipo"], "etiqueta": f["etiqueta_es"],  # descripción
                                       "transformación": f["transformacion"],  # preprocesamiento de A
                                       "cita del diccionario": f["cita_diccionario"]}  # literal de Kaggle
                         for f in features_cfg["features"]}).T  # una fila por variable


# %%
def tabla_psi(post: dict, por: str = "test") -> pd.DataFrame:
    """PSI de cada variable de ``por`` frente a train, de mayor a menor."""
    psi = post["psi_vs_train"]  # {split: {variable: psi}}
    if por not in psi:  # p. ej. train (no se compara consigo)
        raise ValueError(f"no hay PSI de {por!r} frente a train; hay {sorted(psi)}")  # cuáles
    serie = pd.Series(psi[por], name=f"PSI {por} vs train").sort_values(ascending=False)  # descendente
    return serie.rename_axis("variable").to_frame()  # tabla


# %% [markdown]
# ## Búsqueda de hiperparámetros
#
# La rejilla de C de la logística (con exactamente un C elegido), el espacio de la búsqueda del MLP con la escala de
# `src/pd/hpo.py`, los mejores ensayos completos (orden por log-loss de val; empate → menor número, como B-pd §3.3) y
# el resumen de las dos búsquedas.

# %%
def tabla_busqueda_c(busqueda: dict) -> pd.DataFrame:
    """Rejilla de C ordenada por C con la columna 'elegido' (exactamente una fila)."""
    t = pd.DataFrame(busqueda["filas"]).sort_values("C", ignore_index=True)  # ordenada por C
    t["elegido"] = t["C"] == busqueda["C_elegido"]  # igualdad exacta: el C elegido sale de la misma rejilla
    if t["elegido"].sum() != 1:  # fuera de la rejilla o repetido
        raise ValueError(f"C_elegido={busqueda['C_elegido']} debe aparecer exactamente una vez en la rejilla; "
                         f"aparece {int(t['elegido'].sum())}")  # cuántas
    return t  # tabla


# %%
def tabla_espacio(hpo: dict) -> pd.DataFrame:
    """Parámetro del espacio: rango [mín, máx] o lista de valores, y escala (entero, lineal, log u opciones)."""
    ajenas = [k for k in hpo["espacio"] if k not in CLAVES_ESPACIO]  # no son del espacio de B-pd §3.3
    if ajenas:  # p. ej. momento
        raise ValueError(f"claves del espacio que no se buscan: {ajenas}")  # cuáles
    filas = {}  # parámetro → fila
    for clave, valores in hpo["espacio"].items():  # en el orden del JSON
        if clave in RANGOS:  # rango [mín, máx]
            filas[clave] = {"rango o valores": f"[{valores[0]}, {valores[1]}]", "escala": RANGOS[clave]}  # rango
        else:  # lista de opciones
            filas[clave] = {"rango o valores": ", ".join(map(str, valores)), "escala": ESCALA_OPCIONES}  # lista
    return pd.DataFrame(filas).T  # una fila por parámetro


# %%
def _k_valido(k) -> int:
    """``k`` si es un entero ≥ 1 (bool y float no)."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:  # True es int en Python
        raise ValueError(f"k debe ser un entero ≥ 1; se recibió {k!r}")  # tipo o rango
    return k  # válido


# %%
def tabla_mejores_ensayos(hpo: dict, k: int = K_ENSAYOS) -> pd.DataFrame:
    """Los ``k`` mejores ensayos COMPLETE por (log-loss de val, número), con parámetros y métricas."""
    k = _k_valido(k)  # validado antes de todo
    completos = [e for e in hpo["ensayos"] if e["estado"] == ESTADO_COMPLETO]  # sin FAIL
    if not completos:  # todos fallaron
        raise ValueError(f"no hay ensayos {ESTADO_COMPLETO} en hpo_ensayos.json")  # vacío
    mejores = sorted(completos, key=lambda e: (e["logloss_val"], e["numero"]))[:k]  # orden de B-pd §3.3
    filas = {}  # número → fila
    for e in mejores:  # en orden
        fila = {p: e["parametros"][p] for p in PARAMETROS_ENSAYO}  # parámetros armados
        fila["capas"] = str(fila["capas"])  # lista como texto (una celda)
        fila.update({titulo: e[clave] for clave, titulo in COLUMNAS_ENSAYO.items()})  # métricas
        filas[e["numero"]] = fila  # fila lista
    return pd.DataFrame.from_dict(filas, orient="index").rename_axis("ensayo")  # índice = número del ensayo


# %%
def _texto_borde(en_borde: dict) -> str:
    """'lr: no · l2: max · …' a partir de en_borde (null = no está en el borde)."""
    return " · ".join(f"{p}: {NO if v is None else v}" for p, v in en_borde.items())  # una línea


# %%
def tabla_resumen_busquedas(busqueda: dict, hpo: dict) -> pd.DataFrame:
    """Logística y MLP: método, configuraciones, criterio, mejor valor, tiempo (Σ fit) y en_borde."""
    elegida = tabla_busqueda_c(busqueda).query("elegido").iloc[0]  # fila del C elegido (valida la rejilla)
    m = hpo["muestreador"]  # TPE
    metodo_mlp = (f"{m['nombre']} de Optuna · {m['n_startup_trials']} ensayos aleatorios al inicio · "
                  f"multivariado: {SI if m['multivariate'] else NO}")  # muestreador registrado
    return pd.DataFrame({
        "logistica": {"método": "rejilla de C", "configuraciones": len(busqueda["filas"]),  # B3
                      "criterio": busqueda["criterio"], "mejor valor": elegida[busqueda["criterio"]],  # elegido
                      "tiempo (s)": busqueda["tiempo_total_s"],  # tiempo
                      "en el borde": SI if busqueda["en_borde"] else NO},  # C en un extremo
        "mlp": {"método": metodo_mlp, "configuraciones": hpo["n_ensayos"], "criterio": hpo["criterio"],  # B4
                "mejor valor": hpo["mejor"][hpo["criterio"]], "tiempo (s)": hpo["tiempo_total_s"],  # mejor y tiempo
                "en el borde": _texto_borde(hpo["en_borde"])}}).T  # lr, l2, dropout


# %% [markdown]
# ## Modelo final y comparación justa
#
# Config ganadora y parámetros frente a la logística; las semillas del MLP (media, desv., mejor y peor) frente al valor
# único de la logística; la diferencia MLP − logística con el "mejor" según `SENTIDO` (empate explícito); en cuántas
# semillas el MLP es estrictamente mejor (un empate no cuenta); la curva de aprendizaje de la semilla mediana y la
# lectura de la calibración.

# %%
def tabla_arquitectura(resumen: dict) -> pd.DataFrame:
    """Claves de la config del MLP, ensayo de la búsqueda y parámetros entrenables frente a los de la logística."""
    mlp = resumen["mlp"]  # bloque del MLP
    filas = {clave: {"MLP": str(v) if isinstance(v, list) else v, "logística": VACIO}  # config del MLP
             for clave, v in mlp["config"].items()}  # en su orden
    ensayo = mlp["busqueda"]["ensayo"] if mlp["config_de_busqueda"] else VACIO  # procedencia
    filas["ensayo de la búsqueda"] = {"MLP": ensayo, "logística": VACIO}  # ensayo de pd-hpo
    filas["parámetros entrenables"] = {"MLP": mlp["n_parametros"],  # complejidad del MLP
                                       "logística": resumen["logistica"]["n_parametros"]}  # y de la logística
    return pd.DataFrame(filas).T  # una fila por clave


# %%
def tabla_semillas(resumen: dict) -> pd.DataFrame:
    """Por métrica: media, desv., mejor y peor del MLP, número de semillas y valor de la logística."""
    log, mlp = resumen["logistica"]["metricas"], resumen["mlp"]["metricas"]  # bloques
    return pd.DataFrame({k: {"MLP media": b["media"], "MLP desv": b["desv"], "MLP mejor": b["mejor"],  # MLP
                             "MLP peor": b["peor"], "semillas": b["n"], "logística": log[k]["media"]}  # referencia
                         for k, b in mlp.items()}).T  # una fila por métrica


# %%
def _sentido(nombre: str) -> str:
    """'max' o 'min' de la métrica ``nombre`` ({métrica}_{split}); error si no se conoce."""
    metrica, _, split = nombre.rpartition("_")  # p. ej. ('auc', 'test')
    if metrica not in SENTIDO or split not in SPLITS_MEDIDOS:  # métrica o split desconocidos
        raise ValueError(f"métrica desconocida {nombre!r}: se esperan {sorted(SENTIDO)} en {SPLITS_MEDIDOS}")  # qué
    return SENTIDO[metrica]  # sentido


# %%
def _mejor(mlp: float, log: float, sentido: str) -> str:
    """'mlp', 'logistica' o EMPATE según el sentido de la métrica."""
    if mlp == log:  # iguales
        return EMPATE  # sin ganador
    return "mlp" if (mlp > log) == (sentido == "max") else "logistica"  # mayor es mejor sólo en 'max'


# %%
def comparacion(resumen: dict) -> pd.DataFrame:
    """Por métrica: logística, MLP (media), MLP − logística y quién es mejor según SENTIDO."""
    log, mlp = resumen["logistica"]["metricas"], resumen["mlp"]["metricas"]  # bloques
    distintas = sorted(set(log) ^ set(mlp))  # métricas que solo tiene uno
    if distintas:  # no hay comparación justa
        raise ValueError(f"la logística y el MLP no tienen las mismas métricas: {distintas}")  # cuáles
    filas = {}  # métrica → fila
    for k in mlp:  # en el orden del summary
        s = _sentido(k)  # valida el nombre
        a, b = log[k]["media"], mlp[k]["media"]  # logística y MLP
        filas[k] = {"logística": a, "MLP (media)": b, "MLP − logística": b - a, "mejor": _mejor(b, a, s)}  # fila
    return pd.DataFrame(filas).T  # una fila por métrica


# %%
def victorias_por_semilla(semillas: list[dict], resumen: dict) -> pd.DataFrame:
    """Por métrica: en cuántas semillas el MLP es estrictamente mejor que la logística, y cuántas semillas hay."""
    esperadas = sorted(resumen["mlp"]["semillas"])  # las de summary.json
    vistas = sorted(r["seed"] for r in semillas)  # las de los registros
    if vistas != esperadas:  # faltan, sobran o se repiten
        raise ValueError(f"las semillas de los registros {vistas} no son las de summary.json {esperadas}")  # cuáles
    filas = {}  # métrica → fila
    for k, bloque in resumen["logistica"]["metricas"].items():  # cada métrica
        s = _sentido(k)  # valida el nombre
        sin = [r["seed"] for r in semillas if k not in r["metricas"]]  # registros incompletos
        if sin:  # no se cuenta como derrota
            raise ValueError(f"las semillas {sin} no tienen la métrica {k}")  # cuáles
        gana = sum(_mejor(r["metricas"][k], bloque["media"], s) == "mlp" for r in semillas)  # estrictas
        filas[k] = {"victorias del MLP": gana, "semillas": len(semillas)}  # fila
    return pd.DataFrame(filas).T  # una fila por métrica


# %%
def ficha_aprendizaje(semillas: list[dict], resumen: dict) -> dict:
    """Semilla mediana: épocas corridas, mejor época, paciencia, máximo de épocas y mejor pérdida de val."""
    k = resumen["mlp"]["semilla_mediana"]  # semilla de pd_mlp.pt
    registro = next((r for r in semillas if r["seed"] == k), None)  # su registro
    if registro is None:  # no está entre las semillas
        raise ValueError(f"no hay registro de la semilla mediana {k}")  # cuál
    return {"semilla": k, "épocas": registro["epocas"], "mejor época": registro["mejor_epoca"],  # parada temprana
            "paciencia": registro["config"]["paciencia"], "máx. épocas": registro["config"]["max_epocas"],  # config
            "mejor pérdida de val": registro["mejor_perdida_val"]}  # BCE de val restaurada


# %%
def resumen_calibracion(calib: dict) -> pd.DataFrame:
    """Por modelo (pd-calibracion.json): PD media, tasa observada, bins con tasa > PD media y número de bins."""
    return pd.DataFrame({m: {"PD media": c["pd_media_global"], "tasa observada": c["tasa_global"],  # globales
                             "bins con tasa > PD media": sum(t > p for t, p in zip(c["tasa_observada"],  # encima
                                                                                    c["pd_media"])),  # de la diag.
                             "bins": len(c["pd_media"])}  # total
                         for m, c in calib["modelos"].items()}).T  # una fila por modelo
