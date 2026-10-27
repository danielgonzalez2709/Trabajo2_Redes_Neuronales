# %% [markdown]
# # B9 — pruebas de `src/pd/tablas.py` (tablas y textos del notebook `02_pd`, B-pd §3.7)
#
# Insumos mínimos con las claves reales de B3–B5 (núcleo §5.4) y de la Parte A; al final, las mismas funciones contra
# los JSON reales versionados (se omiten si faltan). Ninguna prueba escribe en `data/`, `models/`, `results/`,
# `configs/` ni `blog/`.

# %%
import copy  # copias para comprobar que nada se muta
import json  # lectura de los JSON reales
import shlex  # ida y vuelta del comando de terminal
from pathlib import Path  # rutas

import numpy as np  # NaN y comparaciones
import pandas as pd  # tablas
import pytest  # aserciones de excepciones y omisiones

from src.common.cli import build_parser  # mismo parser que run.py
from src.common.config import apply_overrides, load_config  # mismo cargador que el CLI
from src.common.notebooks import config_notebook  # cargador del notebook
from src.pd import tablas  # módulo bajo prueba

RAIZ = Path(__file__).resolve().parents[1]  # raíz del repositorio
DEMO = "configs/pd/demo.yaml"  # config de la demo (B6)
METRICAS = ("auc", "gini", "ks", "brier", "logloss")  # nombres de métricas (núcleo §5.4)
SPLITS = ("val", "test")  # splits medidos
LOG = {"auc": 0.66, "gini": 0.32, "ks": 0.24, "brier": 0.19, "logloss": 0.56}  # valores de la logística
MLP = {"auc": 0.67, "gini": 0.34, "ks": 0.24, "brier": 0.18, "logloss": 0.55}  # medias del MLP (ks empata)


# %% [markdown]
# ## Insumos sintéticos

# %%
def _met(media: float, desv: float = 0.0, n: int = 1) -> dict:
    """Bloque {media, desv, mejor, peor, n} de summary.json."""
    return {"media": media, "desv": desv, "mejor": media + 0.001, "peor": media - 0.001, "n": n}  # mejor ≠ peor


# %%
def _bloque(valores: dict, desv: float, n: int, **cambios) -> dict:
    """Las 10 métricas {m}_{s} de summary.json con los valores de ``valores`` y ``cambios`` encima."""
    planas = {f"{m}_{s}": valores[m] for s in SPLITS for m in METRICAS}  # 10 claves, split por fuera
    planas.update(cambios)  # p. ej. logloss_val distinta
    return {k: _met(v, desv, n) for k, v in planas.items()}  # bloques


# %%
def _resumen() -> dict:
    """summary.json con logística (n = 1) y MLP de 3 semillas que viene de la búsqueda."""
    config = {"capas": [32, 16, 32], "activacion": "relu", "dropout": 0.1, "l2": 1e-5, "optimizador": "adam",
              "lr": 5e-3, "momento": 0.9, "batch": 512, "max_epocas": 4, "paciencia": 2, "device": "cpu",
              "pos_weight": 1.0}  # bloque mlp
    return {"logistica": {"n_parametros": 49, "n_configuraciones_busqueda": 13, "tiempo_total_s": 60.0,  # B3
                          "metricas": _bloque(LOG, 0.0, 1)},  # n = 1
            "mlp": {"n_parametros": 2673, "n_configuraciones_busqueda": 40, "tiempo_total_s": 900.0,  # B5
                    "metricas": _bloque(MLP, 0.003, 3, logloss_val=0.57),  # la logística gana en logloss_val
                    "semillas": [0, 1, 2], "semilla_mediana": 1, "config": config, "config_de_busqueda": True,  # B5
                    "busqueda": {"tipo": "optuna", "ensayo": 11, "n_ensayos": 40}}}  # procedencia


# %%
def _resumen_sin_busqueda() -> dict:
    """El mismo resumen con un MLP fijado a mano (presupuesto de la búsqueda null)."""
    r = _resumen()  # base
    r["mlp"].update(n_configuraciones_busqueda=None, config_de_busqueda=False, busqueda=None)  # sin_busqueda
    return r  # resumen


# %%
def _semilla(seed: int, **cambios) -> dict:
    """mlp_seed<seed>.json con las 10 métricas iguales a las de la logística salvo ``cambios``."""
    metricas = {f"{m}_{s}": LOG[m] for s in SPLITS for m in METRICAS}  # empate en todo
    metricas.update(cambios)  # lo que cambia en esta semilla
    historial = [{"epoca": e, "perdida_train": 0.5, "perdida_val": 0.5} for e in (1, 2, 3)]  # 3 épocas
    return {"modelo": "mlp", "seed": seed, "metricas": metricas, "epocas": 3, "mejor_epoca": 1,  # registro
            "mejor_perdida_val": 0.5, "historial": historial, "config": {"paciencia": 2, "max_epocas": 4},  # B5
            "n": {"train": 10, "val": 5, "test": 6}}  # filas por split


# %%
def _semillas() -> list[dict]:
    """Tres semillas: AUC test gana 1 (otra empata) y Brier test gana 1 (otra empata)."""
    return [_semilla(0, auc_test=0.65, brier_test=0.18),  # pierde AUC, gana Brier
            _semilla(1, auc_test=0.67, brier_test=0.20),  # gana AUC, pierde Brier
            _semilla(2, auc_test=0.66, brier_test=0.19)]  # empata las dos


# %%
def _hpo() -> dict:
    """hpo_ensayos.json con un FAIL, un empate de log-loss y el espacio real de B-pd §3.3."""
    def ensayo(numero, estado, logloss, capas):  # un ensayo con sus claves de §5.4
        """Registro de un ensayo."""
        return {"numero": numero, "estado": estado,  # identidad
                "parametros": {"capas": capas, "activacion": "relu", "dropout": 0.1, "l2": 1e-4,  # armados
                               "optimizador": "adam", "lr": 1e-3, "batch": 512},  # resto de parámetros
                "logloss_val": logloss, "auc_val": None if logloss is None else 0.6,  # métricas
                "perdida_val_entrenar": logloss, "epocas": 5, "mejor_epoca": 2, "n_parametros": 100,  # fit
                "tiempo_s": 1.0}  # tiempo
    ensayos = [ensayo(0, "COMPLETE", 0.52, [16]), ensayo(1, "FAIL", None, [8]),  # FAIL sin métricas
               ensayo(2, "COMPLETE", 0.50, [32]), ensayo(3, "COMPLETE", 0.50, [64]),  # empate: gana el 2
               ensayo(4, "COMPLETE", 0.51, [128])]  # tercero
    espacio = {"n_capas": [1, 4], "neuronas": [16, 32], "activacion": ["relu", "tanh"], "dropout": [0.0, 0.5],
               "l2": [1e-6, 1e-2], "optimizador": ["adam", "sgd"], "lr": [1e-4, 1e-2], "batch": [256, 512]}  # §3.3
    return {"criterio": "logloss_val", "muestreador": {"nombre": "TPESampler", "semilla": 0,  # TPE
                                                       "n_startup_trials": 2, "multivariate": True},  # valores
            "espacio": espacio, "n_ensayos": 5, "estados": {"COMPLETE": 4, "FAIL": 1},  # presupuesto
            "tiempo_total_s": 5.0, "mejor": {"numero": 2, "logloss_val": 0.50},  # ganador
            "en_borde": {"lr": None, "l2": "max", "dropout": None}, "ensayos": ensayos}  # borde y ensayos


# %%
def _busqueda() -> dict:
    """busqueda_logistica.json con las filas desordenadas por C."""
    filas = [{"C": c, "logloss_val": ll, "auc_val": 0.6, "n_iter": 10, "tiempo_s": 0.5}  # una fila por C
             for c, ll in ((1.0, 0.53), (1e-4, 0.55), (1e-2, 0.52))]  # desordenadas
    return {"criterio": "logloss_val", "filas": filas, "C_elegido": 1e-2, "en_borde": False,  # elegido interno
            "tiempo_total_s": 1.5}  # tiempo


# %%
def _resumen_datos() -> dict:
    """resumen_datos.json de la Parte A con los años como texto y desordenados."""
    anios = {"2015": (40, 0.9, 0.25), "2009": (10, 0.0, 0.10), "2012": (20, 0.1, 0.15),  # año: (n, NA, tasa)
             "2013": (30, 0.4, 0.20), "2014": (35, 0.6, 0.30)}  # desordenados
    return {"particion": {"train_hasta": "2012-12-31", "val_hasta": "2013-12-31", "test_hasta": "2014-12-31"},
            "n_por_split": {"train": 25, "val": 18, "test": 14},  # filas modeladas
            "tasa_malos_por_split": {"train": 0.14, "val": 0.20, "test": 0.30},  # tasas
            "n_por_anio": {a: v[0] for a, v in anios.items()},  # créditos por año
            "fraccion_na_por_anio": {a: v[1] for a, v in anios.items()},  # sin resolver
            "tasa_malos_por_anio": {a: v[2] for a, v in anios.items()}}  # tasa entre resueltos


# %% [markdown]
# ## 1. Moldes de la config

# %%
def test_moldes_cubren_la_config_menos_las_fijadas():
    """Las claves de los tres moldes juntas = las de la config sin salida, modelos ni busqueda."""
    cfg = load_config(RAIZ / DEMO)  # config real de la demo
    moldes = tablas.moldes_de(cfg)  # los tres moldes
    assert set(moldes) == {"comunes", "logistica", "mlp"}  # tres grupos
    claves = [k for m in moldes.values() for k in m]  # todas las claves
    assert len(claves) == len(set(claves))  # ninguna en dos moldes
    hojas = {"base_seed", "datos.muestra", "semillas.n"} | {f"logistica.{k}" for k in cfg["logistica"]} | {
        f"mlp.{k}" for k in cfg["mlp"]}  # hojas de la config sin las fijadas
    assert set(claves) == hojas  # exactamente esas
    assert set(moldes["comunes"]) == {"base_seed", "datos.muestra", "semillas.n"}  # comunes de §3.7
    assert not set(tablas.CLAVES_FIJADAS) & {k.split(".")[0] for k in claves}  # ninguna fijada


# %%
def test_moldes_conservan_valor_y_tipo():
    """Los valores salen tal cual de la config (float, lista, None)."""
    cfg = {"base_seed": 0, "datos": {"muestra": None}, "semillas": {"n": 3},  # comunes
           "logistica": {"rejilla_C": [1e-3, 1e-2], "tol": 1e-6}, "mlp": {"l2": 1e-3, "capas": [8, 4]},  # modelos
           "salida": "x", "modelos": "y", "busqueda": {"tipo": "sin_busqueda"}}  # fijadas
    moldes = tablas.moldes_de(cfg)  # moldes
    assert moldes["mlp"] == {"mlp.l2": 1e-3, "mlp.capas": [8, 4]}  # valores
    assert isinstance(moldes["mlp"]["mlp.l2"], float) and isinstance(moldes["mlp"]["mlp.capas"], list)  # tipos
    assert moldes["comunes"]["datos.muestra"] is None  # null se conserva
    assert moldes["logistica"]["logistica.rejilla_C"] == [1e-3, 1e-2]  # lista de floats


# %%
@pytest.mark.parametrize("extra", [{"nueva": 1}, {"logistica_extra": {"a": 1}}])  # clave suelta o bloque
def test_moldes_clave_sin_clasificar_es_error(extra):
    """Una clave nueva en la config obliga a actualizar los moldes."""
    cfg = {**load_config(RAIZ / DEMO), **extra}  # demo + una clave de más
    with pytest.raises(ValueError, match="molde"):  # no se clasifica
        tablas.moldes_de(cfg)  # error


# %%
def test_moldes_no_mutan_la_config():
    """moldes_de no cambia la config ni comparte listas con ella."""
    cfg = load_config(RAIZ / DEMO)  # config real
    antes = copy.deepcopy(cfg)  # copia
    moldes = tablas.moldes_de(cfg)  # moldes
    moldes["mlp"]["mlp.capas"].append(99)  # se muta el molde
    assert cfg == antes  # la config no cambió


# %% [markdown]
# ## 2. Comando de terminal: ida y vuelta por el parser del CLI

# %%
def _config_por_cli(comando: str) -> dict:
    """La config que obtendría run.py con ``comando`` (sin ejecutar nada)."""
    partes = shlex.split(comando)  # como lo separa la shell
    assert partes[:2] == ["python", "run.py"]  # prefijo
    args = build_parser().parse_args(partes[2:])  # mismo parser
    assert args.comando == "pd"  # subcomando
    return apply_overrides(load_config(RAIZ / args.config), args.overrides)  # mismo cargador


# %%
CAMBIOS = [  # cambios con espacios, notación científica, null, texto y booleanos
    {},  # sin cambios
    {"mlp.capas": [64, 32]},  # lista con espacio
    {"mlp.lr": 1e-3, "mlp.l2": 4.451065531349953e-05},  # floats
    {"datos.muestra": None},  # null
    {"mlp.activacion": "tanh", "mlp.device": "cpu"},  # texto
    {"salida": "results/data/pd 2_demo"},  # texto con espacio
    {"mlp.pos_weight": True},  # booleano (el CLI lo acepta; la validación es de correr_pd)
    {"logistica.rejilla_C": [1e-3, 3.1623e-3]},  # lista de floats
]  # casos


# %%
@pytest.mark.parametrize("cambios", CAMBIOS)  # cada caso
def test_comando_terminal_ida_y_vuelta(cambios, monkeypatch):
    """shlex → parser del CLI → apply_overrides da la misma config que config_notebook."""
    monkeypatch.chdir(RAIZ)  # rutas relativas a la raíz
    comando = tablas.comando_terminal(DEMO, cambios)  # texto
    assert comando.startswith(f"python run.py pd --config {DEMO}")  # forma
    assert _config_por_cli(comando) == config_notebook(DEMO, cambios, imprimir=False)  # misma config


# %%
def test_comando_terminal_clave_inexistente_falla_como_el_cli(monkeypatch):
    """Una clave que no existe da ValueError por los dos caminos."""
    monkeypatch.chdir(RAIZ)  # rutas relativas
    comando = tablas.comando_terminal(DEMO, {"mlp.capaz": [8]})  # error de tipeo
    with pytest.raises(ValueError, match="Clave inexistente"):  # CLI
        _config_por_cli(comando)  # error
    with pytest.raises(ValueError, match="Clave inexistente"):  # notebook
        config_notebook(DEMO, {"mlp.capaz": [8]}, imprimir=False)  # mismo error


# %%
def test_comando_terminal_cita_la_ruta_de_la_config():
    """Una ruta de config con espacio va citada."""
    comando = tablas.comando_terminal("configs/pd/mi demo.yaml", {})  # ruta con espacio
    assert shlex.split(comando)[4] == "configs/pd/mi demo.yaml"  # un solo argumento


# %% [markdown]
# ## 3. Salidas de la demo redirigidas

# %%
def test_redirigir_salidas_no_muta_e_ignora_set(tmp_path):
    """salida y modelos quedan dentro de la carpeta aunque SET los haya cambiado; el original no cambia."""
    cfg = {"salida": "results/data/pd", "modelos": "models", "mlp": {"capas": [8]}}  # SET apuntó a lo real
    antes = copy.deepcopy(cfg)  # copia
    nueva = tablas.redirigir_salidas(cfg, tmp_path)  # redirigida
    assert cfg == antes  # sin mutar
    assert Path(nueva["salida"]) == tmp_path / "resultados"  # resultados en la carpeta
    assert Path(nueva["modelos"]) == tmp_path / "modelos"  # modelos en la carpeta
    assert isinstance(nueva["salida"], str) and isinstance(nueva["modelos"], str)  # rutas_de exige texto
    nueva["mlp"]["capas"].append(4)  # la copia es profunda
    assert cfg["mlp"]["capas"] == [8]  # el original sigue igual


# %%
def test_redirigir_salidas_acepta_texto(tmp_path):
    """La carpeta puede llegar como texto."""
    nueva = tablas.redirigir_salidas({"salida": "a", "modelos": "b"}, str(tmp_path))  # str
    assert Path(nueva["salida"]).parent == tmp_path  # dentro


# %% [markdown]
# ## 4. Métricas de la demo

# %%
def test_tabla_metricas_columnas_y_valores():
    """Logística y MLP con media y desv. de las 10 métricas, semillas, parámetros, configuraciones y tiempo."""
    t = tablas.tabla_metricas(_resumen())  # tabla
    assert list(t.index) == ["logistica", "mlp"]  # filas
    assert list(t.columns) == list(tablas.COLUMNAS_METRICAS)  # orden fijo
    assert t.columns[0] == "AUC val" and t.columns[1] == "AUC val (desv)"  # primera métrica
    assert list(t.columns[-4:]) == ["semillas", "parámetros", "configuraciones", "tiempo (s)"]  # presupuesto
    assert t.loc["logistica", "Brier test (desv)"] == 0.0  # n = 1 → desv 0
    assert t.loc["mlp", "AUC test"] == pytest.approx(0.67)  # media
    assert t.loc["mlp", "AUC test (desv)"] == pytest.approx(0.003)  # desv
    assert t.loc["mlp", "semillas"] == 3 and t.loc["logistica", "semillas"] == 1  # corridas
    assert t.loc["mlp", "configuraciones"] == 40 and t.loc["mlp", "parámetros"] == 2673  # presupuesto


# %%
def test_tabla_metricas_sin_busqueda_deja_configuraciones_vacias():
    """MLP sin_busqueda → configuraciones vacías (no 0 ni NaN)."""
    t = tablas.tabla_metricas(_resumen_sin_busqueda())  # tabla
    assert t.loc["mlp", "configuraciones"] == ""  # vacío
    assert t.loc["logistica", "configuraciones"] == 13  # la logística conserva su rejilla


# %%
def test_tabla_metricas_ignora_rutas_de_correr_pd():
    """El resultado de correr_pd trae además 'rutas': no es un modelo."""
    r = {**_resumen(), "rutas": object()}  # como lo devuelve correr_pd
    assert list(tablas.tabla_metricas(r).index) == ["logistica", "mlp"]  # solo modelos


# %%
def test_describir_mlp():
    """Arquitectura, parámetros, semilla mediana y procedencia en una línea."""
    texto = tablas.describir_mlp(_resumen()["mlp"])  # texto
    assert "[32, 16, 32]" in texto and "2673" in texto  # arquitectura y parámetros
    assert "semilla mediana 1 de 3" in texto and "búsqueda: sí" in texto  # mediana y procedencia
    assert "búsqueda: no" in tablas.describir_mlp(_resumen_sin_busqueda()["mlp"])  # fijada a mano
    assert "\n" not in texto  # una línea


# %%
def test_filas_por_particion():
    """Filas por split de la corrida, iguales en todas las semillas."""
    assert tablas.filas_por_particion(_semillas()) == {"train": 10, "val": 5, "test": 6}  # de seed["n"]
    distinta = _semillas()  # copia
    distinta[1]["n"] = {"train": 9, "val": 5, "test": 6}  # otra muestra
    with pytest.raises(ValueError, match="filas"):  # semillas incoherentes
        tablas.filas_por_particion(distinta)  # error
    with pytest.raises(ValueError, match="semillas"):  # sin semillas
        tablas.filas_por_particion([])  # error


# %% [markdown]
# ## 5. Comparación y victorias por semilla

# %%
def test_comparacion_signo_y_mejor_segun_sentido():
    """MLP − logística con signo; 'mejor' según SENTIDO (Brier menor es mejor) y empate explícito."""
    c = tablas.comparacion(_resumen())  # tabla
    assert list(c.index) == [f"{m}_{s}" for s in SPLITS for m in METRICAS]  # orden del summary
    assert c.loc["auc_test", "MLP − logística"] == pytest.approx(0.01)  # sube
    assert c.loc["auc_test", "mejor"] == "mlp"  # max
    assert c.loc["brier_test", "MLP − logística"] == pytest.approx(-0.01)  # baja
    assert c.loc["brier_test", "mejor"] == "mlp"  # min: menor es mejor
    assert c.loc["logloss_val", "mejor"] == "logistica"  # 0.57 > 0.56
    assert c.loc["ks_test", "mejor"] == "empate"  # iguales


# %%
def test_comparacion_metrica_desconocida_es_error():
    """Una métrica fuera de SENTIDO o de val/test no se compara a ciegas."""
    for nombre in ("foo_test", "auc_train"):  # métrica y split desconocidos
        r = _resumen()  # base
        r["mlp"]["metricas"][nombre] = _met(0.5)  # de más en el MLP
        r["logistica"]["metricas"][nombre] = _met(0.5)  # y en la logística
        with pytest.raises(ValueError, match=nombre):  # nombrada
            tablas.comparacion(r)  # error


# %%
def test_comparacion_metricas_distintas_es_error():
    """Si los dos modelos no tienen las mismas métricas, no hay comparación justa."""
    r = _resumen()  # base
    del r["mlp"]["metricas"]["ks_val"]  # falta una
    with pytest.raises(ValueError, match="ks_val"):  # nombrada
        tablas.comparacion(r)  # error


# %%
def test_victorias_por_semilla_empate_no_cuenta():
    """Victorias estrictas del MLP; n = número de semillas."""
    v = tablas.victorias_por_semilla(_semillas(), _resumen())  # tabla
    assert v.loc["auc_test", "victorias del MLP"] == 1  # semilla 1; la 2 empata
    assert v.loc["brier_test", "victorias del MLP"] == 1  # semilla 0 (menor es mejor); la 2 empata
    assert v.loc["gini_val", "victorias del MLP"] == 0  # todas empatan
    assert (v["semillas"] == 3).all()  # n


# %%
def test_victorias_por_semilla_semillas_ajenas_es_error():
    """Las semillas deben ser exactamente las de summary.json (no una carpeta vieja)."""
    with pytest.raises(ValueError, match="semillas"):  # falta una
        tablas.victorias_por_semilla(_semillas()[:2], _resumen())  # error
    duplicada = _semillas() + [_semilla(2)]  # semilla repetida
    with pytest.raises(ValueError, match="semillas"):  # duplicada
        tablas.victorias_por_semilla(duplicada, _resumen())  # error


# %%
def test_victorias_metrica_que_falta_en_una_semilla_es_error():
    """Una semilla sin una de las métricas no se cuenta como derrota."""
    semillas = _semillas()  # base
    del semillas[0]["metricas"]["ks_test"]  # le falta una
    with pytest.raises(ValueError, match="ks_test"):  # nombrada
        tablas.victorias_por_semilla(semillas, _resumen())  # error


# %%
def test_tabla_semillas():
    """Media, desv, mejor y peor del MLP y el valor de la logística por métrica."""
    t = tablas.tabla_semillas(_resumen())  # tabla
    assert t.loc["auc_test", "MLP media"] == pytest.approx(0.67)  # media
    assert t.loc["auc_test", "MLP mejor"] == pytest.approx(0.671)  # mejor
    assert t.loc["auc_test", "MLP peor"] == pytest.approx(0.669)  # peor
    assert t.loc["auc_test", "logística"] == pytest.approx(0.66)  # referencia
    assert (t["semillas"] == 3).all()  # n


# %%
def test_tabla_arquitectura():
    """Config ganadora, ensayo de la búsqueda y parámetros frente a la logística."""
    t = tablas.tabla_arquitectura(_resumen())  # tabla
    assert t.loc["capas", "MLP"] == "[32, 16, 32]"  # arquitectura
    assert t.loc["parámetros entrenables", "MLP"] == 2673  # MLP
    assert t.loc["parámetros entrenables", "logística"] == 49  # logística
    assert t.loc["ensayo de la búsqueda", "MLP"] == 11  # procedencia
    assert tablas.tabla_arquitectura(_resumen_sin_busqueda()).loc["ensayo de la búsqueda", "MLP"] == ""  # fijada


# %%
def test_ficha_aprendizaje():
    """Épocas, mejor época y paciencia de la semilla mediana."""
    f = tablas.ficha_aprendizaje(_semillas(), _resumen())  # ficha
    assert f == {"semilla": 1, "épocas": 3, "mejor época": 1, "paciencia": 2, "máx. épocas": 4,  # entrenamiento
                 "mejor pérdida de val": 0.5}  # pérdida
    r = _resumen()  # base
    r["mlp"]["semilla_mediana"] = 7  # no está entre las semillas
    with pytest.raises(ValueError, match="mediana"):  # sin registro
        tablas.ficha_aprendizaje(_semillas(), r)  # error


# %% [markdown]
# ## 6. Mejores ensayos de la búsqueda

# %%
def test_mejores_ensayos_excluye_fail_y_desempata_por_numero():
    """Solo COMPLETE, orden (log-loss, número)."""
    t = tablas.tabla_mejores_ensayos(_hpo(), k=3)  # tres mejores
    assert list(t.index) == [2, 3, 4]  # empate 2–3 → menor número primero
    assert 1 not in t.index  # el FAIL no aparece
    assert t.loc[2, "capas"] == "[32]" and t.loc[2, "log-loss val"] == pytest.approx(0.50)  # contenido


# %%
def test_mejores_ensayos_k_mayor_que_n_y_k_invalido():
    """k > n devuelve todos los completos; k < 1 o no entero → error."""
    assert len(tablas.tabla_mejores_ensayos(_hpo(), k=99)) == 4  # los 4 completos
    for k in (0, -1, 2.0, True):  # inválidos
        with pytest.raises(ValueError, match="k"):  # nombrado
            tablas.tabla_mejores_ensayos(_hpo(), k=k)  # error


# %%
def test_mejores_ensayos_sin_completos_es_error():
    """Si todos fallaron, no hay tabla."""
    hpo = _hpo()  # base
    for e in hpo["ensayos"]:  # todos
        e["estado"] = "FAIL"  # fallidos
    with pytest.raises(ValueError, match="COMPLETE"):  # sin completos
        tablas.tabla_mejores_ensayos(hpo)  # error


# %%
def test_tabla_espacio():
    """Cada parámetro del espacio con su rango o valores y su escala (la de src/pd/hpo.py)."""
    t = tablas.tabla_espacio(_hpo())  # tabla
    assert list(t.index) == list(_hpo()["espacio"])  # mismo orden
    assert t.loc["lr", "escala"] == "log" and t.loc["dropout", "escala"] == "lineal"  # rangos
    assert t.loc["n_capas", "escala"] == "entero"  # entero
    assert t.loc["lr", "rango o valores"] == "[0.0001, 0.01]"  # rango
    assert t.loc["batch", "rango o valores"] == "256, 512"  # opciones
    hpo = _hpo()  # base
    hpo["espacio"]["momento"] = [0.5, 0.9]  # no se busca (B-pd §3.3)
    with pytest.raises(ValueError, match="momento"):  # nombrado
        tablas.tabla_espacio(hpo)  # error


# %%
def test_tabla_resumen_busquedas():
    """Las dos búsquedas: método, configuraciones, criterio, mejor valor, tiempo y en_borde."""
    t = tablas.tabla_resumen_busquedas(_busqueda(), _hpo())  # tabla
    assert list(t.index) == ["logistica", "mlp"]  # filas
    assert t.loc["logistica", "configuraciones"] == 3 and t.loc["mlp", "configuraciones"] == 5  # presupuesto
    assert t.loc["logistica", "mejor valor"] == pytest.approx(0.52)  # logloss del C elegido
    assert t.loc["mlp", "mejor valor"] == pytest.approx(0.50)  # mejor ensayo
    assert t.loc["logistica", "criterio"] == "logloss_val" == t.loc["mlp", "criterio"]  # criterio
    assert t.loc["logistica", "en el borde"] == "no"  # C interno
    assert t.loc["mlp", "en el borde"] == "lr: no · l2: max · dropout: no"  # dict de en_borde
    assert "TPESampler" in t.loc["mlp", "método"]  # muestreador
    assert t.loc["mlp", "tiempo (s)"] == pytest.approx(5.0)  # Σ fit


# %% [markdown]
# ## 7. Búsqueda de C

# %%
def test_tabla_busqueda_c_ordena_y_marca_un_elegido():
    """Ordenada por C con exactamente un elegido."""
    t = tablas.tabla_busqueda_c(_busqueda())  # tabla
    assert list(t["C"]) == [1e-4, 1e-2, 1.0]  # ordenada
    assert t["elegido"].sum() == 1 and t.loc[t["elegido"], "C"].item() == 1e-2  # un elegido


# %%
@pytest.mark.parametrize("c", [3e-2, 1.00001e-2])  # fuera de la rejilla (también por poco)
def test_tabla_busqueda_c_elegido_fuera_de_la_rejilla_es_error(c):
    """C_elegido que no está en la rejilla → error."""
    b = _busqueda()  # base
    b["C_elegido"] = c  # fuera
    with pytest.raises(ValueError, match="C_elegido"):  # nombrado
        tablas.tabla_busqueda_c(b)  # error


# %%
def test_tabla_busqueda_c_duplicado_es_error():
    """Un C repetido en la rejilla haría ambiguo el elegido."""
    b = _busqueda()  # base
    b["filas"].append(dict(b["filas"][2]))  # 1e-2 dos veces
    with pytest.raises(ValueError, match="C_elegido"):  # dos elegidos
        tablas.tabla_busqueda_c(b)  # error


# %% [markdown]
# ## 8. Datos por año y partición

# %%
def test_tabla_por_anio_orden_numerico_y_split():
    """Años como texto → orden numérico; split por los cortes; 2015 fuera."""
    t = tablas.tabla_por_anio(_resumen_datos())  # tabla
    assert list(t.index) == [2009, 2012, 2013, 2014, 2015]  # numérico
    assert list(t["partición"]) == ["train", "train", "val", "test", "fuera"]  # por los cortes
    assert t.loc[2014, "tasa de malos (resueltos)"] == pytest.approx(0.30)  # tasa
    assert t.loc[2013, "créditos"] == 30 and t.loc[2013, "fracción sin resolver (NA)"] == pytest.approx(0.4)  # n


# %%
def test_tabla_por_anio_texto_no_ordena_como_texto():
    """'999' va antes que '1000' (como texto iría después)."""
    r = _resumen_datos()  # base
    for campo in ("n_por_anio", "fraccion_na_por_anio", "tasa_malos_por_anio"):  # los tres
        r[campo] = {"1000": 1, "999": 1}  # años de 3 y 4 cifras
    assert list(tablas.tabla_por_anio(r).index) == [999, 1000]  # numérico


# %%
def test_tabla_por_anio_anio_de_mas_es_error():
    """Una tasa de un año sin conteo es un resumen incoherente."""
    r = _resumen_datos()  # base
    r["tasa_malos_por_anio"]["2001"] = 0.5  # año sin n
    with pytest.raises(ValueError, match="2001"):  # nombrado
        tablas.tabla_por_anio(r)  # error


# %%
def test_tabla_particion():
    """Corte, n y tasa por split en el orden train, val, test."""
    t = tablas.tabla_particion(_resumen_datos())  # tabla
    assert list(t.index) == ["train", "val", "test"]  # orden
    assert t.loc["val", "hasta"] == "2013-12-31" and t.loc["test", "n"] == 14  # corte y n
    assert t.loc["train", "tasa de malos"] == pytest.approx(0.14)  # tasa


# %%
def test_tabla_variables():
    """Las variables de features.yaml con tipo, etiqueta, transformación y cita literal."""
    feats = {"features": [{"nombre": "a", "tipo": "num", "etiqueta_es": "A", "transformacion": "log1p",  # num
                           "cita_diccionario": "Texto A.", "disponible_en_originacion": True},  # cita
                          {"nombre": "b", "tipo": "cat", "etiqueta_es": "B", "transformacion": "ninguna",  # cat
                           "cita_diccionario": "Texto B.", "disponible_en_originacion": True}]}  # cita
    t = tablas.tabla_variables(feats)  # tabla
    assert list(t.index) == ["a", "b"] and t.loc["b", "cita del diccionario"] == "Texto B."  # literal
    feats["features"][1]["disponible_en_originacion"] = False  # posterior a la originación
    with pytest.raises(ValueError, match="b"):  # fuga (núcleo §5.7)
        tablas.tabla_variables(feats)  # error


# %% [markdown]
# ## 9. PSI

# %%
def test_tabla_psi_orden_descendente_y_split_inexistente():
    """PSI frente a train de mayor a menor; un split que no está → error."""
    post = {"psi_vs_train": {"val": {"a": 0.1, "b": 0.3}, "test": {"a": 0.2, "b": 0.05, "c": 0.4}}}  # PSI
    t = tablas.tabla_psi(post)  # test por defecto
    assert list(t.index) == ["c", "a", "b"]  # descendente
    assert list(tablas.tabla_psi(post, por="val").index) == ["b", "a"]  # val
    with pytest.raises(ValueError, match="train"):  # train no se compara consigo
        tablas.tabla_psi(post, por="train")  # error


# %% [markdown]
# ## Calibración (lectura de B-pd §3.6 impresa, no escrita a mano)

# %%
def test_resumen_calibracion_cuenta_bins_por_encima():
    """Por modelo: PD media, tasa global y cuántos bins tienen tasa > PD media (por encima de la diagonal)."""
    calib = {"split": "test", "n_bins": 3, "modelos": {  # pd-calibracion.json
        "logistica": {"pd_media": [0.1, 0.2, 0.3], "tasa_observada": [0.15, 0.2, 0.4], "n": [1, 1, 1],  # 2 de 3
                      "pd_media_global": 0.2, "tasa_global": 0.25},  # global
        "mlp": {"pd_media": [0.1, 0.2, 0.3], "tasa_observada": [0.05, 0.1, 0.2], "n": [1, 1, 1],  # 0 de 3
                "pd_media_global": 0.2, "tasa_global": 0.12}}}  # global
    t = tablas.resumen_calibracion(calib)  # tabla
    assert t.loc["logistica", "bins con tasa > PD media"] == 2  # el empate no cuenta
    assert t.loc["mlp", "bins con tasa > PD media"] == 0  # todos debajo
    assert (t["bins"] == 3).all() and t.loc["logistica", "tasa observada"] == pytest.approx(0.25)  # n y tasa


# %% [markdown]
# ## 10. Contra los JSON reales (detecta que una fixture se aleje del esquema de verdad)

# %%
REALES = {"summary": "results/data/pd/summary.json", "busqueda": "results/data/pd/busqueda_logistica.json",
          "hpo": "results/data/pd/hpo_ensayos.json", "datos": "data/processed/resumen_datos.json",
          "post": "results/data/eda/post_preprocesamiento.json",
          "calib": "blog/docs/figures/pd-calibracion.json"}  # insumos versionados
FALTAN = [r for r in REALES.values() if not (RAIZ / r).is_file()]  # los que no están


# %%
@pytest.mark.skipif(bool(FALTAN), reason=f"faltan {FALTAN}")  # clon sin resultados
def test_funciones_aceptan_los_json_reales():
    """Cada función corre sobre los JSON reales sin error y con formas esperadas."""
    j = {k: json.loads((RAIZ / r).read_text(encoding="utf-8")) for k, r in REALES.items()}  # cargados
    semillas = [json.loads(p.read_text(encoding="utf-8"))  # registros del MLP
                for p in (RAIZ / "results/data/pd/semillas").glob("mlp_seed*.json")]  # todas
    assert len(tablas.tabla_metricas(j["summary"])) == 2  # dos modelos
    assert tablas.describir_mlp(j["summary"]["mlp"])  # texto no vacío
    assert len(tablas.comparacion(j["summary"])) == 10  # 10 métricas
    v = tablas.victorias_por_semilla(semillas, j["summary"])  # victorias
    assert (v["semillas"] == len(j["summary"]["mlp"]["semillas"])).all()  # todas
    assert len(tablas.tabla_semillas(j["summary"])) == 10  # 10 métricas
    assert "parámetros entrenables" in tablas.tabla_arquitectura(j["summary"]).index  # fila
    assert tablas.ficha_aprendizaje(semillas, j["summary"])["semilla"] == j["summary"]["mlp"]["semilla_mediana"]
    assert tablas.filas_por_particion(semillas) == j["datos"]["n_por_split"]  # todos los datos
    assert tablas.tabla_busqueda_c(j["busqueda"])["elegido"].sum() == 1  # un elegido
    assert len(tablas.tabla_espacio(j["hpo"])) == len(j["hpo"]["espacio"])  # todo el espacio
    assert len(tablas.tabla_mejores_ensayos(j["hpo"])) == 5  # k = 5
    assert tablas.tabla_mejores_ensayos(j["hpo"], k=1).index[0] == j["hpo"]["mejor"]["numero"]  # = mejor de B4
    assert len(tablas.tabla_resumen_busquedas(j["busqueda"], j["hpo"])) == 2  # dos búsquedas
    assert list(tablas.tabla_particion(j["datos"]).index) == ["train", "val", "test"]  # splits
    anios = tablas.tabla_por_anio(j["datos"])  # por año
    assert anios.index.is_monotonic_increasing and "fuera" in set(anios["partición"])  # 2015 fuera
    assert len(tablas.tabla_variables(load_config(RAIZ / "configs/features.yaml"))) == len(j["datos"]["variables"])
    psi = tablas.tabla_psi(j["post"])  # PSI de test
    assert psi.iloc[:, 0].is_monotonic_decreasing  # descendente
    assert set(tablas.resumen_calibracion(j["calib"]).index) == {"logistica", "mlp"}  # dos modelos
    assert tablas.moldes_de(load_config(RAIZ / DEMO))  # la demo real se clasifica


# %%
def test_tabla_metricas_sin_nan():
    """Una tabla con configuraciones vacías no lleva NaN (se ve como 'nan' en el notebook)."""
    t = tablas.tabla_metricas(_resumen_sin_busqueda())  # tabla
    assert not pd.isna(t.loc["mlp", "configuraciones"])  # vacío, no NaN
    assert not np.any(pd.isna(t.drop(columns="configuraciones").to_numpy(dtype=float)))  # sin NaN en lo numérico


# %% [markdown]
# ## Infraestructura: constantes copiadas, imports y tipos

# %%
def test_constantes_del_espacio_iguales_a_las_de_hpo():
    """Las constantes copiadas de src/pd/hpo.py no se desvían de las originales."""
    from src.pd import hpo  # importa Optuna (aquí no importa: no es Jupyter)
    assert tablas.CLAVES_ESPACIO == hpo.CLAVES_ESPACIO  # claves del espacio
    assert tablas.RANGOS == hpo.RANGOS  # escala de cada rango
    assert tablas.ESTADO_COMPLETO == hpo.ESTADO_COMPLETO  # estado de un ensayo terminado
    assert set(tablas.PARAMETROS_ENSAYO) == set(hpo.CLAVES_BUSCADAS)  # parámetros armados del ensayo


# %%
def test_tablas_no_importa_optuna():
    """Importar tablas no arrastra Optuna (en Jupyter imprime un aviso de tqdm con la ruta local)."""
    import subprocess  # proceso limpio: este ya pudo importar Optuna
    import sys  # intérprete actual
    codigo = "import sys; import src.pd.tablas; print('optuna' in sys.modules, 'tqdm.auto' in sys.modules)"  # chequeo
    salida = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True, text=True,  # ejecuta
                            check=True).stdout.split()  # ['False', 'False']
    assert salida == ["False", "False"]  # ni Optuna ni tqdm.auto (el que avisa en Jupyter)


# %%
def test_columnas_enteras_conservan_su_tipo():
    """Semillas, bins y victorias salen como enteros (no 30.0) al armar la tabla por filas."""
    assert pd.api.types.is_integer_dtype(tablas.tabla_semillas(_resumen())["semillas"])  # n
    assert pd.api.types.is_integer_dtype(tablas.victorias_por_semilla(_semillas(), _resumen())["semillas"])  # n
    calib = {"modelos": {"m": {"pd_media": [0.1], "tasa_observada": [0.2], "pd_media_global": 0.1,  # un bin
                               "tasa_global": 0.2}}}  # global
    assert pd.api.types.is_integer_dtype(tablas.resumen_calibracion(calib)["bins"])  # bins
    assert pd.api.types.is_integer_dtype(tablas.tabla_particion(_resumen_datos())["n"])  # n por split
