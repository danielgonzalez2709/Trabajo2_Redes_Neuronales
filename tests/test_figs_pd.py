"""B7 — figuras de la PD regenerables desde results/ sin reentrenar (núcleo §3 "Arquitectura" y §7.1; B-pd §3.4)."""

import hashlib  # huellas SHA-256 de las carpetas versionadas
import json  # insumos sintéticos y serialización estricta
from pathlib import Path  # rutas

import matplotlib.pyplot as plt  # cierra las figuras que captura la espía de la prueba 7c
import numpy as np  # datos sintéticos y comparaciones
import pandas as pd  # pred_test sintético
import pytest  # aserciones de excepciones y marcadores

from src.common import metricas  # referencia de las curvas y métricas (B1)
from src.viz import figs_pd  # módulo bajo prueba

RAIZ = Path(__file__).resolve().parents[1]  # raíz del repositorio
CARPETAS_PROTEGIDAS = [RAIZ / "blog" / "docs" / "figures", RAIZ / "results", RAIZ / "configs"]  # no se escriben
PNG_SIN_HPO = sorted(n for n in figs_pd.ARCHIVOS if n != "pd-busqueda-mlp.png")  # salida sin hpo_ensayos.json
N_SEMILLAS = 3  # semillas del MLP en los insumos sintéticos
SEMILLA_MEDIANA = 1  # semilla mediana declarada en el summary sintético


# ---------------------------------------------------------------- insumos sintéticos (claves reales de B3–B5)

def _metricas_resumen(valor: float) -> dict:
    """Bloque de métricas de summary.json con el mismo valor en todas (n = 1, como la logística)."""
    nombres = [f"{m}_{s}" for s in ("val", "test") for m in ("auc", "gini", "ks", "brier", "logloss")]  # 10 claves
    return {k: {"media": valor, "desv": 0.0, "mejor": valor, "peor": valor, "n": 1} for k in nombres}  # n = 1


def _semilla(seed: int) -> dict:
    """mlp_seed<seed>.json con las claves que usan las figuras (y algunas de más, como el real)."""
    historial = [{"epoca": e, "perdida_train": 0.5 - e / 100,  # train baja con la época
                  "perdida_val": 0.52 + abs(e - 2) / 100}  # val con mínimo en la época 2
                 for e in range(1, 6)]  # 5 épocas
    return {"modelo": "mlp", "seed": seed, "config": {"capas": [4], "paciencia": 3, "batch": 8},  # config parcial
            "metricas": {"auc_test": 0.66 + seed / 100, "ks_test": 0.24 + seed / 100,  # suben con la semilla
                         "brier_test": 0.18 + seed / 1000, "logloss_val": 0.51 + seed / 1000},  # ídem: distintas
            "n_parametros": 25, "epocas": 5, "mejor_epoca": 2, "historial": historial,  # entrenamiento
            "clave_extra": "se ignora"}  # una clave de más no rompe nada


def _hpo() -> dict:
    """hpo_ensayos.json con un ensayo FAIL (sin log-loss) y el mejor declarado."""
    ensayos = [{"numero": 0, "estado": "COMPLETE", "logloss_val": 0.52},  # primero
               {"numero": 1, "estado": "COMPLETE", "logloss_val": 0.51},  # mejora
               {"numero": 2, "estado": "FAIL", "logloss_val": None},  # falla: se excluye
               {"numero": 3, "estado": "COMPLETE", "logloss_val": 0.53},  # empeora
               {"numero": 4, "estado": "COMPLETE", "logloss_val": 0.50}]  # mejor
    return {"criterio": "logloss_val", "muestreador": {"nombre": "TPESampler", "n_startup_trials": 2},  # TPE
            "mejor": {"numero": 4, "logloss_val": 0.50}, "ensayos": ensayos}  # ganador y ensayos


def _busqueda() -> dict:
    """busqueda_logistica.json con las filas desordenadas a propósito."""
    filas = [{"C": c, "logloss_val": 0.52 + abs(np.log10(c) + 2) / 100,  # mínima en C = 1e-2
              "auc_val": 0.66 - abs(np.log10(c) + 2) / 100}  # máxima en C = 1e-2
             for c in (1.0, 1e-4, 1e-2)]  # desordenadas por C
    return {"criterio": "logloss_val", "filas": filas, "C_elegido": 1e-2, "en_borde": False}  # elegido interno


def _pred(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """pred_test.parquet sintético: y depende de la PD para que las curvas tengan forma."""
    rng = np.random.default_rng(seed)  # azar solo en los datos de prueba
    p_log = rng.uniform(0.05, 0.6, n)  # PD de la logística
    p_mlp = np.clip(p_log + rng.normal(0, 0.05, n), 0.01, 0.99)  # PD del MLP, parecida
    y = (rng.uniform(size=n) < p_log).astype(np.int64)  # desenlace coherente con la PD
    return pd.DataFrame({"id": [str(i) for i in range(n)], "y": y,  # columnas de pred_test (núcleo §5.4)
                         "pred_logistica": p_log, "pred_mlp": p_mlp})  # una por modelo


def _resumen() -> dict:
    """summary.json sintético con la semilla mediana declarada."""
    return {"logistica": {"n_parametros": 5, "metricas": _metricas_resumen(0.25)},  # logística (n = 1)
            "mlp": {"n_parametros": 25, "metricas": _metricas_resumen(0.3),  # MLP
                    "semillas": list(range(N_SEMILLAS)), "semilla_mediana": SEMILLA_MEDIANA}}  # semillas


def _escribir_insumos(raiz: Path, con_hpo: bool = True) -> Path:
    """Escribe los insumos sintéticos en raiz/results/data/pd y devuelve esa carpeta."""
    base = raiz / figs_pd.ENTRADA  # results/data/pd dentro de tmp_path
    (base / "semillas").mkdir(parents=True)  # carpeta de semillas
    _pred().to_parquet(base / "pred_test.parquet")  # predicciones de test
    (base / "summary.json").write_text(json.dumps(_resumen()), encoding="utf-8")  # resumen
    (base / "busqueda_logistica.json").write_text(json.dumps(_busqueda()), encoding="utf-8")  # búsqueda de C
    if con_hpo:  # la demo no tiene búsqueda del MLP
        (base / "hpo_ensayos.json").write_text(json.dumps(_hpo()), encoding="utf-8")  # búsqueda Optuna
    for s in (2, 0, 1):  # orden de escritura distinto al de las semillas
        (base / "semillas" / f"mlp_seed{s}.json").write_text(json.dumps(_semilla(s)), encoding="utf-8")  # semilla
    (base / "semillas" / "logistica_seed0.json").write_text("{}", encoding="utf-8")  # no es del MLP: se ignora
    return base  # carpeta de la corrida


def _huellas(carpetas: list[Path]) -> dict[str, str]:
    """SHA-256 de cada archivo de las carpetas (recursivo); una carpeta inexistente no aporta nada."""
    return {r.as_posix(): hashlib.sha256(r.read_bytes()).hexdigest()  # huella por ruta
            for c in carpetas if c.exists() for r in sorted(c.rglob("*")) if r.is_file()}  # todos los archivos


def _datos_en_memoria(con_hpo: bool = True) -> dict:
    """`datos` para generar() sin leer archivos."""
    return {"resumen": _resumen(), "pred": _pred(), "semillas": [_semilla(s) for s in range(N_SEMILLAS)],  # base
            "hpo": _hpo() if con_hpo else None, "busqueda": _busqueda()}  # búsquedas


# ---------------------------------------------------------------- 1. ROC

def test_1_datos_roc_caso_a_mano():
    """4 créditos: la ROC pasa por (0,0)…(1,1) y AUC/Gini son los de metricas (0,75 y 0,5 a mano)."""
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])  # un bueno (0,4) supera a un malo (0,35)
    d = figs_pd.datos_roc(y, p)  # función pura
    np.testing.assert_allclose(d["fpr"], [0, 0, 0.5, 0.5, 1])  # umbrales +inf, 0,8, 0,4, 0,35, 0,1
    np.testing.assert_allclose(d["tpr"], [0, 0.5, 0.5, 1, 1])  # ídem
    assert d["auc"] == pytest.approx(0.75) and d["auc"] == metricas.auc(y, p)  # 3 de 4 pares bien ordenados
    assert d["gini"] == pytest.approx(0.5) and d["gini"] == metricas.gini(y, p)  # 2·0,75 − 1


# ---------------------------------------------------------------- 2. KS

def test_2_datos_ks_caso_a_mano():
    """Separación perfecta: F₁ = [0, 0, ½, 1], F₀ = [½, 1, 1, 1]; KS = 1 en t = 0,2 (argmax único)."""
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.8])  # todos los malos por encima
    d = figs_pd.datos_ks(y, p)  # función pura
    np.testing.assert_allclose(d["umbrales"], [0.1, 0.2, 0.3, 0.8])  # valores únicos ascendentes
    np.testing.assert_allclose(d["f_malos"], [0, 0, 0.5, 1])  # F₁(t)
    np.testing.assert_allclose(d["f_buenos"], [0.5, 1, 1, 1])  # F₀(t)
    assert d["ks"] == pytest.approx(1.0) and d["ks"] == metricas.ks(y, p)  # igual a metricas.ks
    assert d["umbral_ks"] == pytest.approx(0.2)  # argmax de |F₁ − F₀|


def test_2b_datos_ks_umbral_es_el_argmax_en_datos_aleatorios():
    """En datos sintéticos, |F₁ − F₀| en umbral_ks es el KS."""
    pred = _pred()  # 400 créditos
    d = figs_pd.datos_ks(pred["y"], pred["pred_mlp"])  # función pura
    i = d["i_ks"]  # índice de t* que devuelve la pura
    assert d["umbrales"][i] == d["umbral_ks"]  # el índice apunta al umbral del KS
    assert abs(d["f_malos"][i] - d["f_buenos"][i]) == pytest.approx(d["ks"])  # ahí está la separación máxima


# ---------------------------------------------------------------- 3. calibración

def test_3_datos_calibracion_coincide_con_metricas_y_es_json_nativo():
    """Puntos = curva_calibracion (10 bins); medias globales; floats e ints nativos, sin NaN."""
    pred = _pred()  # 400 créditos
    d = figs_pd.datos_calibracion(pred, semilla_mediana=7)  # función pura
    assert set(d) == {"split", "n_bins", "semilla_mediana", "modelos"}  # claves exactas (las lee el video)
    assert (d["split"], d["n_bins"], d["semilla_mediana"]) == ("test", 10, 7)  # cabecera
    assert set(d["modelos"]) == {"logistica", "mlp"}  # los dos modelos
    for m in ("logistica", "mlp"):  # cada modelo
        c = metricas.curva_calibracion(pred["y"], pred[f"pred_{m}"], n_bins=10)  # referencia (B1)
        bloque = d["modelos"][m]  # salida de la pura
        assert set(bloque) == {"pd_media", "tasa_observada", "n", "pd_media_global", "tasa_global"}  # claves
        np.testing.assert_allclose(bloque["pd_media"], c.pd_media)  # PD media por bin
        np.testing.assert_allclose(bloque["tasa_observada"], c.tasa_observada)  # tasa por bin
        assert bloque["n"] == c.n_por_bin.tolist()  # créditos por bin
        assert bloque["pd_media_global"] == pytest.approx(pred[f"pred_{m}"].mean())  # media global de la PD
        assert bloque["tasa_global"] == pytest.approx(pred["y"].mean())  # tasa global de malos
        assert all(type(v) is float for v in bloque["pd_media"] + bloque["tasa_observada"])  # float nativo
        assert all(type(v) is int for v in bloque["n"])  # int nativo
        assert type(bloque["pd_media_global"]) is float and type(bloque["tasa_global"]) is float  # globales
    json.dumps(d, allow_nan=False)  # serializa sin NaN ni tipos de numpy


def test_3b_datos_calibracion_con_bins_vacios_omite_y_conserva_longitudes():
    """Con empates masivos hay bins vacíos: se omiten y las tres listas siguen alineadas y suman n."""
    n = 100  # créditos
    pred = pd.DataFrame({"y": np.r_[np.zeros(50, int), np.ones(50, int)],  # mitad malos
                         "pred_logistica": np.r_[np.full(90, 0.2), np.linspace(0.3, 0.9, 10)],  # 90 empates
                         "pred_mlp": np.full(n, 0.25)})  # PD constante: un solo bin
    d = figs_pd.datos_calibracion(pred, semilla_mediana=0)  # función pura
    for m in ("logistica", "mlp"):  # cada modelo
        b = d["modelos"][m]  # bloque
        assert len(b["pd_media"]) == len(b["tasa_observada"]) == len(b["n"]) < 10  # bins omitidos
        assert sum(b["n"]) == n  # nadie se pierde
    assert d["modelos"]["mlp"]["n"] == [n]  # PD constante → un único bin
    json.dumps(d, allow_nan=False)  # sin NaN


def test_3c_datos_calibracion_sin_columna_de_un_modelo_da_error():
    """pred_test sin pred_mlp → ValueError que nombra la columna."""
    with pytest.raises(ValueError, match="pred_mlp"):  # columna que falta
        figs_pd.datos_calibracion(_pred().drop(columns="pred_mlp"), semilla_mediana=0)  # sin el MLP


# ---------------------------------------------------------------- 4. métricas por semilla

def test_4_metricas_por_semilla_ordena_y_marca_la_mediana():
    """Ordenado por semilla; la mediana es la de summary.json; la logística sale de summary.logistica."""
    semillas = [_semilla(s) for s in (2, 0, 1)]  # desordenadas
    d = figs_pd.metricas_por_semilla(semillas, _resumen())  # función pura
    assert list(d) == ["auc_test", "ks_test", "brier_test"]  # nombres por defecto, en orden
    for nombre, bloque in d.items():  # cada métrica
        assert bloque["seeds"] == [0, 1, 2]  # ordenado por semilla
        assert bloque["valores"] == [_semilla(s)["metricas"][nombre] for s in (0, 1, 2)]  # alineado
        assert bloque["mediana"] == (SEMILLA_MEDIANA, _semilla(SEMILLA_MEDIANA)["metricas"][nombre])  # (seed, valor)
        assert bloque["logistica"] == _resumen()["logistica"]["metricas"][nombre]["media"]  # referencia


def test_4b_metricas_por_semilla_sin_la_mediana_da_error():
    """La semilla mediana de summary.json no está entre las semillas leídas → ValueError."""
    semillas = [_semilla(s) for s in (0, 2)]  # falta la 1
    with pytest.raises(ValueError, match="mediana"):  # mensaje claro
        figs_pd.metricas_por_semilla(semillas, _resumen())  # función pura


def test_4c_metricas_por_semilla_con_semilla_duplicada_da_error():
    """Dos archivos con la misma semilla (p. ej. copias) → ValueError en vez de puntos repetidos."""
    semillas = [_semilla(s) for s in (0, 1, 1, 2)]  # la 1 dos veces
    with pytest.raises(ValueError, match="repetid"):  # mensaje claro
        figs_pd.metricas_por_semilla(semillas, _resumen())  # función pura


# ---------------------------------------------------------------- 5. aprendizaje

def test_5_datos_aprendizaje_usa_la_semilla_mediana():
    """Curvas, mejor época y paciencia de la semilla mediana."""
    semillas = [_semilla(s) for s in range(N_SEMILLAS)]  # todas
    semillas[SEMILLA_MEDIANA]["historial"][0]["perdida_val"] = 0.99  # marca la mediana para distinguirla
    d = figs_pd.datos_aprendizaje(semillas, _resumen())  # función pura
    assert d["seed"] == SEMILLA_MEDIANA  # semilla de summary.json
    assert d["epocas"] == [1, 2, 3, 4, 5]  # épocas desde 1
    assert d["perdida_val"][0] == 0.99  # es la historia de la mediana
    assert d["perdida_train"] == [h["perdida_train"] for h in _semilla(1)["historial"]]  # train
    assert (d["mejor_epoca"], d["paciencia"]) == (2, 3)  # de la semilla y de su config
    assert d["fin_paciencia"] == 5  # 2 + 3 = 5, que también es la última época


def test_5b_datos_aprendizaje_sin_la_mediana_da_error():
    """La mediana no está → ValueError."""
    with pytest.raises(ValueError, match="mediana"):  # mensaje claro
        figs_pd.datos_aprendizaje([_semilla(0)], _resumen())  # solo la semilla 0


# ---------------------------------------------------------------- 6. búsqueda del MLP

def test_6_trayectoria_busqueda_minimo_acumulado_a_mano_sin_fail():
    """[0,52, 0,51, 0,53, 0,50] → [0,52, 0,51, 0,51, 0,50]; el FAIL se excluye; ganador y startup del JSON."""
    d = figs_pd.trayectoria_busqueda(_hpo())  # función pura
    assert d["numeros"] == [0, 1, 3, 4]  # sin el ensayo 2 (FAIL)
    np.testing.assert_allclose(d["logloss_val"], [0.52, 0.51, 0.53, 0.50])  # log-loss por ensayo
    np.testing.assert_allclose(d["mejor_hasta_ahora"], [0.52, 0.51, 0.51, 0.50])  # mínimo acumulado
    assert d["mejor_numero"] == _hpo()["mejor"]["numero"] == 4  # ganador declarado
    assert d["n_startup"] == 2  # fin de la fase aleatoria del TPE


def test_6b_trayectoria_busqueda_sin_ensayos_completos_da_error():
    """Todos FAIL → ValueError."""
    hpo = _hpo()  # copia
    for e in hpo["ensayos"]:  # todos fallan
        e["estado"] = "FAIL"  # sin COMPLETE
    with pytest.raises(ValueError, match="COMPLETE"):  # mensaje claro
        figs_pd.trayectoria_busqueda(hpo)  # función pura


def test_6c_trayectoria_busqueda_ganador_incoherente_da_error():
    """El ganador declarado no es un COMPLETE con el mínimo → ValueError (el JSON no es de esta búsqueda)."""
    hpo = _hpo()  # copia
    hpo["mejor"]["numero"] = 1  # 0,51 no es el mínimo (0,50)
    with pytest.raises(ValueError, match="mejor"):  # mensaje claro
        figs_pd.trayectoria_busqueda(hpo)  # función pura
    hpo["mejor"]["numero"] = 2  # el FAIL
    with pytest.raises(ValueError, match="mejor"):  # mensaje claro
        figs_pd.trayectoria_busqueda(hpo)  # función pura


# ---------------------------------------------------------------- 7. búsqueda de C

def test_7_datos_busqueda_c_ordena_por_c_y_c_elegido_en_la_rejilla():
    """Ordenado por C; las otras columnas se reordenan igual; C elegido es uno de la rejilla."""
    d = figs_pd.datos_busqueda_c(_busqueda())  # función pura
    assert d["C"] == [1e-4, 1e-2, 1.0]  # ascendente
    por_c = {f["C"]: f for f in _busqueda()["filas"]}  # filas originales por C
    assert d["logloss_val"] == [por_c[c]["logloss_val"] for c in d["C"]]  # alineado
    assert d["auc_val"] == [por_c[c]["auc_val"] for c in d["C"]]  # alineado
    assert d["C_elegido"] in d["C"] and d["en_borde"] is False  # dentro de la rejilla


def test_7b_datos_busqueda_c_con_c_elegido_fuera_de_la_rejilla_da_error():
    """C elegido que no está en las filas → ValueError."""
    busqueda = _busqueda()  # copia
    busqueda["C_elegido"] = 0.5  # no se probó
    with pytest.raises(ValueError, match="C_elegido"):  # mensaje claro
        figs_pd.datos_busqueda_c(busqueda)  # función pura


# ---------------------------------------------------------------- 8–10. generar

def test_8_generar_escribe_exactamente_archivos_y_no_toca_lo_versionado(tmp_path):
    """Escribe ARCHIVOS (PNG no vacíos y el JSON) solo en destino; blog, results y configs reales intactos."""
    raiz = tmp_path / "repo"  # repo falso
    _escribir_insumos(raiz)  # insumos sintéticos
    destino = tmp_path / "figuras"  # destino temporal
    antes = _huellas(CARPETAS_PROTEGIDAS)  # huellas de lo versionado
    antes_tmp = {r for r in tmp_path.rglob("*") if r.is_file()}  # archivos del tmp antes
    rutas = figs_pd.generar(raiz=raiz, destino=destino)  # genera
    assert sorted(r.name for r in rutas) == sorted(figs_pd.ARCHIVOS)  # exactamente ARCHIVOS
    assert {r for r in tmp_path.rglob("*") if r.is_file()} - antes_tmp == set(rutas)  # nada más en tmp
    assert all(r.parent == destino and r.stat().st_size > 0 for r in rutas)  # no vacíos y en destino
    calib = json.loads((destino / "pd-calibracion.json").read_text(encoding="utf-8"))  # JSON escrito
    assert calib["semilla_mediana"] == SEMILLA_MEDIANA  # la del summary
    assert _huellas(CARPETAS_PROTEGIDAS) == antes  # nada versionado cambió


def test_8b_generar_resuelve_salida_al_llamarse(tmp_path, monkeypatch):
    """Sin destino, usa raiz / SALIDA leído al llamar (así test_no_network lo redirige)."""
    raiz = tmp_path / "repo"  # repo falso
    _escribir_insumos(raiz)  # insumos
    monkeypatch.setattr(figs_pd, "SALIDA", Path("otra/carpeta"))  # redirección posterior al import
    rutas = figs_pd.generar(raiz=raiz)  # sin destino
    assert {r.parent for r in rutas} == {raiz / "otra" / "carpeta"}  # respetó el SALIDA actual


def test_9_generar_con_datos_sin_hpo_omite_la_busqueda_del_mlp(tmp_path):
    """datos= en memoria y hpo None → 6 PNG + JSON (sin pd-busqueda-mlp.png)."""
    rutas = figs_pd.generar(raiz=tmp_path / "no-existe", destino=tmp_path, datos=_datos_en_memoria(False))  # sin hpo
    assert sorted(r.name for r in rutas) == PNG_SIN_HPO  # 6 PNG + JSON


def test_9b_insumos_de_corrida_sin_hpo_devuelve_none(tmp_path):
    """Carpeta de la demo (sin hpo_ensayos.json): hpo None; ignora logistica_seed*.json; semillas ordenadas."""
    base = _escribir_insumos(tmp_path, con_hpo=False)  # como la demo
    datos = figs_pd.insumos_de_corrida(base)  # lectura
    assert set(datos) == {"resumen", "pred", "semillas", "hpo", "busqueda"}  # claves
    assert datos["hpo"] is None  # sin búsqueda del MLP
    assert [s["seed"] for s in datos["semillas"]] == [0, 1, 2]  # solo mlp_seed*, por semilla


def test_9c_generar_con_datos_con_claves_de_mas_o_de_menos_da_error(tmp_path):
    """datos= con una clave mal escrita (de más) o sin una clave → ValueError que las nombra."""
    datos = _datos_en_memoria()  # completos
    datos["hpo_"] = datos.pop("hpo")  # error de tipeo
    with pytest.raises(ValueError, match="hpo"):  # nombra la clave
        figs_pd.generar(destino=tmp_path, datos=datos)  # genera
    assert not any(tmp_path.iterdir())  # no escribió nada


@pytest.mark.parametrize("archivo,comando", [("summary.json", "run.py pd"),  # resumen (B5)
                                             ("pred_test.parquet", "run.py pd"),  # PD de test (B5)
                                             ("busqueda_logistica.json", "run.py pd")])  # rejilla de C (B3)
def test_10_insumo_que_falta_nombra_el_comando(tmp_path, archivo, comando):
    """Un insumo obligatorio que falta → FileNotFoundError con el archivo y el comando que lo genera."""
    base = _escribir_insumos(tmp_path)  # insumos
    (base / archivo).unlink()  # quita uno
    with pytest.raises(FileNotFoundError, match=f"falta .*{archivo}.*{comando}"):  # nombra archivo y comando
        figs_pd.generar(raiz=tmp_path, destino=tmp_path / "figs")  # genera


def test_10b_sin_semillas_da_error(tmp_path):
    """Sin mlp_seed*.json → ValueError que menciona las semillas y no escribe nada."""
    base = _escribir_insumos(tmp_path)  # insumos
    for r in (base / "semillas").glob("mlp_seed*.json"):  # borra las del MLP
        r.unlink()  # sin semillas
    with pytest.raises(ValueError, match="semillas"):  # mensaje claro
        figs_pd.generar(raiz=tmp_path, destino=tmp_path / "figs")  # genera
    assert not (tmp_path / "figs").exists()  # no dejó figuras a medias


# ---------------------------------------------------------------- ronda 2: lógica sacada de los dibujos y ramas

def test_2c_datos_ks_devuelve_el_indice_de_t():
    """Caso a mano de la prueba 2: t* = 0,2 está en la posición 1."""
    d = figs_pd.datos_ks(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.8]))  # separación perfecta
    assert d["i_ks"] == 1 and type(d["i_ks"]) is int  # índice nativo de t*


def test_3d_datos_calibracion_a_mano_con_dos_bins():
    """4 créditos, n_bins = 2: bordes Q(0, ½, 1) = 0,1, 0,4, 0,8 (½ interpola 0,2 y 0,6).

    Bin 0 (p ≤ 0,4): p = 0,1 y 0,2 con y = 0 y 1 → PD media 0,15, tasa 0,5, n 2.
    Bin 1 (p > 0,4): p = 0,6 y 0,8 con y = 0 y 1 → PD media 0,7, tasa 0,5, n 2.
    Globales: PD media (0,1 + 0,2 + 0,6 + 0,8)/4 = 0,425; tasa 2/4 = 0,5.
    """
    pred = pd.DataFrame({"y": [0, 1, 0, 1], "pred_logistica": [0.1, 0.2, 0.6, 0.8],  # caso a mano
                         "pred_mlp": [0.1, 0.2, 0.6, 0.8]})  # mismo caso para el MLP
    b = figs_pd.datos_calibracion(pred, semilla_mediana=0, n_bins=2)["modelos"]["mlp"]  # bloque del MLP
    np.testing.assert_allclose(b["pd_media"], [0.15, 0.7])  # medias por bin
    np.testing.assert_allclose(b["tasa_observada"], [0.5, 0.5])  # tasas por bin
    assert b["n"] == [2, 2]  # créditos por bin
    assert b["pd_media_global"] == pytest.approx(0.425) and b["tasa_global"] == pytest.approx(0.5)  # globales


@pytest.mark.parametrize("maximo,esperado", [(3 * 0.1, 0.3),  # 0,30000000000000004: no sube a 0,4
                                             (0.31, 0.4),  # sube al siguiente múltiplo
                                             (0.6000000000000001, 0.6),  # error de coma flotante tolerado
                                             (0.4776, 0.5)])  # máximo real de la calibración (tasa del MLP)
def test_limite_ejes_redondea_hacia_arriba_al_paso(maximo, esperado):
    """ceil(max/paso − 1e-9)·paso con paso 0,1, la misma regla que Honesta.tsx."""
    assert figs_pd.limite_ejes([0.05, maximo]) == esperado  # igualdad exacta tras redondear a 10 decimales


def _semilla_con_epocas(n_epocas: int, mejor: int, paciencia: int) -> dict:
    """Registro de la semilla mediana con `n_epocas`, su mejor época y su paciencia."""
    r = _semilla(SEMILLA_MEDIANA)  # base
    r["historial"] = [{"epoca": e, "perdida_train": 0.5, "perdida_val": 0.5} for e in range(1, n_epocas + 1)]  # planas
    r["mejor_epoca"], r["config"]["paciencia"] = mejor, paciencia  # parada temprana
    return r  # registro


def test_5c_fin_paciencia_corta_por_paciencia():
    """Mejor 2, paciencia 3, 10 épocas → la franja termina en 2 + 3 = 5."""
    d = figs_pd.datos_aprendizaje([_semilla_con_epocas(10, 2, 3)], _resumen())  # función pura
    assert d["fin_paciencia"] == 5  # mejor + paciencia


def test_5d_fin_paciencia_corta_por_la_ultima_epoca():
    """Mejor 8, paciencia 5, 10 épocas → 8 + 5 = 13 pasa del final: termina en la época 10."""
    d = figs_pd.datos_aprendizaje([_semilla_con_epocas(10, 8, 5)], _resumen())  # función pura
    assert d["fin_paciencia"] == 10  # última época


def test_4d_metricas_por_semilla_con_metrica_desconocida_da_error():
    """Una métrica sin título en español → ValueError que la nombra."""
    with pytest.raises(ValueError, match="auc_vall"):  # error de tipeo
        figs_pd.metricas_por_semilla([_semilla(s) for s in range(N_SEMILLAS)], _resumen(), nombres=("auc_vall",))


def test_7c_titulo_avisa_si_c_quedo_en_el_borde(tmp_path, monkeypatch):
    """en_borde = True → el título general lo dice; False → no."""
    titulos = []  # títulos capturados

    def espia(fig, ruta):  # reemplaza a _guardar: captura el título sin escribir
        """Guarda el título general de ``fig``, la cierra y devuelve ``ruta``."""
        titulos.append(fig.get_suptitle())  # API pública de matplotlib (≥ 3.8)
        plt.close(fig)  # _guardar la cerraría: no deja figuras abiertas
        return ruta  # misma salida que _guardar

    monkeypatch.setattr(figs_pd, "_guardar", espia)  # no escribe: solo captura
    for borde in (True, False):  # las dos ramas
        bc = {**figs_pd.datos_busqueda_c(_busqueda()), "en_borde": borde}  # datos con el borde pedido
        figs_pd.figura_busqueda_logistica(bc, tmp_path)  # dibuja (sin escribir)
    assert "en el borde" in titulos[0] and "en el borde" not in titulos[1]  # solo la rama True avisa


def test_9d_insumos_de_corrida_sin_carpeta_semillas_da_lista_vacia(tmp_path):
    """Sin semillas/ → semillas = [] (generar luego falla con su ValueError)."""
    base = _escribir_insumos(tmp_path)  # insumos
    for r in (base / "semillas").iterdir():  # vacía la carpeta
        r.unlink()  # borra cada archivo
    (base / "semillas").rmdir()  # y la quita
    assert figs_pd.insumos_de_corrida(base)["semillas"] == []  # lista vacía


def test_9e_generar_desde_archivos_sin_hpo_omite_la_busqueda_del_mlp(tmp_path):
    """Carpeta tipo demo (sin hpo_ensayos.json) leída de disco → 6 PNG + JSON."""
    _escribir_insumos(tmp_path / "repo", con_hpo=False)  # como la demo
    rutas = figs_pd.generar(raiz=tmp_path / "repo", destino=tmp_path / "figs")  # lee archivos
    assert sorted(r.name for r in rutas) == PNG_SIN_HPO  # sin pd-busqueda-mlp.png


def test_9f_generar_sin_columna_y_da_error_sin_escribir(tmp_path):
    """pred_test sin `y` → ValueError con el comando que lo genera, antes de escribir nada."""
    datos = _datos_en_memoria()  # completos
    datos["pred"] = datos["pred"].drop(columns="y")  # sin desenlace
    with pytest.raises(ValueError, match=r"\['y'\].*run.py pd"):  # nombra la columna y el comando
        figs_pd.generar(destino=tmp_path / "figs", datos=datos)  # genera
    assert not (tmp_path / "figs").exists()  # no escribió nada


# ---------------------------------------------------------------- 11. humo con los resultados reales

@pytest.mark.lento
def test_11_generar_con_resultados_reales(tmp_path):
    """Resultados reales de B3–B5 → tmp_path: 10 bins por modelo y semilla mediana 20 (summary.json de B5)."""
    if not (RAIZ / figs_pd.ENTRADA / "summary.json").exists():  # clon sin results/
        pytest.skip("faltan los resultados reales de results/data/pd/")  # se salta
    rutas = figs_pd.generar(raiz=RAIZ, destino=tmp_path)  # genera con los reales
    assert sorted(r.name for r in rutas) == sorted(figs_pd.ARCHIVOS)  # las 7 + JSON
    calib = json.loads((tmp_path / "pd-calibracion.json").read_text(encoding="utf-8"))  # JSON
    assert calib["semilla_mediana"] == 20  # mediana de B5
    assert all(len(calib["modelos"][m]["pd_media"]) == 10 for m in ("logistica", "mlp"))  # 10 bins cada uno
