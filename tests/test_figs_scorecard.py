"""C7 — figuras de la scorecard regenerables desde results/ sin reentrenar (núcleo §0.7 y §7.1; C §3.1–§3.3)."""

import hashlib  # huellas SHA-256 de las carpetas versionadas
import json  # insumos sintéticos
import math  # ln de las odds
from pathlib import Path  # rutas

import numpy as np  # datos sintéticos y comparaciones
import pandas as pd  # loans_pd sintético
import pytest  # aserciones de excepciones y marcadores

from src.scorecard.scorecard import parametros_scorecard, score_desde_pd  # fórmulas de C1
from src.viz import figs_scorecard as fs  # módulo bajo prueba

RAIZ = Path(__file__).resolve().parents[1]  # raíz del repositorio
CARPETAS_PROTEGIDAS = [RAIZ / "blog" / "docs" / "figures", RAIZ / "results", RAIZ / "configs"]  # no se escriben
SC_REAL = RAIZ / "results" / "data" / "scorecard"  # resultados reales de C3 y C4
LOANS_REAL = RAIZ / "data" / "processed" / "loans_pd.parquet"  # datos reales de A
HAY_REALES = all(r.exists() for r in (SC_REAL / "scorecard.json", SC_REAL / "scorecard_mlp.json",
                                      SC_REAL / "shap_mlp.json", LOANS_REAL))  # insumos reales presentes
PARAMS = parametros_scorecard({"base": 600, "odds_base": 50, "pdo": 20})  # 600 en 50:1, PDO 20 (C §3.1)

# Numérica sintética calculada a mano: mediana 5, recorte [0, 10], sin log1p, media 5, desv 2, −factor·β = −4.
#   puntos(v) = −4 · (clip(v, 0, 10) − 5) / 2  →  v = 0: 10;  v = 5: 0;  v = 10: −10;  v = 12 (recortado): −10.
TRANSF = {"mediana": 5.0, "recorte": [0.0, 10.0], "log1p": False, "media": 5.0, "desv": 2.0}  # receta sintética
TABLA = [{"percentil": 1, "valor": 0.0, "puntos": 10.0}, {"percentil": 50, "valor": 5.0, "puntos": 0.0},
         {"percentil": 99, "valor": 10.0, "puntos": -10.0}]  # tabla a mano
NUMERICAS = ("annual_inc", "revol_util", "inq_last_6mths", "dti", "open_acc")  # numéricas sintéticas
JUGADORES = ("annual_inc", "term", "revol_util", "inq_last_6mths", "dti", "open_acc")  # orden de las columnas de phi


# ---------------------------------------------------------------- insumos sintéticos (claves reales de C3 y C4)

def _entrada_log(nombre: str) -> dict:
    """Entrada numérica de scorecard.json (logística)."""
    return {"nombre": nombre, "tipo": "num", "etiqueta_es": f"Etiqueta de {nombre}", "beta": 0.1,
            "puntos_por_unidad_estandarizada": -4.0, "transformacion": dict(TRANSF), "tabla": [dict(t) for t in TABLA],
            "rango_puntos": 20.0}


def _entrada_mlp(nombre: str) -> dict:
    """Entrada numérica de scorecard_mlp.json con colas abiertas, un tramo vacío y dos interiores con n > 0."""
    tramos = [{"desde": None, "hasta": 0.0, "n": 3, "puntos_medios": 9.0},  # cola izquierda: no se dibuja
              {"desde": 0.0, "hasta": 0.0, "n": 0, "puntos_medios": None},  # vacío: no se dibuja
              {"desde": 0.0, "hasta": 5.0, "n": 4, "puntos_medios": 5.0},  # interior
              {"desde": 5.0, "hasta": 10.0, "n": 2, "puntos_medios": -6.0},  # interior
              {"desde": 10.0, "hasta": None, "n": 1, "puntos_medios": -9.0}]  # cola derecha: no se dibuja
    return {"nombre": nombre, "tipo": "num", "etiqueta_es": f"Etiqueta de {nombre}", "tramos": tramos,
            "n_sin_valor": 0, "rango_puntos": 11.0}


def _hist(bordes: list, conteos: list) -> dict:
    """Bloque histograma de distribucion_scores."""
    return {"ancho": 10, "bordes": bordes, "conteos": conteos}


def _bandas(desplazamiento: float) -> list:
    """3 bandas (0-based, como el JSON real)."""
    return [{"banda": b, "desde": None if b == 0 else 500 + 10 * b, "hasta": None if b == 2 else 510 + 10 * b,
             "n": 10 + b, "puntaje_medio": 505 + 10 * b, "pd_media": 0.3 - 0.1 * b + desplazamiento,
             "tasa_malos": 0.35 - 0.1 * b + desplazamiento} for b in range(3)]


def _scorecard(modelo: str) -> dict:
    """scorecard.json o scorecard_mlp.json sintético con las claves reales."""
    entrada = _entrada_log if modelo == "logistica" else _entrada_mlp  # receta por modelo
    cat = {"nombre": "term", "tipo": "cat", "etiqueta_es": "Plazo", "categorias": [], "rango_puntos": 5.0}  # cat
    hist = _hist([480.0, 490.0, 500.0], [3, 4]) if modelo == "logistica" else _hist([490.0, 500.0, 510.0, 520.0],
                                                                                     [1, 2, 3])  # bordes distintos
    return {**PARAMS, "modelo": modelo, "puntos_base": 520.0,
            "aditividad": {"error_max": 0.0, "tolerancia": 1e-9},
            "puntos_por_variable": [entrada(n) for n in NUMERICAS] + [cat],
            "distribucion_scores": {"cortes_bandas": [510.0, 520.0], "n_bandas": 3, "ancho_bin": 10,
                                    "test": {"n": 7, "histograma": hist,
                                             "bandas": _bandas(0.0 if modelo == "logistica" else 0.01)}}}


# media |φ| por jugador: term (cat) la mayor; numéricas: dti > annual_inc > open_acc > revol_util > inq_last_6mths.
MEDIA_ABS = {"term": 0.9, "dti": 0.5, "annual_inc": 0.4, "open_acc": 0.3, "revol_util": 0.2, "inq_last_6mths": 0.1}
IDS = ["30", "10", "20"]  # explicados en un orden distinto al del parquet


def _shap() -> dict:
    """shap_mlp.json sintético: importancia y permutación desordenadas; phi con una columna por jugador."""
    orden = sorted(MEDIA_ABS, key=lambda n: -MEDIA_ABS[n])  # puestos 1…6
    importancia = [{"nombre": n, "media_abs_phi": MEDIA_ABS[n], "puesto": orden.index(n) + 1}
                   for n in reversed(orden)]  # desordenada a propósito
    direccion = {n: ({"tipo": "num", "spearman": 0.1 * (i + 1), "n": 3} if n != "term" else
                     {"tipo": "cat", "phi_medio": {"36 months": -0.1}, "n": {"36 months": 3}})
                 for i, n in enumerate(JUGADORES)}  # ρ distinto por variable
    variables = lambda f: [{"nombre": n, "caida_auc_media": f * MEDIA_ABS[n], "caida_auc_desv": f / 100,  # noqa: E731
                            "aumento_logloss_media": 0.0, "aumento_logloss_desv": 0.0} for n in JUGADORES]
    perm_mlp = variables(0.02)  # orden de JUGADORES
    phi = [[100 * k + j for j in range(len(JUGADORES))] for k in range(len(IDS))]  # φ[k][j] = 100k + j
    return {"modelo": "mlp", "metodo": {"jugadores": list(JUGADORES)}, "phi0": -1.0,
            "importancia_global": importancia, "direccion": direccion,
            "permutacion": {"logistica": {"auc_base": 0.67, "variables": list(reversed(variables(0.01)))},
                            "mlp": {"auc_base": 0.68, "variables": perm_mlp}},
            "explicados": {"ids": list(IDS), "phi": phi, "logit": [0.0, 0.0, 0.0]}}


def _loans() -> pd.DataFrame:
    """loans_pd sintético: ids en otro orden, filas de más y un NaN en dti del id 20."""
    filas = {"id": ["10", "20", "30", "40"], "split": ["test"] * 4, "term": ["36 months"] * 4}
    for i, n in enumerate(NUMERICAS):  # valor = 1000·(i+1) + id
        filas[n] = [1000.0 * (i + 1) + float(v) for v in filas["id"]]
    filas["dti"][1] = np.nan  # crudo vacío: se excluye
    return pd.DataFrame(filas)


def _escribir_insumos(raiz: Path) -> None:
    """Escribe los cuatro insumos sintéticos dentro de `raiz`."""
    base = raiz / fs.ENTRADA  # results/data/scorecard
    base.mkdir(parents=True)
    for nombre, d in (("scorecard.json", _scorecard("logistica")), ("scorecard_mlp.json", _scorecard("mlp")),
                      ("shap_mlp.json", _shap())):
        (base / nombre).write_text(json.dumps(d), encoding="utf-8")
    (raiz / fs.RUTA_LOANS).parent.mkdir(parents=True)
    _loans().to_parquet(raiz / fs.RUTA_LOANS)


def _huellas(carpetas: list[Path]) -> dict[str, str]:
    """SHA-256 de cada archivo de las carpetas (recursivo)."""
    return {r.as_posix(): hashlib.sha256(r.read_bytes()).hexdigest()
            for c in carpetas if c.exists() for r in sorted(c.rglob("*")) if r.is_file()}


def _datos_en_memoria() -> dict:
    """`datos` para generar() sin leer archivos."""
    return {"scorecard_logistica": _scorecard("logistica"), "scorecard_mlp": _scorecard("mlp"), "shap": _shap(),
            "loans": _loans()}


# ---------------------------------------------------------------- 1. transformación PD → puntaje

def test_1_transformacion_pasa_por_el_ancla_y_duplicar_odds_suma_pdo():
    """Cada curva vale base en p = 1/(1+odds_base) y offset en p = ½; duplicar las odds suma su PDO."""
    d = fs.datos_transformacion(_scorecard("logistica"))
    assert [c["pdo"] for c in d["curvas"]] == [20.0, fs.PDO_ALTERNATIVO]  # la del JSON y la alternativa
    assert d["ancla"]["pd"] == pytest.approx(1 / 51) and d["ancla"]["score"] == 600.0
    for c in d["curvas"]:
        assert score_desde_pd(1 / 51, c["params"]) == pytest.approx(600.0)  # ancla
        assert score_desde_pd(0.5, c["params"]) == pytest.approx(c["params"]["offset"])  # p = ½ → offset
        o = 9.0  # odds buenas: p = 1/(1+o); con 2o, p' = 1/(1+2o)
        dif = score_desde_pd(1 / (1 + 2 * o), c["params"]) - score_desde_pd(1 / (1 + o), c["params"])
        assert dif == pytest.approx(c["pdo"])  # PDO
    assert d["curvas"][0]["params"]["offset"] == pytest.approx(487.1228762045055)  # 600 − 20·log₂ 50


def test_1b_pdo_40_duplica_la_pendiente_en_ln_odds():
    """Con PDO 40 la pendiente del puntaje frente a ln(odds) es el doble que con PDO 20; ancla igual."""
    d = fs.datos_transformacion(_scorecard("logistica"))
    pend = []  # pendiente por curva
    for c in d["curvas"]:
        ln_o = np.log((1 - d["pd"]) / d["pd"])  # ln odds buenas
        pend.append(np.polyfit(ln_o, c["score"], 1)[0])  # recta exacta: pendiente = factor
    assert pend[1] == pytest.approx(2 * pend[0])
    assert pend[0] == pytest.approx(20 / math.log(2))
    assert np.all((d["pd"] > 0) & (d["pd"] < 1)) and np.all(np.diff(d["pd"]) > 0)  # rejilla válida y creciente


# ---------------------------------------------------------------- 2. puntos por variable

def test_2_curva_logistica_reproduce_la_tabla_sintetica():
    """En los valores de la tabla la curva da los puntos de la tabla; fuera del recorte queda plana."""
    e = _entrada_log("annual_inc")
    np.testing.assert_allclose(fs.curva_puntos_logistica(e, [t["valor"] for t in TABLA]), [10.0, 0.0, -10.0])
    np.testing.assert_allclose(fs.curva_puntos_logistica(e, [12.0, -3.0]), [-10.0, 10.0])  # recortados


@pytest.mark.skipif(not HAY_REALES, reason="faltan los resultados reales de C3/C4 o loans_pd.parquet")
def test_2b_curva_logistica_reproduce_la_tabla_real():
    """Con scorecard.json real: la curva en cada `valor` de la tabla da su `puntos` (todas las numéricas)."""
    sc = json.loads((SC_REAL / "scorecard.json").read_text(encoding="utf-8"))
    nums = [e for e in sc["puntos_por_variable"] if e["tipo"] == "num"]
    assert nums
    for e in nums:
        got = fs.curva_puntos_logistica(e, [t["valor"] for t in e["tabla"]])
        np.testing.assert_allclose(got, [t["puntos"] for t in e["tabla"]], rtol=0, atol=1e-9, err_msg=e["nombre"])


def test_2c_tramos_del_mlp_solo_interiores_con_n_positivo():
    """Las colas abiertas y los tramos con n = 0 no se dibujan."""
    t = fs.tramos_interiores(_entrada_mlp("annual_inc")["tramos"])
    assert [(x["desde"], x["hasta"], x["puntos_medios"]) for x in t] == [(0.0, 5.0, 5.0), (5.0, 10.0, -6.0)]


def test_2d_datos_puntos_variable_rejilla_cubre_la_tabla_y_un_poco_mas():
    """Una entrada por VARIABLES_PUNTOS; la rejilla empieza en el mínimo de la tabla y pasa del máximo."""
    d = fs.datos_puntos_variable(_scorecard("logistica"), _scorecard("mlp"))
    assert [v["nombre"] for v in d] == list(fs.VARIABLES_PUNTOS)
    for v in d:
        assert v["x"][0] == 0.0 and v["x"][-1] > 10.0  # p1 … más allá del p99
        np.testing.assert_allclose(v["puntos_logistica"], fs.curva_puntos_logistica(_entrada_log(v["nombre"]), v["x"]))
        assert len(v["tramos_mlp"]) == 2 and v["etiqueta"] == f"Etiqueta de {v['nombre']}"


def test_2e_variable_ausente_da_error():
    """Si una de VARIABLES_PUNTOS no está en un JSON → ValueError que la nombra."""
    sc = _scorecard("mlp")
    sc["puntos_por_variable"] = [e for e in sc["puntos_por_variable"] if e["nombre"] != "revol_util"]
    with pytest.raises(ValueError, match="revol_util"):
        fs.datos_puntos_variable(_scorecard("logistica"), sc)


# ---------------------------------------------------------------- 3. importancia

def test_3_importancia_ordena_por_puesto_y_alinea_por_nombre():
    """Orden por puesto SHAP; la caída de AUC de cada modelo corresponde a la variable por nombre."""
    d = fs.datos_importancia(_shap())
    esperado = sorted(MEDIA_ABS, key=lambda n: -MEDIA_ABS[n])
    assert d["nombres"] == esperado
    assert d["media_abs_phi"] == [MEDIA_ABS[n] for n in esperado]
    for m, f in (("logistica", 0.01), ("mlp", 0.02)):
        assert d["caida"][m]["media"] == pytest.approx([f * MEDIA_ABS[n] for n in esperado])
        assert d["caida"][m]["desv"] == pytest.approx([f / 100] * len(esperado))


def test_3b_importancia_con_variables_distintas_da_error():
    """Si la permutación no tiene las mismas variables que la importancia → ValueError."""
    s = _shap()
    s["permutacion"]["mlp"]["variables"].pop()
    with pytest.raises(ValueError, match="mlp"):
        fs.datos_importancia(s)


# ---------------------------------------------------------------- 4. dependencia SHAP

def test_4_dependencia_top4_numericas_union_por_id_y_nan_fuera():
    """Top-4 numéricas por media |φ| (term no entra); valores unidos por id; NaN excluido."""
    d = fs.datos_dependencia(_shap(), _loans())
    assert [v["nombre"] for v in d] == ["dti", "annual_inc", "open_acc", "revol_util"]
    j = {n: i for i, n in enumerate(JUGADORES)}
    for v in d:
        i = NUMERICAS.index(v["nombre"])
        esperados = [(1000.0 * (i + 1) + float(k), 100 * kk + j[v["nombre"]]) for kk, k in enumerate(IDS)]
        if v["nombre"] == "dti":  # id 20 tiene dti NaN
            esperados = [e for e, k in zip(esperados, IDS) if k != "20"]
        assert list(zip(v["valores"], v["phi"])) == esperados
        assert v["spearman"] == _shap()["direccion"][v["nombre"]]["spearman"]


def test_4b_dependencia_id_ausente_da_error():
    """Un id explicado que no está en loans_pd → ValueError claro."""
    loans = _loans()
    loans = loans[loans["id"] != "20"]
    with pytest.raises(ValueError, match="20"):
        fs.datos_dependencia(_shap(), loans)


def test_4d_dependencia_phi_con_filas_distintas_a_ids_da_error():
    """phi con menos filas que explicados.ids → ValueError que nombra shap_mlp.json."""
    s = _shap()
    s["explicados"]["phi"].pop()
    with pytest.raises(ValueError, match=r"shap_mlp\.json.*phi"):
        fs.datos_dependencia(s, _loans())


def test_4e_dependencia_variable_fuera_de_jugadores_da_error():
    """Una numérica de importancia_global que no está en metodo.jugadores → ValueError que la nombra."""
    s = _shap()
    s["metodo"]["jugadores"] = [n if n != "dti" else "otra" for n in JUGADORES]
    with pytest.raises(ValueError, match=r"shap_mlp\.json.*dti"):
        fs.datos_dependencia(s, _loans())


def test_4c_dependencia_ids_enteros_en_el_parquet_se_unen_como_texto():
    """El id se compara como str aunque el parquet lo traiga como entero."""
    loans = _loans()
    loans["id"] = loans["id"].astype(int)
    assert len(fs.datos_dependencia(_shap(), loans)) == fs.N_DEPENDENCIA


# ---------------------------------------------------------------- 5. bandas

def test_5_bandas_tal_cual_de_los_json():
    """Histogramas y bandas de test copiados de distribucion_scores de cada modelo."""
    sl, sm = _scorecard("logistica"), _scorecard("mlp")
    d = fs.datos_bandas(sl, sm)
    for m, sc in (("logistica", sl), ("mlp", sm)):
        t = sc["distribucion_scores"]["test"]
        assert d[m]["bordes"] == t["histograma"]["bordes"] and d[m]["conteos"] == t["histograma"]["conteos"]
        assert d[m]["banda"] == [b["banda"] for b in t["bandas"]]
        assert d[m]["tasa_malos"] == [b["tasa_malos"] for b in t["bandas"]]
        assert d[m]["pd_media"] == [b["pd_media"] for b in t["bandas"]]


# ---------------------------------------------------------------- 6. generar

def test_6_generar_escribe_exactamente_archivos_y_no_toca_lo_versionado(tmp_path):
    """Escribe ARCHIVOS (no vacíos) solo en destino; blog, results y configs reales intactos."""
    raiz = tmp_path / "repo"
    _escribir_insumos(raiz)
    destino = tmp_path / "figuras"
    antes = _huellas(CARPETAS_PROTEGIDAS)
    antes_tmp = {r for r in tmp_path.rglob("*") if r.is_file()}
    rutas = fs.generar(raiz=raiz, destino=destino)
    assert [r.name for r in rutas] == list(fs.ARCHIVOS)  # en el orden de ARCHIVOS
    assert {r for r in tmp_path.rglob("*") if r.is_file()} - antes_tmp == set(rutas)
    assert all(r.parent == destino and r.stat().st_size > 0 for r in rutas)
    assert _huellas(CARPETAS_PROTEGIDAS) == antes


def test_6b_generar_resuelve_salida_al_llamarse(tmp_path, monkeypatch):
    """Sin destino, usa raiz / SALIDA leído al llamar."""
    raiz = tmp_path / "repo"
    _escribir_insumos(raiz)
    monkeypatch.setattr(fs, "SALIDA", Path("otra/carpeta"))
    rutas = fs.generar(raiz=raiz)
    assert {r.parent for r in rutas} == {raiz / "otra" / "carpeta"}


def test_6c_generar_con_datos_en_memoria(tmp_path):
    """datos= en memoria: mismas 5 figuras sin leer archivos."""
    rutas = fs.generar(raiz=tmp_path / "no-existe", destino=tmp_path / "f", datos=_datos_en_memoria())
    assert [r.name for r in rutas] == list(fs.ARCHIVOS)


def test_6d_generar_con_claves_de_datos_erroneas_da_error(tmp_path):
    """datos= con una clave de más o de menos → ValueError y nada escrito."""
    datos = _datos_en_memoria()
    datos["shap_"] = datos.pop("shap")
    with pytest.raises(ValueError, match="shap"):
        fs.generar(destino=tmp_path / "f", datos=datos)
    assert not (tmp_path / "f").exists()


@pytest.mark.parametrize("insumo,comando", [
    ("scorecard.json", r"C3: python run\.py scorecard"),
    ("scorecard_mlp.json", r"C4: python run\.py scorecard --config configs/scorecard/mlp\.yaml"),
    ("shap_mlp.json", r"C4: python run\.py scorecard --config configs/scorecard/mlp\.yaml"),
    ("loans_pd.parquet", r"python scripts/preparar_datos\.py")])
def test_7_insumo_faltante_nombra_el_comando(tmp_path, insumo, comando):
    """Cada insumo que falta → FileNotFoundError con el archivo y el comando; sin figuras a medias."""
    _escribir_insumos(tmp_path)
    ruta = tmp_path / (fs.RUTA_LOANS if insumo == "loans_pd.parquet" else fs.ENTRADA / insumo)
    ruta.unlink()
    with pytest.raises(FileNotFoundError, match=f"falta .*{insumo}.*{comando}"):
        fs.generar(raiz=tmp_path, destino=tmp_path / "figs")
    assert not (tmp_path / "figs").exists()


@pytest.mark.skipif(not HAY_REALES, reason="faltan los resultados reales de C3/C4 o loans_pd.parquet")
def test_8_generar_con_los_resultados_reales(tmp_path):
    """Con results/ y loans_pd reales: las 5 figuras en tmp_path."""
    rutas = fs.generar(raiz=RAIZ, destino=tmp_path)
    assert [r.name for r in rutas] == list(fs.ARCHIVOS)
    assert all(r.stat().st_size > 0 for r in rutas)
