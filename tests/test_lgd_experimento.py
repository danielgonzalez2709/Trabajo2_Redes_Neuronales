"""Pruebas de D4: búsqueda Optuna, semillas, ``summary.json`` y predicciones de la LGD (Parte D §3.3–§3.4;
núcleo §5.4, §6).

Datos sintéticos pequeños escritos en ``tmp_path`` con las mismas rutas relativas que el repo (no se tocan los datos ni
los resultados reales).
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.common import cli
from src.common.config import apply_overrides, load_config
from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json, save_json
from src.lgd import experimento as exp
from src.lgd.metricas import mae, rmse
from src.lgd.mlp import MLPLGD

RAIZ = Path(__file__).resolve().parents[1]
FEATURES = [
    {"nombre": "loan_amnt", "tipo": "num", "transformacion": "log1p"},
    {"nombre": "dti", "tipo": "num"},
    {"nombre": "term", "tipo": "cat"},
]
N_POR_SPLIT = {"train": 120, "val": 80, "test": 70}
N_PD_TEST = 150
# Overrides para que el MLP entrene en décimas de segundo (no son valores del proyecto).
RAPIDO = ["mlp.capas=[8]", "mlp.max_epocas=4", "mlp.paciencia=2", "mlp.batch=64", "mlp.device=cpu",
          "datos.muestra=null"]


# ---------------------------------------------------------------------------------------------- datos sintéticos
def _features_df(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n),
        "dti": rng.uniform(0, 35, n),
        "term": rng.choice([" 36 months", " 60 months"], n),
    })


def _escribir_datos(raiz: Path) -> None:
    """``loans_lgd.parquet``, ``preprocesador_lgd.json`` (ajustado con train) y ``loans_pd.parquet`` en ``raiz``."""
    rng = np.random.default_rng(123)
    partes = []
    for split, n in N_POR_SPLIT.items():
        df = _features_df(rng, n)
        df.insert(0, "split", split)
        df.insert(0, "id", [f"{split}-{i}" for i in range(n)])
        partes.append(df)
    lgd = pd.concat(partes, ignore_index=True)
    z = 0.8 * (lgd["dti"] - 17) / 10 + 0.5 * (lgd["term"] == " 60 months") + rng.normal(0, 0.5, len(lgd))
    lgd["lgd"] = np.clip(1 / (1 + np.exp(-z)), 0, 1)
    lgd.loc[lgd.index % 10 == 0, "lgd"] = 1.0   # la LGD real tiene masa en 1 y en 0
    lgd.loc[lgd.index % 10 == 5, "lgd"] = 0.0
    lgd["ead"] = 1000.0
    (raiz / "data/processed").mkdir(parents=True)
    lgd.to_parquet(raiz / "data/processed/loans_lgd.parquet", index=False)
    pre = Preprocesador((1.0, 99.0)).fit(lgd[lgd["split"] == "train"], FEATURES)
    save_json(raiz / "models/preprocesador_lgd.json", pre.to_dict())

    pdf = _features_df(rng, N_PD_TEST + 40)
    pdf.insert(0, "split", ["test"] * N_PD_TEST + ["train"] * 40)
    pdf.insert(0, "id", [f"pd-{i}" for i in range(len(pdf))])
    pdf["target"] = rng.integers(0, 2, len(pdf))
    pdf.to_parquet(raiz / "data/processed/loans_pd.parquet", index=False)


@pytest.fixture
def raiz(tmp_path: Path) -> Path:
    _escribir_datos(tmp_path)
    return tmp_path


SIN_BUSQUEDA = {"tipo": "sin_busqueda", "origen": "prueba", "ensayo": None, "n_ensayos": None, "tiempo_s": None,
                "hash_particion": None, "hash_preprocesador": None, "muestra": None, "fecha": None}


def _cfg(nombre: str = "base.yaml", extra: list[str] | None = None, carpeta: Path = RAIZ / "configs/lgd",
         rapido: list[str] = RAPIDO) -> dict:
    """Config del repo (o de ``carpeta``) con overrides rápidos. Con la del repo, la procedencia se fija en "sin
    búsqueda" para no depender de lo que tenga hoy ``configs/lgd/mlp_final.yaml``."""
    cfg = load_config(carpeta / nombre)
    if carpeta == RAIZ / "configs/lgd":
        cfg["busqueda"] = dict(SIN_BUSQUEDA)
    return apply_overrides(cfg, rapido + (extra or []))


def _copiar_configs(raiz: Path) -> Path:
    """Copia configs/lgd/ a ``raiz`` (con mlp_final "sin búsqueda") para que lgd-hpo lo reescriba ahí."""
    destino = raiz / "configs/lgd"
    destino.mkdir(parents=True)
    for nombre in ("base.yaml", "hpo.yaml", "demo.yaml"):
        shutil.copy(RAIZ / "configs/lgd" / nombre, destino / nombre)
    exp.escribir_mlp_final(destino / "mlp_final.yaml", load_config(RAIZ / "configs/lgd/demo.yaml")["mlp"],
                           dict(SIN_BUSQUEDA))
    return destino


def _sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------- métricas
def test_mae_rmse_a_mano_y_contra_sklearn():
    y = np.array([0.0, 0.5, 1.0, 0.25])
    p = np.array([0.1, 0.5, 0.7, 0.25])
    # |e| = 0.1, 0, 0.3, 0 → MAE = 0.4/4 = 0.1; e² = 0.01, 0, 0.09, 0 → RMSE = sqrt(0.1/4) = sqrt(0.025)
    assert mae(y, p) == pytest.approx(0.1)
    assert rmse(y, p) == pytest.approx(math.sqrt(0.025))
    rng = np.random.default_rng(0)
    y2, p2 = rng.uniform(size=50), rng.uniform(size=50)
    assert mae(y2, p2) == pytest.approx(mean_absolute_error(y2, p2))
    assert rmse(y2, p2) == pytest.approx(math.sqrt(mean_squared_error(y2, p2)))


@pytest.mark.parametrize("y, p", [([0.1, 0.2], [0.1]), ([0.1, np.nan], [0.1, 0.2]), ([0.1, 0.2], [0.1, np.inf]),
                                  ([[0.1, 0.2]], [[0.1, 0.2]]), ([], [])])
def test_metricas_validan_forma_y_finitud(y, p):
    with pytest.raises(ValueError):
        mae(y, p)
    with pytest.raises(ValueError):
        rmse(y, p)


# ---------------------------------------------------------------------------------------------- configs
def test_configs_presupuesto_y_espacio():
    base = load_config(RAIZ / "configs/lgd/base.yaml")
    hpo = load_config(RAIZ / "configs/lgd/hpo.yaml")
    demo = load_config(RAIZ / "configs/lgd/demo.yaml")
    assert base["semillas"]["n"] == 30 and hpo["hpo"]["n_ensayos"] == 30 and demo["hpo"]["n_ensayos"] == 5
    e = base["hpo"]["espacio"]
    assert e["n_capas"] == [1, 4]
    assert e["neuronas"] == [16, 32, 64, 128, 256]
    assert e["activacion"] == ["relu", "leaky_relu", "tanh"]
    assert e["dropout"] == [0.0, 0.5]
    assert e["l2"] == [1.0e-6, 1.0e-2]
    assert e["optimizador"] == ["adam", "sgd"]
    assert e["lr"] == [1.0e-4, 1.0e-2]
    assert e["batch"] == [256, 512, 1024, 2048]
    # el bloque mlp llega por herencia desde mlp_final.yaml, con su procedencia
    assert base["mlp"]["perdida"] == "bce_fraccional" and "salida" in base["datos"]
    assert base["mlp"] == load_config(RAIZ / "configs/lgd/mlp_final.yaml")["mlp"]
    assert base["busqueda"]["tipo"] in ("sin_busqueda", "optuna")
    assert base["hpo"]["destino"] == "configs/lgd/mlp_final.yaml"
    assert "demo" in demo["hpo"]["destino"] and demo["busqueda"]["tipo"] == "sin_busqueda"


def test_demo_escribe_en_otra_carpeta_y_otros_modelos():
    base = load_config(RAIZ / "configs/lgd/base.yaml")
    demo = load_config(RAIZ / "configs/lgd/demo.yaml")
    assert demo["salida"] != base["salida"]
    assert "demo" in demo["salida"] and not Path(demo["salida"]).is_relative_to(Path(base["salida"]))
    for m in ("lineal", "mlp"):
        assert demo["modelos"][m] != base["modelos"][m]
    assert base["modelos"] == {"lineal": "models/lgd_lineal.joblib", "mlp": "models/lgd_mlp.pt"}
    assert demo["datos"]["muestra"] is not None and demo["semillas"]["n"] < base["semillas"]["n"]


def test_cli_conecta_lgd_y_lgd_hpo():
    assert cli.SUBCOMANDOS["lgd"].manejador is cli._lgd
    assert cli.SUBCOMANDOS["lgd-hpo"].manejador is cli._lgd_hpo


# ---------------------------------------------------------------------------------------------- entrenamiento
def test_entrenar_semillas_escribe_contratos(raiz: Path):
    cfg = _cfg(extra=["semillas.n=3"])
    resumen = exp.entrenar_semillas(cfg, raiz)
    salida = raiz / cfg["salida"]

    # semillas/: lineal 1 vez + MLP n veces, con base_seed + k
    semillas = sorted(p.name for p in (salida / "semillas").glob("*.json"))
    b = cfg["base_seed"]
    assert semillas == sorted([f"lineal_seed{b}.json"] + [f"mlp_seed{b + k}.json" for k in range(3)])
    for nombre in semillas:
        d = load_json(salida / "semillas" / nombre)
        for clave in ("modelo", "seed", "config", "metricas", "n_parametros", "tiempo_s", "device", "epocas",
                      "evidencia"):
            assert clave in d, (nombre, clave)
        assert set(exp.METRICAS) <= set(d["metricas"])

    s = load_json(salida / "summary.json")
    assert s["modelos"].keys() == {"lineal", "mlp"}
    for m, n in (("lineal", 1), ("mlp", 3)):
        r = s["modelos"][m]
        assert r["n_parametros"] > 0 and r["tiempo_total_s"] > 0
        assert r["n_configuraciones_busqueda"] == 1  # sin búsqueda previa
        for met in exp.METRICAS:
            assert set(r["metricas"][met]) == {"media", "desv", "mejor", "peor", "n"}
            assert r["metricas"][met]["n"] == n
            assert r["metricas"][met]["mejor"] <= r["metricas"][met]["peor"]
    assert set(s["modelos"]["lineal"]["fuera_de_rango"]) == {"val", "test", "pd_test"}
    assert s["modelos"]["lineal"]["fuera_de_rango"]["pd_test"]["n"] == N_PD_TEST
    assert resumen == s

    pt = pd.read_parquet(salida / "pred_test.parquet")
    assert list(pt.columns) == ["id", "y", "pred_lineal", "pred_mlp"] and len(pt) == N_POR_SPLIT["test"]
    assert pt["id"].map(type).eq(str).all() and pt["id"].str.startswith("test-").all()
    assert pt["pred_mlp"].between(0, 1).all()
    ppd = pd.read_parquet(salida / "pred_pd_test.parquet")
    assert list(ppd.columns) == ["id", "pred_lineal", "pred_mlp"] and len(ppd) == N_PD_TEST
    assert ppd["id"].str.startswith("pd-").all()

    # semilla mediana leída de semillas/*.json; el modelo guardado es esa semilla y reproduce pred_test
    corridas = sorted((load_json(p)["metricas"]["mae_test"], load_json(p)["seed"])
                      for p in (salida / "semillas").glob("mlp_seed*.json"))
    esperada = corridas[(len(corridas) - 1) // 2][1]
    assert s["modelos"]["mlp"]["semilla_mediana"] == esperada
    mlp = MLPLGD.cargar(raiz / cfg["modelos"]["mlp"])
    assert mlp.seed_ == esperada
    datos = exp.cargar_datos(cfg, raiz)
    np.testing.assert_allclose(mlp.predict(datos["X"]["test"]), pt["pred_mlp"].to_numpy(), rtol=0, atol=1e-7)
    assert (raiz / cfg["modelos"]["lineal"]).is_file()

    # evidencia (núcleo §5.4, §6) recalculada aquí: líneas "{id},{split}\n" en orden y bytes del preprocesador
    df = pd.read_parquet(raiz / "data/processed/loans_lgd.parquet")
    texto = "".join(f"{i},{s_}\n" for i, s_ in zip(df["id"], df["split"]))
    esperada = {"sha256_particion": hashlib.sha256(texto.encode("utf-8")).hexdigest(),
                "sha256_preprocesador": _sha(raiz / "models/preprocesador_lgd.json")}
    for m in ("lineal", "mlp"):
        assert s["modelos"][m]["evidencia"] == esperada
    for p in (salida / "semillas").glob("*.json"):
        assert load_json(p)["evidencia"] == esperada
    assert not any(k.startswith("hash_") for k in s), "las huellas viejas no deben quedar en el summary"


def test_misma_semilla_mismas_metricas(tmp_path: Path):
    resultados = []
    for i in range(2):
        r = tmp_path / str(i)
        r.mkdir()
        _escribir_datos(r)
        resultados.append(exp.entrenar_semillas(_cfg(extra=["semillas.n=2"]), r))
    assert resultados[0]["modelos"]["mlp"]["metricas"] == resultados[1]["modelos"]["mlp"]["metricas"]
    assert resultados[0]["modelos"]["lineal"]["metricas"] == resultados[1]["modelos"]["lineal"]["metricas"]


def test_no_reajusta_el_preprocesador(raiz: Path, monkeypatch):
    antes = _sha(raiz / "models/preprocesador_lgd.json")

    def prohibido(*a, **k):
        raise AssertionError("D4 no debe reajustar el preprocesador")

    monkeypatch.setattr(Preprocesador, "fit", prohibido)
    exp.entrenar_semillas(_cfg(extra=["semillas.n=1"]), raiz)
    assert _sha(raiz / "models/preprocesador_lgd.json") == antes


def test_muestra_estratificada_por_split(raiz: Path):
    cfg = _cfg(extra=["datos.muestra=100"])
    datos = exp.cargar_datos(cfg, raiz)
    n = {s: len(datos["y"][s]) for s in ("train", "val", "test")}
    assert sum(n.values()) == 100 and all(v > 0 for v in n.values())
    total = sum(N_POR_SPLIT.values())
    for s in n:
        assert abs(n[s] - 100 * N_POR_SPLIT[s] / total) <= 1
    assert exp.cargar_datos(cfg, raiz)["ids"]["test"].tolist() == datos["ids"]["test"].tolist()


# ---------------------------------------------------------------------------------------------- evidencia y semillas
def _es_hex64(valor) -> bool:
    return isinstance(valor, str) and len(valor) == 64 and all(c in "0123456789abcdef" for c in valor)


def test_evidencia_igual_en_semillas_summary_y_hpo(raiz: Path):
    """D9: ``evidencia`` = {sha256_particion, sha256_preprocesador} (núcleo §5.4) en semillas/, en cada modelo del
    summary y en hpo_ensayos.json; 64 dígitos hex y el mismo valor en las tres salidas."""
    carpeta = _copiar_configs(raiz)
    h = exp.buscar(_cfg("hpo.yaml", ["hpo.n_ensayos=1"], carpeta), raiz)
    cfg = _cfg(extra=["semillas.n=2"])
    s = exp.entrenar_semillas(cfg, raiz)
    salida = raiz / cfg["salida"]
    evidencias = [h["evidencia"], load_json(salida / "hpo_ensayos.json")["evidencia"]]
    evidencias += [s["modelos"][m]["evidencia"] for m in ("lineal", "mlp")]
    evidencias += [load_json(p)["evidencia"] for p in (salida / "semillas").glob("*.json")]
    assert len(evidencias) == 2 + 2 + 3
    for ev in evidencias:
        assert set(ev) == {"sha256_particion", "sha256_preprocesador"}
        assert _es_hex64(ev["sha256_particion"]) and _es_hex64(ev["sha256_preprocesador"])
        assert ev == evidencias[0]
    assert evidencias[0]["sha256_preprocesador"] == _sha(raiz / "models/preprocesador_lgd.json")


def test_evidencia_con_muestra_es_la_del_df_muestreado(raiz: Path):
    """La huella de la partición describe el mismo DataFrame del que salen las matrices (tras ``datos.muestra``)."""
    from src.pd.experimento import huella_particion
    cfg = _cfg(extra=["datos.muestra=100"])
    d = exp.cargar_datos(cfg, raiz)
    df = exp._muestrear(pd.read_parquet(raiz / "data/processed/loans_lgd.parquet"), 100, cfg["base_seed"])
    assert d["evidencia"]["sha256_particion"] == huella_particion(df["id"], df["split"])
    completo = exp.cargar_datos(_cfg(), raiz)
    assert completo["evidencia"]["sha256_particion"] != d["evidencia"]["sha256_particion"]


def test_semilla_mediana_a_mano():
    c = [{"seed": 10, "mae_test": 0.3}, {"seed": 11, "mae_test": 0.1}, {"seed": 12, "mae_test": 0.2}]
    assert exp._semilla_mediana(c) == 12
    # n par → mediana inferior: orden 0.1 (11), 0.2 (12), 0.3 (10), 0.4 (13) → posición 1 → 12
    assert exp._semilla_mediana(c + [{"seed": 13, "mae_test": 0.4}]) == 12
    # empate → la semilla menor
    assert exp._semilla_mediana([{"seed": 5, "mae_test": 0.2}, {"seed": 3, "mae_test": 0.2},
                                 {"seed": 4, "mae_test": 0.2}]) == 4
    assert exp._semilla_mediana([{"seed": 9, "mae_test": 0.2}, {"seed": 3, "mae_test": 0.2}]) == 3


def test_base_seed_mas_k(raiz: Path):
    cfg = _cfg(extra=["base_seed=7", "semillas.n=3"])
    exp.entrenar_semillas(cfg, raiz)
    nombres = sorted(p.name for p in (raiz / cfg["salida"] / "semillas").glob("*.json"))
    assert nombres == ["lineal_seed7.json", "mlp_seed7.json", "mlp_seed8.json", "mlp_seed9.json"]
    assert [load_json(raiz / cfg["salida"] / "semillas" / f"mlp_seed{k}.json")["seed"] for k in (7, 8, 9)] == [7, 8, 9]


def test_muestra_que_vacia_un_split_es_error():
    df = pd.DataFrame({"id": [str(i) for i in range(104)], "split": ["train"] * 100 + ["val"] * 2 + ["test"] * 2})
    with pytest.raises(ValueError, match="datos.muestra.*vac"):
        exp._muestrear(df, 5, 0)


# ---------------------------------------------------------------------------------------------- búsqueda
def test_buscar_reescribe_mlp_final_y_lgd_lo_hereda(raiz: Path):
    carpeta = _copiar_configs(raiz)
    cfg = _cfg("hpo.yaml", ["hpo.n_ensayos=3", "semillas.n=1"], carpeta)
    h = exp.buscar(cfg, raiz)
    salida = raiz / cfg["salida"]
    assert load_json(salida / "hpo_ensayos.json") == json.loads(json.dumps(h))
    assert len(h["ensayos"]) == 3 and h["n_ensayos"] == 3
    for e in h["ensayos"]:
        assert {"numero", "parametros", "valor_val", "mae_val", "tiempo_s", "estado"} <= set(e)
    assert set(h["evidencia"]) == {"sha256_particion", "sha256_preprocesador"}
    assert h["semilla_muestreador"] == cfg["base_seed"]
    assert h["tiempo_total_s"] == pytest.approx(sum(e["tiempo_s"] for e in h["ensayos"]))

    texto = (carpeta / "mlp_final.yaml").read_text(encoding="utf-8")
    assert "lgd-hpo" in texto.splitlines()[0] and "No se edita a mano" in texto
    final = load_config(carpeta / "mlp_final.yaml")
    completos = [e for e in h["ensayos"] if e["estado"] == "COMPLETE"]
    mejor = min(completos, key=lambda e: e["valor_val"])
    b = final["busqueda"]
    assert b["tipo"] == "optuna" and b["ensayo"] == mejor["numero"] and b["n_ensayos"] == 3
    assert b["tiempo_s"] == pytest.approx(h["tiempo_total_s"])
    assert b["hash_particion"] == h["evidencia"]["sha256_particion"] and b["muestra"] is None
    assert b["hash_preprocesador"] == h["evidencia"]["sha256_preprocesador"]
    for clave, valor in mejor["parametros"].items():
        assert final["mlp"][clave] == valor

    # run.py lgd con base.yaml (que hereda el mlp_final reescrito), sin ningún --set mlp.*
    cfg_lgd = _cfg("base.yaml", ["semillas.n=1"], carpeta, rapido=["datos.muestra=null"])
    assert cfg_lgd["mlp"] == final["mlp"]
    s = exp.entrenar_semillas(cfg_lgd, raiz)
    assert s["modelos"]["mlp"]["n_configuraciones_busqueda"] == 3
    assert s["modelos"]["mlp"]["tiempo_busqueda_s"] == pytest.approx(h["tiempo_total_s"])
    # Presupuesto de cómputo total (núcleo §6): búsqueda + semillas.
    mlp = s["modelos"]["mlp"]
    assert mlp["tiempo_busqueda_s"] > 0
    assert mlp["tiempo_total_s"] == pytest.approx(mlp["tiempo_busqueda_s"] + mlp["tiempo_semillas_s"])
    semilla = load_json(salida / "semillas" / f"mlp_seed{cfg['base_seed']}.json")
    for clave, valor in mejor["parametros"].items():
        assert semilla["config"]["mlp"][clave] == valor


def test_set_mlp_gana_sobre_la_busqueda(raiz: Path):
    carpeta = _copiar_configs(raiz)
    exp.buscar(_cfg("hpo.yaml", ["hpo.n_ensayos=2"], carpeta), raiz)
    final = load_config(carpeta / "mlp_final.yaml")
    assert final["mlp"]["capas"] != [64, 32]  # si la búsqueda diera [64, 32] la prueba no diría nada
    cfg = _cfg("base.yaml", ["semillas.n=1", "mlp.capas=[64, 32]", "mlp.dropout=0.3"], carpeta)
    s = exp.entrenar_semillas(cfg, raiz)
    semilla = load_json(raiz / cfg["salida"] / "semillas" / f"mlp_seed{cfg['base_seed']}.json")
    assert semilla["config"]["mlp"]["capas"] == [64, 32] and semilla["config"]["mlp"]["dropout"] == 0.3
    n_entradas = MLPLGD.cargar(raiz / cfg["modelos"]["mlp"]).arquitectura_["n_entradas"]
    esperado = (n_entradas * 64 + 64) + (64 * 32 + 32) + (32 + 1)
    assert semilla["n_parametros"] == esperado == s["modelos"]["mlp"]["n_parametros"]


@pytest.mark.parametrize("clave, valor", [("hash_particion", "otro"), ("hash_preprocesador", "otro"),
                                          ("muestra", 100)])
def test_procedencia_de_otros_datos_es_error(raiz: Path, clave, valor):
    carpeta = _copiar_configs(raiz)
    exp.buscar(_cfg("hpo.yaml", ["hpo.n_ensayos=1"], carpeta), raiz)
    cfg = _cfg("base.yaml", ["semillas.n=1", f"busqueda.{clave}={valor}"], carpeta)
    with pytest.raises(ValueError, match=clave):
        exp.entrenar_semillas(cfg, raiz)


def test_busqueda_con_muestra_y_final_sin_muestra_es_error(raiz: Path):
    carpeta = _copiar_configs(raiz)
    exp.buscar(_cfg("hpo.yaml", ["hpo.n_ensayos=1", "datos.muestra=200"], carpeta), raiz)
    with pytest.raises(ValueError, match="muestra"):
        exp.entrenar_semillas(_cfg("base.yaml", ["semillas.n=1"], carpeta), raiz)


def test_ensayo_fallido_queda_fail(raiz: Path, monkeypatch):
    carpeta = _copiar_configs(raiz)
    original = MLPLGD.fit
    llamadas = {"n": 0}

    def fit_que_falla_una_vez(self, *a, **k):
        llamadas["n"] += 1
        if llamadas["n"] == 2:
            raise RuntimeError("divergió (simulado)")
        return original(self, *a, **k)

    monkeypatch.setattr(MLPLGD, "fit", fit_que_falla_una_vez)
    h = exp.buscar(_cfg("hpo.yaml", ["hpo.n_ensayos=3"], carpeta), raiz)
    estados = [e["estado"] for e in h["ensayos"]]
    assert estados == ["COMPLETE", "FAIL", "COMPLETE"]
    assert h["ensayos"][1]["valor_val"] is None
    assert h["mejor"]["numero"] in (0, 2)


def test_todos_los_ensayos_fallan_es_error(raiz: Path, monkeypatch):
    carpeta = _copiar_configs(raiz)

    def siempre_falla(self, *a, **k):
        raise RuntimeError("divergió (simulado)")

    monkeypatch.setattr(MLPLGD, "fit", siempre_falla)
    antes = (carpeta / "mlp_final.yaml").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="ningún ensayo"):
        exp.buscar(_cfg("hpo.yaml", ["hpo.n_ensayos=2"], carpeta), raiz)
    assert (carpeta / "mlp_final.yaml").read_text(encoding="utf-8") == antes


def test_buscar_es_reproducible(tmp_path: Path):
    params = []
    for i in range(2):
        r = tmp_path / str(i)
        r.mkdir()
        _escribir_datos(r)
        carpeta = _copiar_configs(r)
        cfg = _cfg("hpo.yaml", ["hpo.n_ensayos=2"], carpeta)
        params.append([e["parametros"] for e in exp.buscar(cfg, r)["ensayos"]])
    assert params[0] == params[1]


# ---------------------------------------------------------------------------------------------- CLI y demo
def test_cli_demo_con_set_semillas_no_toca_la_carpeta_real(raiz: Path, monkeypatch, capsys):
    monkeypatch.chdir(raiz)
    argv = ["lgd", "--config", str(RAIZ / "configs/lgd/demo.yaml"), "--set", "semillas.n=2"]
    for o in RAPIDO:
        argv += ["--set", o]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "Tiempo de ejecución" in out
    demo = load_config(RAIZ / "configs/lgd/demo.yaml")
    base = load_config(RAIZ / "configs/lgd/base.yaml")
    assert len(list((raiz / demo["salida"] / "semillas").glob("mlp_seed*.json"))) == 2
    assert not (raiz / base["salida"] / "summary.json").exists()
    assert not (raiz / base["salida"] / "semillas").exists()
    for m in ("lineal", "mlp"):
        assert (raiz / demo["modelos"][m]).is_file() and not (raiz / base["modelos"][m]).exists()


def test_cli_lgd_hpo(raiz: Path, monkeypatch, capsys):
    monkeypatch.chdir(raiz)
    argv = ["lgd-hpo", "--config", str(RAIZ / "configs/lgd/demo.yaml"), "--set", "hpo.n_ensayos=2"]
    for o in RAPIDO:
        argv += ["--set", o]
    assert cli.main(argv) == 0
    assert "Tiempo de ejecución" in capsys.readouterr().out
    demo = load_config(RAIZ / "configs/lgd/demo.yaml")
    assert len(load_json(raiz / demo["salida"] / "hpo_ensayos.json")["ensayos"]) == 2
    assert load_config(raiz / demo["hpo"]["destino"])["busqueda"]["tipo"] == "optuna"
    assert not (raiz / "configs/lgd/mlp_final.yaml").exists()


# ---------------------------------------------------------------------------------------------- referencias constantes
def test_referencias_constantes_a_mano():
    y_tr = np.array([0.0, 0.5, 1.0, 1.0])          # media 0.625, mediana 0.75
    y_val = np.array([0.0, 1.0])
    y_test = np.array([0.5, 0.75, 1.0])
    r = exp._referencias_constantes(y_tr, y_val, y_test)
    assert set(r) == {"constante_media", "constante_mediana"}
    m, md = r["constante_media"], r["constante_mediana"]
    assert m["valor"] == pytest.approx(0.625) and md["valor"] == pytest.approx(0.75)
    # media 0.625: val |e| = 0.625, 0.375 → MAE 0.5; e² = 0.390625, 0.140625 → RMSE sqrt(0.265625)
    assert m["metricas"]["mae_val"] == pytest.approx(0.5)
    assert m["metricas"]["rmse_val"] == pytest.approx(math.sqrt(0.265625))
    # test |e| = 0.125, 0.125, 0.375 → MAE 0.625/3; e² = 0.015625, 0.015625, 0.140625 → RMSE sqrt(0.171875/3)
    assert m["metricas"]["mae_test"] == pytest.approx(0.625 / 3)
    assert m["metricas"]["rmse_test"] == pytest.approx(math.sqrt(0.171875 / 3))
    # mediana 0.75: val |e| = 0.75, 0.25 → MAE 0.5, RMSE sqrt((0.5625 + 0.0625)/2) = sqrt(0.3125)
    assert md["metricas"]["mae_val"] == pytest.approx(0.5)
    assert md["metricas"]["rmse_val"] == pytest.approx(math.sqrt(0.3125))
    # test |e| = 0.25, 0, 0.25 → MAE 0.5/3; RMSE sqrt(0.125/3)
    assert md["metricas"]["mae_test"] == pytest.approx(0.5 / 3)
    assert md["metricas"]["rmse_test"] == pytest.approx(math.sqrt(0.125 / 3))
    for ref in r.values():
        assert ref["n_parametros"] == 1 and ref["n_configuraciones_busqueda"] == 1


def test_referencias_constantes_usan_solo_train():
    y_tr = np.array([0.0, 0.5, 1.0, 1.0])
    a = exp._referencias_constantes(y_tr, np.array([0.0, 1.0]), np.array([0.5]))
    b = exp._referencias_constantes(y_tr, np.array([0.9, 0.9, 0.9]), np.array([0.1, 0.2]))
    for k in ("constante_media", "constante_mediana"):
        assert a[k]["valor"] == b[k]["valor"]


def test_summary_trae_las_referencias(raiz: Path):
    cfg = _cfg(extra=["semillas.n=1"])
    s = exp.entrenar_semillas(cfg, raiz)
    disco = load_json(raiz / cfg["salida"] / "summary.json")
    assert set(disco["referencias"]) == {"constante_media", "constante_mediana"}
    assert set(disco["modelos"]) == {"lineal", "mlp"}
    y_tr = pd.read_parquet(raiz / "data/processed/loans_lgd.parquet").query("split == 'train'")["lgd"]
    assert disco["referencias"]["constante_mediana"]["valor"] == pytest.approx(float(y_tr.median()))
    assert disco["referencias"]["constante_media"]["valor"] == pytest.approx(float(y_tr.mean()))
    assert s["referencias"] == disco["referencias"]
    pt = pd.read_parquet(raiz / cfg["salida"] / "pred_test.parquet")
    assert list(pt.columns) == ["id", "y", "pred_lineal", "pred_mlp"]
