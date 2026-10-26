"""Notebooks (núcleo §5.8): parámetros con el cargador del CLI y ejecución de punta a punta."""

from pathlib import Path

import pytest

from src.common.notebooks import config_notebook, ejecutar, fuentes

RAIZ = Path(__file__).resolve().parents[1]


def test_config_notebook_aplica_cambios_como_set():
    cfg = config_notebook(RAIZ / "configs" / "datos.yaml",
                          {"particion.train_hasta": "2011-12-31", "preprocesamiento.percentiles_recorte": [5.0, 95.0]},
                          imprimir=False)
    assert cfg["particion"]["train_hasta"] == "2011-12-31"
    assert cfg["preprocesamiento"]["percentiles_recorte"] == [5.0, 95.0]


def test_config_notebook_clave_inexistente_es_error():
    with pytest.raises(ValueError):
        config_notebook(RAIZ / "configs" / "datos.yaml", {"particion.train_hsta": "2011-12-31"}, imprimir=False)


def test_cada_notebook_tiene_celda_de_parametros():
    for fuente in fuentes(RAIZ / "notebooks"):
        texto = fuente.read_text(encoding="utf-8")
        assert '# %% tags=["parameters"]' in texto and "CONFIG =" in texto and "SET =" in texto, fuente.name


@pytest.mark.lento
@pytest.mark.parametrize("fuente", fuentes(RAIZ / "notebooks"), ids=lambda p: p.name)
def test_notebook_corre_de_punta_a_punta(fuente, tmp_path, monkeypatch):
    copia = tmp_path / fuente.name
    copia.write_text(fuente.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(RAIZ)
    destino = ejecutar(copia, RAIZ)
    assert destino.is_file()


def test_estado_demos_marca_las_pendientes():
    from src.common.notebooks import estado_demos

    cfg = config_notebook(RAIZ / "configs" / "sustentacion.yaml", imprimir=False)
    estados = {e["demo"]: e["estado"] for e in estado_demos(cfg)}
    assert estados["eda"] == "listo"
    assert all(e == "listo" or e.startswith("pendiente") for e in estados.values())
    # la E[L] tiene config pero necesita las predicciones de B y D
    if not (RAIZ / "results" / "data" / "pd" / "pred_test.parquet").is_file():
        assert estados["el"].startswith("pendiente") and "pred_test.parquet" in estados["el"]


def test_correr_demo_no_escribe_en_los_resultados_versionados(monkeypatch, capsys):
    """La demo de la Parte A escribe en una carpeta temporal y los JSON versionados quedan intactos."""
    from src.common.notebooks import correr_demo

    monkeypatch.chdir(RAIZ)
    cfg = config_notebook(RAIZ / "configs" / "sustentacion.yaml", imprimir=False)
    antes = {p: p.stat().st_mtime_ns for p in (RAIZ / "results" / "data" / "eda").glob("*.json")}
    r = correr_demo(cfg, "eda", {"eda.umbral_monotonia": 0.8})
    assert r["estado"] == "ok", r
    assert "eda.umbral_monotonia=0.8" in capsys.readouterr().out
    assert {p: p.stat().st_mtime_ns for p in (RAIZ / "results" / "data" / "eda").glob("*.json")} == antes


def test_correr_demo_pendiente_no_falla(tmp_path):
    from src.common.notebooks import correr_demo

    cfg = {"demos": {"x": {"parte": "B", "subcomando": "pd", "config": str(tmp_path / "no.yaml"), "salida_temporal": None}}}
    assert correr_demo(cfg, "x") == {"estado": "pendiente"}


def _huellas_versionadas(raiz: Path) -> dict:
    """(tamaño, mtime) de cada archivo de las carpetas que el notebook no debe tocar (núcleo §5.8)."""
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for carpeta in ("data", "models", "results", "blog")
            for p in (raiz / carpeta).rglob("*") if p.is_file()}


TIEMPO_MAX_NOTEBOOK_S = 120  # tarea E11 (Parte E): cada notebook corre en < 2 min
ENTRADAS_05 = ("results/data/pd/pred_test.parquet", "results/data/lgd/pred_pd_test.parquet",
               "data/processed/montos.parquet", "data/processed/loans_pd.parquet")  # B, D y A


@pytest.mark.lento
def test_notebook_05_perdida_esperada_no_escribe_en_lo_versionado(tmp_path, monkeypatch):
    """E11: el notebook 05 corre la E[L] y dibuja sus figuras en una carpeta temporal en < 2 min; data/, models/,
    results/ y blog/ quedan intactos (results/data/el/ lo consumen el blog y la app) y el .ipynb no expone la ruta local
    de la carpeta temporal."""
    import tempfile
    import time

    fuente = RAIZ / "notebooks" / "05_perdida_esperada.py"
    assert fuente.is_file()
    faltan = [r for r in ENTRADAS_05 if not (RAIZ / r).is_file()]
    if faltan:
        pytest.skip("faltan entradas de la E[L]: " + ", ".join(faltan))
    copia = tmp_path / fuente.name
    copia.write_text(fuente.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(RAIZ)
    antes = _huellas_versionadas(RAIZ)
    inicio = time.perf_counter()
    destino = ejecutar(copia, RAIZ)
    assert time.perf_counter() - inicio < TIEMPO_MAX_NOTEBOOK_S
    assert _huellas_versionadas(RAIZ) == antes
    texto = destino.read_text(encoding="utf-8")
    temporal = Path(tempfile.gettempdir()).resolve()
    for forma in (str(temporal), temporal.as_posix(), str(temporal).replace("\\", "\\\\")):
        assert forma not in texto
