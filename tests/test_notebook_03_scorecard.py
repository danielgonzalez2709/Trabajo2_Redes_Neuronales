"""Notebook 03 de la Parte C (tarea C8, C §4): corre en < 2 min, no escribe en lo versionado y SET cambia los puntajes.

Imita ``test_notebook_05_perdida_esperada_no_escribe_en_lo_versionado`` de ``tests/test_notebooks.py`` (Parte E): copia
la fuente a ``tmp_path`` y la ejecuta con ``src.common.notebooks.ejecutar``. Con ``SET = {"scorecard.pdo": 40}`` (se
sustituye la línea ``SET = {}`` de la copia) los puntos base impresos deben ser los de esa config
(offset − factor · β₀), distintos de los de PDO 20.
"""

import re
import tempfile
import time
from pathlib import Path

import pytest

from src.common.notebooks import config_notebook, ejecutar
from src.pd.logistica import LogisticaPD
from src.scorecard.puntos import puntos_base
from src.scorecard.scorecard import bloque_scorecard, parametros_scorecard

RAIZ = Path(__file__).resolve().parents[1]
FUENTE = RAIZ / "notebooks" / "03_scorecard.py"
TIEMPO_MAX_NOTEBOOK_S = 120  # aceptación de C8 (C §4): < 2 min en CPU
CARPETAS_INTACTAS = ("data", "models", "results", "configs", "blog")  # el notebook no escribe aquí (núcleo §5.8)
ENTRADAS = ("data/processed/loans_pd.parquet", "models/preprocesador_pd.json", "models/pd_logistica.joblib",
            "results/data/scorecard/shap_mlp.json", "results/data/scorecard/scorecard_mlp.json",
            "results/data/scorecard/hipotesis.json")  # A, B y C (C3, C4, C6)
TOL_IMPRESO = 1e-4  # "Puntos base" se imprime con 4 decimales
LINEA_SET = "SET = {}"  # línea de la celda de parámetros que se sustituye
PATRON_PUNTOS_BASE = re.compile(r"Puntos base: ([-+0-9.eE]+)")  # línea estable impresa por la sección 3


def _huellas(raiz: Path) -> dict:
    """(tamaño, mtime) de cada archivo de las carpetas que el notebook no debe tocar."""
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for carpeta in CARPETAS_INTACTAS
            for p in (raiz / carpeta).rglob("*") if p.is_file()}


@pytest.mark.lento
@pytest.mark.parametrize("valor_set,cambios", [("{}", {}), ('{"scorecard.pdo": 40}', {"scorecard.pdo": 40})],
                         ids=["set_vacio", "pdo_40"])
def test_notebook_03_scorecard(valor_set, cambios, tmp_path, monkeypatch):
    assert FUENTE.is_file()
    faltan = [r for r in ENTRADAS if not (RAIZ / r).is_file()]
    if faltan:
        pytest.skip("faltan entradas de la scorecard: " + ", ".join(faltan))
    texto_fuente = FUENTE.read_text(encoding="utf-8")
    assert texto_fuente.count(LINEA_SET) == 1
    copia = tmp_path / FUENTE.name
    copia.write_text(texto_fuente.replace(LINEA_SET, f"SET = {valor_set}"), encoding="utf-8")
    monkeypatch.chdir(RAIZ)
    antes = _huellas(RAIZ)
    inicio = time.perf_counter()
    destino = ejecutar(copia, RAIZ)
    assert time.perf_counter() - inicio < TIEMPO_MAX_NOTEBOOK_S
    assert _huellas(RAIZ) == antes
    texto = destino.read_text(encoding="utf-8")
    temporal = Path(tempfile.gettempdir()).resolve()
    for forma in (str(temporal), temporal.as_posix(), str(temporal).replace("\\", "\\\\")):
        assert forma not in texto
    encontrados = PATRON_PUNTOS_BASE.findall(texto)
    assert encontrados, "el notebook no imprimió 'Puntos base: …'"
    impreso = float(encontrados[0])
    # los puntos base impresos son los de la config con ese SET (offset − factor · β₀ de la logística de B3)
    cfg = config_notebook(RAIZ / "configs" / "scorecard" / "base.yaml", cambios, imprimir=False)
    _, b0 = LogisticaPD.cargar(RAIZ / cfg["rutas"]["modelo"]).coeficientes()
    esperado = puntos_base(b0, parametros_scorecard(bloque_scorecard(cfg)))
    assert impreso == pytest.approx(esperado, abs=TOL_IMPRESO)  # con PDO 40 el esperado es otro: SET cambia el puntaje
