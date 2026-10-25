"""D5 (Parte D §5): la demo de la LGD corre por CLI en < 2 min en CPU y deja sus artefactos.

Lenta (usa los datos reales): ``pytest tests/test_demo_lgd.py -m lento``. Corre en ``tmp_path`` con enlaces a los
datos procesados y al preprocesador, así que no escribe nada en el repo.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from src.common.config import load_config
from src.common.results import load_json

RAIZ = Path(__file__).resolve().parents[1]
LIMITE_S = 120  # D §5: demo < 2 min en CPU


@pytest.mark.lento
def test_demo_lgd_menos_de_2_minutos(tmp_path: Path):
    demo = load_config(RAIZ / "configs/lgd/demo.yaml")
    entradas = [demo["datos"]["salida"]["loans_lgd"], demo["datos"]["salida"]["preprocesador"],
                demo["datos"]["loans_pd"]]
    if not all((RAIZ / e).is_file() for e in entradas):
        pytest.skip("faltan los datos procesados reales (D2 / Parte A)")
    for e in entradas:
        (tmp_path / e).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / e).symlink_to(RAIZ / e)

    inicio = time.perf_counter()
    res = subprocess.run([sys.executable, str(RAIZ / "run.py"), "lgd", "--config", str(RAIZ / "configs/lgd/demo.yaml"),
                          "--set", "mlp.device=cpu"],
                         cwd=tmp_path, capture_output=True, encoding="utf-8", check=False)
    duracion = time.perf_counter() - inicio
    assert res.returncode == 0, res.stderr
    assert duracion < LIMITE_S
    assert "Tiempo de ejecución" in res.stdout

    salida = tmp_path / demo["salida"]
    s = load_json(salida / "summary.json")
    assert s["modelos"]["mlp"]["metricas"]["mae_test"]["n"] == demo["semillas"]["n"]
    assert len(list((salida / "semillas").glob("mlp_seed*.json"))) == demo["semillas"]["n"]
    assert list(pd.read_parquet(salida / "pred_test.parquet").columns) == ["id", "y", "pred_lineal", "pred_mlp"]
    assert list(pd.read_parquet(salida / "pred_pd_test.parquet").columns) == ["id", "pred_lineal", "pred_mlp"]
    for m in ("lineal", "mlp"):
        assert (tmp_path / demo["modelos"][m]).is_file()
    assert not (tmp_path / "results/data/lgd").exists()
