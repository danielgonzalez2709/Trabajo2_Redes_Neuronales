"""E5 — figuras de la E[L] (blog y notebook 05): se regeneran desde results/data/el/summary.json sin recalcular."""

import json
from pathlib import Path

import pytest

from src.viz.figs_el import generar

NOMBRES = ["el-bandas.png", "el-pd-vs-tasa.png"]


def _resumen():
    def base(b):
        return {"n": 10, "el": 1e6 * (11 - b), "ead": 1e7, "tasa_malos": 0.05 * (11 - b) / 2,
                "pd_media": 0.04 * (11 - b) / 2, "lgd_media": 0.9}

    bandas = [{"banda": b, "score_min": 480.0 + 20 * b, "score_max": 499.0 + 20 * b, "total": base(b),
               "resueltos": {**base(b), "perdida_observada": 1.2e6 * (11 - b),
                             "razon_el_sobre_perdida_observada": 1 / 1.2},
               "en_mora": {**base(b), "perdida_provisional": 1e5}} for b in range(1, 11)]
    return {"modelo_pd": "logistica", "modelo_lgd": "mlp", "por_banda": bandas,
            "scorecard": {"base": 600.0, "odds_base": 50.0, "pdo": 20.0, "factor": 28.85, "offset": 487.12}}


def test_generar_desde_summary_json(tmp_path):
    entrada = tmp_path / "results" / "data" / "el"
    entrada.mkdir(parents=True)
    (entrada / "summary.json").write_text(json.dumps(_resumen()), encoding="utf-8")
    rutas = generar(tmp_path, tmp_path / "figuras")
    assert sorted(r.name for r in rutas) == NOMBRES
    assert all(r.is_file() and r.stat().st_size > 0 for r in rutas)


def test_generar_con_el_resumen_en_memoria(tmp_path):
    """El notebook 05 dibuja una corrida sin escribir en results/: pasa el resumen y un destino temporal."""
    rutas = generar(tmp_path, tmp_path / "figuras", resumen=_resumen())
    assert sorted(r.name for r in rutas) == NOMBRES


def test_sin_summary_json_lo_dice_y_no_inventa_figuras(tmp_path):
    with pytest.raises(FileNotFoundError, match="run.py el"):
        generar(tmp_path, tmp_path / "figuras")
    assert not (tmp_path / "figuras").exists()
