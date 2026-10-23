"""Las figuras de la Parte A se regeneran desde results/ sin reentrenar (núcleo §7.1)."""

from pathlib import Path

from src.viz.figs_datos import generar

RAIZ = Path(__file__).resolve().parents[1]


def test_generar_escribe_las_cuatro_figuras(tmp_path):
    rutas = generar(RAIZ, tmp_path)
    assert sorted(r.name for r in rutas) == ["datos-anio.png", "eda-tasa-variables.png", "post-correlacion.png",
                                             "post-psi.png"]
    assert all(r.stat().st_size > 10_000 for r in rutas)
