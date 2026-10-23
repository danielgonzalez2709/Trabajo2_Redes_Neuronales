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
