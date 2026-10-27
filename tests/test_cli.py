"""CLI de `run.py` (núcleo §4.1): registro de subcomandos, `--set` y errores claros."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.common import cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    ruta = tmp_path / "cfg.yaml"
    ruta.write_text(
        yaml.safe_dump({"base_seed": 0, "mlp": {"dropout": 0.1, "capas": [32, 16]}, "scorecard": {"pdo": 20}}),
        encoding="utf-8",
    )
    return ruta


@pytest.fixture
def recibido(monkeypatch) -> dict:
    """Sustituye los manejadores del registro por uno falso que guarda la config recibida."""
    llamadas: dict = {}
    for nombre, sub in list(cli.SUBCOMANDOS.items()):

        def falso(cfg, nombre=nombre):
            llamadas[nombre] = cfg
            return 0

        monkeypatch.setitem(cli.SUBCOMANDOS, nombre, sub._replace(manejador=falso))
    return llamadas


def _config_impresa(salida: str) -> dict:
    return yaml.safe_load(salida.split("---", 1)[1].split("...", 1)[0])


def test_registro_de_subcomandos_del_nucleo():
    usa_config = {nombre: sub.usa_config for nombre, sub in cli.SUBCOMANDOS.items()}
    assert usa_config == {
        "eda": True, "pd": True, "pd-hpo": True, "pd-sensibilidad": True, "lgd": True, "lgd-hpo": True, "scorecard": True, "el": True,
        "export": False, "figures": False, "notebooks": False,
    }


def test_set_imprime_y_entrega_la_config_final(cfg_file: Path, recibido: dict, capsys):
    argv = ["pd", "--config", str(cfg_file), "--set", "mlp.dropout=0.3", "--set", "mlp.capas=[64, 32]"]
    assert cli.main(argv) == 0
    impreso = _config_impresa(capsys.readouterr().out)
    assert impreso["mlp"] == {"dropout": 0.3, "capas": [64, 32]}
    assert recibido["pd"] == impreso


def test_clave_inexistente_es_error_con_uso_del_subcomando(cfg_file: Path, recibido: dict, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["scorecard", "--config", str(cfg_file), "--set", "scorecard.pod=40"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "scorecard.pod" in err and "usage: run.py scorecard" in err
    assert recibido == {}


def test_subcomando_sin_config(recibido: dict):
    assert cli.main(["figures"]) == 0
    assert recibido == {"figures": None}


def test_run_py_ayuda_en_otro_proceso():
    res = subprocess.run([sys.executable, "run.py", "--help"], cwd=ROOT, capture_output=True,
                         encoding="utf-8", check=False)
    assert res.returncode == 0, res.stderr
    assert "scorecard" in res.stdout and "notebooks" in res.stdout
