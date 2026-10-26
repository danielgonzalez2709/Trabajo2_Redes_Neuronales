"""Los experimentos no usan internet (enunciado: "que la ejecución no dependa de conexión a internet ni de APIs
externas"; núcleo §7.1). Portada del Trabajo 1 (T0.5). Corre los comandos de run.py con la red bloqueada y revisa que el código de src/ no importe
bibliotecas de red. Ningún script usa internet: el crudo se descarga a mano (data/fuentes.md).

Cada parte agrega su subcomando a la lista cuando lo conecte.
"""
import ast
import socket
import urllib.request
from pathlib import Path

import pytest

from src.common import cli

RAIZ = Path(__file__).resolve().parents[1]
MODULOS_DE_RED = {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib"}


class RedBloqueada(RuntimeError):
    pass


@pytest.fixture
def sin_red(monkeypatch):
    """Cualquier intento de abrir una conexión (o de resolver un nombre) lanza RedBloqueada."""

    def _bloqueado(*args, **kwargs):
        raise RedBloqueada("intento de usar la red durante un experimento")

    monkeypatch.setattr(socket.socket, "connect", _bloqueado)
    monkeypatch.setattr(socket.socket, "connect_ex", _bloqueado)
    monkeypatch.setattr(socket, "create_connection", _bloqueado)
    monkeypatch.setattr(socket, "getaddrinfo", _bloqueado)


def test_el_bloqueo_si_detecta_un_acceso_a_la_red(sin_red):
    """Control de la propia prueba: si el bloqueo no funcionara, las demás pasarían sin probar nada."""
    with pytest.raises(RedBloqueada):
        urllib.request.urlopen("https://example.org", timeout=5)


@pytest.mark.parametrize(
    "argv",
    [["eda", "--config", "configs/datos.yaml"], ["figures"], ["pd-hpo", "--config", "configs/pd/hpo.yaml"]],
    ids=["eda", "figures", "pd-hpo"],
)
def test_comandos_de_run_py_corren_sin_red(sin_red, monkeypatch, tmp_path, argv):
    """Cada parte agrega aquí su comando cuando lo conecte. `notebooks` no entra: el kernel de Jupyter se comunica
    con el proceso por sockets locales (no es internet)."""
    monkeypatch.chdir(RAIZ)
    if argv[0] == "eda":  # escribe resultados: a un directorio temporal, nunca a results/
        argv = [*argv, "--set", f"eda.salida={tmp_path.as_posix()}"]
    if argv[0] == "pd-hpo":  # escribe resultados y mlp_final.yaml: a tmp_path, nunca a configs/ ni a results/
        argv = [*argv, "--set", "hpo.n_ensayos=1", "--set", "mlp.max_epocas=1",
                "--set", f"hpo.destino={(tmp_path / 'mlp_final.yaml').as_posix()}",
                "--set", f"hpo.salida={tmp_path.as_posix()}"]
    if argv[0] == "figures":
        from src.viz import figs_datos

        monkeypatch.setattr(figs_datos, "SALIDA", tmp_path)
    assert cli.main(argv) == 0


def _modulos_importados(archivo: Path) -> set[str]:
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    nombres = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            nombres.update(alias.name.split(".")[0] for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            nombres.add(nodo.module.split(".")[0])
    return nombres


@pytest.mark.parametrize("archivo", sorted((RAIZ / "src").rglob("*.py")) + [RAIZ / "run.py"],
                         ids=lambda p: p.relative_to(RAIZ).as_posix())
def test_el_codigo_de_los_experimentos_no_importa_bibliotecas_de_red(archivo):
    prohibidos = _modulos_importados(archivo) & MODULOS_DE_RED
    assert not prohibidos, f"{archivo.name} importa {prohibidos}: los experimentos no usan internet"
