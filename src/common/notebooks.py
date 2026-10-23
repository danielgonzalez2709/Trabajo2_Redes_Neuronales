"""Notebooks como capa delgada sobre src/ (núcleo §5.8).

- Fuente: `notebooks/NN_nombre.py` en formato *percent* de Jupytext (celdas `# %%`).
- `python run.py notebooks` convierte cada fuente a `.ipynb` y la ejecuta (modo demo); el `.ipynb` con salidas se versiona.
- Primera celda de código = parámetros (`CONFIG`, `SET`); la segunda llama a `config_notebook(CONFIG, SET)`, que usa el
  mismo cargador del CLI (una clave inexistente es un error) e imprime la config resuelta.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

from src.common.config import apply_overrides, load_config

CARPETA = Path("notebooks")
TIEMPO_MAX_CELDA_S = 600


def config_notebook(config: str | Path, cambios: dict | None = None, imprimir: bool = True) -> dict:
    """Carga `config` y aplica `cambios` ({"a.b": valor}) como si fueran `--set a.b=valor`."""
    overrides = [f"{clave}={yaml.safe_dump(valor, default_flow_style=True).strip().removesuffix('...').strip()}"
                 for clave, valor in (cambios or {}).items()]
    cfg = apply_overrides(load_config(config), overrides)
    if imprimir:
        print("Configuración final resuelta:")
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), end="")
    return cfg


def fuentes(carpeta: Path = CARPETA) -> list[Path]:
    return sorted(carpeta.glob("[0-9][0-9]_*.py"))


def ejecutar(fuente: Path, raiz: Path = Path(".")) -> Path:
    """Convierte `fuente` (.py percent) a .ipynb junto a ella y lo ejecuta con la raíz del repo como directorio."""
    import jupytext
    from nbclient import NotebookClient

    nb = jupytext.read(fuente)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    anterior = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(raiz.resolve()) + (os.pathsep + anterior if anterior else "")
    try:
        NotebookClient(nb, timeout=TIEMPO_MAX_CELDA_S, kernel_name="python3",
                       resources={"metadata": {"path": str(raiz.resolve())}}).execute()
    finally:
        if anterior is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = anterior
    destino = fuente.with_suffix(".ipynb")
    jupytext.write(nb, destino)
    return destino


def manejador(cfg: dict | None) -> int:
    for fuente in fuentes():
        inicio = time.perf_counter()
        destino = ejecutar(fuente)
        print(f"{destino.as_posix()}: {time.perf_counter() - inicio:.1f} s")
    return 0
