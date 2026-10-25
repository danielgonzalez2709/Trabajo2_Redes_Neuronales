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


def _valor_set(valor) -> str:
    return yaml.safe_dump(valor, default_flow_style=True).strip().removesuffix("...").strip()


def _faltantes(demo: dict) -> list[str]:
    return [r for r in [demo["config"], *demo.get("requiere", [])] if not Path(r).is_file()]


def estado_demos(cfg: dict) -> list[dict]:
    """Para el notebook 00: cada demo de `configs/sustentacion.yaml` y si ya se puede correr (existen su config y lo que
    requiere de otras partes)."""
    return [{"demo": nombre, "parte": d["parte"], "comando": _texto_comando(d, {}),
             "estado": "listo" if not _faltantes(d) else "pendiente: falta " + ", ".join(_faltantes(d))}
            for nombre, d in cfg["demos"].items()]


def _argumentos(demo: dict, cambios: dict, salida_temporal: str | None) -> list[str]:
    args = [demo["subcomando"], "--config", demo["config"]]
    sets = dict(cambios)
    if demo.get("salida_temporal") and salida_temporal:
        sets.setdefault(demo["salida_temporal"], salida_temporal)
    for clave, valor in sets.items():
        args += ["--set", f"{clave}={_valor_set(valor)}"]
    return args


def _texto_comando(demo: dict, cambios: dict) -> str:
    return "python run.py " + " ".join(_argumentos(demo, cambios, None))


def correr_demo(cfg: dict, nombre: str, cambios: dict | None = None, timeout_s: int = 115) -> dict:
    """Corre `python run.py <subcomando> --config ... [--set ...]` en otro proceso, como se haría en la terminal.

    Si la demo declara `salida_temporal` (la clave de su carpeta de salida), escribe en una carpeta temporal: correr la
    demo en la sustentación nunca sobrescribe los resultados versionados que usan las demás partes.
    """
    import subprocess
    import sys
    import tempfile

    demo = cfg["demos"][nombre]
    if _faltantes(demo):
        print(f"Pendiente (Parte {demo['parte']}): todavía no existe " + ", ".join(_faltantes(demo)))
        return {"estado": "pendiente"}
    args = _argumentos(demo, cambios or {}, tempfile.mkdtemp(prefix=f"demo_{nombre}_"))
    print("$ python run.py " + " ".join(args))
    inicio = time.perf_counter()
    proceso = subprocess.run([sys.executable, "run.py", *args], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=timeout_s, check=False)
    segundos = time.perf_counter() - inicio
    print(proceso.stdout[-6000:])
    if proceso.returncode != 0:
        print(proceso.stderr[-3000:])
    return {"estado": "ok" if proceso.returncode == 0 else "error", "codigo": proceso.returncode,
            "segundos": round(segundos, 1), "salida": proceso.stdout}


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
    total = time.perf_counter()
    for fuente in fuentes():
        inicio = time.perf_counter()
        destino = ejecutar(fuente)
        print(f"{destino.as_posix()}: {time.perf_counter() - inicio:.1f} s")
    print(f"Tiempo de ejecución: {time.perf_counter() - total:.1f} s")
    return 0
