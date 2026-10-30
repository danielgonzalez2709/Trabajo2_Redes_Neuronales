# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # `scripts/umbral_pd.py`: de la PD a la decisión (B11 [EXTRA], B-pd §3.9)
#
# ```
# python scripts/umbral_pd.py --config configs/pd/umbral.yaml [--set clave=valor …]
# ```
#
# Escribe `umbral.json` y `pred_umbral.parquet` en `salida` y regenera `pd-umbral-sensibilidad.png` y
# `pd-umbral-costo.png` en `figuras`. Es un script aparte (como `scripts/diagnostico_nivel_pd.py`) para no tocar
# `cli.py` ni `run.py`, que son compartidos. Demo (< 2 min): `--set semillas.n=3 --set
# salida=results/data/pd_umbral_demo --set figuras=results/data/pd_umbral_demo/figuras`.
#
# Código 0 al terminar; ante un error esperado (config, rutas, datos, control), una línea en stderr y código 2.

# %%
"""CLI de B11: lee la config (con --set), corre src/pd/umbral_experimento.py y devuelve el código de salida."""

import argparse  # argumentos de la línea de comandos
import sys  # sys.path y código de salida
from pathlib import Path  # rutas

RAIZ = Path(__file__).resolve().parents[1]  # raíz del repositorio (scripts/ → raíz)
if str(RAIZ) not in sys.path:  # src/ importable desde cualquier directorio
    sys.path.insert(0, str(RAIZ))  # antes que el resto

from src.common.config import apply_overrides, load_config  # noqa: E402  mismo cargador que run.py
from src.pd import experimento as ex  # noqa: E402  errores esperados y una_linea (B5/B6)
from src.pd import umbral_experimento as ue  # noqa: E402  orquestación de B11


# %% [markdown]
# ## Argumentos
#
# `--config` (obligatorio) y `--set clave=valor` repetible (el último gana), como `run.py`.

# %%
def construir_parser() -> argparse.ArgumentParser:
    """Parser con --config y --set repetible."""
    parser = argparse.ArgumentParser(prog="scripts/umbral_pd.py",  # nombre en la ayuda
                                     description="B11: de la PD a la decisión (umbral, matriz de confusión y costo).")
    parser.add_argument("--config", required=True, help="YAML de B11 (p. ej. configs/pd/umbral.yaml)")  # config
    parser.add_argument("--set", action="append", default=[], metavar="clave=valor",  # overrides
                        help="cambia una clave de la config (repetible)")  # ayuda
    return parser  # parser


# %% [markdown]
# ## Config resuelta
#
# `load_config` (con `hereda`) y luego los `--set` en orden. Un archivo que no existe o una clave inexistente son
# errores esperados.

# %%
def leer_config(ruta: str, cambios: list[str]) -> dict:
    """Config con herencia y --set aplicados."""
    return apply_overrides(load_config(ruta), cambios)  # el último --set gana


# %% [markdown]
# ## Punto de entrada
#
# La lectura de la config y la corrida comparten el manejo de errores de `umbral_experimento.manejador` (código 2 y
# una línea). `raiz` y `dep` se inyectan solo en las pruebas.

# %%
def main(argv=None, raiz=None, dep=None) -> int:
    """Código de salida: 0 al terminar, 2 ante un error esperado."""
    args = construir_parser().parse_args(argv)  # argumentos
    try:  # la config también puede fallar
        cfg = leer_config(args.config, args.set)  # config resuelta
    except ex.ERRORES_DE_PROGRAMACION:  # bugs: con traza
        raise  # se propagan
    except ex.ERRORES_ESPERADOS as exc:  # archivo que no existe o clave inexistente
        print(f"{ue.PREFIJO_ERROR}{type(exc).__name__}: {ex.una_linea(exc)}", file=sys.stderr)  # una línea
        return ex.CODIGO_ERROR  # 2
    return ue.manejador(cfg, raiz=raiz, dep=dep)  # corrida e informe


# %% [markdown]
# Ejecución directa.

# %%
if __name__ == "__main__":  # python scripts/umbral_pd.py …
    sys.exit(main())  # código de salida
