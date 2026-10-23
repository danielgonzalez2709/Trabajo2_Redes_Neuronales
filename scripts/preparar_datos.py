"""Crudo → data/processed (Parte A). Uso: python scripts/preparar_datos.py [--config configs/datos.yaml] [--set clave=valor]

Requiere el CSV de Kaggle en la ruta de `crudo.csv` (ver data/fuentes.md). No usa internet.
"""

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.common.config import apply_overrides, load_config  # noqa: E402
from src.datos.preparar import preparar  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datos.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(RAIZ / args.config), args.overrides)
    inicio = time.perf_counter()
    r = preparar(cfg, RAIZ)
    print(f"Registros crudos: {r['n_registros_crudo']:,} · variables: {r['n_variables_crudo']} · "
          f"periodo {r['periodo']['min']} a {r['periodo']['max']}")
    print(f"Para modelar: {r['n_modelado']:,} · por split: {r['n_por_split']}")
    print(f"Tasa de malos por split: { {k: round(v, 4) for k, v in r['tasa_malos_por_split'].items()} }")
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
