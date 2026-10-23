"""Punto de entrada único (núcleo §4.1). Uso: python run.py {pd,pd-hpo,lgd,lgd-hpo,scorecard,el,export,figures,notebooks} --help"""

import sys

from src.common.cli import main

if __name__ == "__main__":
    sys.exit(main())
