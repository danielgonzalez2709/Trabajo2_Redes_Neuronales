"""Configuración común de pytest: hace importables el paquete src/ (desde la raíz) y scripts/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for ruta in (SCRIPTS, ROOT):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))
