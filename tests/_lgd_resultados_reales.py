"""Ayudante compartido por las pruebas de la Parte D que leen los resultados reales de ``results/data/lgd/``.

TODO(retirar-tras-D4b): hasta que D4b regenere los resultados, el ``summary.json`` versionado tiene el formato anterior
a D10 (raíz ``modelos``); las pruebas con datos reales se saltan si falta el archivo o si tiene ese formato.
"""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_SUMMARY = RAIZ / "results/data/lgd/summary.json"
MOTIVO_SALTO = "faltan los resultados de D4 o tienen el formato anterior a D10 (los regenera D4b)"


def summary_real_con_formato_d10() -> bool:
    """True si existe ``results/data/lgd/summary.json`` y ya es por modelo (sin la raíz ``modelos``)."""
    return RUTA_SUMMARY.is_file() and "modelos" not in json.loads(RUTA_SUMMARY.read_text(encoding="utf-8"))
