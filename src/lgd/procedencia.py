"""Procedencia de los resultados de la LGD: hashes de la partición y del preprocesador (Parte D; núcleo §5.4).

Módulo ligero (sin torch ni sklearn) para que lo importen ``src/lgd/experimento.py`` (D4) y ``src/lgd/descriptivo.py``
(D1b) con una sola definición:

- ``hash_particion``: SHA-256 de las líneas ``id,split`` (unidas con salto de línea) en el orden del archivo.
- ``hash_preprocesador``: SHA-256 del JSON canónico (claves ordenadas, sin espacios, UTF-8) del preprocesador.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd


def _sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def hash_particion(df: pd.DataFrame) -> str:
    """SHA-256 de las líneas ``id,split`` (unidas con salto de línea) en el orden del archivo."""
    return _sha256_texto("\n".join(df["id"].astype(str) + "," + df["split"].astype(str)))


def hash_preprocesador(pre_dict: dict) -> str:
    """SHA-256 del JSON canónico (claves ordenadas, sin espacios, UTF-8) del preprocesador."""
    return _sha256_texto(json.dumps(pre_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
