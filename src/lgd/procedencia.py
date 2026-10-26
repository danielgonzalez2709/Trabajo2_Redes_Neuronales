"""Procedencia de los resultados de la LGD: campo ``evidencia`` (Parte D; núcleo §5.4 y §6, "Mismos datos").

Módulo ligero (sin torch ni sklearn, y sin importar ``src/pd``) para que lo importen ``src/lgd/experimento.py`` (D4) y
``src/lgd/descriptivo.py`` (D1b) con una sola definición, la del núcleo §6 (idéntica a ``huella_particion`` y
``huella_archivo`` de ``src/pd/experimento.py``):

- ``huella_particion``: SHA-256 (hex) del texto UTF-8 de las líneas ``"{id},{split}\\n"`` en el orden del DataFrame.
- ``huella_archivo``: SHA-256 (hex) de los bytes del archivo (para el JSON del preprocesador).
- ``evidencia``: ``{sha256_particion, sha256_preprocesador}`` (núcleo §5.4).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

CODIFICACION = "utf-8"  # texto de la huella de la partición (núcleo §6)


def huella_particion(df: pd.DataFrame) -> str:
    """SHA-256 (hex) del texto UTF-8 de las líneas ``f"{id},{split}\\n"`` en el orden de ``df``."""
    texto = "".join(f"{i},{s}\n" for i, s in zip(df["id"], df["split"], strict=True))
    return hashlib.sha256(texto.encode(CODIFICACION)).hexdigest()


def huella_archivo(ruta) -> str:
    """SHA-256 (hex) de los bytes del archivo ``ruta``."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def evidencia(df: pd.DataFrame, ruta_preprocesador) -> dict:
    """``{sha256_particion, sha256_preprocesador}`` de ``df`` y del archivo del preprocesador (núcleo §5.4, §6)."""
    return {"sha256_particion": huella_particion(df), "sha256_preprocesador": huella_archivo(ruta_preprocesador)}
