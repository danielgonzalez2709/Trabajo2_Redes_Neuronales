"""Pruebas de las huellas de la LGD (tarea D9; núcleo §5.4 y §6, "Mismos datos").

``src/lgd/procedencia.py`` debe dar exactamente las huellas del §6, las mismas que ``src/pd/experimento.py``.
"""
import ast
import hashlib
from pathlib import Path

import pandas as pd

from src.lgd import procedencia
from src.pd.experimento import huella_archivo, huella_particion

RAIZ = Path(__file__).resolve().parents[1]


def _df() -> pd.DataFrame:
    return pd.DataFrame({"id": ["10", "7", "ñ-3"], "split": ["train", "val", "test"], "lgd": [0.1, 0.5, 1.0]})


def test_huella_particion_a_mano_y_igual_a_b():
    df = _df()
    esperado = hashlib.sha256("10,train\n7,val\nñ-3,test\n".encode("utf-8")).hexdigest()
    assert procedencia.huella_particion(df) == esperado
    assert procedencia.huella_particion(df) == huella_particion(df["id"], df["split"])
    # el orden del DataFrame cuenta
    assert procedencia.huella_particion(df.iloc[::-1]) != esperado


def test_huella_archivo_son_los_bytes(tmp_path: Path):
    ruta = tmp_path / "pre.json"
    ruta.write_bytes('{\n  "b": 1,\n  "a": "ñ"\n}'.encode("utf-8"))
    esperado = hashlib.sha256(ruta.read_bytes()).hexdigest()
    assert procedencia.huella_archivo(ruta) == esperado == huella_archivo(ruta)


def test_evidencia_tiene_las_claves_del_spec(tmp_path: Path):
    ruta = tmp_path / "pre.json"
    ruta.write_text('{"a": 1}', encoding="utf-8")
    df = _df()
    ev = procedencia.evidencia(df, ruta)
    assert ev == {"sha256_particion": huella_particion(df["id"], df["split"]),
                  "sha256_preprocesador": huella_archivo(ruta)}
    for valor in ev.values():
        assert len(valor) == 64 and all(c in "0123456789abcdef" for c in valor)


def test_procedencia_es_ligero():
    """Sin torch, sklearn ni ``src.pd`` (lo importan módulos ligeros como ``descriptivo``)."""
    arbol = ast.parse((RAIZ / "src/lgd/procedencia.py").read_text(encoding="utf-8"))
    modulos = {a.name for n in ast.walk(arbol) if isinstance(n, ast.Import) for a in n.names}
    modulos |= {n.module for n in ast.walk(arbol) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m.split(".")[0] in {"torch", "sklearn"} or m.startswith("src.pd") for m in modulos), modulos
