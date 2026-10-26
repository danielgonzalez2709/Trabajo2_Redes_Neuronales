"""Interfaz de línea de comandos de `run.py` (núcleo §4.1). Portada del Trabajo 1 (T0.2).

    python run.py eda       --config configs/datos.yaml
    python run.py pd        --config configs/pd/demo.yaml [--set clave.subclave=valor ...]
    python run.py pd-hpo    --config configs/pd/hpo.yaml
    python run.py lgd       --config configs/lgd/demo.yaml
    python run.py lgd-hpo   --config configs/lgd/hpo.yaml
    python run.py scorecard --config configs/scorecard/base.yaml --set scorecard.pdo=40
    python run.py el        --config configs/el/base.yaml
    python run.py export
    python run.py figures
    python run.py notebooks

Para conectar un subcomando, cambiar el `manejador` de su entrada en `SUBCOMANDOS` por una
función `(cfg: dict | None) -> int` (recibe la config final, o None si el subcomando no la usa)
que importe su módulo al llamarse (así `run.py --help` no carga torch).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import NamedTuple

import yaml

from src.common.config import apply_overrides, load_config

Manejador = Callable[[dict | None], int]

CABECERA_CONFIG = "Configuración final resuelta:"
DESTINO_PARSER = "parser_subcomando"
# Tildes y "ñ" intactas aunque se redirija la salida (en Windows sería cp1252).
CODIFICACION_SALIDA = "utf-8"
AYUDA_SET = (
    "Sobrescribe un parámetro existente; el valor se interpreta como YAML (repetible).\n"
    "Ej.: --set mlp.dropout=0.3 --set scorecard.pdo=40\n"
    'En PowerShell, los valores con espacios van entre comillas: --set "mlp.capas=[64, 32]"'
)


class Subcomando(NamedTuple):
    ayuda: str
    usa_config: bool  # True: --config obligatorio y --set repetible
    manejador: Manejador


def _pendiente(nombre: str, tarea: str) -> Manejador:
    """Manejador provisional: avisa qué tarea implementará el subcomando."""

    def manejador(cfg: dict | None) -> int:
        print(f"Subcomando {nombre}: pendiente ({tarea})")
        return 0

    return manejador


def _eda(cfg: dict | None) -> int:
    """Parte A (A6/A6b). Import perezoso: cargar la CLI no importa pandas."""
    from src.datos.experimento_eda import manejador

    return manejador(cfg)


def _lgd(cfg: dict | None) -> int:
    """Parte D (D4): lineal + MLP de LGD con semillas."""
    from src.lgd.experimento import manejador

    return manejador(cfg)


def _lgd_hpo(cfg: dict | None) -> int:
    """Parte D (D4): búsqueda Optuna del MLP de LGD."""
    from src.lgd.experimento import manejador_hpo

    return manejador_hpo(cfg)


def _el(cfg: dict | None) -> int:
    """Parte E (E5): pérdida esperada."""
    from src.el.experimento import manejador

    return manejador(cfg)


def _figures(cfg: dict | None) -> int:
    """E8: cada parte agrega aquí su módulo de figuras. Por ahora, la Parte A y la Parte D (D6)."""
    import time

    from src.viz import figs_datos, figs_lgd

    inicio = time.perf_counter()
    for ruta in [*figs_datos.generar(), *figs_lgd.generar()]:
        print(ruta.as_posix())
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0


def _export(cfg: dict | None) -> int:
    """Parte C (C2): pd_model.json y casos_paridad.json para la app. Lee `exportar_app.RUTA_CONFIG` al llamarse."""
    import time

    from src.export import exportar_app

    inicio = time.perf_counter()
    cfg = load_config(exportar_app.RUTA_CONFIG)
    _imprimir_config(cfg)
    for ruta in exportar_app.exportar(cfg):
        print(ruta.as_posix())
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")
    return 0


def _notebooks(cfg: dict | None) -> int:
    """E10: notebooks/NN_*.py → .ipynb ejecutado (núcleo §5.8)."""
    from src.common.notebooks import manejador

    return manejador(cfg)


# Registro único nombre -> (ayuda, usa_config, manejador). Cada parte conecta el suyo.
SUBCOMANDOS: dict[str, Subcomando] = {
    "eda": Subcomando("Parte A: EDA inicial, hipótesis y descriptivo después del preprocesamiento", True, _eda),
    "pd": Subcomando("Parte B: logística + MLP de PD", True, _pendiente("pd", "B3/B5")),
    "pd-hpo": Subcomando("Parte B: búsqueda Optuna del MLP de PD", True, _pendiente("pd-hpo", "B4")),
    "lgd": Subcomando("Parte D: lineal + MLP de LGD", True, _lgd),
    "lgd-hpo": Subcomando("Parte D: búsqueda Optuna del MLP de LGD", True, _lgd_hpo),
    "scorecard": Subcomando("Parte C: scorecard y SHAP", True, _pendiente("scorecard", "C1/C4")),
    "el": Subcomando("Parte E: pérdida esperada", True, _el),
    "export": Subcomando("Parte C: modelos → app/public/model/*.json", False, _export),
    "figures": Subcomando("Regenera todas las figuras desde results/", False, _figures),
    "notebooks": Subcomando("Convierte y ejecuta los notebooks en modo demo", False, _notebooks),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Punto de entrada único del Trabajo 2 (núcleo §4.1).",
    )
    sub = parser.add_subparsers(dest="comando", required=True)
    for nombre, subcomando in SUBCOMANDOS.items():
        p = sub.add_parser(
            nombre,
            help=subcomando.ayuda,
            description=subcomando.ayuda,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        if not subcomando.usa_config:
            continue
        # Se guarda el parser del subcomando en el Namespace para que main() emita los errores de
        # configuración con el uso de ESTE subcomando (y no el del parser raíz).
        p.set_defaults(**{DESTINO_PARSER: p})
        p.add_argument("--config", required=True, help="Ruta al archivo YAML de configuración")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="CLAVE.SUBCLAVE=VALOR",
            help=AYUDA_SET,
        )
    return parser


def _imprimir_config(cfg: dict) -> None:
    print(CABECERA_CONFIG)
    texto = yaml.safe_dump(
        cfg, sort_keys=False, allow_unicode=True, explicit_start=True, explicit_end=True
    )
    print(texto, end="")


def _forzar_utf8() -> None:
    """Al redirigir en Windows Python usa la codificación local (cp1252); se fuerza UTF-8."""
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding=CODIFICACION_SALIDA)


def main(argv: list[str] | None = None) -> int:
    _forzar_utf8()
    args = build_parser().parse_args(argv)
    subcomando = SUBCOMANDOS[args.comando]
    cfg: dict | None = None
    if subcomando.usa_config:
        try:
            cfg = apply_overrides(load_config(args.config), args.overrides)
        except (FileNotFoundError, ValueError) as exc:
            # El Namespace trae el parser del subcomando: su .error() imprime "usage: run.py pd ..."
            # y sale con código 2.
            getattr(args, DESTINO_PARSER).error(str(exc))
        _imprimir_config(cfg)
    return subcomando.manejador(cfg)
