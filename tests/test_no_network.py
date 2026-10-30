"""Los experimentos no usan internet (enunciado: "que la ejecución no dependa de conexión a internet ni de APIs
externas"; núcleo §7.1). Portada del Trabajo 1 (T0.5). Corre los comandos de run.py con la red bloqueada y revisa que el código de src/ no importe
bibliotecas de red. Ningún script usa internet: el crudo se descarga a mano (data/fuentes.md).

Cada parte agrega su subcomando a la lista cuando lo conecte.
"""
import ast
import hashlib
import importlib
import pkgutil
import socket
import urllib.request
from pathlib import Path

import pytest

import src.viz
from src.common import cli

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS_VERSIONADAS = RAIZ / "blog" / "docs" / "figures"
MODELO_APP_VERSIONADO = RAIZ / "app" / "public" / "model"
RESULTADOS_SCORECARD = RAIZ / "results" / "data" / "scorecard"
CONFIG_SCORECARD_MLP = "configs/scorecard/mlp.yaml"
SHAP_PEQUENO = ["shap.n_explicar=30", "shap.n_estabilidad=10", "shap.n_fondo=20", "shap.max_evals=62",
                "shap.k_permutacion=2"]  # tamaños pequeños de la scorecard del MLP (C4)
MODULOS_DE_RED = {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib"}


class RedBloqueada(RuntimeError):
    pass


@pytest.fixture
def sin_red(monkeypatch):
    """Cualquier intento de abrir una conexión (o de resolver un nombre) lanza RedBloqueada."""

    def _bloqueado(*args, **kwargs):
        raise RedBloqueada("intento de usar la red durante un experimento")

    monkeypatch.setattr(socket.socket, "connect", _bloqueado)
    monkeypatch.setattr(socket.socket, "connect_ex", _bloqueado)
    monkeypatch.setattr(socket, "create_connection", _bloqueado)
    monkeypatch.setattr(socket, "getaddrinfo", _bloqueado)


def test_el_bloqueo_si_detecta_un_acceso_a_la_red(sin_red):
    """Control de la propia prueba: si el bloqueo no funcionara, las demás pasarían sin probar nada."""
    with pytest.raises(RedBloqueada):
        urllib.request.urlopen("https://example.org", timeout=5)


@pytest.mark.parametrize(
    "argv",
    [["eda", "--config", "configs/datos.yaml"], ["figures"], ["export"],
     ["pd-hpo", "--config", "configs/pd/hpo.yaml"], ["pd", "--config", "configs/pd/base.yaml"],
     ["scorecard", "--config", "configs/scorecard/base.yaml"],
     ["scorecard", "--config", "configs/scorecard/mlp.yaml"],
     ["pd-sensibilidad", "--config", "configs/pd/sensibilidad.yaml"]],
    ids=["eda", "figures", "export", "pd-hpo", "pd", "scorecard", "scorecard-mlp", "pd-sensibilidad"],
)
def test_comandos_de_run_py_corren_sin_red(sin_red, monkeypatch, tmp_path, argv):
    """Cada parte agrega aquí su comando cuando lo conecte. `notebooks` no entra: el kernel de Jupyter se comunica
    con el proceso por sockets locales (no es internet).

    Ningún comando puede escribir en los archivos versionados: `eda` escribe en un directorio temporal y en `figures`
    se redirige el `SALIDA` de **todos** los módulos `src/viz/figs_*.py` (también los que se agreguen después) a una
    subcarpeta temporal por módulo. Después se comprueba que `blog/docs/figures/` quedó idéntico (SHA-256 antes y
    después: ningún archivo cambiado, nuevo ni borrado) y que cada módulo dejó al menos un .png en su subcarpeta.

    `export` (Parte C, C2) no recibe `--config`: lee `src.export.exportar_app.RUTA_CONFIG`, que se apunta a un YAML
    temporal que hereda `configs/scorecard/export.yaml` y solo cambia `export.destino` a `tmp_path/app`. Se comprueba
    que escribió allí EXACTAMENTE los archivos de `archivos_modelo(m)` para cada `m` de `MODELOS` (C5: pd_model y casos
    de paridad de los dos modelos) y que `app/public/model/` quedó idéntico (si no existe, huellas vacías).

    `scorecard` (Parte C, C3) recibe `--set salida=tmp_path` (como `eda`): se comprueba que escribió
    `tmp_path/scorecard.json` y que `results/data/scorecard/` quedó idéntico (si no existe, huellas vacías).

    `scorecard` con `configs/scorecard/mlp.yaml` (Parte C, C4: SHAP del MLP) recibe además tamaños pequeños por
    `--set` (`SHAP_PEQUENO`); se comprueba que escribió `scorecard_mlp.json` y `shap_mlp.json` en `tmp_path` (y no
    `scorecard.json`) y que `results/data/scorecard/` quedó idéntico.
    """
    monkeypatch.chdir(RAIZ)
    if argv[0] == "eda":  # escribe resultados: a un directorio temporal, nunca a results/
        argv = [*argv, "--set", f"eda.salida={tmp_path.as_posix()}"]
    if argv[0] == "pd-hpo":  # escribe resultados y mlp_final.yaml: a tmp_path, nunca a configs/ ni a results/
        argv = [*argv, "--set", "hpo.n_ensayos=1", "--set", "mlp.max_epocas=1",
                "--set", f"hpo.destino={(tmp_path / 'mlp_final.yaml').as_posix()}",
                "--set", f"hpo.salida={tmp_path.as_posix()}"]
    if argv[0] == "pd":  # escribe resultados y modelos: a tmp_path, nunca a results/ ni a models/
        argv = [*argv, "--set", f"salida={(tmp_path / 'res').as_posix()}",
                "--set", f"modelos={(tmp_path / 'models').as_posix()}",
                "--set", "semillas.n=1", "--set", "mlp.max_epocas=1", "--set", "logistica.rejilla_C=[1.0e-2]"]
    if argv[0] == "scorecard":  # scorecard.json: a un directorio temporal, nunca a results/data/scorecard/
        argv = [*argv, "--set", f"salida={tmp_path.as_posix()}"]
    if argv[0] == "pd-sensibilidad":  # B10: unión, preprocesador, resultados y modelos a tmp_path; muestra pequeña
        sens = tmp_path / "sens"
        argv = [*argv, "--set", f"sensibilidad.union={(sens / 'res' / 'union.parquet').as_posix()}",
                "--set", f"sensibilidad.preprocesador={(sens / 'models' / 'pre.json').as_posix()}",
                "--set", f"sensibilidad.salida={(sens / 'res').as_posix()}",
                "--set", f"sensibilidad.modelos={(sens / 'models').as_posix()}",
                "--set", "datos.muestra=5000", "--set", "semillas.n=1", "--set", "hpo.n_ensayos=1",
                "--set", "mlp.max_epocas=1", "--set", "logistica.rejilla_C=[1.0e-2]"]
    es_mlp = argv[0] == "scorecard" and argv[2] == CONFIG_SCORECARD_MLP
    if es_mlp:  # SHAP del MLP con tamaños pequeños (la config aprobada tarda minutos)
        argv = [*argv, *(x for clave in SHAP_PEQUENO for x in ("--set", clave))]
    destinos = _redirigir_figuras(monkeypatch, tmp_path) if argv[0] == "figures" else {}
    destino_app = _redirigir_export(monkeypatch, tmp_path) if argv[0] == "export" else None
    antes = _huellas(FIGURAS_VERSIONADAS)
    antes_app = _huellas(MODELO_APP_VERSIONADO)
    antes_scorecard = _huellas(RESULTADOS_SCORECARD)
    assert cli.main(argv) == 0
    alterados_scorecard = _diferencias(antes_scorecard, _huellas(RESULTADOS_SCORECARD))
    assert not alterados_scorecard, f"`run.py {argv[0]}` alteró results/data/scorecard/: {alterados_scorecard}"
    if es_mlp:
        from src.scorecard.shap_pd import ARCHIVO_SCORECARD_MLP, ARCHIVO_SHAP

        escritos = sorted(r.name for r in tmp_path.iterdir() if r.is_file())
        assert escritos == sorted([ARCHIVO_SCORECARD_MLP, ARCHIVO_SHAP]), escritos
    elif argv[0] == "scorecard":
        from src.scorecard.puntos import ARCHIVO_SCORECARD

        assert (tmp_path / ARCHIVO_SCORECARD).is_file(), "`run.py scorecard` no escribió scorecard.json en tmp_path"
    alterados_app = _diferencias(antes_app, _huellas(MODELO_APP_VERSIONADO))
    assert not alterados_app, f"`run.py {argv[0]}` alteró app/public/model/: {alterados_app}"
    if destino_app is not None:
        from src.export.exportar_app import MODELOS, archivos_modelo

        escritos = sorted(r.name for r in destino_app.rglob("*") if r.is_file())
        esperados = sorted(nombre for modelo in MODELOS for nombre in archivos_modelo(modelo))
        assert escritos == esperados, escritos
    alterados = _diferencias(antes, _huellas(FIGURAS_VERSIONADAS))
    assert not alterados, (f"`run.py {argv[0]}` alteró figuras versionadas de blog/docs/figures/: {alterados}; "
                           "restaurar con `git checkout -- blog/docs/figures/`")
    sin_figuras = [f"src/viz/{nombre}.py" for nombre, destino in destinos.items() if not any(destino.rglob("*.png"))]
    assert not sin_figuras, (f"`run.py figures` no generó ningún .png de {sin_figuras}: ¿falta conectarlos en "
                             "src/common/cli.py::_figures?")
    if argv[0] == "figures":  # B7: la Parte B escribe exactamente sus 7 figuras y pd-calibracion.json
        from src.viz import figs_pd  # import local: si falta el módulo, solo falla el caso figures

        escritos_pd = sorted(r.name for r in destinos["figs_pd"].rglob("*") if r.is_file())  # lo escrito por B
        assert escritos_pd == sorted(figs_pd.ARCHIVOS), escritos_pd  # ni de más ni de menos
    if argv[0] == "figures":  # B11: la Parte B escribe exactamente sus 2 figuras de umbral y costo
        from src.viz import figs_umbral_pd  # import local: si falta el módulo, solo falla el caso figures

        escritos_umbral = sorted(r.name for r in destinos["figs_umbral_pd"].rglob("*") if r.is_file())  # B11
        assert escritos_umbral == sorted(figs_umbral_pd.ARCHIVOS), escritos_umbral  # ni de más ni de menos
    if argv[0] == "figures":  # C7: la Parte C escribe exactamente sus 5 figuras
        from src.viz import figs_scorecard  # import local: si falta el módulo, solo falla el caso figures

        escritos_sc = sorted(r.name for r in destinos["figs_scorecard"].rglob("*") if r.is_file())  # lo escrito por C
        assert escritos_sc == sorted(figs_scorecard.ARCHIVOS), escritos_sc  # ni de más ni de menos


def _modulos_de_figuras() -> list[str]:
    """Nombres de los módulos `src/viz/figs_*.py` (los de las partes presentes y futuras)."""
    return sorted(m.name for m in pkgutil.iter_modules(src.viz.__path__) if m.name.startswith("figs_"))


def _redirigir_figuras(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    """Apunta el `SALIDA` de cada módulo de figuras a `tmp_path/<módulo>` y devuelve esos destinos.

    Un módulo sin `SALIDA` hace fallar la prueba antes de correr nada: no habría cómo redirigirlo y escribiría sus
    figuras en el repositorio. La convención (figs_datos, figs_lgd) es `SALIDA` + `generar(raiz, destino)`.
    """
    destinos = {}
    for nombre in _modulos_de_figuras():
        modulo = importlib.import_module(f"src.viz.{nombre}")
        assert hasattr(modulo, "SALIDA"), (f"src/viz/{nombre}.py no tiene la constante SALIDA: la prueba no puede "
                                           "redirigir sus figuras a un directorio temporal (seguir la convención de "
                                           "figs_datos.py: SALIDA + generar(raiz, destino))")
        destinos[nombre] = tmp_path / nombre
        destinos[nombre].mkdir()
        monkeypatch.setattr(modulo, "SALIDA", destinos[nombre])
    assert destinos, "no se encontró ningún módulo src/viz/figs_*.py"
    return destinos


def _redirigir_export(monkeypatch, tmp_path: Path) -> Path:
    """Apunta `RUTA_CONFIG` del exportador a un YAML temporal que hereda la config real (ruta absoluta; `load_config`
    fusiona en profundidad) y cambia solo `export.destino`. Devuelve ese destino temporal.

    El import va aquí y no arriba: así, mientras `src.export` no exista, solo falla el caso `export`.
    """
    import src.export.exportar_app as exportar_app

    destino = tmp_path / "app"
    yaml_tmp = tmp_path / "export_tmp.yaml"
    base = (RAIZ / "configs" / "scorecard" / "export.yaml").as_posix()
    yaml_tmp.write_text(f'hereda: "{base}"\nexport:\n  destino: "{destino.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setattr(exportar_app, "RUTA_CONFIG", yaml_tmp)
    return destino


def _huellas(carpeta: Path) -> dict[str, str]:
    """SHA-256 de cada archivo de `carpeta` (recursivo), por ruta relativa; carpeta inexistente → {}."""
    if not carpeta.exists():
        return {}
    return {ruta.relative_to(carpeta).as_posix(): hashlib.sha256(ruta.read_bytes()).hexdigest()
            for ruta in sorted(carpeta.rglob("*")) if ruta.is_file()}


def _diferencias(antes: dict[str, str], despues: dict[str, str]) -> dict[str, list[str]]:
    """Archivos cambiados, nuevos y borrados entre dos juegos de huellas (solo las categorías no vacías)."""
    cambios = {"cambiados": sorted(r for r in antes.keys() & despues.keys() if antes[r] != despues[r]),
               "nuevos": sorted(despues.keys() - antes.keys()),
               "borrados": sorted(antes.keys() - despues.keys())}
    return {tipo: rutas for tipo, rutas in cambios.items() if rutas}


def _modulos_importados(archivo: Path) -> set[str]:
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    nombres = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            nombres.update(alias.name.split(".")[0] for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            nombres.add(nodo.module.split(".")[0])
    return nombres


@pytest.mark.parametrize("archivo", sorted((RAIZ / "src").rglob("*.py")) + [RAIZ / "run.py"],
                         ids=lambda p: p.relative_to(RAIZ).as_posix())
def test_el_codigo_de_los_experimentos_no_importa_bibliotecas_de_red(archivo):
    prohibidos = _modulos_importados(archivo) & MODULOS_DE_RED
    assert not prohibidos, f"{archivo.name} importa {prohibidos}: los experimentos no usan internet"
