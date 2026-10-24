"""Configuración YAML, herencia, overrides `--set a.b=c` y `run.py` (núcleo §4.1). Portado del Trabajo 1 (T0.2)."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.common import cli
from src.common.config import apply_overrides, cargar_yaml, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg() -> dict:
    return {
        "runs": 30,
        "base_seed": 0,
        "w": [25],
        "methods": ["gdfijo", "pso"],
        "pso": {"w": 0.7298, "vecindario": {"tipo": "gbest", "k": 3}},
        "gd": {"armijo": False, "eta": 0.002},
        "costo": {"valor_hora": 25},
    }


def _escribir(ruta: Path, datos: dict | str) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(datos, dict):
        datos = yaml.safe_dump(datos, sort_keys=False, allow_unicode=True)
    ruta.write_text(datos, encoding="utf-8")
    return ruta


@pytest.fixture
def cfg_file(tmp_path: Path, cfg: dict) -> Path:
    return _escribir(tmp_path / "cfg.yaml", cfg)


@pytest.fixture
def recibido(monkeypatch) -> dict:
    """Sustituye los manejadores del registro por uno falso que guarda la config recibida."""
    llamadas: dict = {}
    for nombre, sub in list(cli.SUBCOMANDOS.items()):

        def falso(cfg, nombre=nombre):
            llamadas[nombre] = cfg
            return 0

        monkeypatch.setitem(cli.SUBCOMANDOS, nombre, sub._replace(manejador=falso))
    return llamadas


def _config_impresa(salida: str) -> dict:
    return yaml.safe_load(salida.split("---", 1)[1].split("...", 1)[0])


# ------------------------------------------------- cargador (YAML 1.2 core)

INF = float("inf")


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        # flotantes de YAML 1.2 (PyYAML/YAML 1.1 los dejaría como texto si no llevan punto)
        ("1e-3", 0.001),
        ("2E5", 200000.0),
        ("-1.5e-8", -1.5e-8),
        (".5", 0.5),
        ("1.0e-3", 0.001),
        (".inf", INF),
        # enteros: decimales sin octal implícito; octal solo con 0o; hexadecimal con 0x
        ("5", 5),
        ("010", 10),
        ("0o17", 15),
        ("0x1F", 31),
        # booleanos solo true/false
        ("true", True),
        ("False", False),
        ("yes", "yes"),
        ("off", "off"),
        # lo que no es número sigue siendo texto (versiones, guiones bajos, sexagesimales)
        ("1.2.3", "1.2.3"),
        ("1e", "1e"),
        ("1_000", "1_000"),
        ("1:30", "1:30"),
        ("null", None),
        ("[1e-3, 5, x]", [0.001, 5, "x"]),
    ],
)
def test_cargador_esquema_core_yaml12(texto: str, esperado):
    valor = cargar_yaml(texto)
    assert valor == esperado
    assert type(valor) is type(esperado)


def test_cargador_no_convierte_fechas_en_numeros():
    """Las fechas siguen siendo lo que eran con SafeLoader (no float)."""
    texto = "fecha: 2026-10-01\n"
    assert cargar_yaml(texto) == yaml.safe_load(texto)


def test_cargador_no_altera_safeloader_global():
    """El cargador propio no debe cambiar el comportamiento de yaml.safe_load en otros módulos."""
    assert yaml.safe_load("1e-3") == "1e-3"


def test_notacion_cientifica_en_archivo_y_en_set(tmp_path: Path):
    cfg = load_config(_escribir(tmp_path / "num.yaml", "eta: 1e-3\n"))
    assert cfg["eta"] == 0.001 and type(cfg["eta"]) is float
    out = apply_overrides(cfg, ["eta=1e-3"])
    assert out["eta"] == 0.001 and type(out["eta"]) is float


# ---------------------------------------------------------------- load_config


def test_load_config_lee_un_mapeo(cfg_file: Path, cfg: dict):
    assert load_config(cfg_file) == cfg


def test_load_config_archivo_inexistente(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no_existe.yaml"):
        load_config(tmp_path / "no_existe.yaml")


@pytest.mark.parametrize("contenido", ["- a\n- b\n", "42\n", ""])
def test_load_config_rechaza_lo_que_no_es_mapeo(tmp_path: Path, contenido: str):
    with pytest.raises(ValueError, match="mapeo"):
        load_config(_escribir(tmp_path / "malo.yaml", contenido))


def test_load_config_yaml_invalido_resume_el_error(tmp_path: Path):
    ruta = _escribir(tmp_path / "roto.yaml", "a: 1\nb: [1, 2\nc: 3\n")
    with pytest.raises(ValueError) as exc:
        load_config(ruta)
    mensaje = str(exc.value)
    assert "roto.yaml" in mensaje and "línea" in mensaje and "columna" in mensaje
    assert "\n" not in mensaje  # resumen de una línea, no el volcado completo de PyYAML


# ------------------------------------------------------------------- herencia


def test_herencia_simple_hija_gana_y_fusion_profunda(tmp_path: Path):
    _escribir(tmp_path / "base.yaml", {"a": 1, "b": {"c": 2, "d": 3}, "l": [1, 2, 3], "s": "x"})
    hija = _escribir(
        tmp_path / "hija.yaml", {"hereda": "base.yaml", "b": {"c": 20}, "l": [9], "nueva": True}
    )
    # las listas se reemplazan, no se concatenan; 'hereda' desaparece
    assert load_config(hija) == {"a": 1, "b": {"c": 20, "d": 3}, "l": [9], "s": "x", "nueva": True}


def test_herencia_encadenada_y_rutas_relativas_a_la_carpeta(tmp_path: Path, monkeypatch):
    _escribir(tmp_path / "abuela.yaml", {"a": 1, "b": {"c": 1, "d": 1, "e": 1}})
    _escribir(tmp_path / "sub" / "madre.yaml", {"hereda": "../abuela.yaml", "b": {"d": 2, "e": 2}})
    hija = _escribir(tmp_path / "sub" / "hija.yaml", {"hereda": "madre.yaml", "b": {"e": 3}})
    otra = tmp_path / "otra"
    otra.mkdir()
    monkeypatch.chdir(otra)  # 'hereda' es relativa al archivo, no al directorio de trabajo
    assert load_config(hija) == {"a": 1, "b": {"c": 1, "d": 2, "e": 3}}


def test_herencia_ciclo_es_error(tmp_path: Path):
    _escribir(tmp_path / "a.yaml", {"hereda": "b.yaml", "x": 1})
    _escribir(tmp_path / "b.yaml", {"hereda": "a.yaml", "y": 2})
    with pytest.raises(ValueError, match="ciclo") as exc:
        load_config(tmp_path / "a.yaml")
    assert "a.yaml" in str(exc.value) and "b.yaml" in str(exc.value)


def test_herencia_de_si_misma_es_ciclo(tmp_path: Path):
    _escribir(tmp_path / "a.yaml", {"hereda": "a.yaml"})
    with pytest.raises(ValueError, match="ciclo"):
        load_config(tmp_path / "a.yaml")


def test_herencia_base_inexistente_es_error(tmp_path: Path):
    hija = _escribir(tmp_path / "hija.yaml", {"hereda": "no_hay.yaml", "x": 1})
    with pytest.raises(FileNotFoundError) as exc:
        load_config(hija)
    assert "no_hay.yaml" in str(exc.value) and "hija.yaml" in str(exc.value)


@pytest.mark.parametrize("valor", [5, None, "", ["base.yaml"]])
def test_herencia_valor_invalido_es_error(tmp_path: Path, valor):
    hija = _escribir(tmp_path / "hija.yaml", {"hereda": valor})
    with pytest.raises(ValueError, match="hereda"):
        load_config(hija)


@pytest.mark.parametrize(
    "bloque", ["pso:\n", "pso: 3\n", "pso: [1, 2]\n", "pso:\n  vecindario: gbest\n"]
)
def test_herencia_no_reemplaza_un_bloque_por_un_valor(tmp_path: Path, bloque: str):
    """§4.1 [DECIDIDO 9-oct]: un mapeo de la base no se reemplaza por un valor (vacío, escalar)."""
    _escribir(tmp_path / "base.yaml", {"pso": {"w": 0.7, "vecindario": {"tipo": "gbest"}}})
    hija = _escribir(tmp_path / "hija.yaml", "hereda: base.yaml\n" + bloque)
    with pytest.raises(ValueError, match=r"usa pso\.") as exc:
        load_config(hija)
    assert "hija.yaml" in str(exc.value)


def test_herencia_bloque_vacio_como_mapeo_no_cambia_nada(tmp_path: Path):
    _escribir(tmp_path / "base.yaml", {"pso": {"w": 0.7}})
    hija = _escribir(tmp_path / "hija.yaml", "hereda: base.yaml\npso: {}\n")
    assert load_config(hija) == {"pso": {"w": 0.7}}


# ------------------------------------------------------------ apply_overrides


def test_override_anidado(cfg: dict):
    out = apply_overrides(cfg, ["pso.vecindario.k=5", "costo.valor_hora=40"])
    assert out["pso"]["vecindario"] == {"tipo": "gbest", "k": 5}
    assert out["costo"]["valor_hora"] == 40
    assert out["pso"]["w"] == cfg["pso"]["w"]


@pytest.mark.parametrize(
    ("override", "ruta", "esperado"),
    [  # los cuatro ejemplos literales de §4.1
        ("pso.w=0.9", ("pso", "w"), 0.9),
        ("runs=5", ("runs",), 5),
        ("gd.armijo=true", ("gd", "armijo"), True),
        ("w=[0,5,10]", ("w",), [0, 5, 10]),
    ],
)
def test_override_tipos_de_la_spec(cfg: dict, override: str, ruta: tuple, esperado):
    valor = apply_overrides(cfg, [override])
    for parte in ruta:
        valor = valor[parte]
    assert valor == esperado
    assert type(valor) is type(esperado)


def test_override_valor_vacio_es_null(cfg: dict):
    assert apply_overrides(cfg, ["runs="])["runs"] is None


def test_override_varios_y_el_ultimo_gana(cfg: dict):
    assert apply_overrides(cfg, ["runs=5", "runs=7"])["runs"] == 7


def test_override_valor_con_igual(cfg: dict):
    """Solo el primer '=' separa clave y valor."""
    assert apply_overrides(cfg, ["pso.vecindario.tipo=a=b"])["pso"]["vecindario"]["tipo"] == "a=b"


@pytest.mark.parametrize(
    ("override", "disponibles"),
    [
        ("costo.valor_hra=30", "valor_hora"),
        ("nada=1", "runs"),
        ("pso.vecindario.kk=1", "tipo"),
        ("aco.alpha=1", "costo"),  # ruta intermedia inexistente
        ("runs.x=1", "runs"),  # ruta intermedia que no es un bloque
    ],
)
def test_override_clave_inexistente_es_error(cfg: dict, override: str, disponibles: str):
    clave = override.split("=")[0]
    with pytest.raises(ValueError, match="Clave inexistente en la configuración") as exc:
        apply_overrides(cfg, [override])
    assert f"'{clave}'" in str(exc.value) and disponibles in str(exc.value)


@pytest.mark.parametrize("override", ["pso=3", "pso={w: 1}", "pso.vecindario=gbest", "pso="])
def test_override_no_reemplaza_un_bloque(cfg: dict, override: str):
    """§4.1 [DECIDIDO 9-oct]: `--set pso=3` es un error que sugiere `pso.<clave>=valor`."""
    clave = override.split("=")[0]
    with pytest.raises(ValueError, match=rf"usa {clave}\.<clave>=valor"):
        apply_overrides(cfg, [override])


@pytest.mark.parametrize("override", ["runs", "pso.w", "", "=5", "pso..w=1", ".runs=1"])
def test_override_mal_formado_es_error(cfg: dict, override: str):
    with pytest.raises(ValueError, match="mal formado"):
        apply_overrides(cfg, [override])


def test_override_valor_yaml_invalido(cfg: dict):
    with pytest.raises(ValueError, match="methods") as exc:
        apply_overrides(cfg, ["methods=[1, 2"])
    assert "\n" not in str(exc.value)


def test_override_no_muta_la_entrada(cfg: dict):
    original = copy.deepcopy(cfg)
    sin_cambios = apply_overrides(cfg, [])
    assert sin_cambios == cfg and sin_cambios is not cfg
    out = apply_overrides(cfg, ["pso.vecindario.k=9", "runs=1", "methods=[x]"])
    assert cfg == original
    out["w"].append(99)
    assert cfg["w"] == original["w"]


