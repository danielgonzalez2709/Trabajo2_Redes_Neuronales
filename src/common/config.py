"""Configuración YAML, herencia y overrides `--set a.b=valor` (núcleo §4.1; portado del Trabajo 1, T0.2).

- Archivos y valores de `--set` se leen con `CargadorYAML`: SafeLoader con el esquema core de
  YAML 1.2 para bool/int/float (PyYAML sigue YAML 1.1: `1e-3` sería texto, `yes` sería True y
  `010` sería 8).
- Herencia: la clave `hereda` (ruta relativa a la carpeta del archivo) carga otra config como
  base y se fusiona en profundidad (la hija gana; las listas se reemplazan). Los overrides se
  aplican después, sobre el resultado.
- Una clave que no existe es un error, y un bloque (mapeo) no se puede reemplazar por un valor.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import IO, Any

import yaml

SEPARADOR_CLAVE = "."
SEPARADOR_VALOR = "="
CLAVE_HEREDA = "hereda"

ETIQUETA_BOOL = "tag:yaml.org,2002:bool"
ETIQUETA_INT = "tag:yaml.org,2002:int"
ETIQUETA_FLOAT = "tag:yaml.org,2002:float"

# Esquema core de YAML 1.2 (https://yaml.org/spec/1.2.2/#1032-tag-resolution).
BOOL_YAML12 = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
INT_YAML12 = re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$")
# Flotantes sin los enteros puros (los resuelve INT_YAML12): 1e-3, 2E5, -1.5e-8, .5, 1.0e-3, 1.
FLOAT_YAML12 = re.compile(
    r"""^(?:[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9]+[eE][-+]?[0-9]+
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)
PRIMEROS_BOOL = list("tTfF")
PRIMEROS_NUMERO = list("-+0123456789.")


class CargadorYAML(yaml.SafeLoader):
    """SafeLoader con el esquema core de YAML 1.2 para bool/int/float.

    null y fechas se resuelven igual que en SafeLoader. No modifica `yaml.SafeLoader`:
    `yaml.safe_load` sigue igual en el resto del proyecto.
    """


def _construir_int(loader: CargadorYAML, nodo: yaml.ScalarNode) -> int:
    """Enteros de YAML 1.2: decimal sin octal implícito (010 → 10); 0o octal, 0x hexadecimal."""
    texto = loader.construct_scalar(nodo)
    return int(texto, 0) if texto.startswith(("0o", "0x")) else int(texto)


def _esquema_core_yaml12() -> None:
    """Configura `CargadorYAML` (se llama una vez, al importar el módulo)."""
    reglas_11 = (ETIQUETA_BOOL, ETIQUETA_INT, ETIQUETA_FLOAT)
    # 1) Copia propia de los resolvedores de SafeLoader sin sus reglas YAML 1.1 de bool/int/float.
    CargadorYAML.yaml_implicit_resolvers = {
        primero: [(etiqueta, regex) for etiqueta, regex in reglas if etiqueta not in reglas_11]
        for primero, reglas in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    # 2) Reglas de YAML 1.2 para esas tres etiquetas.
    CargadorYAML.add_implicit_resolver(ETIQUETA_BOOL, BOOL_YAML12, PRIMEROS_BOOL)
    CargadorYAML.add_implicit_resolver(ETIQUETA_INT, INT_YAML12, PRIMEROS_NUMERO)
    CargadorYAML.add_implicit_resolver(ETIQUETA_FLOAT, FLOAT_YAML12, PRIMEROS_NUMERO)
    # 3) Constructor de enteros propio: el de SafeLoader lee "010" como octal (8).
    #    bool y float sirven tal cual: el de bool busca el texto en minúsculas (true/false) y el
    #    de float hace float(texto) tras tratar .inf/.nan; su rama sexagesimal nunca se alcanza
    #    porque FLOAT_YAML12 no admite ':'.
    CargadorYAML.add_constructor(ETIQUETA_INT, _construir_int)


_esquema_core_yaml12()


def _resumir_error_yaml(exc: yaml.YAMLError) -> str:
    """Resumen de una línea: el problema y su posición (línea/columna, desde 1)."""
    problema = getattr(exc, "problem", None) or str(exc).splitlines()[0]
    marca = getattr(exc, "problem_mark", None)
    if marca is None:
        return problema
    return f"{problema} (línea {marca.line + 1}, columna {marca.column + 1})"


def cargar_yaml(texto: str | IO[str]) -> Any:
    """Interpreta texto (o un flujo) YAML con `CargadorYAML`."""
    return yaml.load(texto, Loader=CargadorYAML)  # noqa: S506 — deriva de SafeLoader


def _leer_mapeo(ruta: Path) -> dict:
    """Lee un archivo YAML cuya raíz debe ser un mapeo (sin resolver la herencia)."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo de configuración: {ruta}")
    with ruta.open(encoding="utf-8") as fh:
        try:
            datos = cargar_yaml(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML inválido en {ruta}: {_resumir_error_yaml(exc)}") from exc
    if not isinstance(datos, dict):
        raise ValueError(
            f"La configuración {ruta} debe ser un mapeo (clave: valor) en la raíz; "
            f"se obtuvo {type(datos).__name__}"
        )
    return datos


def _error_bloque(texto_clave: str) -> str:
    return (
        f"No se puede reemplazar el bloque '{texto_clave}' por un valor: "
        f"usa {texto_clave}.<clave>=valor (solo las subclaves que cambian)"
    )


def _fusionar(base: dict, hija: dict, origen: Path, prefijo: str = "") -> dict:
    """Fusión profunda de `hija` sobre `base`: la hija gana, los mapeos se fusionan y las listas
    y escalares se reemplazan. Un mapeo de la base no se puede reemplazar por un valor.

    MUTA `base` a propósito y la devuelve: `_cargar_con_herencia` siempre le pasa diccionarios
    recién leídos del disco que nadie más referencia, así que copiarlos sería trabajo inútil.
    """
    for clave, valor in hija.items():
        texto_clave = f"{prefijo}{clave}"
        if isinstance(base.get(clave), dict):
            if not isinstance(valor, dict):
                raise ValueError(f"En {origen}: {_error_bloque(texto_clave)}")
            _fusionar(base[clave], valor, origen, texto_clave + SEPARADOR_CLAVE)
        else:
            base[clave] = valor
    return base


def _cargar_con_herencia(ruta: Path, cadena: tuple[Path, ...]) -> dict:
    """Carga `ruta` resolviendo `hereda` recursivamente; `cadena` = archivos ya visitados."""
    identidad = ruta.resolve()
    if identidad in cadena:
        recorrido = " -> ".join(str(p) for p in (*cadena, identidad))
        raise ValueError(f"Herencia de configuraciones con ciclo ('{CLAVE_HEREDA}'): {recorrido}")
    datos = _leer_mapeo(ruta)
    if CLAVE_HEREDA not in datos:
        return datos
    base_rel = datos.pop(CLAVE_HEREDA)
    if not isinstance(base_rel, str) or not base_rel.strip():
        raise ValueError(
            f"En {ruta}, '{CLAVE_HEREDA}' debe ser la ruta (texto) de la config base, relativa "
            f"a su carpeta; se obtuvo {base_rel!r}"
        )
    ruta_base = ruta.parent / base_rel
    if not ruta_base.is_file():
        raise FileNotFoundError(
            f"{ruta} declara '{CLAVE_HEREDA}: {base_rel}', pero no existe el archivo base "
            f"{ruta_base}"
        )
    base = _cargar_con_herencia(ruta_base, (*cadena, identidad))
    return _fusionar(base, datos, ruta)


def load_config(path: str | Path) -> dict:
    """Lee una config YAML (raíz = mapeo), resuelve `hereda` y la devuelve sin esa clave."""
    return _cargar_con_herencia(Path(path), ())


def _parsear_override(override: str) -> tuple[list[str], str, Any]:
    """Separa "a.b.c=valor" en (["a", "b", "c"], "a.b.c", valor interpretado como YAML)."""
    if SEPARADOR_VALOR not in override:
        raise ValueError(f"Override mal formado {override!r}: se espera 'clave.subclave=valor'")
    texto_clave, texto_valor = override.split(SEPARADOR_VALOR, 1)
    partes = [parte.strip() for parte in texto_clave.split(SEPARADOR_CLAVE)]
    if not all(partes):
        raise ValueError(f"Override mal formado {override!r}: clave vacía o segmentos vacíos")
    texto_clave = SEPARADOR_CLAVE.join(partes)
    try:
        valor = cargar_yaml(texto_valor)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Valor YAML inválido en el override '{texto_clave}': {texto_valor!r} "
            f"({_resumir_error_yaml(exc)})"
        ) from exc
    return partes, texto_clave, valor


def _clave_inexistente(texto_clave: str, detalle: str) -> ValueError:
    return ValueError(f"Clave inexistente en la configuración: '{texto_clave}'. {detalle}")


def _disponibles(nodo: dict, partes_nivel: list[str]) -> str:
    nivel = SEPARADOR_CLAVE.join(partes_nivel) or "(raíz)"
    return f"Claves disponibles en {nivel}: {', '.join(sorted(map(str, nodo))) or '(ninguna)'}"


def _nodo_padre(cfg: dict, partes: list[str], texto_clave: str) -> dict:
    """Baja por partes[:-1] y devuelve el mapeo que contiene la hoja.

    Error si algún tramo no existe o no es un bloque.
    """
    nodo = cfg
    for i, parte in enumerate(partes[:-1]):
        if parte not in nodo:
            raise _clave_inexistente(texto_clave, _disponibles(nodo, partes[:i]))
        nodo = nodo[parte]
        if not isinstance(nodo, dict):
            tramo = SEPARADOR_CLAVE.join(partes[: i + 1])
            raise _clave_inexistente(texto_clave, f"'{tramo}' no es un bloque (vale {nodo!r}).")
    return nodo


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Copia de `cfg` con los overrides aplicados en orden (el último gana). No muta `cfg`."""
    resultado = copy.deepcopy(cfg)
    for override in overrides:
        partes, texto_clave, valor = _parsear_override(override)
        padre = _nodo_padre(resultado, partes, texto_clave)
        hoja = partes[-1]
        if hoja not in padre:
            raise _clave_inexistente(texto_clave, _disponibles(padre, partes[:-1]))
        if isinstance(padre[hoja], dict):
            raise ValueError(_error_bloque(texto_clave))
        padre[hoja] = valor
    return resultado
