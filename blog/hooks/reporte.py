"""Hook de MkDocs del reporte (Spec §8): arma la entrada de blog y hace cumplir las reglas del enunciado.

Sintaxis que usan las secciones (detalle en blog/README.md):
  --8<-- "secciones/archivo.md"          incluye una sección (ruta relativa a blog/)
  ::: figura <id>                       bloque de figura APA; dentro: "Título: …", "Fuente: …" y el contenido
  ::: tabla <id>                        igual, para tablas (el contenido es una tabla Markdown)
  :::                                   cierra el bloque
  incluir: figures/x.html               (dentro de una figura) incrusta un fragmento HTML (p. ej. Plotly)
  [[fig:<id>]]  [[tab:<id>]]            referencia cruzada → "Figura N" / "Tabla N" con enlace
  [@clave]  [@a; @b, p. 3]  @clave      citas APA 7 (pandoc citeproc + blog/apa.csl); claves = apellidoAÑO
  \\bibliografia                         lugar donde se escribe la lista de referencias

El build FALLA (requisitos del enunciado) si: una figura/tabla no tiene título o fuente, un id se repite, una
referencia cruzada no existe, una figura/tabla nunca se cita en el texto, o una cita no está en references.bib.
"""
from __future__ import annotations

import re
from pathlib import Path

import pypandoc
from mkdocs.exceptions import PluginError

RAIZ_BLOG = Path(__file__).resolve().parents[1]
BIB = RAIZ_BLOG / "references.bib"
CSL = RAIZ_BLOG / "apa.csl"
MARCA_BIBLIOGRAFIA = "\\bibliografia"
PROFUNDIDAD_MAX_INCLUSION = 3

_INCLUSION = re.compile(r'^--8<-- "(?P<ruta>[^"]+)"[ \t]*$', re.M)
_BLOQUE = re.compile(
    r"^::: (?P<tipo>figura|tabla) (?P<id>[a-z0-9-]+)[ \t]*\n(?P<cuerpo>.*?)\n:::[ \t]*$", re.M | re.S
)
_REFERENCIA = re.compile(r"\[\[(?P<tipo>fig|tab):(?P<id>[a-z0-9-]+)\]\]")
_CITA = re.compile(r"\[@[^\]\n]+\]|(?<![\w\[@/.])@[a-z]+\d{4}[a-z]?")
_CLAVE = re.compile(r"@([a-z]+\d{4}[a-z]?)")
_ENTRADA_BIB = re.compile(r"^@\w+\{([^,\s]+),", re.M)
_COMENTARIO = re.compile(r"<!--.*?-->", re.S)

_NOMBRE = {"figura": "Figura", "tabla": "Tabla"}
_PREFIJO = {"figura": "fig", "tabla": "tab"}


class ErrorReporte(ValueError):
    """Violación de una regla del reporte (se convierte en error del build)."""


# ---------------------------------------------------------------- 1. inclusiones
def expandir_inclusiones(texto: str, raiz: Path = RAIZ_BLOG, profundidad: int = 0) -> str:
    """Reemplaza cada línea --8<-- "ruta" por el contenido del archivo (ruta relativa a `raiz`)."""
    if profundidad > PROFUNDIDAD_MAX_INCLUSION:
        raise ErrorReporte("demasiados niveles de inclusión (¿una sección se incluye a sí misma?)")

    def _incluir(m: re.Match) -> str:
        ruta = raiz / m.group("ruta")
        if not ruta.is_file():
            raise ErrorReporte(f"no existe el archivo incluido: {m.group('ruta')}")
        return expandir_inclusiones(ruta.read_text(encoding="utf-8"), raiz, profundidad + 1)

    return _INCLUSION.sub(_incluir, texto)


# ---------------------------------------------------------------- 2. figuras y tablas
def _leer_bloque(tipo: str, ident: str, cuerpo: str, raiz: Path) -> tuple[str, str, str]:
    """Separa Título, Fuente y contenido de un bloque; valida que existan."""
    titulo = fuente = None
    contenido = []
    for linea in cuerpo.splitlines():
        if titulo is None and linea.startswith("Título:"):
            titulo = linea.removeprefix("Título:").strip()
        elif fuente is None and linea.startswith("Fuente:"):
            fuente = linea.removeprefix("Fuente:").strip()
        elif linea.startswith("incluir:"):
            ruta = raiz / "docs" / linea.removeprefix("incluir:").strip()
            if not ruta.is_file():
                raise ErrorReporte(f"{tipo} '{ident}': no existe el archivo a incluir {ruta.name}")
            # markdown="0": el HTML incluido (p. ej. Plotly, con sangría y JS) no se procesa como Markdown; si no,
            # las líneas con sangría se vuelven un bloque de código y la figura sale en blanco.
            contenido.append(f'<div class="apa-html" markdown="0">\n{ruta.read_text(encoding="utf-8")}\n</div>')
        else:
            contenido.append(linea)
    if not titulo:
        raise ErrorReporte(f"{tipo} '{ident}' sin 'Título:' (APA exige título)")
    if not fuente:
        raise ErrorReporte(f"{tipo} '{ident}' sin 'Fuente:' (el enunciado exige la fuente)")
    return titulo, fuente, "\n".join(contenido).strip()


def numerar_bloques(texto: str, raiz: Path = RAIZ_BLOG) -> tuple[str, dict[str, int]]:
    """Convierte los bloques ::: figura/tabla en HTML con formato APA y devuelve {'fig:id': N, …}."""
    numeros: dict[str, int] = {}
    contador = {"figura": 0, "tabla": 0}

    def _formatear(m: re.Match) -> str:
        tipo, ident = m.group("tipo"), m.group("id")
        clave = f"{_PREFIJO[tipo]}:{ident}"
        if clave in numeros:
            raise ErrorReporte(f"id repetido: {clave}")
        titulo, fuente, contenido = _leer_bloque(tipo, ident, m.group("cuerpo"), raiz)
        contador[tipo] += 1
        numeros[clave] = contador[tipo]
        etiqueta = "figure" if tipo == "figura" else "div"
        return (
            f'<{etiqueta} class="apa-{tipo}" id="{_PREFIJO[tipo]}-{ident}" markdown="1">\n'
            f'<p class="apa-numero"><strong>{_NOMBRE[tipo]} {contador[tipo]}</strong></p>\n'
            f'<p class="apa-titulo" markdown="span"><em>{titulo}</em></p>\n\n'
            f"{contenido}\n\n"
            f'<p class="apa-nota" markdown="span"><em>Nota.</em> {fuente}</p>\n'
            f"</{etiqueta}>"
        )

    return _BLOQUE.sub(_formatear, texto), numeros


def resolver_referencias(texto: str, numeros: dict[str, int]) -> str:
    """Cambia [[fig:id]] por "[Figura N](#fig-id)"; exige que exista y que toda figura/tabla se cite."""
    citadas = set()

    def _enlace(m: re.Match) -> str:
        clave = f"{m.group('tipo')}:{m.group('id')}"
        if clave not in numeros:
            raise ErrorReporte(f"referencia a un id que no existe: [[{clave}]]")
        citadas.add(clave)
        nombre = "Figura" if m.group("tipo") == "fig" else "Tabla"
        return f"[{nombre} {numeros[clave]}](#{m.group('tipo')}-{m.group('id')})"

    texto = _REFERENCIA.sub(_enlace, texto)
    huerfanas = sorted(set(numeros) - citadas)
    if huerfanas:
        raise ErrorReporte(f"figuras/tablas que nunca se citan en el texto: {', '.join(huerfanas)}")
    return texto


# ---------------------------------------------------------------- 3. citas APA 7
def claves_bib(bib: Path = BIB) -> set[str]:
    return set(_ENTRADA_BIB.findall(bib.read_text(encoding="utf-8")))


def _pandoc(texto: str, formato: str, suprimir_bibliografia: bool, bib: Path, csl: Path) -> str:
    args = ["--citeproc", f"--bibliography={bib}", f"--csl={csl}", "--metadata=lang:es-ES", "--wrap=none"]
    if suprimir_bibliografia:
        args.append("--metadata=suppress-bibliography:true")
    salida = pypandoc.convert_text(texto, formato, format="markdown", extra_args=args)
    return salida.replace("\r\n", "\n")  # en Windows pandoc devuelve CRLF


def resolver_citas(texto: str, bib: Path = BIB, csl: Path = CSL) -> str:
    """Reemplaza las citas por su forma APA 7 en el texto y escribe la bibliografía en \\bibliografia."""
    citas = list(dict.fromkeys(_CITA.findall(texto)))  # únicas, en orden de aparición
    if not citas:
        return texto.replace(MARCA_BIBLIOGRAFIA, "")
    claves = list(dict.fromkeys(k for c in citas for k in _CLAVE.findall(c)))
    faltan = [k for k in claves if k not in claves_bib(bib)]
    if faltan:
        raise ErrorReporte(f"citas que no están en references.bib: {', '.join(faltan)}")
    if MARCA_BIBLIOGRAFIA not in texto:
        raise ErrorReporte(f"hay citas pero falta la marca {MARCA_BIBLIOGRAFIA} para la lista de referencias")

    # Una sola llamada a pandoc para todas las citas: cada una en su propio párrafo.
    renderizadas = _pandoc("\n\n".join(citas), "plain", True, bib, csl).strip().split("\n\n")
    if len(renderizadas) != len(citas):
        raise ErrorReporte("pandoc devolvió un número inesperado de citas")
    reemplazo = dict(zip(citas, (r.strip() for r in renderizadas)))
    texto = _CITA.sub(lambda m: reemplazo[m.group(0)], texto)

    nocite = ", ".join(f"@{k}" for k in claves)
    lista = _pandoc(f'---\nnocite: "{nocite}"\n---\n', "html", False, bib, csl).strip()
    return texto.replace(MARCA_BIBLIOGRAFIA, f'<div class="apa-referencias">\n{lista}\n</div>')


# ---------------------------------------------------------------- orquestación
def procesar(texto: str, raiz: Path = RAIZ_BLOG, bib: Path = BIB, csl: Path = CSL) -> str:
    """Aplica, en orden: inclusiones → notas de autor → figuras/tablas → referencias cruzadas → citas."""
    texto = expandir_inclusiones(texto, raiz)
    # Los comentarios HTML son notas para los autores: no se publican y sus citas no cuentan como citadas.
    texto = _COMENTARIO.sub("", texto)
    texto, numeros = numerar_bloques(texto, raiz)
    texto = resolver_referencias(texto, numeros)
    return resolver_citas(texto, bib, csl)


def on_page_markdown(markdown, page, config, files):  # noqa: ARG001 - firma del hook de MkDocs
    try:
        return procesar(markdown)
    except ErrorReporte as exc:
        raise PluginError(f"{page.file.src_path}: {exc}") from exc
