"""Pruebas del hook del reporte (blog/hooks/reporte.py; Spec §8, T0.7).

Cada regla del enunciado se prueba en las dos direcciones: el caso válido pasa y el caso que la viola FALLA
(figura sin fuente, figura huérfana, referencia inexistente, id repetido, cita fuera de references.bib).
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
BLOG = RAIZ / "blog"

_spec = importlib.util.spec_from_file_location("reporte", BLOG / "hooks" / "reporte.py")
reporte = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reporte)
ErrorReporte = reporte.ErrorReporte

FIGURA = """::: figura mapa
Título: Un mapa
Fuente: Elaboración propia.
![alt](figures/x.png)
:::"""


# ---------------------------------------------------------------- figuras, tablas y referencias cruzadas
def test_html_incluido_no_lo_procesa_markdown(tmp_path):
    # Los fragmentos de Plotly traen líneas con sangría y JS: dentro de la figura (markdown="1") Markdown los
    # convertía en bloque de código y la figura salía en blanco. El HTML incluido debe llegar intacto.
    import markdown

    (tmp_path / "docs" / "figures").mkdir(parents=True)
    fragmento = '<div id="g"></div>\n        <script>\n        var a_b_c = 1 * 2 * 3;\n        </script>'
    (tmp_path / "docs" / "figures" / "g.html").write_text(fragmento, encoding="utf-8")
    texto = "Ver [[fig:g]].\n\n::: figura g\nTítulo: G\nFuente: F.\nincluir: figures/g.html\n:::"
    salida, _ = reporte.numerar_bloques(texto, raiz=tmp_path)
    html = markdown.markdown(salida, extensions=["md_in_html"])
    assert "<pre>" not in html and "<code>" not in html and "&lt;script" not in html
    assert "var a_b_c = 1 * 2 * 3;" in html


def test_numera_figuras_y_tablas_por_separado_y_en_orden():
    texto = (
        "Ver [[fig:a]], [[tab:t]] y [[fig:b]].\n\n"
        + FIGURA.replace("mapa", "a") + "\n\n"
        + "::: tabla t\nTítulo: T\nFuente: F.\n| x |\n|---|\n| 1 |\n:::\n\n"
        + FIGURA.replace("mapa", "b")
    )
    salida, numeros = reporte.numerar_bloques(texto)
    assert numeros == {"fig:a": 1, "tab:t": 1, "fig:b": 2}
    salida = reporte.resolver_referencias(salida, numeros)
    assert "[Figura 1](#fig-a)" in salida and "[Figura 2](#fig-b)" in salida and "[Tabla 1](#tab-t)" in salida


def test_formato_apa_numero_titulo_y_nota():
    salida, _ = reporte.numerar_bloques(FIGURA)
    assert '<strong>Figura 1</strong>' in salida
    assert "<em>Un mapa</em>" in salida
    assert "<em>Nota.</em> Elaboración propia." in salida
    assert 'id="fig-mapa"' in salida


@pytest.mark.parametrize("quitar, mensaje", [("Título: Un mapa\n", "Título"), ("Fuente: Elaboración propia.\n", "Fuente")])
def test_figura_sin_titulo_o_sin_fuente_falla(quitar, mensaje):
    with pytest.raises(ErrorReporte, match=mensaje):
        reporte.numerar_bloques(FIGURA.replace(quitar, ""))


def test_id_repetido_falla():
    with pytest.raises(ErrorReporte, match="repetido"):
        reporte.numerar_bloques(FIGURA + "\n\n" + FIGURA)


def test_figura_huerfana_falla():
    salida, numeros = reporte.numerar_bloques(FIGURA)
    with pytest.raises(ErrorReporte, match="nunca se citan"):
        reporte.resolver_referencias(salida, numeros)


def test_referencia_inexistente_falla():
    with pytest.raises(ErrorReporte, match="no existe"):
        reporte.resolver_referencias("Ver [[fig:fantasma]].", {})


# ---------------------------------------------------------------- inclusiones
def test_inclusion_y_archivo_inexistente(tmp_path):
    (tmp_path / "sec.md").write_text("contenido incluido", encoding="utf-8")
    assert reporte.expandir_inclusiones('a\n--8<-- "sec.md"\nb', tmp_path) == "a\ncontenido incluido\nb"
    with pytest.raises(ErrorReporte, match="no existe"):
        reporte.expandir_inclusiones('--8<-- "no.md"', tmp_path)


def test_inclusion_circular_falla(tmp_path):
    (tmp_path / "a.md").write_text('--8<-- "a.md"', encoding="utf-8")
    with pytest.raises(ErrorReporte, match="niveles"):
        reporte.expandir_inclusiones('--8<-- "a.md"', tmp_path)


# ---------------------------------------------------------------- citas APA 7
def test_citas_parentetica_narrativa_y_multiple():
    texto = "Uno [@kennedy1995]. Según @storn1997, dos. Tres [@clerc2002; @shi1998].\n\n\\bibliografia"
    salida = reporte.resolver_citas(texto)
    assert "(Kennedy & Eberhart, 1995)" in salida
    assert "Storn & Price (1997)" in salida
    assert "(Clerc & Kennedy, 2002; Shi & Eberhart, 1998)" in salida
    listadas = re.findall(r'id="ref-([^"]+)"', salida)
    assert sorted(listadas) == ["clerc2002", "kennedy1995", "shi1998", "storn1997"]  # solo las citadas
    assert "https://doi.org/10.1109/ICNN.1995.488968" in salida


def test_cita_fuera_de_la_bibliografia_falla():
    with pytest.raises(ErrorReporte, match="no están en references.bib"):
        reporte.resolver_citas("Ver [@inventado2024].\n\\bibliografia")


def test_citas_sin_marca_de_bibliografia_falla():
    with pytest.raises(ErrorReporte, match="bibliografia"):
        reporte.resolver_citas("Ver [@kennedy1995].")


def test_correos_y_comentarios_no_son_citas():
    texto = "Contacto: alguien@unal.edu.co\n<!-- nota para autores: @storn1997 -->\n\\bibliografia"
    assert "storn" not in reporte.procesar(texto)


# ---------------------------------------------------------------- bibliografía y build completo
def test_toda_entrada_de_la_bibliografia_tiene_doi_y_clave_apellido_anio():
    bib = (BLOG / "references.bib").read_text(encoding="utf-8")
    entradas = re.split(r"\n(?=@)", bib[bib.index("@"):])
    for entrada in entradas:
        clave = re.match(r"@\w+\{([^,]+),", entrada).group(1)
        assert re.fullmatch(r"[a-z]+\d{4}[a-z]?", clave), clave
        assert re.search(r"^\s*doi\s*=", entrada, re.M), f"{clave} sin DOI verificable"


def test_build_estricto_del_blog(tmp_path):
    proceso = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-f", str(BLOG / "mkdocs.yml"), "-d", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert proceso.returncode == 0, proceso.stderr[-2000:]
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Figura 1</strong>" in html and "Tabla 1</strong>" in html
    assert 'class="apa-referencias"' in html
