# %% [markdown]
# # B9 — pruebas del notebook `02_pd` (núcleo §5.8, B-pd §3.7)
#
# Rápidas: los moldes y el comando de las celdas de texto coinciden con `configs/pd/demo.yaml`, la primera celda de
# código es la de parámetros, las secciones están en el orden de §3.7 y el texto no trae cifras escritas a mano.
# Lentas (`-m lento`): el notebook corre de punta a punta en < 2 min sin tocar lo versionado y sin exponer la ruta
# temporal, y la celda de parámetros cambia la arquitectura sin tocar código (con una sola semilla).

# %%
import ast  # lectura de los moldes y de la celda de parámetros sin ejecutarlos
import re  # bloques de código y cifras en el texto
import shlex  # separación del comando como la shell
import subprocess  # git check-ignore
import tempfile  # carpeta temporal del sistema (no debe aparecer en el .ipynb)
import time  # duración del notebook
from pathlib import Path  # rutas

import pytest  # marcadores

from src.common.cli import build_parser  # parser de run.py
from src.common.config import apply_overrides, load_config  # cargador del CLI
from src.common.notebooks import config_notebook, ejecutar  # cargador y ejecución del notebook
from src.pd.tablas import comando_terminal, moldes_de  # moldes y comando esperados
from src.viz import figs_pd  # figuras de la PD (B7)

RAIZ = Path(__file__).resolve().parents[1]  # raíz del repositorio
FUENTE = RAIZ / "notebooks" / "02_pd.py"  # fuente percent del notebook
TIEMPO_MAX_S = 120  # núcleo §5.8: < 2 min en CPU
CARPETAS_PROTEGIDAS = ("data", "models", "results", "blog")  # el notebook no las escribe
CARPETA_EJEMPLO = {"salida": "results/data/pd2_demo", "modelos": "models/pd2_demo"}  # del comando de ejemplo
SET_UNA_SEMILLA = '{"mlp.capas": [8], "semillas.n": 1}'  # prueba 14: otra arquitectura y una semilla
SECCIONES = ("1", "1.0", "1.1", "1.2", "1.3", "2", "2.1", "2.2", "2.3", "3", "3.1", "3.2", "3.3", "3.4", "3.5",
             "4", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6")  # B-pd §3.7, en orden
ENTRADAS = ("data/processed/loans_pd.parquet", "models/preprocesador_pd.json", "results/data/pd/summary.json",
            "results/data/pd/hpo_ensayos.json", "configs/pd/mlp_final.yaml")  # lo que el notebook necesita


# %% [markdown]
# ## Lectura del `.py` percent

# %%
def _celdas(texto: str) -> list[tuple[str, str]]:
    """[(tipo, cuerpo)] de las celdas percent: 'markdown' (sin el '# ' inicial) o 'codigo' (con la cabecera)."""
    celdas = []  # resultado
    for bloque in re.split(r"^(?=# %%)", texto, flags=re.M)[1:]:  # cada celda empieza con '# %%'
        cabecera, _, cuerpo = bloque.partition("\n")  # primera línea = marcador
        if "[markdown]" in cabecera:  # celda de texto
            lineas = [re.sub(r"^# ?", "", ln) for ln in cuerpo.splitlines()]  # quita el comentario
            celdas.append(("markdown", "\n".join(lineas)))  # texto
        else:  # celda de código
            celdas.append(("codigo", bloque))  # con la cabecera (lleva la etiqueta)
    return celdas  # en orden


# %% [markdown]
# Bloques cercados de un lenguaje dentro del texto.

# %%
def _bloques(md: str, lenguaje: str) -> list[str]:
    """Contenido de los bloques ```<lenguaje> … ``` de un texto markdown."""
    return re.findall(rf"^```{lenguaje}\n(.*?)^```", md, flags=re.M | re.S)  # bloques cercados


# %% [markdown]
# Fuente del notebook.

# %%
def _texto() -> str:
    """Fuente del notebook."""
    return FUENTE.read_text(encoding="utf-8")  # UTF-8


# %% [markdown]
# `CONFIG` y `SET` leídos de la celda de parámetros sin ejecutarla.

# %%
def _parametros() -> dict:
    """{CONFIG, SET} de la celda de parámetros, leídos sin ejecutar."""
    codigo = next(c for t, c in _celdas(_texto()) if t == "codigo")  # primera celda de código
    arbol = ast.parse(codigo)  # sin ejecutar
    return {n.targets[0].id: ast.literal_eval(n.value) for n in arbol.body if isinstance(n, ast.Assign)}  # asign.


# %% [markdown]
# ## 11. Moldes y comando = `demo.yaml`

# %%
def test_moldes_del_texto_coinciden_con_demo_yaml():
    """Los tres diccionarios de texto (comunes, logística, MLP) = moldes_de(demo.yaml), claves y valores."""
    md = "\n".join(c for t, c in _celdas(_texto()) if t == "markdown")  # todo el texto
    moldes = [ast.literal_eval(ast.parse(b, mode="eval").body) for b in _bloques(md, "python")]  # sin comentarios
    esperado = moldes_de(load_config(RAIZ / _parametros()["CONFIG"]))  # de la config real
    assert moldes == [esperado["comunes"], esperado["logistica"], esperado["mlp"]]  # mismo orden y valores
    tipos = {clave: type(v) for m in esperado.values() for clave, v in m.items()}  # clave → tipo esperado
    for molde in moldes:  # tipos exactos (1e-3 float, listas)
        for clave, valor in molde.items():  # cada clave
            assert type(valor) is tipos[clave], clave  # mismo tipo que en demo.yaml


# %% [markdown]
# Cada clave de los moldes lleva su comentario de una línea.

# %%
def test_cada_clave_del_molde_tiene_su_comentario():
    """Un comentario de una línea por clave (para saber qué hace al copiarla al SET)."""
    md = "\n".join(c for t, c in _celdas(_texto()) if t == "markdown")  # texto
    for bloque in _bloques(md, "python"):  # cada molde
        for linea in bloque.splitlines():  # cada línea
            if linea.strip().startswith('"'):  # línea de una clave
                assert re.search(r",\s+# \S", linea), linea  # valor, coma y comentario


# %% [markdown]
# El comando de texto sale de `comando_terminal` y el CLI lo entiende igual que el notebook.

# %%
def test_comando_del_texto_es_el_de_comando_terminal(monkeypatch):
    """La primera línea = comando_terminal(CONFIG, {}); cada línea da por el CLI la misma config que el notebook."""
    monkeypatch.chdir(RAIZ)  # rutas relativas
    md = "\n".join(c for t, c in _celdas(_texto()) if t == "markdown")  # texto
    (bloque,) = _bloques(md, "bash")  # un solo bloque de terminal
    lineas = [ln for ln in bloque.splitlines() if ln.strip()]  # comandos
    config = _parametros()["CONFIG"]  # config de la celda de parámetros
    assert lineas[0] == comando_terminal(config, {})  # comando base
    ejemplos = [shlex.split(ln) for ln in lineas[1:]]  # modificaciones de ejemplo
    assert ejemplos and all("salida=" + CARPETA_EJEMPLO["salida"] in e  # otra carpeta de resultados (B-pd §3.5)
                            and "modelos=" + CARPETA_EJEMPLO["modelos"] in e for e in ejemplos)  # y de modelos
    for linea in lineas:  # ida y vuelta
        args = build_parser().parse_args(shlex.split(linea)[2:])  # parser del CLI
        assert args.config == config  # misma config base
        cfg_cli = apply_overrides(load_config(args.config), args.overrides)  # config del CLI
        claves = [o.partition("=")[0] for o in args.overrides]  # claves de cada --set
        cambios = {k: _hoja(cfg_cli, k) for k in claves}  # el mismo cambio escrito como SET
        assert cfg_cli == config_notebook(config, cambios, imprimir=False)  # misma config por los dos caminos
        assert comando_terminal(config, cambios) == linea  # el texto es el que arma comando_terminal


# %% [markdown]
# Valor de una config en una clave con punto.

# %%
def _hoja(cfg: dict, clave: str):
    """Valor de ``cfg`` en la clave con punto ``clave``."""
    nodo = cfg  # raíz
    for parte in clave.split("."):  # baja
        nodo = nodo[parte]  # nivel
    return nodo  # hoja


# %% [markdown]
# Las carpetas del comando de ejemplo deben estar ignoradas por git: pegar el comando no ensucia el repositorio.

# %%
@pytest.mark.parametrize("clave", sorted(CARPETA_EJEMPLO))  # salida y modelos
def test_carpetas_del_ejemplo_ignoradas_por_git(clave):
    """`git check-ignore` reconoce la carpeta del ejemplo (results/data/*_demo/, models/*_demo*)."""
    ruta = CARPETA_EJEMPLO[clave] + "/x.json"  # un archivo que escribiría la corrida
    r = subprocess.run(["git", "check-ignore", "-q", ruta], cwd=RAIZ, check=False)  # 0 = ignorado
    assert r.returncode == 0, ruta  # ignorada


# %% [markdown]
# ## Figuras citadas
#
# Cada figura del blog que el notebook muestra existe en `blog/docs/figures/`, y las de la demo son exactamente las
# que dibuja `figs_pd.generar` sin `hpo_ensayos.json` (todas menos `pd-busqueda-mlp.png`).

# %%
def test_figuras_del_blog_citadas_existen():
    """Cada FIGURAS_BLOG / "…" del notebook es un archivo de blog/docs/figures/."""
    citadas = re.findall(r'FIGURAS_BLOG / "([^"]+)"', _texto())  # nombres citados
    assert "pd-calibracion.json" in citadas and len(citadas) >= 10  # figuras y JSON de la calibración
    faltan = [n for n in citadas if not (RAIZ / figs_pd.SALIDA / n).is_file()]  # no existen
    assert not faltan, faltan  # todas existen


# %% [markdown]
# `FIGURAS_DEMO` se lee del `.py` sin ejecutarlo y se compara con `figs_pd.ARCHIVOS`.

# %%
def test_figuras_demo_son_las_de_generar_sin_hpo():
    """FIGURAS_DEMO = PNG de figs_pd.ARCHIVOS menos pd-busqueda-mlp.png (sin repetir)."""
    asignacion = re.search(r"^FIGURAS_DEMO = (\(.*?\))", _texto(), flags=re.M | re.S).group(1)  # tupla
    demo = ast.literal_eval(re.sub(r"#[^\n]*", "", asignacion))  # sin comentarios
    sin_hpo = {n for n in figs_pd.ARCHIVOS if n.endswith(".png")} - {figs_pd.FIGURA_BUSQUEDA_MLP}  # de generar
    assert len(demo) == len(set(demo)) and set(demo) == sin_hpo  # exactamente esas


# %% [markdown]
# ## 12. Orden de las celdas

# %%
def test_primera_celda_de_codigo_es_la_de_parametros():
    """Etiqueta parameters, CONFIG de la demo y SET vacío; los moldes van antes, solo en texto."""
    celdas = _celdas(_texto())  # celdas
    primera = next(i for i, (t, _) in enumerate(celdas) if t == "codigo")  # índice
    assert celdas[primera][1].startswith('# %% tags=["parameters"]')  # etiqueta
    assert _parametros() == {"CONFIG": "configs/pd/demo.yaml", "SET": {}}  # solo esas dos
    antes = "\n".join(c for _, c in celdas[:primera])  # texto previo
    assert len(_bloques(antes, "python")) == 3 and len(_bloques(antes, "bash")) == 1  # moldes y comando antes
    despues = "\n".join(c for t, c in celdas[primera + 1:] if t == "markdown")  # texto posterior
    assert not _bloques(despues, "python") and not _bloques(despues, "bash")  # ninguno después


# %% [markdown]
# Títulos numerados en el orden de B-pd §3.7.

# %%
def test_secciones_en_el_orden_de_la_spec():
    """Títulos '## N.' y '### N.M' en el orden de B-pd §3.7; 1.0 justo antes de la celda de parámetros."""
    celdas = _celdas(_texto())  # celdas
    titulos = [m.group(1) for t, c in celdas if t == "markdown"  # textos
               for m in re.finditer(r"^#{2,3} (\d+(?:\.\d+)?)\.? ", c, flags=re.M)]  # números
    assert tuple(titulos) == SECCIONES  # exactamente esos, en orden
    primera = next(i for i, (t, _) in enumerate(celdas) if t == "codigo")  # celda de parámetros
    assert re.search(r"^### 1\.0 ", celdas[primera - 1][1], flags=re.M)  # 1.0 inmediatamente antes


# %% [markdown]
# El texto no trae cifras: todas salen impresas de los JSON.

# %%
def test_texto_sin_cifras_escritas_a_mano():
    """Fuera de títulos, bloques, código en línea y referencias (§, enlaces, secciones), no hay números decimales."""
    md = "\n".join(c for t, c in _celdas(_texto()) if t == "markdown")  # texto
    md = re.sub(r"^```.*?^```", "", md, flags=re.M | re.S)  # sin bloques
    md = re.sub(r"^#+ .*$", "", md, flags=re.M)  # sin títulos
    md = re.sub(r"`[^`]*`|\]\([^)]*\)|§\s?\d+(\.\d+)*", "", md)  # sin código, destinos de enlaces ni §
    md = re.sub(r"sección \d+\.\d+", "", md)  # sin referencias a otras secciones del notebook
    assert not re.findall(r"\d+[.,]\d+", md)  # ninguna cifra decimal


# %% [markdown]
# Cada sección enlaza a una sección del blog que existe.

# %%
def test_cada_seccion_enlaza_al_blog():
    """Cada sección 1–4 remite a una sección del blog que existe."""
    md = "\n".join(c for t, c in _celdas(_texto()) if t == "markdown")  # texto
    for parte in re.split(r"^## (?=\d)", md, flags=re.M)[1:]:  # cada sección numerada
        enlaces = re.findall(r"\]\(\.\./(blog/secciones/0\d-[\w-]+\.md)\)", parte)  # enlaces al blog
        assert enlaces, parte.splitlines()[0]  # al menos uno
        for ruta in enlaces:  # cada uno
            assert (RAIZ / ruta).is_file(), ruta  # existe


# %% [markdown]
# ## 13–14. Ejecución (lentas)

# %%
def _huellas(raiz: Path) -> dict:
    """(tamaño, mtime) de cada archivo de las carpetas protegidas."""
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for c in CARPETAS_PROTEGIDAS  # cada carpeta
            for p in (raiz / c).rglob("*") if p.is_file()}  # cada archivo


# %% [markdown]
# Ejecuta una copia del notebook fuera del repo y comprueba que lo versionado no cambie.

# %%
def _correr_copia(tmp_path: Path, monkeypatch, texto: str) -> tuple[str, float]:
    """Ejecuta una copia del notebook con ``texto`` y devuelve (texto del .ipynb, segundos)."""
    faltan = [r for r in ENTRADAS if not (RAIZ / r).is_file()]  # clon sin resultados
    if faltan:  # no se puede correr
        pytest.skip("faltan " + ", ".join(faltan))  # se omite
    copia = tmp_path / FUENTE.name  # fuera del repo
    copia.write_text(texto, encoding="utf-8")  # copia
    monkeypatch.chdir(RAIZ)  # el notebook corre desde la raíz
    antes = _huellas(RAIZ)  # huellas previas
    inicio = time.perf_counter()  # reloj
    destino = ejecutar(copia, RAIZ)  # .py → .ipynb ejecutado (error si una celda falla)
    segundos = time.perf_counter() - inicio  # duración
    assert _huellas(RAIZ) == antes  # nada versionado cambió
    return destino.read_text(encoding="utf-8"), segundos  # salida


# %% [markdown]
# El `.ipynb` no expone la carpeta temporal del sistema.

# %%
def _sin_ruta_temporal(ipynb: str) -> None:
    """El .ipynb no contiene la ruta de la carpeta temporal del sistema en ninguna forma."""
    temporal = Path(tempfile.gettempdir()).resolve()  # p. ej. /tmp
    for forma in (str(temporal), temporal.as_posix(), str(temporal).replace("\\", "\\\\")):  # Linux y Windows
        assert forma not in ipynb  # no aparece


# %% [markdown]
# Prueba 13: el notebook tal cual.

# %%
@pytest.mark.lento  # entrena la demo
def test_notebook_02_corre_sin_tocar_lo_versionado(tmp_path, monkeypatch):
    """Punta a punta en < 2 min; data/, models/, results/ y blog/ intactos; sin la ruta temporal."""
    ipynb, segundos = _correr_copia(tmp_path, monkeypatch, _texto())  # notebook tal cual
    print(f"notebook 02_pd: {segundos:.1f} s")  # para el informe (-s)
    assert segundos < TIEMPO_MAX_S  # núcleo §5.8
    _sin_ruta_temporal(ipynb)  # no se expone
    assert ipynb.count('"image/png"') == 6 + 2 + 2 + 5  # demo, datos, búsqueda y modelo final


# %% [markdown]
# Prueba 14: otra arquitectura y una sola semilla desde la celda de parámetros.

# %%
@pytest.mark.lento  # entrena la demo
def test_set_cambia_la_arquitectura_con_una_semilla(tmp_path, monkeypatch):
    """SET = {mlp.capas: [8], semillas.n: 1} corre sin error y la tabla muestra la arquitectura nueva."""
    texto = _texto().replace("SET = {}", f"SET = {SET_UNA_SEMILLA}", 1)  # celda de parámetros reescrita
    assert texto != _texto()  # el reemplazo ocurrió
    ipynb, segundos = _correr_copia(tmp_path, monkeypatch, texto)  # ejecución
    assert segundos < TIEMPO_MAX_S  # también en < 2 min
    assert "capas [8]" in ipynb and "semilla mediana 0 de 1" in ipynb  # describir_mlp de esta corrida
    assert "búsqueda: no" in ipynb  # fuera de la búsqueda
    _sin_ruta_temporal(ipynb)  # no se expone
