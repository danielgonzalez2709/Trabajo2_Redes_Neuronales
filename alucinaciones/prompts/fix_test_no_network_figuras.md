# Prompts literales — corrección de `tests/test_no_network.py` (figuras de la Parte D reescritas por la suite)

- **Fecha:** 25-oct-2026 · **Orquestador:** sesión de Claude Code (Claude Opus 5.5) · **Rama:** `dev/juanjose`
- Tarea fuera del PLAN; el archivo es de la Parte E y se cambia con permiso de Daniel.

## Prompt a la sesión principal (literal)

```text
Tarea fuera del PLAN: corregir una prueba de la Parte E con permiso de Daniel (dueño del archivo). Rama dev/juanjose; el commit de C1 ya está hecho y no hay cambios pendientes. Windows: .venv/Scripts/python.

PROBLEMA: al correr pytest (incluso -m "not lento") se reescriben las 7 figuras versionadas blog/docs/figures/lgd-*.png de la Parte D. Causa: en tests/test_no_network.py, test_comandos_de_run_py_corren_sin_red[figures] solo hace monkeypatch de figs_datos.SALIDA a tmp_path; pero src/common/cli.py::_figures también llama a figs_lgd.generar(), cuyo destino por defecto es raiz / figs_lgd.SALIDA = blog/docs/figures.

CORRECCIÓN (solo en tests/test_no_network.py; no toques src/, run.py, cli.py ni otras pruebas):
1. En la rama "figures" de esa prueba, en vez de parchear solo figs_datos, recorrer TODOS los módulos src/viz/figs_*.py (por ejemplo con pkgutil.iter_modules sobre src/viz + importlib) y hacer monkeypatch.setattr(modulo, "SALIDA", tmp_path) en cada uno que tenga SALIDA. Así quedan cubiertos los módulos de figuras que se agreguen después (figs_pd de B, figs_scorecard de C).
2. Añadir a esa misma rama una verificación de regresión: calcular el SHA-256 de cada archivo de RAIZ / "blog/docs/figures" antes de cli.main(argv) y comprobar después que ninguno cambió ni apareció uno nuevo; el mensaje de fallo debe nombrar los archivos alterados.
3. Comprobar que la prueba sigue verificando lo que verificaba (cli.main(argv) == 0 con la red bloqueada) y que efectivamente se generan las figuras en tmp_path.

VERIFICACIÓN OBLIGATORIA:
- Antes de cambiar nada: confirma con git status que no hay cambios. Corre .venv/Scripts/python -m pytest -q "tests/test_no_network.py" y muestra que, SIN la corrección, git status reporta las lgd-*.png modificadas; luego descártalas con git checkout -- blog/docs/figures/.
- Después de la corrección: corre .venv/Scripts/python -m pytest -q -m "not lento" completo, pega la salida y muestra git status: no debe aparecer NINGÚN archivo de blog/docs/figures/ modificado.

FLUJO: implementador → revisor-calidad (no hace falta verificador-matematico ni revisor-spec: no hay fórmulas ni contratos). Guarda este prompt literal y el de cada subagente en alucinaciones/prompts/fix_test_no_network_figuras.md antes de enviarlos. Si un subagente comete un error, regístralo como candidato en alucinaciones/registro.md (siguiente número libre).

CIERRE: en docs/MEMORIA.md, reemplaza el pendiente 9 ("C: la suite sin lentas reescribe blog/docs/figures/lgd-*.png...") por una línea de bitácora que diga que se corrigió en tests/test_no_network.py con permiso de Daniel (sin mencionar quién lo hizo, va en el commit), y deja la MEMORIA en ≤ 150 líneas.

NO hagas git add ni commit. Al terminar dame: archivos cambiados, salida real de pytest, git status final, veredicto del revisor-calidad y un mensaje de commit sugerido (summary + description) redactado en primera persona implícita, sin decir "Juan José" ni "decisión de", legible para el equipo.
```

Reproducción del orquestador antes de cambiar nada: `git status` mostraba las 7 `lgd-*.png` ya modificadas (restos de
corridas anteriores, fuera del commit de C1); se descartaron con `git checkout -- blog/docs/figures/` y el árbol quedó limpio.
`pytest -q tests/test_no_network.py` → `48 passed in 51.96s`, y después `git status` mostró de nuevo las 7 `lgd-*.png`
modificadas (las de `figs_datos` no, porque esa sí se parchea). Se descartaron otra vez.

## 1. implementador (literal)

```text
Tarea fuera del PLAN del Trabajo 2 (rama dev/juanjose): corregir tests/test_no_network.py, archivo de la Parte E, con permiso
de Daniel. Windows; Python: .venv/Scripts/python. NO hagas git add ni commit. Toca SOLO tests/test_no_network.py (no src/,
run.py, src/common/cli.py ni otras pruebas). Sin internet.

PROBLEMA (reproducido por el orquestador desde un árbol limpio): `pytest -q tests/test_no_network.py` reescribe las 7
figuras versionadas blog/docs/figures/lgd-*.png de la Parte D. Causa: test_comandos_de_run_py_corren_sin_red[figures] solo
hace monkeypatch de figs_datos.SALIDA a tmp_path, pero src/common/cli.py::_figures también llama a figs_lgd.generar(), cuyo
destino por defecto es raiz / figs_lgd.SALIDA = blog/docs/figures. Lee src/common/cli.py (_figures), src/viz/figs_datos.py y
src/viz/figs_lgd.py (constante SALIDA y generar) para confirmar cómo se resuelve el destino antes de cambiar nada.

CORRECCIÓN, en la rama "figures" de test_comandos_de_run_py_corren_sin_red:
1. En vez de parchear solo figs_datos, recorre TODOS los módulos src/viz/figs_*.py (pkgutil.iter_modules sobre el paquete
   src.viz + importlib.import_module) y haz monkeypatch.setattr(modulo, "SALIDA", tmp_path) en cada uno que tenga el
   atributo SALIDA. Así quedan cubiertos los módulos que se agreguen después (figs_pd de B, figs_scorecard de C). Si un
   figs_*.py no tiene SALIDA, no lo ignores en silencio sin pensarlo: decide si la prueba debe fallar con un mensaje claro
   (módulo de figuras sin destino parcheable) y justifícalo en tu reporte.
2. Verificación de regresión: SHA-256 de cada archivo de RAIZ / "blog/docs/figures" antes de cli.main(argv); después,
   comprobar que ninguno cambió, ninguno desapareció y no apareció uno nuevo; el mensaje de fallo nombra los archivos
   alterados (cambiados/nuevos/borrados).
3. La prueba sigue verificando cli.main(argv) == 0 con la red bloqueada, y además comprueba que se generaron figuras en
   tmp_path (al menos un .png; idealmente que haya figuras de cada módulo parcheado, si se puede saber sin acoplarse a
   nombres concretos — si no, explica qué comprobaste).
Mantén el estilo del archivo (docstrings y comentarios en español, ayudantes privados pequeños si hacen falta) y actualiza el
docstring de la prueba.

PRUEBA DE QUE LA REGRESIÓN SE DETECTA: demuestra que la verificación del punto 2 falla si se rompe la corrección (p. ej.
temporalmente parcheando solo figs_datos como antes) y pega esa salida de fallo; luego restaura tu versión y descarta las
figuras reescritas con `git checkout -- blog/docs/figures/`.

Entrega: salida de `.venv/Scripts/python -m pytest -q tests/test_no_network.py -p no:cacheprovider`, `git status --short`
(no debe aparecer nada de blog/docs/figures/), el diff de tests/test_no_network.py y dudas. No corras la suite completa: la
corre el orquestador al final.
```

Resultado del implementador: `48 passed`; `git status` sin cambios en `blog/docs/figures/`. Con la corrección rota a propósito
(solo `figs_datos`), la nueva verificación falla nombrando las 7 `lgd-*.png` como "cambiados". Decidió que un `figs_*.py`
sin `SALIDA` y un módulo que no deje ningún .png hagan fallar la prueba.

## 2. revisor-calidad (literal)

```text
Revisa la calidad de una corrección fuera del PLAN del Trabajo 2 (rama dev/juanjose, cambio SIN commit): solo
tests/test_no_network.py (archivo de la Parte E, cambiado con permiso de Daniel). No modifiques ningún archivo; sin git add
ni commit. Windows; Python: .venv/Scripts/python.

Contexto: la prueba test_comandos_de_run_py_corren_sin_red[figures] solo parcheaba figs_datos.SALIDA, y src/common/cli.py::
_figures también llama a figs_lgd.generar(), así que cada `pytest` (incluso -m "not lento") reescribía las 7 figuras
versionadas blog/docs/figures/lgd-*.png de la Parte D. La corrección (ver `git diff tests/test_no_network.py`):
- redirige SALIDA de TODOS los src/viz/figs_*.py (pkgutil + importlib) a tmp_path/<módulo>;
- SHA-256 de blog/docs/figures antes y después de cli.main(argv) (también para eda): falla nombrando cambiados/nuevos/borrados;
- sigue exigiendo cli.main(argv) == 0 con la red bloqueada y exige ≥ 1 .png por módulo parcheado;
- un figs_*.py sin SALIDA hace fallar la prueba antes de correr.

Revisa:
1. Que la corrección ataque la causa y que la redirección sea correcta (lee src/common/cli.py::_figures y el generar() /
   SALIDA de src/viz/figs_datos.py y src/viz/figs_lgd.py: ¿`raiz / SALIDA` con SALIDA absoluta da SALIDA en Windows y
   Linux? ¿algún módulo escribe en otro lugar además de SALIDA, p. ej. results/ o una ruta construida sin SALIDA?).
2. Diseño para el equipo: el assert de "≥ 1 .png por módulo" hará fallar la CI si B o C agregan figs_pd.py /
   figs_scorecard.py antes de conectarlo en cli._figures. ¿Es deseable (obliga a conectar) o frena el desarrollo? Igual para
   "módulo sin SALIDA". Da una recomendación concreta.
3. Legibilidad y estilo coherente con el archivo (docstrings/comentarios en español, ayudantes pequeños), nombres, que el
   mensaje de fallo sea útil. ¿La comprobación SHA-256 es robusta (archivos que no son PNG, subcarpetas, .gitkeep)?
4. Que la prueba pueda fallar de verdad: el implementador mostró que con la corrección rota falla nombrando las 7 lgd-*.png;
   si quieres reproducirlo, hazlo sobre una COPIA del archivo fuera del repo o restaura después con
   `git checkout -- blog/docs/figures/` y confirma con `git status --short` que no queda nada de blog/docs/figures/.
5. Ejecuta `.venv/Scripts/python -m pytest -q tests/test_no_network.py -p no:cacheprovider` y después `git status --short`
   (no debe aparecer nada de blog/docs/figures/). No corras la suite completa (la corre el orquestador).

Salida: APROBADO o CAMBIOS REQUERIDOS, con hallazgos priorizados (bloqueante / recomendado) y archivo:línea.
```

Resultado: APROBADO (0 bloqueantes; 4 recomendados menores; mantener las dos comprobaciones y avisar a B y C de la
convención de `src/viz/figs_*.py`).

## 3. implementador — ajustes del revisor (literal, por SendMessage)

```text
El revisor-calidad APROBÓ tu corrección. Aplica estos tres ajustes menores en tests/test_no_network.py (nada más; sin git
add ni commit; no corras la suite completa):
1. Línea ~69: añade al mensaje de fallo cómo restaurar: "restaurar con `git checkout -- blog/docs/figures/`".
2. Línea ~70: `destino.glob("*.png")` → `destino.rglob("*.png")` (un módulo futuro podría guardar en subcarpetas).
3. Líneas ~71-72: en el mensaje, nombra los módulos como `src/viz/<módulo>.py` (igual que el mensaje de la línea ~89).
Deja `import src.viz` como está. Entrega: salida de `.venv/Scripts/python -m pytest -q tests/test_no_network.py
-p no:cacheprovider`, `git status --short` y el diff de las líneas cambiadas.
```

Resultado: ajustes aplicados, `48 passed`. Suite completa sin lentas (orquestador): `926 passed, 2 skipped, 9 deselected`;
`git status` sin nada de `blog/docs/figures/`. Ningún subagente cometió errores que registrar en la cacería.
