# Prompt literal — D1b (implementador)

Fecha: 2026-10-25 · Orquestador: sesión de Camila (`dev/camila`) · Spec v0.12

---

Eres el `implementador` de la tarea **D1b** del Trabajo 2 (Parte D — LGD). Repo: `/Users/softwaresolenium/CoursesExercises/Trabajo2_Redes_Neuronales`, rama `dev/camila`. Python: `.venv/bin/python`.

**Reglas duras**
- **CERO escrituras en git.** Al final reporta `git log --oneline -1`.
- TDD: `tests/test_lgd_descriptivo.py` primero, verla fallar, luego implementar.
- Toca solo: `src/lgd/descriptivo.py` (nuevo), `tests/test_lgd_descriptivo.py` (nuevo), un bloque `descriptivo:` en `configs/lgd/base.yaml` si hace falta (no toques `datos:` ni `hereda:`), y escribe `results/data/lgd/descriptivo.json`. Nada de `src/common/`, `src/datos/`, `src/viz/`, `run.py`.
- **Los modelos no dibujan** (núcleo §3, Arquitectura): esta tarea produce **solo datos** en JSON; las figuras son D6 (`src/viz/figs_lgd.py`) y se dibujarán desde este JSON. Deja en el JSON todo lo necesario para dibujarlas sin recalcular (p. ej. conteos de histograma con sus bordes).
- Sin internet. No inventes cifras ni afirmaciones sobre "la forma típica" de la LGD (D §6): todo lo que reportes sale de ejecutar.

## Tarea (D §4)
| ID | Tarea | Depende | Aceptación | + Verificador |
|---|---|---|---|---|
| D1b | Descriptivo de la LGD observada y después del preprocesamiento (§3.2b) | D1 (A6b para la parte posterior al preprocesamiento) | `descriptivo.json`; figuras citadas en el blog | analista-experimentos |

## Spec D §3.2b (literal)
- **Variable objetivo observada (antes de modelar):** histograma de la LGD, masa en 0 y en 1, casos recortados, EAD y LGD por año de originación y por plazo. Es la evidencia para justificar la pérdida y la activación de salida, que el enunciado pide justificar "a partir de la naturaleza de la variable objetivo".
- **Después del preprocesamiento:** las funciones de la Parte A (A §3.8) aplicadas al subconjunto de LGD: efecto de las transformaciones, estabilidad entre splits y relación de cada variable con la LGD.
- Salida: `results/data/lgd/descriptivo.json` + figuras; subsección en el blog 08 y en el notebook 04.

Decisión vigente (D §3.2, [DECIDIDO, Camila, 24-oct]): la distribución distinta entre particiones se acepta y se documenta ("consistente con censura", no causa probada). El descriptivo debe dejarla **medida** (por split y por año).

## Insumos
- `data/processed/loans_lgd.parquet` (D2): `id`, `issue_d`, `split`, 15 variables de `features.yaml` (incluye `term`), campos de la fórmula, `ead`, `lgd`, `lgd_sin_recorte`. 43 236 filas.
- `models/preprocesador_lgd.json` (D2; `Preprocesador.from_dict(load_json(...))`; **no reajustar**).
- `results/data/lgd/resumen_objetivo.json` (D2): conteos de población, recortes, masa en 0 y 1. Reutiliza; no lo dupliques.
- **Parte A (A §3.8)**: `src/datos/eda.py::descriptivo_post(df, pre, objetivo="target", umbral=...)` devuelve los cinco puntos (efecto de transformaciones, estandarización, PSI vs. train, correlaciones, relación con el objetivo por decil). Acepta cualquier columna objetivo numérica: **llámala con `objetivo="lgd"`** en vez de reimplementarla. Si algo de esa función no tiene sentido para un objetivo continuo (p. ej. el nombre `tasa_objetivo_por_split`), documéntalo en el JSON, no la modifiques.

## Qué entregar
1. `src/lgd/descriptivo.py`:
   - `descriptivo_observado(df) -> dict` (función pura): histograma de `lgd` (bordes y conteos, p. ej. 20 bins en [0, 1]) total y por split; masa exacta en 0 y en 1; `lgd_sin_recorte` < 0 y > 1; resumen (n, media, mediana, desv, cuantiles) de `lgd` y de `ead` por split, por año de originación (`issue_d.dt.year`) y por plazo (`term`, limpiando espacios como hace el preprocesador); fracción con LGD = 1 por año.
   - `descriptivo_preprocesado(df, pre) -> dict`: `descriptivo_post(df, pre, objetivo="lgd")`.
   - `generar(cfg, raiz) -> dict` que lee los insumos, arma `{observado, preprocesado, fuente: {hash_particion, hash_preprocesador}}` y escribe `results/data/lgd/descriptivo.json` con `save_json`. Para los hashes, reutiliza las funciones públicas de `src/lgd/experimento.py` si existen (si son privadas, no las importes: calcula con la misma definición y documenta).
   - Punto de entrada `python -m src.lgd.descriptivo [--config configs/lgd/base.yaml] [--set …]` que imprima config y tiempo, igual que `src/lgd/preparar.py`.
2. `tests/test_lgd_descriptivo.py` con un sintético pequeño: histograma suma n; masa en 0/1 exacta; resumen por año y por plazo con valores a mano; que se usa el preprocesador sin reajustar; esquema del JSON. Una prueba `@pytest.mark.lento` sobre los datos reales (n = 43 236; suma de conteos del histograma = n).
3. Ejecuta sobre los datos reales y reporta (leído del JSON): masa en 0 y 1; fracción con LGD = 1 por split y por año; LGD y EAD medias por plazo; las 3 variables con mayor |PSI| entre train y test; las variables con relación monótona con la LGD (si alguna).
4. Suite `-m "not lento"` y tu prueba lenta; archivos tocados; `git log --oneline -1`.

---

# Ronda 2 — mejoras (implementador, mismo agente)

Fecha: 2026-10-25. revisor-spec APROBADO, revisor-calidad APROBADO (4 de 9 mutaciones sobreviven). **CERO escrituras en git.** TDD: cada prueba nueva debe fallar con la mutación correspondiente. Otro subagente revisa D6 (`src/viz/figs_lgd.py`) en paralelo: no lo toques; las cifras de `descriptivo.json` no deben cambiar (solo se añaden notas).

1. M1: histogramas por split con los mismos bordes que el total: `len(c) == n_bins` para cada split y la suma elemento a elemento de los splits == conteos del total.
2. M6: llamar con `umbral=1.01` y comprobar `monotona is False` (el umbral se propaga).
3. M7: parametrizar la prueba de cuadre sobre las 4 claves de `CLAVES_CUADRE` y además `n_lgd_valida`.
4. M8: comprobar `fraccion_lgd_0` por año en el sintético (valores a mano).
5. Mover `_sha256_texto`, `hash_particion`, `hash_preprocesador` de `src/lgd/experimento.py` a un módulo ligero nuevo `src/lgd/procedencia.py` (sin torch/sklearn). `experimento.py` y `descriptivo.py` los importan de ahí (una sola definición). Mantén los nombres públicos; actualiza las pruebas que los importen. Verifica que `import src.lgd.descriptivo` ya no carga torch (`"torch" not in sys.modules`, en un subproceso).
6. Quitar el `json.dumps(to_jsonable(...))` redundante antes de `save_json` (y los imports que queden sin uso).
7. `umbral: float = UMBRAL_MONOTONIA` (importado de `src.datos.eda`) en la firma; llamada directa a `descriptivo_post(..., umbral=umbral)`.
8. Sacar `_histograma(df, bordes, splits) -> dict` para que `descriptivo_observado` se lea como la lista del Spec; comentario en `fraccion_lgd_1_por_anio` ("atajo para D6").
9. `_limpiar_plazo`: docstring que cite `src/common/preprocesamiento.py` `Preprocesador._limpiar_cat` como la limpieza que replica.
10. En `NOTAS_OBJETIVO_CONTINUO` (y por lo tanto en el JSON), añadir: con `n_grupos == 2`, `monotona` es siempre verdadera (ρ = signo de la diferencia; `src/datos/eda.py:74-75`) y no es evidencia (caso `pub_rec`). Añadir también: `descriptivo_post` mide la relación con el objetivo solo para las numéricas; las categóricas `emp_length`, `home_ownership`, `purpose` no están cubiertas (`term` se cubre con `observado.por_plazo`).

Regenera `results/data/lgd/descriptivo.json` con `python -m src.lgd.descriptivo` y confirma que solo cambian las notas (compara con una copia previa en el scratchpad). Suite `-m "not lento"` + `tests/test_lgd_descriptivo.py tests/test_lgd_experimento.py -m lento`; archivos tocados; `git log --oneline -1`.
