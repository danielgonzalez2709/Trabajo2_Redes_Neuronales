# Prompt literal enviado a `revisor-spec` — revisión de la Parte A (A1–A7, A9) y E10

- **Fecha:** 23-oct-2026 · **Orquestador:** sesión de Claude Code de Daniel (Claude Opus 5.5) · **Rama:** `dev/daniel`

```text
Revisa la Parte A del Trabajo 2 (rama dev/daniel, commits locales) contra el Spec. Repositorio: el directorio de trabajo actual.

Lee SOLO esto como contexto:
- docs/spec/SPEC.md: §0, §3, §4.1, §5.1, §5.2, §5.4, §5.7, §5.8, §6 y §7.1, y del Anexo los apartados "Datos" y "Parte 1"
  (variable objetivo y tareas 1, 2 y 5) y "Entregables" 1–3.
- docs/spec/partes/A-datos.md (completo).

Código y artefactos a revisar:
- src/datos/objetivo.py, cargar.py, particion.py, preparar.py, eda.py, experimento_eda.py; scripts/preparar_datos.py
- src/common/preprocesamiento.py, src/common/notebooks.py, src/common/cli.py (subcomandos eda, figures, notebooks)
- configs/datos.yaml, configs/features.yaml, configs/hipotesis.yaml
- data/processed/loans_pd.parquet, data/processed/resumen_datos.json, models/preprocesador_pd.json, data/fuentes.md
- results/data/eda/*.json, src/viz/figs_datos.py, notebooks/01_datos.py
- blog/secciones/01-introduccion.md, 02-datos.md, 03-descriptivo.md
- tests/ (las pruebas de lo anterior)

Comprueba, citando la sección del Spec en cada hallazgo:
1. Contratos §5.1 y §5.2: columnas y tipos del parquet, campos de resumen_datos.json, firma del Preprocesador, operaciones
   permitidas y ajuste SOLO con train.
2. Fuga (§5.7): ninguna variable de features.yaml es posterior a la originación; verifica cada una contra el diccionario
   (data/raw/credit-risk-dataset/LCDataDictionary.xlsx, hoja LoanStats) y que la cita_diccionario sea textual.
3. Objetivo: mapeo idéntico a la tabla del Anexo; estado desconocido = error.
4. Partición temporal (A §3.5): cortes de la config, sin solapamiento, 2015 excluido y contado.
5. Que toda cifra de las secciones 01–03 del blog coincida con resumen_datos.json o results/data/eda/*.json (recalcúlalas).
6. §5.8: notebook sin lógica propia y con celda de parámetros.
7. Nada añadido que el Spec no pida.
8. Ejecuta: .venv/Scripts/python -m pytest -q y .venv/Scripts/python run.py eda --config configs/datos.yaml --set eda.salida=<carpeta temporal>.

No modifiques ningún archivo. Salida: APROBADO o RECHAZADO + lista de hallazgos (archivo:línea, sección del Spec, qué falta o
sobra, gravedad bloqueante/recomendado).
```

## Ronda 2 (literal, enviada al mismo revisor el 23-oct-2026)

```text
Segunda ronda de la revisión de la Parte A: el commit e9243c0 en dev/daniel aplica tus hallazgos. Vuelve a revisar SOLO lo que señalaste (B1, B2, R1–R9) con el mismo contexto de la primera ronda, más el Spec v0.7 (núcleo §4.1, §5.1, §5.2) y A §3.7. Qué cambió: B1 en blog/secciones/02-datos.md; B2 en src/datos/eda.py (distribucion_antes_despues, distribucion_discretas), results/data/eda/post_preprocesamiento.json, figura post-transformaciones y blog 03; R1 en blog 03 y eda.py; R2 en scripts/preparar_datos.py, src/common/cli.py, src/common/notebooks.py y núcleo §4.1; R3 en src/datos/experimento_eda.py (eda_en_memoria) y notebooks/01_datos.py; R4 en tests/; R5 en src/common/preprocesamiento.py (to_dict) y §5.2; R6 en §5.1; R7 en configs/features.yaml; R8 en blog 03; R9 en configs/datos.yaml y A §3.7. Ejecuta de nuevo pytest y run.py eda con eda.salida en una carpeta temporal. No modifiques archivos. Salida: APROBADO o RECHAZADO + hallazgos que sigan abiertos o nuevos.
```
