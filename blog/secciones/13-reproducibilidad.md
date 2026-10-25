<!-- Responsable: Daniel (E). Enunciado (Anexo del núcleo), Entregables 2 y 3. Cada parte agrega su comando y el tiempo
medido cuando lo conecte. Guía: blog/README.md -->
## 13. Reproducibilidad

Todo el trabajo está en https://github.com/danielgonzalez2709/Trabajo2_Redes_Neuronales. El repositorio guarda los datos
procesados y los modelos entrenados, de modo que nada depende de internet ni de reentrenar.

**Instalación** (Python 3.12, versiones fijas en `requirements.txt`):

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**Comandos** ([[tab:comandos]]). Todos imprimen la configuración resuelta antes de ejecutar y el tiempo al terminar, y
aceptan `--set clave.subclave=valor` para cambiar cualquier parámetro sin editar el código.

::: tabla comandos
Título: Comandos del repositorio y tiempo medido
Fuente: Elaboración propia; tiempos medidos en un portátil con Windows 11 y CPU, octubre de 2026.
| Comando | Qué hace | Tiempo |
|---|---|---|
| `python scripts/preparar_datos.py` | CSV de Kaggle → `data/processed/` (requiere el crudo, ver `data/fuentes.md`) | ≈ 5 s |
| `python run.py eda --config configs/datos.yaml` | Análisis descriptivo, hipótesis y datos después del preprocesamiento | ≈ 5 s |
| `python run.py figures` | Regenera las figuras del reporte desde `results/` | ≈ 3 s |
| `python run.py notebooks` | Convierte y ejecuta los notebooks de `notebooks/` | ≈ 13 s (notebook 01) |
| `python run.py el --config configs/el/base.yaml` | Pérdida esperada de cada crédito de prueba | pendiente |
| `python -m pytest -q` | Pruebas automáticas (también corren en cada *pull request*) | ≈ 40 s |
:::
<!-- Parte B: agregar `run.py pd` (demo y completo), `run.py pd-hpo`. Parte C: `run.py scorecard`, `run.py export`.
Parte D: `run.py lgd` (demo y completo), `run.py lgd-hpo`. Parte E: la app con `npm --prefix app run dev`. -->

**Semillas.** La semilla efectiva de cada corrida es `base_seed + run_id` (`src/common/seeds.py`), y `base_seed` está en
cada archivo de `configs/`. Con la misma semilla y el mismo dispositivo los resultados se repiten.

**Notebooks.** Cada parte tiene un notebook en `notebooks/` que solo llama funciones del repositorio. Su primera celda
(`CONFIG` y `SET`) acepta las mismas claves que `--set`, y se puede modificar en vivo sin escribir sobre los archivos que usan
las demás partes.
