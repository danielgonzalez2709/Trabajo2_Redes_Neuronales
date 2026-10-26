<!-- Responsable: Daniel (E). Enunciado (Anexo del núcleo), Entregables 2 y 3. Cada parte agrega su comando y el tiempo
medido cuando lo conecte. Guía: blog/README.md -->
<!-- Tiempos medidos el 26-oct-2026 en una copia del repositorio (sin .git) en una carpeta temporal, con las salidas de el,
scorecard, eda y la demo de LGD apuntadas a carpetas temporales con --set; export, figures, notebooks y preparar_datos
corrieron sobre la copia para no reescribir archivos versionados. Tiempos de las corridas largas: leídos de
results/data/pd/summary.json, results/data/pd/hpo_ensayos.json, results/data/lgd/summary.json y
results/data/lgd/hpo_ensayos.json. -->
## 13. Reproducibilidad

Todo el trabajo está en https://github.com/danielgonzalez2709/Trabajo2_Redes_Neuronales. El repositorio guarda los datos
procesados y los modelos entrenados, de modo que ningún experimento depende de internet ni de reentrenar (instalar el
entorno y descargar el crudo de Kaggle sí lo necesitan).

**Instalación** (Python 3.12, versiones fijas en `requirements.txt`):

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**Comandos** ([[tab:comandos]]). Todos se corren desde la raíz del repositorio con el entorno activado. Los subcomandos de
`run.py` que reciben `--config`, `run.py export` y `scripts/preparar_datos.py` imprimen la configuración resuelta antes de
ejecutar; los de `run.py` imprimen además el tiempo al terminar (`figures` y `notebooks` no reciben configuración, ni
tampoco `pytest`, `mkdocs` y `npm`). Los que reciben `--config` aceptan `--set clave.subclave=valor` para cambiar cualquier
parámetro sin editar el código (por ejemplo `--set mlp.capas=[64,32]`, `--set semillas.n=5` o `--set scorecard.pdo=40`).

::: tabla comandos
Título: Comandos del repositorio y tiempo de cada uno
Fuente: Elaboración propia. "Medido": corrido el 26 de octubre de 2026 en un portátil con Windows 11, Intel Core i5-11400H (12 hilos), 7,7 GB de RAM, Python 3.12.6 y PyTorch 2.14.1 en CPU; la primera cifra es el tiempo que imprime el comando y la segunda, entre paréntesis, el tiempo de reloj del proceso completo (incluye importar las bibliotecas). "Registrado": tiempo guardado por la corrida oficial en el archivo indicado, en la máquina de quien la corrió (Partes B y D), no en la de las mediciones; son tiempos de ajuste de los modelos, no la duración total del comando (no incluyen cargar los datos ni escribir los resultados). En las búsquedas, `tiempo_total_s` es la suma de los ajustes de los ensayos y, en la de la PD, `tiempo_reloj_s` es el tiempo de `estudio.optimize` (`src/pd/hpo.py`), tampoco el del proceso completo.
| Comando | Qué hace | Tiempo | Origen del tiempo |
|---|---|---|---|
| `python scripts/preparar_datos.py` | CSV de Kaggle → 5 de los 6 archivos de `data/processed/` y `models/preprocesador_pd.json` (requiere el crudo, ver `data/fuentes.md`) | 6,4 s (7,6 s) | Medido |
| `python run.py eda --config configs/datos.yaml` | Análisis descriptivo, hipótesis y datos después del preprocesamiento | 4,5 s (7,0 s) | Medido |
| `python run.py pd --config configs/pd/demo.yaml` | Demostración de la PD (< 2 min) | pendiente: `configs/pd/demo.yaml` todavía no existe (tarea B6) | — |
| `python run.py pd --config configs/pd/base.yaml` | Logística (13 valores de $C$) y MLP final con 30 semillas | Ajuste: 60,8 s (logística) + 238,0 s (30 semillas del MLP) | Registrado: `results/data/pd/summary.json` (`logistica.tiempo_total_s`, `mlp.tiempo_semillas_s`) |
| `python run.py pd-hpo --config configs/pd/hpo.yaml` | Búsqueda TPE del MLP de PD (40 ensayos); reescribe `configs/pd/mlp_final.yaml` | Suma de los ajustes: 702,4 s; `estudio.optimize`: 706,8 s | Registrado: `results/data/pd/hpo_ensayos.json` (`tiempo_total_s`, `tiempo_reloj_s`) |
| `python run.py lgd --config configs/lgd/demo.yaml` | Demostración de la LGD: muestra de 20 000 créditos y 3 semillas | 5,3 s (10,2 s) | Medido |
| `python run.py lgd --config configs/lgd/base.yaml` | Regresión lineal y MLP de LGD con 30 semillas | Ajuste: 0,01 s (lineal) + 13,1 s (30 semillas del MLP) | Registrado: `results/data/lgd/summary.json` (`modelos.lineal.tiempo_total_s`, `modelos.mlp.tiempo_semillas_s`) |
| `python run.py lgd-hpo --config configs/lgd/hpo.yaml` | Búsqueda TPE del MLP de LGD (30 ensayos); reescribe `configs/lgd/mlp_final.yaml` | Suma de los ajustes: 32,5 s | Registrado: `results/data/lgd/hpo_ensayos.json` (`tiempo_total_s`) |
| `python run.py scorecard --config configs/scorecard/base.yaml --set scorecard.pdo=40` | Scorecard de la logística | 1,7 s (6,4 s) | Medido |
| `python run.py el --config configs/el/base.yaml` | Pérdida esperada de cada crédito de prueba | 0,9 s (6,1 s) | Medido |
| `python run.py export` | Los dos modelos de PD → `app/public/model/*.json` y casos de paridad | 1,3 s (6,5 s) | Medido |
| `python run.py figures` | Regenera las figuras del reporte desde `results/` (por ahora las de las Partes A, D y E) | 4,7 s (6,7 s) | Medido |
| `python run.py notebooks` | Convierte y ejecuta los notebooks 00, 01, 04 y 05 | 45,5 s (45,7 s) | Medido |
| `npm --prefix app run dev` | Aplicación web local | pendiente (Parte E) | — |
| `mkdocs build --strict -f blog/mkdocs.yml` | Construye este reporte (`mkdocs serve -f blog/mkdocs.yml` lo sirve en local) | 3,8 s de reloj | Medido |
| `python -m pytest -q -m "not lento"` | Pruebas automáticas que corre el CI | 105,0 s (107,6 s) | Medido |
| `python -m pytest -q` | Todas las pruebas, incluidas las lentas | 201,3 s (203,6 s) | Medido |
:::
<!-- Pendientes: Parte B, `configs/pd/demo.yaml` (B6), figs_pd en `run.py figures` y el notebook 02; Parte C, figs_scorecard y
el notebook 03; Parte E, la app con `npm --prefix app run dev`. `jupyter lab notebooks/` abre los notebooks y no tiene tiempo
que medir. -->
<!-- revisar: autor (Andrés y Camila: confirmar en qué máquina y con cuántos hilos corrieron las corridas registradas; la PD
dice hilos_torch = 4 en results/data/pd/summary.json) -->

La corrida completa de la PD y su búsqueda no caben en dos minutos; por eso cada tarea debe llevar una configuración de
demostración que se lance con un solo comando. La de la LGD tardó 5,3 s; la de la PD todavía no existe. Las otras demos de la
sustentación (`eda`, `scorecard` y `el`, según `configs/sustentacion.yaml`) tardaron menos de 10 s de reloj cada una, y
también `export` y `figures`. Los
notebooks tardaron 10,2 s (00, sustentación), 11,7 s (01, datos), 15,4 s (04, LGD) y 8,2 s (05, pérdida esperada).

**Comprobación en una copia.** Al correr `python scripts/preparar_datos.py` sobre una copia del repositorio con el crudo de
Kaggle, los cinco archivos que escribe en `data/processed/` (`loans_pd.parquet`, `montos.parquet`,
`sensibilidad_lc.parquet`, `resumen_datos.json` y `diccionario_loanstats.csv`; `loans_lgd.parquet` lo escribe la Parte D) y
`models/preprocesador_pd.json` salieron idénticos byte a byte a los versionados, y
lo mismo pasó con las figuras de `python run.py figures`. `python run.py export` solo cambió dos campos de cada
`pd_model_{modelo}.json`: la fecha de generación (`generado`) y el commit (`commit`). Esta comprobación se hizo en la misma
máquina de desarrollo; la prueba en un clon limpio en otra máquina, que pide el enunciado, va aparte.
<!-- revisar: autor (Daniel: registrar aquí el resultado del clon limpio en otra máquina cuando se haga) -->

**Pruebas e integración continua.** `.github/workflows/ci.yml` corre `python -m pytest -q -m "not lento"` en Linux (Ubuntu,
Python 3.12, PyTorch de CPU) en cada *pull request* y en cada cambio de `main`, y el *check* `pruebas` debe estar en verde para
fusionar en `main`. En la medición de Windows de la [[tab:comandos]] pasaron 1 336 pruebas, 1 se omitió y fallaron 2:
`test_07_texto_mlp_final_se_lee_igual` y `test_17_mlp_final`, de `tests/test_hpo.py`, que esperan permisos 0o644 en el
archivo que escribe la búsqueda y en Windows reciben 0o666 (reportadas a la Parte B). Con las pruebas lentas falla además
`test_demo_lgd_menos_de_2_minutos` (`tests/test_demo_lgd.py`), porque crea enlaces simbólicos y Windows no lo permite sin
privilegios (WinError 1314; pendiente de la Parte D); en esa corrida pasaron 1 348.

**Publicación del reporte.** `.github/workflows/blog.yml` construye el blog con `mkdocs build --strict` y lo publica en GitHub
Pages (https://danielgonzalez2709.github.io/Trabajo2_Redes_Neuronales/) cada vez que un cambio en `blog/` llega a `main`.

**Semillas.** La semilla efectiva de cada corrida es `base_seed + run_id` (`src/common/seeds.py`). `base_seed` está
declarado en `configs/datos.yaml`, `configs/pd/base.yaml`, `configs/lgd/base.yaml`, `configs/el/base.yaml` y en el bloque
`export` de `configs/scorecard/export.yaml`; lo usan la PD y la LGD (semillas, búsquedas y muestra) y la exportación (casos
de paridad), mientras que la preparación de los datos y la pérdida esperada son deterministas. Con la misma semilla y el
mismo dispositivo los resultados se repiten.

**Notebooks.** Cada parte tiene un notebook en `notebooks/` (por ahora están 00, 01, 04 y 05; faltan el 02 de la PD y el 03 de
la scorecard) que solo llama funciones del repositorio. Su primera celda
(`CONFIG` y `SET`) acepta las mismas claves que `--set`, y se puede modificar en vivo sin escribir sobre los archivos que usan
las demás partes.
