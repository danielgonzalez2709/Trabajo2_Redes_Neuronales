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
Fuente: Elaboración propia. "Medido": corrido el 26 de octubre de 2026 en un portátil con Windows 11, Intel Core i5-11400H (12 hilos), 7,7 GB de RAM, Python 3.12.6 y PyTorch 2.14.1 en CPU; la primera cifra es el tiempo que imprime el comando y la segunda, entre paréntesis, el tiempo de reloj del proceso completo (incluye importar las bibliotecas). "Registrado": tiempo guardado por la corrida oficial en el archivo indicado, en la máquina de quien la corrió (Partes B y D), no en la de las mediciones; son tiempos de ajuste de los modelos, no la duración total del comando (no incluyen cargar los datos ni escribir los resultados). En las búsquedas, `tiempo_total_s` es la suma de los ajustes de los ensayos y, en la de la PD, `tiempo_reloj_s` es el tiempo de `estudio.optimize` (`src/pd/hpo.py`), tampoco el del proceso completo. "Clon limpio": tiempo que imprime el comando (o, en la app, el de "Ready" de Next.js) en la auditoría del 28 de octubre de 2026 en un *runner* de GitHub con Ubuntu 24.04 (`.github/workflows/auditoria.yml`; ver "Clon limpio en otra máquina").
| Comando | Qué hace | Tiempo | Origen del tiempo |
|---|---|---|---|
| `python scripts/preparar_datos.py` | CSV de Kaggle → 5 de los 6 archivos de `data/processed/` y `models/preprocesador_pd.json` (requiere el crudo, ver `data/fuentes.md`) | 6,4 s (7,6 s) | Medido |
| `python run.py eda --config configs/datos.yaml` | Análisis descriptivo, hipótesis y datos después del preprocesamiento | 4,5 s (7,0 s) | Medido |
| `python run.py pd --config configs/pd/demo.yaml` | Demostración de la PD: muestra de 50 000 créditos, 3 valores de $C$ y 3 semillas | 5,0 s | Clon limpio |
| `python run.py pd --config configs/pd/base.yaml` | Logística (13 valores de $C$) y MLP final con 30 semillas | Ajuste: 60,8 s (logística) + 238,0 s (30 semillas del MLP) | Registrado: `results/data/pd/summary.json` (`logistica.tiempo_total_s`, `mlp.tiempo_semillas_s`) |
| `python run.py pd-hpo --config configs/pd/hpo.yaml` | Búsqueda TPE del MLP de PD (40 ensayos); reescribe `configs/pd/mlp_final.yaml` | Suma de los ajustes: 702,4 s; `estudio.optimize`: 706,8 s | Registrado: `results/data/pd/hpo_ensayos.json` (`tiempo_total_s`, `tiempo_reloj_s`) |
| `python run.py lgd --config configs/lgd/demo.yaml` | Demostración de la LGD: muestra de 20 000 créditos y 3 semillas | 5,3 s (10,2 s) | Medido |
| `python run.py lgd --config configs/lgd/base.yaml` | Regresión lineal y MLP de LGD con 30 semillas | Ajuste: 0,03 s (lineal) + 39,2 s (30 semillas del MLP) | Registrado: `results/data/lgd/summary.json` (`lineal.tiempo_total_s`, `mlp.tiempo_semillas_s`) |
| `python run.py lgd-hpo --config configs/lgd/hpo.yaml` | Búsqueda TPE del MLP de LGD (30 ensayos); reescribe `configs/lgd/mlp_final.yaml` | Suma de los ajustes: 126,1 s | Registrado: `results/data/lgd/hpo_ensayos.json` (`tiempo_total_s`) |
| `python run.py scorecard --config configs/scorecard/base.yaml --set scorecard.pdo=40` | Scorecard de la logística | 1,7 s (6,4 s) | Medido |
| `python run.py el --config configs/el/base.yaml` | Pérdida esperada de cada crédito de prueba | 0,9 s (6,1 s) | Medido |
| `python run.py export` | Los dos modelos de PD → `app/public/model/*.json` y casos de paridad | 1,3 s (6,5 s) | Medido |
| `python run.py figures` | Regenera las 26 figuras del reporte desde `results/` (todas las partes) | 10,3 s | Clon limpio |
| `python run.py notebooks` | Convierte y ejecuta los seis notebooks (00 a 05) | 53,5 s | Clon limpio |
| `npm --prefix app run dev` | Aplicación web local en http://localhost:3000 (antes, una vez, `npm --prefix app ci`: 9 s) | 0,4 s hasta "Ready" | Clon limpio |
| `mkdocs build --strict -f blog/mkdocs.yml` | Construye este reporte (`mkdocs serve -f blog/mkdocs.yml` lo sirve en local) | 3,8 s de reloj | Medido |
| `python -m pytest -q -m "not lento"` | Pruebas automáticas que corre el CI | 105,0 s (107,6 s) | Medido |
| `python -m pytest -q` | Todas las pruebas, incluidas las lentas | 201,3 s (203,6 s); en el clon limpio, 393,2 s | Medido y clon limpio |
:::
Los tiempos "Registrado" vienen de otras máquinas: las corridas de la PD registran `hilos_torch` = 4 y las de la LGD
`hilos_torch` = 8 (`results/data/pd/summary.json` y `results/data/lgd/summary.json`). La Parte B midió su demo en un AMD
Ryzen 5 3500U (4 núcleos y 8 hilos; `docs/spec/partes/B-pd.md`, §3.5), y en la descripción del *pull request* #19 indicó
que las corridas registradas de la PD se hicieron en esa misma máquina. `jupyter lab notebooks/` abre los notebooks y no tiene tiempo que medir.

La corrida completa de la PD y su búsqueda no caben en dos minutos; por eso cada tarea debe llevar una configuración de
demostración que se lance con un solo comando. En el clon limpio la de la PD tardó 5,0 s (de 13,8 a 15,1 s en la máquina de
desarrollo de la Parte B) y la de la LGD 3,6 s. Las otras demos de la sustentación (`eda`, `scorecard` y `el`, según
`configs/sustentacion.yaml`) tardaron menos de 10 s de reloj cada una en Windows, y también `export` y `figures`; en el
clon limpio, `eda`, `scorecard`, `el` y `export` tardaron entre 0,8 y 4,4 s, y `figures` 10,3 s. En el clon limpio los
notebooks tardaron 10,0 s (00, sustentación), 10,4 s (01, datos), 11,7 s (02, PD), 6,1 s (03, scorecard), 10,0 s (04, LGD) y
5,3 s (05, pérdida esperada).

**Comprobación en una copia.** Al correr `python scripts/preparar_datos.py` sobre una copia del repositorio con el crudo de
Kaggle, los cinco archivos que escribe en `data/processed/` (`loans_pd.parquet`, `montos.parquet`,
`sensibilidad_lc.parquet`, `resumen_datos.json` y `diccionario_loanstats.csv`; `loans_lgd.parquet` lo escribe la Parte D) y
`models/preprocesador_pd.json` salieron idénticos byte a byte a los versionados, y
lo mismo pasó con las figuras de `python run.py figures`. `python run.py export` solo cambió dos campos de cada
`pd_model_{modelo}.json`: la fecha de generación (`generado`) y el commit (`commit`). Esta comprobación se hizo en la misma
máquina de desarrollo.

**Clon limpio en otra máquina.** El enunciado pide probarlo todo en un clon limpio en una máquina distinta de la de
desarrollo. Lo hace `.github/workflows/auditoria.yml`, que se lanza a mano desde GitHub Actions. Lo corrimos el 28 de octubre de
2026 en un *runner* de GitHub con Ubuntu 24.04, Python 3.12.15 y Node 22, sobre el commit `ca4121e`:

- **Instalación desde cero**, sin cachés, siguiendo el README: torch de CPU (26 s) y después `requirements.txt` (44 s).
- **Sin internet.** Los experimentos corrieron en un espacio de red sin salida, y la comprobación lo confirma: abrir
  `https://pypi.org` desde ahí falla con "Temporary failure in name resolution".
- **Todo terminó sin errores:** las demos de la PD y de la LGD, `eda`, `scorecard`, `el`, `export`, `figures`, `notebooks`,
  las 1 822 pruebas (1 821 pasan y 1 se omite, incluidas las lentas), `mkdocs build --strict` y la app (`npm ci`, las 104
  pruebas de `npm test`, `npm run build` y `npm run dev`).
- **Archivos versionados al regenerar.** Los resultados en JSON de `results/` salieron idénticos a los versionados, entre
  ellos `scorecard.json` y el `summary.json` de la pérdida esperada. Cambiaron cuatro grupos de archivos:
  - los modelos de la app (`app/public/model/`): en `pd_model_logistica.json` solo cambian 4 líneas, como en Windows; en
    `pd_model_mlp.json` y en los dos `casos_paridad_*.json` cambian más líneas (100, 68 y 44 en el `git diff --stat`). Las
    104 pruebas de la app, que incluyen la paridad, pasaron con esos archivos regenerados. No examinamos la causa;
  - las 26 figuras, en bytes. La Parte B ya había visto que, sin las fuentes de la paleta, matplotlib usa otra fuente;
    no lo comprobamos figura por figura;
  - los seis notebooks, en sus salidas;
  - `results/data/el/el_test.parquet`, con el mismo tamaño y el `summary.json` idéntico. En Windows, volver a correr
    `python run.py el` (28 de octubre) lo deja igual byte a byte. No investigamos la diferencia fila por fila.

**Pruebas e integración continua.** `.github/workflows/ci.yml` corre `python -m pytest -q -m "not lento"` en Linux (Ubuntu,
Python 3.12, PyTorch de CPU) en cada *pull request* y en cada cambio de `main`, y el *check* `pruebas` debe estar en verde para
fusionar en `main`. En la medición de Windows de la [[tab:comandos]] (26 de octubre) pasaron 1 336 pruebas, 1 se omitió y fallaron 2:
`test_07_texto_mlp_final_se_lee_igual` y `test_17_mlp_final`, de `tests/test_hpo.py`, que esperan permisos 0o644 en el
archivo que escribe la búsqueda y en Windows reciben 0o666 (reportadas a la Parte B). Con las pruebas lentas falla además
`test_demo_lgd_menos_de_2_minutos` (`tests/test_demo_lgd.py`), porque creaba enlaces simbólicos y Windows no lo permite sin
privilegios (WinError 1314); en esa corrida pasaron 1 348. La Parte D lo corrigió después: la prueba copia los archivos. En el
clon limpio en Linux pasan todas menos una, que se omite (1 821 pasan).

**Publicación del reporte.** `.github/workflows/blog.yml` construye el blog con `mkdocs build --strict` y lo publica en GitHub
Pages (https://danielgonzalez2709.github.io/Trabajo2_Redes_Neuronales/) cada vez que un cambio en `blog/` llega a `main`.

**Semillas.** La semilla efectiva de cada corrida es `base_seed + run_id` (`src/common/seeds.py`). `base_seed` está
declarado en `configs/datos.yaml`, `configs/pd/base.yaml`, `configs/lgd/base.yaml`, `configs/el/base.yaml` y en el bloque
`export` de `configs/scorecard/export.yaml`; lo usan la PD y la LGD (semillas, búsquedas y muestra) y la exportación (casos
de paridad), mientras que la preparación de los datos y la pérdida esperada son deterministas. Con la misma semilla y el
mismo dispositivo los resultados se repiten.

**Notebooks.** Cada parte tiene un notebook en `notebooks/` (los seis, del 00 al 05) que solo llama funciones del repositorio. Su primera celda
(`CONFIG` y `SET`) acepta las mismas claves que `--set`, y se puede modificar en vivo sin escribir sobre los archivos que usan
las demás partes.
