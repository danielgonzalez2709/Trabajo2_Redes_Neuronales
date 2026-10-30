# Parte B — PD: modelos y evaluación

| | |
|---|---|
| Responsables | **Andrés** (reparto de la pareja, 24-oct; Juan José hace la Parte C) |
| Rama personal | `dev/andres` |
| Carpetas propias | `src/pd/`, `src/common/metricas.py`, `src/common/torch_utils.py`, `configs/pd/`, `src/viz/figs_pd.py`, `src/pd/tablas.py`, `src/pd/espacio.py`, `src/pd/sensibilidad.py`, `src/pd/umbral.py`, `src/pd/umbral_experimento.py`, `src/pd/tablas_umbral.py`, `src/viz/umbral_pd.py`, `scripts/umbral_pd.py`, `notebooks/02_pd.py`, `blog/secciones/04–06` |
| Estado | **B1–B10 cerradas; B11 es un complemento [EXTRA, opcional]** (§3.9) |
| Lee del núcleo | §0, §3, §4.1, §5.1, §5.2, §5.3, §5.4, §6, §7.1 · **Anexo (enunciado):** Parte 1, tareas 3–5; Regla de comparación justa; Discusión; Entregables 1 (fundamentación matemática) y 3 |
| Arranca sin esperar | Sí: métricas y MLP con datos sintéticos (núcleo §5.6) desde el jueves 22 |

## 1. Lo que pide el enunciado
- P1.3 **Regresión logística** como modelo de baja complejidad.
- P1.4 **MLP** para la PD optimizando arquitectura e hiperparámetros: número de capas, neuronas por capa, activaciones,
  regularización (dropout, L2, parada temprana), optimizador y tasa de aprendizaje. Método de búsqueda libre **pero justificado**.
- P1.5 Evaluación: **AUC, Gini, KS** (discriminación); **Brier y curva de calibración** (calibración).
- Comparación justa: mismos datos, presupuesto explícito, **30 semillas** de la configuración final (media, desviación, mejor,
  peor) y número de parámetros entrenables.
- Blog: neurona, MLP, cada activación usada, **derivación de la pérdida por máxima verosimilitud**, retropropagación, ecuaciones
  del optimizador, formulación de cada regularización, definición de cada métrica.

## 2. Consumo / Entrego
| Consumo | De | Cuándo |
|---|---|---|
| Esqueleto, `src/common/`, datos sintéticos | E | jue 22, noche |
| `loans_pd.parquet` + preprocesador reales | A | vie 23, 2 p. m. |

| Entrego | A | Cuándo |
|---|---|---|
| `metricas.py` (lo usan C, D y E) | todos | vie 23 |
| `src/pd/mlp.py` + `entrenar.py` (D reutiliza el bucle de entrenamiento) | D | **vie 23, mediodía** |
| Modelo PD final (`models/pd_*.{pt,joblib}`), `summary.json`, `pred_test.parquet` | C, E | **sáb 24, mediodía** |
| Secciones 04–06 del blog | E | dom 25, 8 p. m. |

## 3. Especificación

### 3.1 Logística (referencia)
`LogisticRegression` con penalización L2 sobre el preprocesamiento común. Búsqueda de `C` en una rejilla logarítmica declarada
(p. ej. 10 valores) usando validación; el número de configuraciones va a `summary.json`. Determinista con `lbfgs`: se reporta una
corrida y se explica por qué su desviación entre semillas es 0.
- Decisiones de B3 [DECIDIDO, Andrés, 25-oct; Spec v0.14]: rejilla de **13** valores de 10⁻⁴ a 10² cada medio orden de magnitud (`configs/pd/base.yaml`, sin duplicados: `n_configuraciones_busqueda = len(rejilla_C)`); criterio = log-loss de validación (`log_loss(..., labels=[0, 1])`), AUC de val al lado; argmin estricto y, en empate exacto, el **menor C**; el modelo final queda entrenado solo con train (sin refit con train + val); `lbfgs`, `max_iter = 1000`, `tol = 1e-6`, un `ConvergenceWarning` es error; sin `class_weight`, `penalty` ni `multi_class`; X en float64. El objetivo de sklearn es (1/n)·Σ BCE + ‖w‖²/(2·C·n) con el intercepto sin penalizar (λ = 1/(C·n) en la L2 de §3.2). `epocas` del JSON de la semilla = iteraciones de lbfgs. `desv` muestral (n = 1 → 0, núcleo §5.4). `en_borde` avisa si C elegido es el mínimo o el máximo. joblib guarda `version_sklearn` y `cargar` falla si no coincide.

### 3.2 MLP (núcleo §5.3)
- `Linear → activación → Dropout` repetido; salida `Linear(·, 1)` con logit; pérdida `BCEWithLogitsLoss` (estable).
- Regularización: dropout, **L2** y parada temprana sobre la pérdida de validación (restaura los mejores pesos).
  ⚠️ L2 ≠ weight decay desacoplado: `AdamW` no es L2. Si el blog dice "penalización L2", el código debe hacer L2: término
  **(λ/2)‖W‖²** en la pérdida (gradiente λW) o, de forma equivalente, `weight_decay = λ` de `Adam`/`SGD`, que suma λW al
  gradiente; los sesgos van sin L2 [DECIDIDO, Andrés, 24-oct; `mlp.l2` = λ]. Ojo: λ‖W‖² (sin el ½) tiene gradiente 2λW y
  **no** equivale a `weight_decay = λ` (corrección v0.13, hallazgo H-08 del verificador-matematico). Loshchilov & Hutter
  (2019), arXiv:1711.05101: L2 y weight decay coinciden en SGD pero no en Adam [DECIDIDO; cita verificada (V-11), en
  `blog/references.bib` y citada en el blog 04].
- Desbalance [DECIDIDO, Andrés, B2]: `pos_weight = 1` en todas las corridas reportadas, para no descalibrar la PD (la PD
  alimenta la scorecard y la E[L]). Otro valor solo como sensibilidad, reportando su efecto en Brier y calibración (no se
  corrió).
- `entrenar.py` es genérico (recibe pérdida y modelo): D lo reutiliza para la LGD.
- Decisiones de B2 [DECIDIDO, Andrés, 24/25-oct]: `pos_weight` va dentro del bloque `mlp` (`mlp.pos_weight`, obligatorio,
  1 por defecto); parada temprana con `min_delta = 0` (mejora = pérdida de validación estrictamente menor); pérdida no finita
  en train o val → `FloatingPointError` con la época y la fase (B4 marca el ensayo como fallido); `predict_logit` en una sola
  pasada; épocas numeradas desde 1; `cargar()` reconstruye en CPU (la config guardada conserva el `device` de
  entrenamiento). `iterar_minilotes(n, batch, rng, device)` recibe el dispositivo (D, si la llama directo).

### 3.3 Búsqueda [DECIDIDO, Andrés, 25-oct; Spec v0.15]
- Optuna TPE, **40 ensayos** (`configs/pd/hpo.yaml`; demo: `--set hpo.n_ensayos=5`). `python run.py pd-hpo` escribe
  `results/data/pd/hpo_ensayos.json` (campos en núcleo §5.4) y reescribe `configs/pd/mlp_final.yaml` (no se edita a mano),
  que `configs/pd/base.yaml` hereda: B5 entrena el ganador sin copiar nada.
- Espacio: capas 1–4; neuronas {16, 32, 64, 128, 256}, **una elección por capa** (sin forma de pirámide); activación {relu,
  leaky_relu, tanh}; dropout [0, 0.5]; l2 log-uniforme [1e-6, 1e-2]; optimizador {adam, sgd con momento}; lr log-uniforme
  [1e-4, 1e-2]; batch {256, 512, 1024, 2048}. Nombres de Optuna: `n_capas`, `neuronas_capa{k}`, `activacion`, `dropout`,
  `l2`, `optimizador`, `lr`, `batch`; el JSON guarda los parámetros ya armados (`capas: [...]`). No se buscan `momento`,
  `max_epocas`, `paciencia`, `device` ni `pos_weight` (= 1, §3.2): salen del bloque `mlp` heredado (`--set mlp.*` los cambia
  en todos los ensayos) y son el `mlp_base` del JSON. Validación antes de leer datos: claves exactas, rangos con mín ≤ máx
  (> 0 en escala log, dropout < 1), listas no vacías y sin duplicados; se aceptan listas de un elemento y rangos con
  mín = máx para fijar un parámetro desde el CLI.
- Muestreador: `TPESampler(seed=base_seed)` con los valores por defecto y `NopPruner`, en memoria, `n_jobs = 1`. **En
  Optuna 5.0.0 el TPE por defecto es multivariado** en estudios de un objetivo (no el TPE univariado de Bergstra et al.);
  el JSON registra `multivariate` (efectivo) y `n_startup_trials`. Misma `base_seed` en el muestreador y en todos los fit.
- Criterio: **log-loss de validación** (regla de puntuación propia: premia discriminación y calibración), con la `logloss`
  de §3.1 sobre `predict_proba(X_val)`; AUC de validación al lado. Mejor = argmin entre los ensayos completos; empate →
  menor número de ensayo. Test no se usa en la búsqueda.
- Fallos: solo `FloatingPointError` (pérdida no finita, §3.2) deja el ensayo FAIL con su error y su tiempo; cualquier otro
  error detiene la búsqueda. Si ningún ensayo completa → error y no se escribe nada.
- Presupuesto (núcleo §6): `n_ensayos` cuenta todos los ensayos; `tiempo_total_s` = Σ del tiempo de cada fit, incluidos los
  fallidos; `tiempo_reloj_s` = todo el estudio, aparte.
- `en_borde` = {lr, l2, dropout: "min" | "max" | null}: el valor elegido está en el borde si queda a ≤ 5 % del ancho del
  rango respecto de un extremo (en log10 para lr y l2, lineal para dropout); con mín = máx → null.
- Justificación citada: TPE (Bergstra et al., 2011) y Optuna (Akiba et al., 2019), verificadas en `blog/references.bib`;
  TPE multivariado: Falkner et al. (2018) [DECIDIDO; verificada, en `blog/references.bib` y citada en el blog 04].
- `datos.muestra` (v0.20): `pd-hpo` lo ignora (`hpo.validar_cfg` no lee `datos`; la búsqueda usa siempre todos los datos). Se rechazará en el refactor posterior a B5 [DECIDIDO, Andrés, 26-oct].

### 3.4 Métricas (`src/common/metricas.py`)
- AUC; **Gini = 2·AUC − 1**; **KS = máx_t |F₁(t) − F₀(t)|** (acumuladas de la PD en malos y buenos).
- **Brier = (1/n) Σ (pᵢ − yᵢ)²**; curva de calibración con 10 bins por cuantiles.
- Contraste en pruebas con `roc_auc_score`, `brier_score_loss`, `scipy.stats.ks_2samp`.

### 3.5 Corridas finales
- MLP final: **30 semillas** (`base_seed + 0…29`), misma partición; `summary.json` con media, desviación, mejor y peor.
- Se guardan `pd_logistica.joblib`, `pd_mlp.pt` (semilla con la métrica mediana) y `pred_test.parquet`.
- **Demo [DECIDIDO, Andrés, 26-oct; Spec v0.20]:** `python run.py pd --config configs/pd/demo.yaml` es el mismo código que la corrida final con menos presupuesto (ningún `if demo`). `demo.yaml` va completa y **sin `hereda`**: heredar `base.yaml` traería el bloque `busqueda` de `mlp_final.yaml`, y `{tipo: sin_busqueda}` no admite otras claves. Valores: `base_seed: 0`; `datos.muestra: 50000`; logística con `rejilla_C: [1e-3, 3.1623e-3, 1e-2]` (los vecinos de la rejilla de §3.1 alrededor del C elegido en B3, 3.1623e-3), `max_iter: 1000`, `tol: 1e-6`; `mlp` = el de `mlp_final.yaml` (ensayo 11 de B4) con `max_epocas: 20` y `paciencia: 5` (en B5 las 30 semillas corrieron entre 11 y 30 épocas); `busqueda: {tipo: sin_busqueda}` (`config_de_busqueda` false, presupuesto de la búsqueda null); `semillas.n: 3`; `salida: results/data/pd_demo` y `modelos: models/pd_demo`, ignoradas por git: la demo nunca pisa `results/data/pd/` ni `models/pd_mlp.pt`, que usan C y E. Sus cifras **no** son resultados del blog. Una prueba exige que `capas, activacion, dropout, l2, optimizador, lr, batch` sigan iguales a los de `mlp_final.yaml` (alarma si `pd-hpo` cambia la configuración final).
- **`datos.muestra`** (núcleo §4.1): `base.yaml` declara `datos: {muestra: null}` (todos los datos); una config sin bloque `datos` equivale a null. `datos` es un mapeo con exactamente la clave `muestra`, que vale null o un entero ≥ 3; bool, flotante o texto → `ValueError` antes de leer datos. La regla es la de la LGD (`src/lgd/experimento.py`): el n total se reparte entre train, val y test en proporción a su tamaño (mayor resto; empate → train, val, test), se elige sin reemplazo con `make_rng(base_seed, 0)` y las filas quedan en el orden del parquet; con n ≥ filas se usan todos los datos; si un split queda vacío → `ValueError`. La selección depende solo de `split` y del orden de las filas, así que el pre-chequeo y el entrenamiento usan los mismos ids. Se aplica antes de transformar (el preprocesador de A nunca se reajusta) y antes de `evidencia`: la huella describe las filas usadas, por lo que una muestra con la config de la búsqueda (p. ej. `base.yaml --set datos.muestra=20000`) → `ValueError` ("la búsqueda se hizo con otros datos"); en vivo, `--set datos.muestra` se usa sobre `demo.yaml`. Reparto: con 50 000, train 19 005, val 15 397 y test 15 598; con 20 000, 7 602, 6 159 y 6 239. La consola imprime las filas por split.
- Volver a correr en la misma `salida` con otra muestra falla en el pre-chequeo (`pred_test.parquet` tiene otros ids): para la modificación en vivo se usa el notebook 00 (carpeta temporal) o `--set salida=<otra carpeta>`.
- **Errores:** `run.py pd` captura `ValueError`, `RuntimeError`, `FloatingPointError` y `FileNotFoundError` (p. ej. un clon sin `data/processed/loans_pd.parquet`), imprime `run.py pd: error: <Tipo>: <mensaje>` (una línea) en stderr y sale con código 2, sin traceback; `NotImplementedError`, `RecursionError`, las demás `OSError` (p. ej. `PermissionError`) y cualquier otro error conservan el traceback (decisión de Andrés tras revisor-calidad, 26-oct).
- **Tiempos medidos** (26-oct, AMD Ryzen 5 3500U, 4 núcleos / 8 hilos, `hilos_torch` = 4; reloj de `time`): demo 13,8–15,1 s (3 corridas; `summary.json` idéntico salvo tiempos); peor caso `--set mlp.lr=1e-4` 17,5 s; `--set semillas.n=5` 19,5 s; `--set datos.muestra=20000` 9,7 s; `--set datos.muestra=null` 41,0 s. Clon limpio con venv nuevo y sin red (`unshare -n`, auditor-reproducibilidad, misma máquina): demo 14,9–16,0 s (3 corridas); las 7 modificaciones en vivo 10,7–22,3 s (peor caso `lr=1e-4`). Otra máquina: pendiente (Daniel o Juan José, clon limpio).
- Decisiones de B5 [DECIDIDO, Andrés, 26-oct; Spec v0.16]: `python run.py pd --config configs/pd/base.yaml` corre la logística (§3.1) y después el MLP con el bloque `mlp` resuelto (herencia de `mlp_final.yaml` y luego `--set`), con `semillas.n` semillas efectivas `base_seed + 0…n−1` (`seeds_for_runs`), un `MLPPD` nuevo por semilla y `tiempo_s` = solo el `fit`. `configs/pd/base.yaml` declara `semillas: {n: 30}`, `salida: results/data/pd` y `modelos: models` (carpetas; los nombres de archivo son los del núcleo §5.4). `semillas.n` entero ≥ 1, `base_seed` entero ≥ 0, el bloque `mlp` y el bloque `busqueda` se validan antes de leer datos y antes de correr la logística (`ValueError`).
- **Semilla mediana:** las corridas se ordenan por (`logloss_val`, `seed`) de menor a mayor y se toma la posición ⌊(n − 1)/2⌋ (n par → mediana inferior; empate → semilla menor). Decide validación, como la búsqueda (§3.3); test nunca decide. Su red (la entrenada, sin reentrenar) va a `models/pd_mlp.pt` y sus PD de test a `pred_mlp`.
- **Config de la búsqueda:** `config_de_busqueda` es true si el bloque `mlp` validado de la corrida es igual al de `configs/pd/mlp_final.yaml` releído y validado, y el bloque `busqueda` también coincide. Si es true, `busqueda.tipo` debe ser `optuna` y `busqueda.evidencia` igual a la de esta corrida (si no → `ValueError`: "la búsqueda se hizo con otros datos; repite pd-hpo"), y el presupuesto del MLP es `n_configuraciones_busqueda = busqueda.n_ensayos`, `tiempo_total_s = busqueda.tiempo_s + Σ tiempo_s`. Si es false (p. ej. `--set mlp.capas=[8]`), la consola avisa qué claves difieren y el presupuesto de la búsqueda queda en null.
- Antes de entrenar se borran solo los archivos `mlp_seed<entero>.json` de `{salida}/semillas`; si un `fit` lanza `FloatingPointError`, el error sale con la semilla y avisa que `summary.json` conserva la entrada `mlp` anterior, que ya no corresponde a `semillas/` (hay que volver a correr); no se escriben `summary.json`, `pred_mlp` ni `pd_mlp.pt` (quedan las semillas terminadas). La huella del pre-chequeo (solo `id, split`) debe ser idéntica a la del DataFrame completo (lo comprueba una prueba). Campos extra de la semilla y del summary: núcleo §5.4 (v0.16).
- Bloques y pre-chequeo [DECIDIDO, Andrés, 26-oct; v0.16, tras revisor-spec]: `semillas` tiene exactamente la clave `n`. `busqueda` es o bien el bloque de `pd-hpo` (exactamente `tipo: optuna`, `ensayo`, `n_ensayos`, `tiempo_s`, `evidencia`, `fecha`) o bien exactamente `{tipo: sin_busqueda}` (config fijada a mano, p. ej. una demo): con `sin_busqueda`, `config_de_busqueda` es false y el presupuesto de la búsqueda queda en null. Si no existe `configs/pd/mlp_final.yaml` → `ValueError` ("corre pd-hpo"). Antes de la logística, `correr_pd` comprueba además el bloque `logistica` y las salidas previas: `summary.json` (si existe) con un diccionario en la raíz y `pred_test.parquet` (si existe) con los mismos ids e `y` de test; cualquier falla → `ValueError` sin entrenar nada. La consola informa cuántos `mlp_seed*.json` anteriores se borraron.

### 3.6 Figuras [DECIDIDO, Andrés, 26-oct; Spec v0.21]
- `src/viz/figs_pd.py` (formato percent): una función **pura** de datos por figura (no lee archivos ni dibuja) y una
  función de dibujo; la lectura (`insumos_de_corrida`) y la escritura (`_guardar`, `_escribir_json`) van aparte.
  `generar(raiz, destino, datos)` resuelve `destino or raiz / SALIDA` al llamarse (`SALIDA = blog/docs/figures`), así que
  `test_no_network` la redirige; `datos=` permite dibujar una corrida en memoria (notebook 02, B9). Los modelos no dibujan:
  todo sale de `results/data/pd/` (núcleo §3).
- Figuras (todas de **prueba** salvo las de búsqueda y aprendizaje; MLP = semilla mediana de §3.5):
  `pd-roc` (ROC de los dos modelos, AUC y Gini), `pd-ks` (acumuladas de la PD en malos y buenos y separación máxima),
  `pd-calibracion` (10 bins por cuantiles, §3.4, con la diagonal), `pd-semillas` (AUC, KS y Brier de las 30 semillas
  frente a la logística), `pd-aprendizaje` (BCE de train y val por época de la semilla mediana, mejor época y paciencia),
  `pd-busqueda-mlp` (log-loss de val de los ensayos TPE completos, mejor acumulado, fin de los `n_startup_trials`
  aleatorios y la logística como referencia) y `pd-busqueda-logistica` (log-loss y AUC de val contra C, C elegido).
- `blog/docs/figures/pd-calibracion.json` = `{split: "test", n_bins, semilla_mediana, modelos: {logistica|mlp:
  {pd_media[], tasa_observada[], n[], pd_media_global, tasa_global}}}` (los puntos de `pd-calibracion.png`; floats
  nativos, sin NaN). Lo importa la escena 7 del video (`video/src/datos.ts`); no es un contrato de `results/`.
- Lectura: en prueba los dos modelos ordenan bien el riesgo pero lo **subestiman**: en los 10 bins la tasa observada
  supera a la PD media, es decir, los puntos quedan **por encima** de la diagonal (PD media 0,187 y 0,204 frente a una
  tasa de 0,262; hallazgo candidato H-22 por haberlo descrito al revés).

### 3.7 Notebook `02_pd` [DECIDIDO, Andrés, 26-oct; Spec v0.22]
- **Propósito:** la demo de la PD lo más a la mano posible (sección 1) y, debajo, un recorrido guiado de la Parte B que
  sirve de repaso para la sustentación. Cumple el núcleo §5.8: fuente `notebooks/02_pd.py` en formato percent, el `.ipynb`
  lo genera `run.py notebooks`, cero lógica propia, < 2 min en CPU. Solo la sección 1 entrena; las secciones 2–4 **leen**
  resultados versionados y no cambian con `SET`.
- **Explicaciones:** de 2 a 4 líneas por subsección con el **qué** y el **por qué**, y el enlace a la sección del blog
  (04–06, o 01–03 para los datos) donde está el desarrollo completo; las derivaciones viven en el blog (B8), no se
  duplican. Las cifras salen **siempre impresas** de los JSON (nunca escritas a mano en el texto). La lectura no afirma
  causas que los resultados no prueben.
- **Encabezado (celdas de texto, antes de la celda de parámetros):** qué lee, que nunca escribe en `data/processed/`,
  `models/`, `results/`, `configs/pd/mlp_final.yaml` ni `blog/`, y cómo usar los moldes:
  - **Molde de la logística**, **molde del MLP** y **molde de las claves comunes** (`base_seed`, `datos.muestra`,
    `semillas.n`): diccionarios con **todas** las claves de `configs/pd/demo.yaml` que se pueden cambiar en vivo, con su
    valor actual y un comentario de una línea sobre lo que hace cada una, listos para copiar las claves al `SET`.
  - **Molde del comando de terminal** equivalente (`python run.py pd --config configs/pd/demo.yaml --set …`, con
    `--set salida=<carpeta>` para no chocar con una corrida anterior de otra muestra, §3.5), listo para pegar.
  - Una prueba exige que las claves y los valores de los moldes coincidan con `demo.yaml` (un molde desactualizado
    daría un error delante del profesor). `salida` y `modelos` no van en los moldes: el notebook los fija.
- **Secciones** (numeradas; subsecciones `N.M`):
  1. **Demo en vivo.** 1.0 celda de parámetros (`CONFIG = "configs/pd/demo.yaml"`, `SET = {}`, primera celda de código,
     etiqueta `parameters`) · 1.1 config resuelta impresa (`config_notebook`) · 1.2 `correr_pd` en una **carpeta
     temporal** (salida y modelos van siempre ahí aunque `SET` cambie `salida` o `modelos`), con el tiempo, las filas por
     partición y la tabla logística frente a MLP (AUC, Gini, KS, Brier y log-loss de val y test; parámetros; semilla
     mediana) · 1.3 figuras de **esta** corrida con `figs_pd.generar(destino=<temporal>, datos=insumos_de_corrida(...))`:
     `pd-roc`, `pd-ks`, `pd-calibracion`, `pd-aprendizaje`, `pd-semillas` y `pd-busqueda-logistica` (la demo no tiene
     `hpo_ensayos.json`, así que no dibuja `pd-busqueda-mlp`).
  2. **Datos que consume la PD (Parte A, solo lectura).** 2.1 partición temporal y tasa de malos por partición y por año
     (`data/processed/resumen_datos.json`, `results/data/eda/post_preprocesamiento.json`, figura `datos-anio.png`) ·
     2.2 las 15 variables de originación y su relación con el malo (`eda-tasa-variables.png`); preprocesador de A ajustado
     solo con train y nunca reajustado por B · 2.3 estabilidad entre particiones (PSI frente a train, del mismo JSON).
     Remite al notebook 01 para el análisis completo.
  3. **Búsqueda de hiperparámetros.** 3.1 logística: rejilla de C, criterio, C elegido y `en_borde`
     (`busqueda_logistica.json`, `pd-busqueda-logistica.png`) · 3.2 MLP con TPE: espacio, muestreador, criterio,
     presupuesto (ensayos, tiempo) y estados (`hpo_ensayos.json`) · 3.3 trayectoria de la búsqueda (`pd-busqueda-mlp.png`)
     · 3.4 tabla de los mejores ensayos (arquitectura, log-loss y AUC de val, épocas, mejor época, parámetros) y
     `en_borde` de lr, l2 y dropout · 3.5 tabla resumen de las dos búsquedas (configuraciones probadas, tiempo, criterio).
  4. **Modelo final y comparación justa.** 4.1 arquitectura ganadora (`configs/pd/mlp_final.yaml`, ensayo elegido) y
     parámetros entrenables frente a los de la logística · 4.2 las semillas de `summary.json`: media ± desv., mejor y peor
     por métrica, y la regla de la semilla mediana (`pd-semillas.png`) · 4.3 MLP − logística por métrica, con signo, y
     en cuántas semillas el MLP es mejor que la logística (desde `semillas/mlp_seed*.json`) · 4.4 ROC y KS oficiales de
     prueba (`pd-roc.png`, `pd-ks.png`) · 4.5 calibración (`pd-calibracion.png`) y su lectura de §3.6, ligada a la tasa
     por año de 2.1 · 4.6 curva de aprendizaje de la semilla mediana y parada temprana (`pd-aprendizaje.png`).
  5. **De la PD a la decisión: el umbral (B11, §3.9; v0.29).** Solo lee `results/data/pd_umbral/umbral.json` y las dos
     figuras `pd-umbral-*`: 5.1 frente al modelo trivial · 5.2 matriz de confusión con el umbral 0,5 · 5.3 sensibilidad al
     umbral (`pd-umbral-sensibilidad.png`) · 5.4 cuota por costo y metas de sensibilidad, con la variación de las 30
     semillas (`pd-umbral-costo.png`) · 5.5 PD absoluta frente a cuota y p* frente al corte observado. Hilo narrativo:
     una línea en 2.1 (la tasa de malos sube → deriva de 5.5) y en 4.5 (calibración → qué implica al elegir un corte).
     Las tablas salen de `src/pd/tablas_umbral.py`.
- **`src/pd/tablas.py`** (con `tests/test_tablas_pd.py`): las funciones que arman las tablas y textos de arriba a partir
  de los JSON ya cargados (puras: no leen archivos ni entrenan), más la que redirige `salida` y `modelos` de la config a
  la carpeta temporal y la que devuelve los moldes desde `demo.yaml`. El notebook solo las llama y muestra el resultado.

### 3.8 Sensibilidad con `grade` e `int_rate` (B10) [DECIDIDO, Andrés, 26/27-oct; Spec v0.23]
- **Pregunta:** cuánto mejoraría la PD si el modelo viera la evaluación de riesgo de Lending Club (`grade`: "LC assigned
  loan grade"; `int_rate`: "Interest Rate on the loan", `configs/features_sensibilidad.yaml`). Es un análisis aparte:
  **no va a la app ni reemplaza al modelo principal**. `sub_grade` (redundante con `grade`) e `installment` (depende de la
  tasa) no entran (A §3.4).
- **Datos:** A ya entrega `data/processed/sensibilidad_lc.parquet` (`id`, `grade`, `int_rate`; mismas filas y orden que
  `loans_pd.parquet`, probado en `test_sensibilidad_aparte_y_alineada`). B los une por `id` (error si los ids difieren
  en conjunto u orden, hay repetidos o vacíos, o una columna ya existe) y ajusta un **preprocesador propio** (el
  `Preprocesador` de A con los percentiles de `configs/datos.yaml`) **solo con train**, sobre las 15 variables de
  `features.yaml` seguidas de las 2 adicionales. Las adicionales se validan como las principales (disponibles en la
  originación, sin repetidos, ninguna prohibida del núcleo §5.7: constante `PROHIBIDAS` en `src/pd/sensibilidad.py`;
  una prueba exige que coincida con `tests/test_fuga.py`; `src/` no importa de `tests/`).
- **Dos modos** (`sensibilidad.modos`, por defecto los dos):
  - `reusar` = **misma arquitectura que el modelo principal**: la logística con la rejilla de 13 C de §3.1 y el MLP con
    el bloque `mlp` de `mlp_final.yaml` y `busqueda: {tipo: sin_busqueda}` (la búsqueda del principal se hizo con otros
    datos: su presupuesto no cuenta aquí);
  - `optuna` = **arquitectura buscada con las 17 variables**: un TPE nuevo con el mismo presupuesto (40 ensayos) y el
    mismo espacio de §3.3, y luego la logística y las 30 semillas del ganador.
- **Mismo código que el principal:** `correr_pd` (con `rutas` inyectadas), `correr_logistica`, `correr_mlp` y
  `hpo.buscar` reciben un `RutasPD` con el parquet unido y el preprocesador de sensibilidad. Nunca se escribe en
  `results/data/pd/`, `models/pd_*` ni `configs/pd/mlp_final.yaml`.
- **Comando y artefactos:** `python run.py pd-sensibilidad --config configs/pd/sensibilidad.yaml` (config completa, sin
  `hereda`, como `demo.yaml`; pruebas de alarma: su `mlp` = `mlp_final.yaml`, `logistica` = `base.yaml`, `hpo` =
  `hpo.yaml`). `results/data/pd_sensibilidad/{loans_pd_sensibilidad.parquet, reusar/, optuna/, comparacion.json}` y
  `models/pd_sensibilidad/{preprocesador_pd_sensibilidad.json, reusar/, optuna/}`; cada modo con los archivos de §3.5
  (y `optuna/` además `hpo_ensayos.json` y su `mlp_final.yaml`).
- **`comparacion.json`:** por modelo (logística, MLP) y métrica de test (AUC, Gini, KS, Brier): media ± desv. del
  principal y de cada modo y Δ = sensibilidad − principal; parámetros, configuraciones y tiempo (núcleo §6);
  `calibracion_global` por modo y modelo (`pd_media_test` de `pred_test.parquet`, MLP = semilla mediana, frente a
  `tasa_test`); `misma_particion` (`sha256_particion` igual al del principal).
- **Blog 06 §6.6:** tabla desde `comparacion.json` y una lectura breve: cuánto de la discriminación ya está en la nota
  de LC; los modos se nombran "misma arquitectura que el modelo principal" y "arquitectura buscada con las 17
  variables"; si la PD media de test se acerca a la tasa de prueba, se lee como **consistente con** la deriva de
  composición entre train y test (participación del grado A e `int_rate` media por partición, medidas desde los
  datos); si no, también puede deberse al sesgo de selección del blog 02 §2.4. No se afirma una causa ni qué
  información usa Lending Club.

### 3.9 De la PD a la decisión: umbral, matriz de confusión y costo (B11) [DECIDIDO, Andrés, 30-oct; Spec v0.29] [EXTRA, opcional]
- **Pregunta:** cómo se convierte la PD en una decisión de aprobar o rechazar y cuánto cambia la matriz de confusión con
  el corte, si prestarle a un malo (FN) cuesta más que rechazar a un bueno (FP). **Complemento que solo agrega:** no
  cambia ninguna cifra de B1–B10, no toca archivos de otras partes, no va a la app y **no fija una política de corte**
  (ilustra cómo cambia la decisión según el costo relativo, que depende del apetito de riesgo del banco; coherente con el
  blog 07, que no fija puntos de corte).
- **Convención:** positivo = malo (`y = 1`). Una política **rechaza** un subconjunto de créditos. VP = malos rechazados,
  FN = malos aprobados (el error caro), FP = buenos rechazados, VN = buenos aprobados. TPR (sensibilidad) = VP/(VP+FN),
  TNR (especificidad) = VN/(VN+FP), tasa de rechazo = (VP+FP)/n, tasa de aprobación = 1 − tasa de rechazo.
- **Dos formas de rechazar** (`src/pd/umbral.py`, lógica pura, sin E/S; entradas validadas con
  `metricas.validar_entradas`):
  - **Por umbral** `t`: rechaza si `p ≥ t`.
  - **Por cuota** `r ∈ [0, 1]`: rechaza los `k = ⌈r·n⌉` créditos de mayor PD, con `k` calculado sobre `r·n` redondeado a
    9 decimales (para que 0,07·100 dé 7 y no 8); empates de PD en el borde: orden **estable** descendente (a igual PD,
    primero la fila que aparece antes). `umbral_de_cuota(p, r)` = la menor PD rechazada (`+inf` si `k = 0`).
- **Costo** en unidades de G (lo que se pierde al rechazar a un bueno): `costo = (L/G)·FN + FP`, y por crédito `/n`. Solo
  la razón `L/G`; **sin dólares ni margen inventado** (`int_rate` está excluida del modelo, núcleo §5.7). Referencia
  teórica `p* = 1/(1 + L/G)`: el corte que minimiza el costo esperado **si la PD está calibrada** (rechazar si
  `p·L > (1 − p)·G`).
- **Elección del corte (principal, cuota):** para cada `L/G`, `r*` = la cuota de la rejilla que minimiza el costo
  **observado en validación** (2013) con sus etiquetas; empate → la **menor** cuota. Se aplica en prueba (2014) como cuota
  sobre la cohorte (`rechazos_por_cuota(p_test, r*)`): es una política **por lotes o de cuota**, no la decisión individual
  en tiempo real de la app, y así se dice en el texto. **Contraste (PD absoluta):** `t_val = umbral_de_cuota(p_val, r*)`
  aplicado tal cual en prueba (`p_test ≥ t_val`); muestra qué tasa de rechazo produce en 2014 por la deriva. La brecha
  entre `p*` y `t_val` mide la descalibración (coherente con §6.4 y Limitaciones del blog 06).
- **Metas de sensibilidad:** para cada meta `m`, la menor cuota de la rejilla con TPR de validación ≥ `m`, y su efecto en
  prueba (matriz y tasas).
- **Modelo trivial:** PD constante `q`. Principal `q` = tasa de malos de **train**; piso teórico `q` = tasa de **prueba**
  (usa etiquetas de prueba). Brier en val y test con `metricas.brier`; prueba: igual a la forma cerrada π(1−π) + (q−π)²
  (π = tasa del conjunto). AUC = 0,5 y matriz degenerada (aprueba o rechaza a todos): se dice en el texto, no va como fila.
- **Matriz inicial:** umbral `0,5` en val y test.
- **Curva de sensibilidad al umbral:** TPR, TNR y tasa de rechazo en una rejilla de umbrales, en val y test.
- **Modelos:** logística (`models/pd_logistica.joblib`) y MLP lado a lado. El análisis principal del MLP usa
  `models/pd_mlp.pt` (semilla mediana de `summary.json`, la de la app); sus PD de prueba deben coincidir con `pred_mlp` de
  `results/data/pd/pred_test.parquet`. **Variación entre semillas:** las `semillas.n` redes de la configuración final
  (`configs/pd/mlp_final.yaml`, semilla efectiva `base_seed + k`, `src/common/seeds.py`) se reentrenan con
  `experimento.entrenar_semilla` y, para cada escenario y meta, se resumen TPR, TNR y tasa de aprobación de prueba con
  `experimento.resumir` (media, desv. muestral, mejor, peor, n). **No se guardan las redes** (solo sus PD).
- **Control de la semilla mediana:** si está entre las reentrenadas, sus PD (val y test) se comparan con las de
  `models/pd_mlp.pt`: `{identica_bit_a_bit, max_abs_dif, dentro_tolerancia}`. En otra máquina basta la tolerancia de §8
  (3·√2·EE sobre las medias de las métricas de prueba); fuera de tolerancia → error. Si no está (demo con pocas
  semillas), `null`.
- **Configuración** `configs/pd/umbral.yaml`: `hereda: base.yaml` (mismo `mlp`, `base_seed` y `semillas`) más
  `salida: results/data/pd_umbral`, `figuras: blog/docs/figures` y el bloque `umbral`: `umbral_inicial: 0.5`,
  `escenarios_razon: [2, 5, 10]`, `curva_razon: {min: 1, max: 20, n: 25}` (rejilla logarítmica), `rejilla_cuotas` y
  `rejilla_umbrales` como `{min: 0.0, max: 1.0, n: 101}` (lineal, pasos de 0,01, redondeada a 9 decimales) y
  `metas_sensibilidad: [0.5, 0.7, 0.8, 0.9]`.
  `datos.muestra` debe ser `null` (B11 usa todos los datos). Claves exactas; una clave desconocida es error.
- **Guarda de rutas** (lección H-26): `salida` resuelta debe ser `results/data/pd_umbral/` o una hermana
  `results/data/pd_umbral_<algo>/` (p. ej. la demo, ignorada por git); `figuras` debe ser `blog/docs/figures/` o una
  carpeta dentro de `salida`. Nunca `results/data/pd/`, `models/` ni rutas fuera del repositorio (`..`, absolutas). La
  raíz es inyectable para las pruebas.
- **Salidas:** `results/data/pd_umbral/umbral.json` (bloques `trivial`, `matriz_inicial`, `sensibilidad_umbral`,
  `escenarios`, `curva_costo`, `metas`, `semillas`, `control_semilla_mediana`, más `config`, `tiempo_s` y `evidencia` =
  la de `experimento.evidencia` + SHA-256 de los dos modelos leídos) y `pred_umbral.parquet` (`id`, `split` ∈ {val, test},
  `y`, `pred_logistica`, `pred_mlp`). Figuras `pd-umbral-sensibilidad.png` y `pd-umbral-costo.png`
  (`src/viz/umbral_pd.py`, con `SALIDA` + `generar(raiz, destino, datos)`; **sin** el prefijo `figs_` a propósito: todo
  `figs_*` debe estar conectado a `run.py figures` (`tests/test_no_network.py`) y eso tocaría `cli.py`, compartido).
- **Comando:** `python scripts/umbral_pd.py --config configs/pd/umbral.yaml [--set clave=valor …]` (como
  `scripts/diagnostico_nivel_pd.py`, para no tocar `cli.py` ni `run.py`, que son compartidos): escribe el JSON, el parquet
  y regenera las 2 figuras; código 2 y una línea en stderr ante un error esperado. Demo: `--set semillas.n=3 --set
  salida=results/data/pd_umbral_demo --set figuras=results/data/pd_umbral_demo/figuras`. En el PR se le propone a Daniel
  conectar `src/viz/umbral_pd.py` a `run.py figures` (renombrándolo `figs_umbral_pd.py`).
- **Código:** formato percent (`# %%`), comentario explicativo por línea, responsabilidad única (`umbral.py` cálculo,
  `umbral_experimento.py` orquestación con dependencias inyectadas, `tablas_umbral.py` tablas, `src/viz/umbral_pd.py`
  figuras, `scripts/umbral_pd.py` CLI) e inversión de dependencias (cargar, entrenar, predecir y escribir se inyectan).
- **Blog:** 05 **§5.8** (convención, matriz, TPR/TNR, relación con el KS = máx(TPR + TNR − 1), costo, derivación de `p*`,
  Brier del trivial) y 06 **§6.8** después de Limitaciones, que sigue siendo §6.7 porque el blog 10 la cita (6.8.1 frente
  al trivial · 6.8.2 matriz con 0,5 · 6.8.3 sensibilidad al umbral · 6.8.4 cuota por costo y metas, con la variación de
  las semillas · 6.8.5 PD absoluta frente a cuota y `p*` · 6.8.6 conclusión y próximos pasos: recalibrar, inferencia de
  rechazados, monitoreo con PSI y fijar `L/G` con datos del negocio; una frase remite a la sección 7: un corte en PD
  equivale a un corte en puntaje). Sin puntos de la scorecard. Blog 07–13 no se editan.

## 4. Tareas
| ID | Tarea | Depende | Aceptación | + Verificador | Fecha | Quién |
|---|---|---|---|---|---|---|
| ✅ B1 | `metricas.py` con datos sintéticos | E0 | `test_metricas.py` coincide con sklearn/scipy | **verificador-matematico** | vie 23 | Andrés |
| ✅ B2 | MLP + `entrenar.py` (parada temprana, dropout, L2, Adam/SGD, `n_parametros`) | E0 | `test_mlp.py`: determinista en CPU, restaura los mejores pesos | verificador-matematico | vie 23, mediodía | Andrés |
| ✅ B3 | Logística con búsqueda de `C` | B1 | Presupuesto registrado; resultados en `results/data/pd/` | revisor-spec | vie 23 | Andrés |
| ✅ B4 | Optuna del MLP (con datos reales) | A5, B2 | `hpo_ensayos.json` con 40 ensayos y tiempo | analista-experimentos | vie 23, noche | Andrés |
| ✅ B5 | 30 semillas + `summary.json` + modelos guardados | B4 | Media, desviación, mejor, peor; parámetros | analista-experimentos | **sáb 24, mediodía** | Andrés |
| ✅ B6 | Demo: `configs/pd/demo.yaml` + `datos.muestra` (§3.5, v0.20) | B2, B3, B5 | **< 2 min en CPU medido**: 13,8–15,1 s en la máquina de desarrollo y 5,0 s en otra máquina (clon limpio en un runner Ubuntu 24.04 de GitHub, sin red; `auditoria.yml`, 28-oct) | auditor-reproducibilidad | sáb 24 | Andrés |
| ✅ B7 | Figuras: ROC, KS, calibración, cajas por semilla + aprendizaje y búsquedas; JSON de calibración para el video (§3.6, v0.21) | B5 | `run.py figures` las regenera | revisor-spec | sáb 24 (hecha 26-oct) | Andrés |
| ✅ B9 | Notebook `02_pd` (núcleo §5.8) con la estructura de §3.7 (v0.22): moldes y comando de terminal, demo en la sección 1, datos, búsqueda y modelo final; `src/pd/tablas.py` | B5, B7 | Corre en < 2 min; la celda de parámetros cambia la arquitectura sin tocar código; los moldes coinciden con `demo.yaml` (prueba); no escribe en `results/`, `models/` ni `blog/` (`test_notebooks.py`) | revisor-spec | dom 25 (hecha 26-oct) | Andrés |
| ✅ B10 | Sensibilidad con `grade` e `int_rate` (Daniel, 23-oct; detalle en §3.8, v0.23): la logística y el MLP con la configuración final, sobre las 15 variables + `configs/features_sensibilidad.yaml` (unir `data/processed/sensibilidad_lc.parquet` por `id`; preprocesador propio ajustado solo con train). Misma partición; 30 semillas del MLP si alcanza el tiempo, si no 5 y se dice. **No va a la app ni reemplaza al modelo principal** | B5 | Tabla en el blog 06: AUC/Gini/KS/Brier del modelo principal vs. con las variables de Lending Club, y la lectura (cuánto de la discriminación ya está en la nota de LC) | analista-experimentos | dom 25 (hecha 27-oct) | Andrés |
| ✅ B8 | Blog 04–06 | B5, B7 | Derivaciones revisadas; cifras desde `summary.json` | redactor-tecnico + verificador-matematico | dom 25 (hecha 27-oct) | Andrés |
| B11 | **[EXTRA, opcional]** De la PD a la decisión (§3.9, v0.29): umbral, matriz de confusión, costo `L/G` y variación en las 30 semillas; script, figuras, notebook 02 §5 y blog §5.8 y §6.8 | B5, B7, B9 | Pruebas en verde; `umbral.json` reproducible con un comando; control de la semilla mediana; no escribe en `results/data/pd/` ni `models/`; blog sin figuras huérfanas | verificador-matematico + analista-experimentos | vie 30 | Andrés |

(E0 = esqueleto del puente, jue 22.)

## 5. Pruebas de la parte
`test_metricas.py`, `test_mlp.py`, `test_logistica.py`, `test_hpo.py`, `test_experimento_pd.py` (B5: semillas, summary, pred_test, `pd_mlp.pt` y `run.py pd`), `test_demo_pd.py` (marcada lenta), `test_tablas_pd.py` (B9: tablas del notebook 02 y moldes = `demo.yaml`), `test_notebook_02_pd.py` (B9; las que ejecutan el notebook, lentas), `test_espacio.py` (B9: `espacio.py` sin Optuna, mismo objeto en `hpo` y `tablas`), `test_sensibilidad_pd.py` (B10), `test_umbral_pd.py`, `test_umbral_experimento.py`, `test_tablas_umbral.py` y `test_viz_umbral_pd.py` (B11).

## 6. Puntos calientes de alucinación
- L2 vs. `AdamW`; "dropout se aplica también al predecir"; `BCELoss` sobre logits sin sigmoide.
- Fórmulas de KS y Gini (KS calculado sobre la curva ROC mal; Gini como 2·AUC sin restar 1).
- Derivación de la BCE: verosimilitud Bernoulli → log-verosimilitud negativa.
- Ecuaciones de Adam (corrección de sesgo de m y v).
- Afirmaciones del tipo "las redes siempre superan a la logística en datos tabulares".

## 7. Blog y fundamentos
04: neurona, MLP, activaciones, BCE por máxima verosimilitud, retropropagación, Adam/SGD, L2, dropout, parada temprana.
05: definición de AUC, Gini, KS, Brier y curva de calibración. 06: resultados de PD y comparación justa (tabla con parámetros,
configuraciones probadas, tiempo y métricas con su variabilidad). B11 (§3.9): 05 §5.8 (matriz de confusión, TPR/TNR,
costo y `p*`) y 06 §6.8 (de la PD a la decisión).

## 8. Definición de terminado
`python run.py pd --config configs/pd/base.yaml` reproduce `summary.json`; la demo corre en < 2 min en CPU; C y E usan el modelo
guardado sin reentrenar; ≥ 1 hallazgo o verificación documentada.

Reproducción de `summary.json` [DECIDIDO, Andrés, 26-oct; v0.16]: en la misma máquina, con la misma CPU, los mismos `hilos_torch` y `version_torch`, todo `summary.json` se reproduce bit a bit salvo los campos de tiempo. En otra máquina, la media de cada métrica debe quedar a menos de 3·√2·EE de la de referencia, con EE = desv/√n del `summary.json` de referencia (3 = convención de tres sigmas; √2 porque se comparan dos medias independientes con la misma varianza).
