# Parte B — PD: modelos y evaluación

| | |
|---|---|
| Responsables | **Andrés** (reparto de la pareja, 24-oct; Juan José hace la Parte C) |
| Rama personal | `dev/andres` |
| Carpetas propias | `src/pd/`, `src/common/metricas.py`, `src/common/torch_utils.py`, `configs/pd/`, `src/viz/figs_pd.py`, `src/pd/tablas.py`, `src/pd/espacio.py`, `notebooks/02_pd.py`, `blog/secciones/04–06` |
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
  (2019), arXiv:1711.05101: L2 y weight decay coinciden en SGD pero no en Adam [VERIFICAR: TODO(cita) en el blog].
- Desbalance [PROPUESTO]: `pos_weight = 1` por defecto para no descalibrar la PD (la PD alimenta la scorecard y la E[L]). Otro
  valor solo como sensibilidad, reportando su efecto en Brier y calibración.
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
  TPE multivariado: TODO(cita) Falkner et al. (2018) [VERIFICAR].
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
- **`src/pd/tablas.py`** (con `tests/test_tablas_pd.py`): las funciones que arman las tablas y textos de arriba a partir
  de los JSON ya cargados (puras: no leen archivos ni entrenan), más la que redirige `salida` y `modelos` de la config a
  la carpeta temporal y la que devuelve los moldes desde `demo.yaml`. El notebook solo las llama y muestra el resultado.

## 4. Tareas
| ID | Tarea | Depende | Aceptación | + Verificador | Fecha | Quién |
|---|---|---|---|---|---|---|
| B1 | `metricas.py` con datos sintéticos | E0 | `test_metricas.py` coincide con sklearn/scipy | **verificador-matematico** | vie 23 | Andrés |
| ✅ B2 | MLP + `entrenar.py` (parada temprana, dropout, L2, Adam/SGD, `n_parametros`) | E0 | `test_mlp.py`: determinista en CPU, restaura los mejores pesos | verificador-matematico | vie 23, mediodía | Andrés |
| ✅ B3 | Logística con búsqueda de `C` | B1 | Presupuesto registrado; resultados en `results/data/pd/` | revisor-spec | vie 23 | Andrés |
| ✅ B4 | Optuna del MLP (con datos reales) | A5, B2 | `hpo_ensayos.json` con 40 ensayos y tiempo | analista-experimentos | vie 23, noche | Andrés |
| ✅ B5 | 30 semillas + `summary.json` + modelos guardados | B4 | Media, desviación, mejor, peor; parámetros | analista-experimentos | **sáb 24, mediodía** | Andrés |
| B6 | Demo: `configs/pd/demo.yaml` + `datos.muestra` (§3.5, v0.20) | B2, B3, B5 | **< 2 min en CPU medido**: 13,8–15,1 s aquí; **pendiente: otra máquina** | auditor-reproducibilidad | sáb 24 | Andrés |
| ✅ B7 | Figuras: ROC, KS, calibración, cajas por semilla + aprendizaje y búsquedas; JSON de calibración para el video (§3.6, v0.21) | B5 | `run.py figures` las regenera | revisor-spec | sáb 24 (hecha 26-oct) | Andrés |
| ✅ B9 | Notebook `02_pd` (núcleo §5.8) con la estructura de §3.7 (v0.22): moldes y comando de terminal, demo en la sección 1, datos, búsqueda y modelo final; `src/pd/tablas.py` | B5, B7 | Corre en < 2 min; la celda de parámetros cambia la arquitectura sin tocar código; los moldes coinciden con `demo.yaml` (prueba); no escribe en `results/`, `models/` ni `blog/` (`test_notebooks.py`) | revisor-spec | dom 25 (hecha 26-oct) | Andrés |
| B10 | Sensibilidad con `grade` e `int_rate` (Daniel, 23-oct): la logística y el MLP con la configuración final, sobre las 15 variables + `configs/features_sensibilidad.yaml` (unir `data/processed/sensibilidad_lc.parquet` por `id`; preprocesador propio ajustado solo con train). Misma partición; 30 semillas del MLP si alcanza el tiempo, si no 5 y se dice. **No va a la app ni reemplaza al modelo principal** | B5 | Tabla en el blog 06: AUC/Gini/KS/Brier del modelo principal vs. con las variables de Lending Club, y la lectura (cuánto de la discriminación ya está en la nota de LC) | analista-experimentos | dom 25 | Andrés |
| B8 | Blog 04–06 | B5, B7 | Derivaciones revisadas; cifras desde `summary.json` | redactor-tecnico + verificador-matematico | dom 25 | Andrés |

(E0 = esqueleto del puente, jue 22.)

## 5. Pruebas de la parte
`test_metricas.py`, `test_mlp.py`, `test_logistica.py`, `test_hpo.py`, `test_experimento_pd.py` (B5: semillas, summary, pred_test, `pd_mlp.pt` y `run.py pd`), `test_demo_pd.py` (marcada lenta), `test_tablas_pd.py` (B9: tablas del notebook 02 y moldes = `demo.yaml`), `test_notebook_02_pd.py` (B9; las que ejecutan el notebook, lentas), `test_espacio.py` (B9: `espacio.py` sin Optuna, mismo objeto en `hpo` y `tablas`).

## 6. Puntos calientes de alucinación
- L2 vs. `AdamW`; "dropout se aplica también al predecir"; `BCELoss` sobre logits sin sigmoide.
- Fórmulas de KS y Gini (KS calculado sobre la curva ROC mal; Gini como 2·AUC sin restar 1).
- Derivación de la BCE: verosimilitud Bernoulli → log-verosimilitud negativa.
- Ecuaciones de Adam (corrección de sesgo de m y v).
- Afirmaciones del tipo "las redes siempre superan a la logística en datos tabulares".

## 7. Blog y fundamentos
04: neurona, MLP, activaciones, BCE por máxima verosimilitud, retropropagación, Adam/SGD, L2, dropout, parada temprana.
05: definición de AUC, Gini, KS, Brier y curva de calibración. 06: resultados de PD y comparación justa (tabla con parámetros,
configuraciones probadas, tiempo y métricas con su variabilidad).

## 8. Definición de terminado
`python run.py pd --config configs/pd/base.yaml` reproduce `summary.json`; la demo corre en < 2 min en CPU; C y E usan el modelo
guardado sin reentrenar; ≥ 1 hallazgo o verificación documentada.

Reproducción de `summary.json` [DECIDIDO, Andrés, 26-oct; v0.16]: en la misma máquina, con la misma CPU, los mismos `hilos_torch` y `version_torch`, todo `summary.json` se reproduce bit a bit salvo los campos de tiempo. En otra máquina, la media de cada métrica debe quedar a menos de 3·√2·EE de la de referencia, con EE = desv/√n del `summary.json` de referencia (3 = convención de tres sigmas; √2 porque se comparan dos medias independientes con la misma varianza).
