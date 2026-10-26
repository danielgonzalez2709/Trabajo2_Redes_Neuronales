# Parte B — PD: modelos y evaluación

| | |
|---|---|
| Responsables | **Andrés** (reparto de la pareja, 24-oct; Juan José hace la Parte C) |
| Rama personal | `dev/andres` |
| Carpetas propias | `src/pd/`, `src/common/metricas.py`, `src/common/torch_utils.py`, `configs/pd/`, `src/viz/figs_pd.py`, `notebooks/02_pd.py`, `blog/secciones/04–06` |
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

### 3.4 Métricas (`src/common/metricas.py`)
- AUC; **Gini = 2·AUC − 1**; **KS = máx_t |F₁(t) − F₀(t)|** (acumuladas de la PD en malos y buenos).
- **Brier = (1/n) Σ (pᵢ − yᵢ)²**; curva de calibración con 10 bins por cuantiles.
- Contraste en pruebas con `roc_auc_score`, `brier_score_loss`, `scipy.stats.ks_2samp`.

### 3.5 Corridas finales
- MLP final: **30 semillas** (`base_seed + 0…29`), misma partición; `summary.json` con media, desviación, mejor y peor.
- Se guardan `pd_logistica.joblib`, `pd_mlp.pt` (semilla con la métrica mediana) y `pred_test.parquet`.
- Demo `configs/pd/demo.yaml`: muestra de datos, MLP pequeño, pocas épocas, 3 semillas, CPU, **< 2 min medido**.
- Decisiones de B5 [DECIDIDO, Andrés, 26-oct; Spec v0.16]: `python run.py pd --config configs/pd/base.yaml` corre la logística (§3.1) y después el MLP con el bloque `mlp` resuelto (herencia de `mlp_final.yaml` y luego `--set`), con `semillas.n` semillas efectivas `base_seed + 0…n−1` (`seeds_for_runs`), un `MLPPD` nuevo por semilla y `tiempo_s` = solo el `fit`. `configs/pd/base.yaml` declara `semillas: {n: 30}`, `salida: results/data/pd` y `modelos: models` (carpetas; los nombres de archivo son los del núcleo §5.4). `semillas.n` entero ≥ 1, `base_seed` entero ≥ 0, el bloque `mlp` y el bloque `busqueda` se validan antes de leer datos y antes de correr la logística (`ValueError`).
- **Semilla mediana:** las corridas se ordenan por (`logloss_val`, `seed`) de menor a mayor y se toma la posición ⌊(n − 1)/2⌋ (n par → mediana inferior; empate → semilla menor). Decide validación, como la búsqueda (§3.3); test nunca decide. Su red (la entrenada, sin reentrenar) va a `models/pd_mlp.pt` y sus PD de test a `pred_mlp`.
- **Config de la búsqueda:** `config_de_busqueda` es true si el bloque `mlp` validado de la corrida es igual al de `configs/pd/mlp_final.yaml` releído y validado, y el bloque `busqueda` también coincide. Si es true, `busqueda.tipo` debe ser `optuna` y `busqueda.evidencia` igual a la de esta corrida (si no → `ValueError`: "la búsqueda se hizo con otros datos; repite pd-hpo"), y el presupuesto del MLP es `n_configuraciones_busqueda = busqueda.n_ensayos`, `tiempo_total_s = busqueda.tiempo_s + Σ tiempo_s`. Si es false (p. ej. `--set mlp.capas=[8]`), la consola avisa qué claves difieren y el presupuesto de la búsqueda queda en null.
- Antes de entrenar se borran solo los archivos `mlp_seed<entero>.json` de `{salida}/semillas`; si un `fit` lanza `FloatingPointError`, el error sale con la semilla y avisa que `summary.json` conserva la entrada `mlp` anterior, que ya no corresponde a `semillas/` (hay que volver a correr); no se escriben `summary.json`, `pred_mlp` ni `pd_mlp.pt` (quedan las semillas terminadas). La huella del pre-chequeo (solo `id, split`) debe ser idéntica a la del DataFrame completo (lo comprueba una prueba). Campos extra de la semilla y del summary: núcleo §5.4 (v0.16).

## 4. Tareas
| ID | Tarea | Depende | Aceptación | + Verificador | Fecha | Quién |
|---|---|---|---|---|---|---|
| B1 | `metricas.py` con datos sintéticos | E0 | `test_metricas.py` coincide con sklearn/scipy | **verificador-matematico** | vie 23 | Andrés |
| ✅ B2 | MLP + `entrenar.py` (parada temprana, dropout, L2, Adam/SGD, `n_parametros`) | E0 | `test_mlp.py`: determinista en CPU, restaura los mejores pesos | verificador-matematico | vie 23, mediodía | Andrés |
| ✅ B3 | Logística con búsqueda de `C` | B1 | Presupuesto registrado; resultados en `results/data/pd/` | revisor-spec | vie 23 | Andrés |
| ✅ B4 | Optuna del MLP (con datos reales) | A5, B2 | `hpo_ensayos.json` con 40 ensayos y tiempo | analista-experimentos | vie 23, noche | Andrés |
| B5 | 30 semillas + `summary.json` + modelos guardados | B4 | Media, desviación, mejor, peor; parámetros | analista-experimentos | **sáb 24, mediodía** | Andrés |
| B6 | Demo | B2, B3 | **< 2 min en CPU medido** | auditor-reproducibilidad | sáb 24 | Andrés |
| B7 | Figuras: ROC, KS, calibración, cajas por semilla | B5 | `run.py figures` las regenera | revisor-spec | sáb 24 | Andrés |
| B9 | Notebook `02_pd` (núcleo §5.8) | B5, B7 | Corre en < 2 min; la celda de parámetros cambia la arquitectura sin tocar código | revisor-spec | dom 25 | Andrés |
| B10 | Sensibilidad con `grade` e `int_rate` (Daniel, 23-oct): la logística y el MLP con la configuración final, sobre las 15 variables + `configs/features_sensibilidad.yaml` (unir `data/processed/sensibilidad_lc.parquet` por `id`; preprocesador propio ajustado solo con train). Misma partición; 30 semillas del MLP si alcanza el tiempo, si no 5 y se dice. **No va a la app ni reemplaza al modelo principal** | B5 | Tabla en el blog 06: AUC/Gini/KS/Brier del modelo principal vs. con las variables de Lending Club, y la lectura (cuánto de la discriminación ya está en la nota de LC) | analista-experimentos | dom 25 | Andrés |
| B8 | Blog 04–06 | B5, B7 | Derivaciones revisadas; cifras desde `summary.json` | redactor-tecnico + verificador-matematico | dom 25 | Andrés |

(E0 = esqueleto del puente, jue 22.)

## 5. Pruebas de la parte
`test_metricas.py`, `test_mlp.py`, `test_logistica.py`, `test_hpo.py`, `test_experimento_pd.py` (B5: semillas, summary, pred_test, `pd_mlp.pt` y `run.py pd`), `test_demo_pd.py` (marcada lenta).

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
