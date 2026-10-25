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

### 3.2 MLP (núcleo §5.3)
- `Linear → activación → Dropout` repetido; salida `Linear(·, 1)` con logit; pérdida `BCEWithLogitsLoss` (estable).
- Regularización: dropout, **L2** y parada temprana sobre la pérdida de validación (restaura los mejores pesos).
  ⚠️ L2 ≠ weight decay desacoplado: `AdamW` no es L2. Si el blog dice "penalización L2", el código debe hacer L2 (término
  λ‖W‖² en la pérdida, o `weight_decay` de `Adam`/`SGD`, que lo suma al gradiente) [VERIFICAR con Loshchilov & Hutter y la
  documentación de PyTorch].
- Desbalance [PROPUESTO]: `pos_weight = 1` por defecto para no descalibrar la PD (la PD alimenta la scorecard y la E[L]). Otro
  valor solo como sensibilidad, reportando su efecto en Brier y calibración.
- `entrenar.py` es genérico (recibe pérdida y modelo): D lo reutiliza para la LGD.

### 3.3 Búsqueda [PROPUESTO]
- Optuna TPE, **40 ensayos** (`configs/pd/hpo.yaml`; demo: 5), semilla del muestreador fija.
- Espacio: capas 1–4; neuronas {16, 32, 64, 128, 256}; activación {relu, leaky_relu, tanh}; dropout [0, 0.5]; l2 log-uniforme
  [1e-6, 1e-2]; optimizador {adam, sgd con momento}; lr log-uniforme [1e-4, 1e-2]; batch {256, 512, 1024, 2048}.
- Criterio: **log-loss de validación** (regla de puntuación propia: premia discriminación y calibración); AUC de validación al lado.
- Justificación citada: TPE (Bergstra et al., 2011) y Optuna (Akiba et al., 2019) [VERIFICAR referencias].

### 3.4 Métricas (`src/common/metricas.py`)
- AUC; **Gini = 2·AUC − 1**; **KS = máx_t |F₁(t) − F₀(t)|** (acumuladas de la PD en malos y buenos).
- **Brier = (1/n) Σ (pᵢ − yᵢ)²**; curva de calibración con 10 bins por cuantiles.
- Contraste en pruebas con `roc_auc_score`, `brier_score_loss`, `scipy.stats.ks_2samp`.

### 3.5 Corridas finales
- MLP final: **30 semillas** (`base_seed + 0…29`), misma partición; `summary.json` con media, desviación, mejor y peor.
- Se guardan `pd_logistica.joblib`, `pd_mlp.pt` (semilla con la métrica mediana) y `pred_test.parquet`.
- Demo `configs/pd/demo.yaml`: muestra de datos, MLP pequeño, pocas épocas, 3 semillas, CPU, **< 2 min medido**.

## 4. Tareas
| ID | Tarea | Depende | Aceptación | + Verificador | Fecha | Quién |
|---|---|---|---|---|---|---|
| B1 | `metricas.py` con datos sintéticos | E0 | `test_metricas.py` coincide con sklearn/scipy | **verificador-matematico** | vie 23 | Andrés |
| B2 | MLP + `entrenar.py` (parada temprana, dropout, L2, Adam/SGD, `n_parametros`) | E0 | `test_mlp.py`: determinista en CPU, restaura los mejores pesos | verificador-matematico | vie 23, mediodía | Andrés |
| B3 | Logística con búsqueda de `C` | B1 | Presupuesto registrado; resultados en `results/data/pd/` | revisor-spec | vie 23 | Andrés |
| B4 | Optuna del MLP (con datos reales, de noche) | A5, B2 | `hpo_ensayos.json` con 40 ensayos y tiempo | analista-experimentos | vie 23, noche | Andrés |
| B5 | 30 semillas + `summary.json` + modelos guardados | B4 | Media, desviación, mejor, peor; parámetros | analista-experimentos | **sáb 24, mediodía** | Andrés |
| B6 | Demo | B2, B3 | **< 2 min en CPU medido** | auditor-reproducibilidad | sáb 24 | Andrés |
| B7 | Figuras: ROC, KS, calibración, cajas por semilla | B5 | `run.py figures` las regenera | revisor-spec | sáb 24 | Andrés |
| B9 | Notebook `02_pd` (núcleo §5.8) | B5, B7 | Corre en < 2 min; la celda de parámetros cambia la arquitectura sin tocar código | revisor-spec | dom 25 | Andrés |
| B10 | Sensibilidad con `grade` e `int_rate` (Daniel, 23-oct): la logística y el MLP con la configuración final, sobre las 15 variables + `configs/features_sensibilidad.yaml` (unir `data/processed/sensibilidad_lc.parquet` por `id`; preprocesador propio ajustado solo con train). Misma partición; 30 semillas del MLP si alcanza el tiempo, si no 5 y se dice. **No va a la app ni reemplaza al modelo principal** | B5 | Tabla en el blog 06: AUC/Gini/KS/Brier del modelo principal vs. con las variables de Lending Club, y la lectura (cuánto de la discriminación ya está en la nota de LC) | analista-experimentos | dom 25 | Andrés |
| B8 | Blog 04–06 | B5, B7 | Derivaciones revisadas; cifras desde `summary.json` | redactor-tecnico + verificador-matematico | dom 25 | Andrés |

(E0 = esqueleto del puente, jue 22.)

## 5. Pruebas de la parte
`test_metricas.py`, `test_mlp.py`, `test_logistica.py`, `test_demo_pd.py` (marcada lenta).

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
