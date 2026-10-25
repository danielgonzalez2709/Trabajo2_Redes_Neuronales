# Parte D — LGD (pérdida dado incumplimiento)

| | |
|---|---|
| Responsables | **Camila y José** |
| Ramas personales | `dev/camila`, `dev/jose` |
| Carpetas propias | `src/lgd/`, `configs/lgd/`, `src/viz/figs_lgd.py`, `notebooks/04_lgd.py`, `blog/secciones/08` |
| Lee del núcleo | §0, §3, §4.1, §5.1, §5.2, §5.3, §5.4, §6, §7.1 · **Anexo (enunciado):** Parte 2, puntos 1–4; Regla de comparación justa; Discusión; Entregables 1 (fundamentación matemática) y 3 |
| Arranca sin esperar | Sí: la variable objetivo se construye desde el CSV crudo (público) y los modelos con datos sintéticos |

## 1. Lo que pide el enunciado
- P2.1 Para los créditos que incumplieron, construir la LGD con variables del conjunto: **fórmula, justificación de cada campo y
  limitaciones**.
- P2.2 Modelo de baja complejidad (p. ej. regresión lineal).
- P2.3 MLP optimizado; **justificar la pérdida y la activación de salida** según la naturaleza de la variable objetivo.
- P2.4 **MAE y RMSE** y comparación de la **distribución** predicha con la observada.
- Comparación justa: mismo preprocesamiento y partición, presupuesto explícito, **30 semillas**, parámetros entrenables.

## 2. Consumo / Entrego
| Consumo | De | Cuándo |
|---|---|---|
| Esqueleto y datos sintéticos | E | jue 22, noche |
| `entrenar.py` del MLP | B | vie 23, mediodía (mientras tanto, un bucle propio mínimo) |
| Preprocesador y partición reales | A | vie 23, 2 p. m. |

| Entrego | A | Cuándo |
|---|---|---|
| `loans_lgd.parquet` | E | vie 23, noche |
| Modelo LGD final, `summary.json`, `pred_test.parquet` | E (E[L]) | **sáb 24, 6 p. m.** |
| Sección 08 del blog | E | dom 25, 8 p. m. |

## 3. Especificación

### 3.1 Población [PROPUESTO]
Créditos con recuperación observable: **`Charged Off`** (incluida la variante "Does not meet the credit policy").
`Default` y `Late (31-120 days)` todavía no terminan su cobranza: se excluyen de la LGD y se documenta.

### 3.2 Variable objetivo [PROPUESTO — verificar cada campo en el diccionario]
- Los campos ya están en `data/processed/montos.parquet` (núcleo §5.1, v0.10): no hace falta leer el CSV crudo. Unir con
  `loans_pd.parquet` por `id`.
- Las definiciones para citar cada campo están en `data/processed/diccionario_loanstats.csv` (núcleo §5.1, v0.11), copiadas
  textualmente del diccionario de Kaggle.
- Además de `pred_test.parquet`, entregar `results/data/lgd/pred_pd_test.parquet` con la LGD predicha para **todos** los
  créditos de test de `loans_pd` (núcleo §5.4): la E[L] la necesita para cada crédito, no solo para los incumplidos.
- **EAD en el incumplimiento = `funded_amnt` − `total_rec_prncp`** (principal pendiente al castigo).
- **LGD = 1 − (`recoveries` − `collection_recovery_fee`) / EAD**, recortada a [0, 1]; se reporta cuántos casos quedan fuera
  antes del recorte (`lgd_sin_recorte`) y qué se hace con EAD ≤ 0.
- Limitaciones para el blog: sin descuento del dinero en el tiempo; solo el costo de cobranza registrado; recuperaciones abiertas
  en castigos recientes (censura); intereses no incluidos; masa de probabilidad en los extremos.

### 3.2b Análisis descriptivo de la LGD
- **Variable objetivo observada (antes de modelar):** histograma de la LGD, masa en 0 y en 1, casos recortados, EAD y LGD por
  año de originación y por plazo. Es la evidencia para justificar la pérdida y la activación de salida, que el enunciado pide
  justificar "a partir de la naturaleza de la variable objetivo".
- **Después del preprocesamiento:** las funciones de la Parte A (A §3.8) aplicadas al subconjunto de LGD: efecto de las
  transformaciones, estabilidad entre splits y relación de cada variable con la LGD.
- Salida: `results/data/lgd/descriptivo.json` + figuras; subsección en el blog 08 y en el notebook 04.

### 3.3 Modelos
- Referencia: `LinearRegression` sobre el mismo preprocesamiento (puede predecir fuera de [0, 1]: se reporta cuántas veces).
- MLP (núcleo §5.3) con salida **sigmoide** (LGD ∈ [0, 1]) y pérdida [PROPUESTO] **entropía cruzada con objetivo fraccional**
  (cuasi-verosimilitud Bernoulli; Papke & Wooldridge, 1996 [VERIFICAR]), derivada por máxima verosimilitud en el blog.
  Contraste opcional: MSE (verosimilitud gaussiana).
- Búsqueda Optuna TPE con **30 ensayos** (demo: 5), mismo espacio que B; **30 semillas** de la configuración final.
- Variables: las de `features.yaml` (información de la originación).

### 3.4 Evaluación
MAE y RMSE en val y test; histograma o densidad y gráfico Q-Q de predicha vs. observada; error por banda de LGD.

## 4. Tareas
| ID | Tarea | Depende | Aceptación | + Verificador | Fecha | Quién |
|---|---|---|---|---|---|---|
| D1 | Población, EAD y LGD desde el CSV crudo, con la cita del diccionario por campo | E0 | `test_lgd.py` con casos hechos a mano; conteo fuera de [0, 1] | **curador-datos** + cazador | vie 23 | por repartir |
| D1b | Descriptivo de la LGD observada y después del preprocesamiento (§3.2b) | D1 (A6b para la parte posterior al preprocesamiento) | `descriptivo.json`; figuras citadas en el blog | analista-experimentos | sáb 24 | por repartir |
| D2 | `loans_lgd.parquet` con el preprocesador y la partición reales | A5, D1 | Misma partición temporal; esquema = núcleo §5.1 | revisor-spec | vie 23, noche | por repartir |
| D3 | Lineal + MLP de LGD (salida sigmoide, pérdida §3.3) | E0 (B2 cuando llegue) | Derivación de la pérdida revisada; `test_lgd_mlp.py` | verificador-matematico | vie 23 | por repartir |
| D4 | Optuna + 30 semillas + `summary.json` | D2, D3 | Presupuesto y semillas según núcleo §6 | analista-experimentos | **sáb 24, 6 p. m.** | por repartir |
| D5 | Demo | D3 | **< 2 min en CPU** | auditor-reproducibilidad | sáb 24 | por repartir |
| D6 | Figuras: distribuciones, Q-Q, error por banda | D4 | `run.py figures` las regenera | revisor-spec | sáb 24 | por repartir |
| D8 | Notebook `04_lgd` (núcleo §5.8) | D4, D6 | Corre en < 2 min; cero lógica propia | revisor-spec | dom 25 | por repartir |
| D7 | Blog 08 | D4, D6 | Cifras desde `summary.json` | redactor-tecnico + verificador-matematico | dom 25 | por repartir |

## 5. Pruebas de la parte
`test_lgd.py` (fórmula, recorte, EAD ≤ 0), `test_lgd_mlp.py` (salida en [0, 1], pérdida con objetivo fraccional),
`test_demo_lgd.py` (marcada lenta).

## 6. Puntos calientes de alucinación
- Fórmulas de LGD "de memoria" con `total_pymnt` o `loan_amnt` como denominador, o sin restar `collection_recovery_fee`.
- Significado de `recoveries` y `collection_recovery_fee` (¿la tarifa ya está descontada?): diccionario + datos.
- "La BCE solo sirve con objetivos 0/1" (falso para la cuasi-verosimilitud fraccional).
- Afirmaciones sobre la forma típica de la distribución de LGD sin medirla.

## 7. Blog y fundamentos
08: fórmula de LGD y EAD con cada campo justificado, limitaciones, derivación de la pérdida por máxima verosimilitud, por qué
sigmoide en la salida, resultados (MAE, RMSE, variabilidad entre semillas, parámetros, presupuesto) y comparación de distribuciones.

## 8. Definición de terminado
`python run.py lgd --config configs/lgd/base.yaml` reproduce `summary.json`; la demo corre en < 2 min; E usa `pred_test` para
la E[L]; ≥ 1 hallazgo o verificación documentada.
