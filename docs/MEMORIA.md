# 🧠 Memoria del proyecto (contexto vivo)

> **Para qué:** que cualquier integrante (o sesión de Claude / subagente) sepa en 2 minutos **dónde estamos, qué se decidió
> y qué sigue**. Se carga automáticamente en cada sesión de Claude Code (vía `CLAUDE.md`).
>
> **Reglas:** (1) actualizar al cerrar una tarea, fusionar un PR o tomar una decisión; (2) ≤ 150 líneas: aquí va el *estado*,
> el detalle vive en el Spec y el PLAN; (3) bitácora de lo más nuevo a lo más viejo; (4) toda decisión se copia al Spec.

---

## 📍 Estado actual
| | |
|---|---|
| Última actualización | 2026-10-25 |
| Fase | En `main`: Parte A (PR #1), reparto B+C (PR #2) y Parte E parcial con `montos.parquet` y contratos v0.10 (PR #3, 24-oct 10:26 p. m.). Spec v0.12 · `dev/andres` (PR #6 abierto, lo revisa Daniel): B1 métricas ✅, B2 MLP + entrenar ✅ y B3 logística ✅; resultados reales de la logística en `results/data/pd/` y `models/pd_logistica.joblib`; Spec v0.13 (L2) y v0.14 (B3) |
| Próxima tarea | Todos: `git pull origin main` en su rama. B (Andrés): B1–B4 (Optuna de noche). C (Juan José): C1–C2. D (Camila y José): D1–D3. Camila: E1 (app). Daniel: blog 09, notebook 05, confirmar H-01 y H-02. Andrés (tras el PR): B4 (Optuna) y B5. Juan José: C3 ya puede usar `models/pd_logistica.joblib` (`LogisticaPD.cargar`, `predict_logit`, `coeficientes()`) |
| Entrega | **mar 27-oct-2026, 23:59** (meta interna 6 p. m.; días restantes: 2) |
| Sustentación | Posterior al 27-oct, por confirmar |

## 👥 Equipo
| Parte | Spec | Quién |
|---|---|---|
| A — Datos y caso de uso | `docs/spec/partes/A-datos.md` | Daniel |
| B — PD: modelos y evaluación | `docs/spec/partes/B-pd.md` | Andrés |
| C — Scorecard y variables de riesgo | `docs/spec/partes/C-scorecard.md` | Juan José |
| D — LGD | `docs/spec/partes/D-lgd.md` | Camila y José |
| E — Integración y producto (puente) | `docs/spec/partes/E-producto.md` | Daniel y Camila (Camila: frontend de la app) |

Ramas personales: `dev/daniel`, `dev/camila`, `dev/andres`, `dev/juanjose`, `dev/jose`.

Integrantes: Edinson Andrés Ariza Mendoza, Maria Camila Agudelo Espinosa, Daniel Gonzalez Henao, Jose Miguel Pulgarin Agudelo,
Juan José González Morales.

## ✅ Decisiones vigentes
- Reparto B+C (Andrés y Juan José, 24-oct): Andrés hace la Parte B y Juan José la Parte C (núcleo §1.1, Spec v0.9).
- Datos (Daniel, 23-oct): partición train ≤ 2012 / val 2013 / test 2014 (2015 fuera: 93 % sin desenlace); 15 variables de
  originación; `grade` e `int_rate` solo en un análisis de sensibilidad aparte (B10, `sensibilidad_lc.parquet`).
- Git: rama personal `dev/<nombre>` con push libre; a `main` solo por PR, **solo Daniel aprueba y fusiona**. Ruleset
  `proteger-main` activo (24-oct): PR con 1 aprobación de Code Owner, check `pruebas` en verde, solo *merge commit*, sin
  force push ni borrado; Daniel fusiona sus propios PR con el bypass de admin.
- Metodología: Spec Driven + Subagent Driven Development, como en el Trabajo 1.
- Spec por partes (22-oct): núcleo único con los contratos + una spec por parte; los contratos **no se copian** a las partes.
- Para no bloquearse: contratos congelados el jue 22 + datos sintéticos con el mismo esquema (núcleo §5.6).
- [PROPUESTO] Notebooks por parte (`notebooks/NN_*.py` en formato percent → `.ipynb` ejecutado con `run.py notebooks`):
  celda de parámetros con las mismas claves que `--set`, cero lógica propia; el CLI se mantiene.
- [PROPUESTO] Python 3.12 + PyTorch + scikit-learn + Optuna TPE (PD 40 ensayos, LGD 30) + SHAP; blog MkDocs con el hook del
  Trabajo 1; app Next.js estática en Vercel con inferencia en el navegador; video con Remotion; Design System `apple-design`.
- B3 (Andrés, 25-oct; Spec v0.14, B-pd §3.1): rejilla de 13 C (10⁻⁴…10², sin duplicados); criterio log-loss de val, empate →
  menor C; sin refit con train+val; ConvergenceWarning = error; campo `evidencia` (SHA-256 de partición y preprocesador)
  en todos los JSON de resultados; `desv` muestral (n = 1 → 0); métricas `{auc,gini,ks,brier,logloss}_{val,test}`;
  `epocas` de la logística = iteraciones de lbfgs; joblib con `version_sklearn`; `en_borde` en la búsqueda.
- B2 (Andrés, 24/25-oct): L2 = (λ/2)‖W‖² ≡ `weight_decay = mlp.l2`, sin L2 en sesgos (Spec v0.13, B-pd §3.2);
  `min_delta = 0`; `mlp.pos_weight = 1` obligatorio; pérdida no finita → `FloatingPointError`; épocas desde 1; se aceptan
  las pruebas extra (documentadas en `tests/test_mlp.py`) y las funciones auxiliares fuera de las firmas del plan.

## ⏳ Hitos (núcleo §11)
jue 22 E0 · vie 23 2 p. m. datos reales · vie 23 noche búsquedas · sáb 24 mediodía PD final · sáb 24 tarde grabación ·
sáb 24 6 p. m. LGD final · sáb 24 noche JSON real · dom 25 8 p. m. blog de cada parte · lun 26 clon limpio y blog publicado ·
mar 27 6 p. m. entrega.

## ⏳ Pendientes (decisiones)
1. Reparto de tareas dentro de cada pareja (D y E; B+C ya repartió).
4. Modelo que va a la app (sáb 24).
5. **PR de `dev/andres`** (B1 + B2 + B3), declarar: (a) añade `src/common/torch_utils.py` y `src/common/metricas.py`
   (compartidos); (b) Spec v0.13 corrige B-pd §3.2 (L2 = (λ/2)‖W‖², H-08): avisar a Daniel y pedirle que confirme el origen
   de H-08; (c) `iterar_minilotes(n, batch, rng, device)` recibe `device`: avisar a D (Camila y José); (d) B3 y Spec v0.14:
   el campo `evidencia` también aplica a los JSON de la LGD (§5.4): avisar a D; C3 puede usar el modelo de la logística.
6. Para B5: conectar `run.py pd` (imprimir la config resuelta y el tiempo total, núcleo §4.1) reutilizando
   `escribir_semilla`, `actualizar_summary`, `actualizar_pred_test`, `resumir` y las huellas de `src/pd/experimento.py`.
   Después de B5: refactor `TODO(refactor-post-B5)` (validadores duplicados de `mlp.py`, `logistica.py`, `experimento.py`).
7. Para B6: la logística con todo train tarda ~65–90 s; la demo necesita `datos.muestra` o C fijo (`--set logistica.rejilla_C=[...]`).
   Para C/E: la PD de Python sale de X float32 → float64; vigilar la paridad < 1e-6 si la app preprocesa en float64.

## 🎯 Cacería (resumen)
Confirmados: 2 (H-03 y H-04, Código, Parte B) · Candidatos: 7 (H-01, H-05, H-06, H-07, H-09 Código; H-02 Datos; H-08 Matemáticas) · Afirmaciones por verificar: V-01…V-11 · Meta ≥ 5 en ≥ 3 categorías, ≥ 1 por parte.

## 📓 Bitácora
- **2026-10-25** — Revisión de lo que B, C, D y E consumen de la Parte A, desde una copia limpia de `main` sin datos crudos:
  todo se carga y funciona. Hueco corregido (Spec v0.12): `categorias_es` en `features.yaml` y `etiquetas_categorias` en el
  JSON de la app (las categorías salían en inglés); avisar a Juan José (exportador) y Camila (app). Siddiqi: 2.ª ed., 2017.
- **2026-10-25** — Spec v0.11: `data/processed/diccionario_loanstats.csv` (copia literal del diccionario, hoja LoanStats) para
  que D cite los campos de la LGD sin el crudo; lo pidió Camila. Prueba: las citas de `features.yaml` coinciden letra por letra.
- **2026-10-24** — PR #3 fusionado (10:26 p. m.): `montos.parquet` y contratos v0.10 en `main`, que Camila necesitaba para D1/D2;
  E5, blog 00 y 11–13, notebook 00 y el proyecto del video.
- **2026-10-24** — PR #2 fusionado: Spec v0.9, Andrés y Juan José reparten la pareja B+C por parte completa (Andrés B,
  Juan José C). Los contratos de la E[L] de `dev/daniel` pasan a ser la v0.10.
- **2026-10-24** — Blog 00 (resumen, sin cifras de modelos todavía) y notebook 00 de la sustentación: demos de `run.py` en
  carpetas temporales, la celda de modificación exige escribir la predicción antes, 3 predicciones de la Parte A verificadas.
  `configs/sustentacion.yaml`: cada parte registra ahí su demo. 192 pruebas.
- **2026-10-24** — E6: proyecto Remotion en `video/` con las 9 escenas del guion (clips como marcadores). E7 parcial: blog 11–13.
  V-05 (licencia de Remotion) verificada.
- **2026-10-24** — E5: `src/el/` + `run.py el` listos con pruebas (falta la PD de B y la LGD de D). Spec v0.10: `montos.parquet`
  (EAD y recuperaciones, solo para LGD y E[L]), columnas `pred_<modelo>` y `lgd/pred_pd_test.parquet` (LGD de todo el test).
  Dato: al incumplir seguía pendiente en promedio el 80 % del monto en test (64 % en train).
- **2026-10-24** — PR #1 (Parte A, 17 commits) fusionado en `main`; CI `pruebas` en verde en Linux; ruleset activo.
  B, C y D ya pueden usar `loans_pd.parquet` y `preprocesador_pd.json` desde `main`.
- **2026-10-25** — B3 (Andrés): `src/pd/logistica.py` (`LogisticaPD` según `ModeloPD`, búsqueda de C con la log-loss de
  val) y `src/pd/experimento.py` (`correr_logistica`, huellas, `resumir`, escritores que no borran lo ajeno);
  `configs/pd/base.yaml`; Spec v0.14 (0.11 en la rama). 124 pruebas rápidas + 1 lenta; 47 de 48 mutantes mueren (1 equivalente).
  revisor-spec, revisor-calidad (2 revisiones) y verificador-matematico APROBADO; 4 rondas de mejoras. Corrida real: C
  elegido 3,16e-3, 49 parámetros, 13 configuraciones; AUC test 0,670, KS 0,247, Brier 0,187 (PD media de test 0,187 frente
  a una tasa observada de 0,262; tasas de malos: train 0,163, val 0,230, test 0,262). Candidato H-09 (prueba débil).
- **2026-10-25** — B2 (Andrés): `src/pd/mlp.py` (red Linear→φ→Dropout con salida logit, `MLPPD` según `ModeloPD`),
  `src/pd/entrenar.py` (bucle genérico, minilotes reproducibles, parada temprana con restauración, L2 por grupos) y
  `src/common/torch_utils.py`; `tests/test_mlp.py` con 91 pruebas que matan 47 mutantes. Percent format, una celda por
  función, un comentario por línea. revisor-spec APROBADO; revisor-calidad RECHAZADO 3 veces y APROBADO en la 4.ª;
  verificador-matematico APROBADO. Candidatos H-05, H-06 (míos), H-07 (implementador) y H-08 (Spec: λ‖W‖² ≠ weight_decay;
  Spec v0.13, 0.10 en la rama). Sin push: Andrés revisa el código antes.
- **2026-10-24** — B1 (Andrés): `src/common/metricas.py` en percent format (fórmula antes de cada función; AUC, Gini, KS,
  Brier, ROC con TPR/FPR aparte, calibración por cuantiles; dependencias inyectadas) + `test_metricas.py` (63 pruebas de caja
  negra contra sklearn/scipy). TDD con 2 rondas: revisor-spec, revisor-calidad y verificador-matematico APROBADO (27
  mutantes mueren). H-03 (convención de borde de `calibration_curve`) y H-04 (AUC errónea con ±inf), confirmados por Andrés. Pendiente:
  TODO(cita) de Murphy (1973); el PR debe declarar que toca `src/common/` (compartido) y esperar a que se fusione el PR #2.
- **2026-10-23** — Daniel aprueba la partición B y las 15 variables, y pide la sensibilidad con `grade`/`int_rate` (Spec v0.8).
- **2026-10-23** — revisor-spec: Parte A RECHAZADA en la 1.ª ronda (2 bloqueantes + 9 recomendaciones), APROBADA en la 2.ª tras
  las correcciones (Spec v0.7: `run.py eda`, limpieza de categóricas en §5.2, EDA en memoria para el notebook). Hallazgo H-02.
- **2026-10-23** — A6/A6b: 11 hipótesis (todas con apoyo; H6 y H10 no monótonas), PSI por variable (los mayores: `purpose` y
  `pub_rec`, 0,27 en test), hallazgo H-01 (PSI = 0 en silencio para variables discretas). Blog portado del Trabajo 1 con
  01–03 escritas; `run.py eda`, `run.py figures` y `run.py notebooks` (01_datos en 14 s). 170 pruebas.
- **2026-10-23** — Parte A: crudo de Kaggle descargado (887 379 × 74, 2007-06 a 2015-12, SHA-256 en `data/fuentes.md`);
  objetivo, 15 variables de originación, partición ≤ 2012 / 2013 / 2014 (2015 tiene 93 % sin resolver), preprocesador y
  `loans_pd.parquet` (237 695 créditos). Entorno `.venv` 3.12 + `requirements.txt` fijado + CI. Los sintéticos ya no hacen falta.
- **2026-10-22** — Spec v0.4: notebooks de demostración en `notebooks/` como capa delgada sobre `src/` (consejo del profesor
  en el Trabajo 1; núcleo §5.8); análisis descriptivo después del preprocesamiento (A §3.8, tarea A6b) y de la LGD (D §3.2b, D1b).
- **2026-10-22** — Spec v0.3: equipo asignado (A Daniel; B y C Andrés y Juan José; D Camila y José; E Daniel y Camila); el
  enunciado literal pasa al Anexo del núcleo (se elimina `docs/enunciado.md`); cualquier integrante puede cambiar el Spec por PR.
- **2026-10-22** — Spec v0.2: el núcleo conserva reglas, decisiones y contratos; cada parte tiene su spec con "consumo /
  entrego", tareas con fecha, pruebas, puntos calientes de alucinación y definición de terminado. Cronograma comprimido al 27.
- **2026-10-19** — Spec v0.1, PLAN, agentes y plantillas creados por el puente con Claude a partir del enunciado y de la
  estructura del Trabajo 1. Repo de GitHub creado (vacío).
