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
| Fase | **Parte A completa y fusionada en `main`** (PR #1). Reparto B+C declarado: Andrés B, Juan José C. Spec v0.10. **B1 métricas ✅ y B2 MLP + entrenar ✅ en `dev/andres`** (sin push ni PR; Andrés revisa B2) |
| Próxima tarea | Daniel: revisar el PR del reparto B+C (v0.9) y activar el ruleset de `main` si falta. Andrés: B3 (logística) y B4 (Optuna); Juan José C1/C2. B, C y D actualizan su rama desde `main` y usan `data/processed/loans_pd.parquet` y `models/preprocesador_pd.json`. Daniel: confirmar H-01 y H-02 |
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
- B2 (Andrés, 24/25-oct): L2 = (λ/2)‖W‖² ≡ `weight_decay = mlp.l2`, sin L2 en sesgos (Spec v0.10, B-pd §3.2);
  `min_delta = 0`; `mlp.pos_weight = 1` obligatorio; pérdida no finita → `FloatingPointError`; épocas desde 1; se aceptan
  las pruebas extra (documentadas en `tests/test_mlp.py`) y las funciones auxiliares fuera de las firmas del plan.
- Reparto B+C (Andrés y Juan José, 24-oct): Andrés hace la Parte B y Juan José la Parte C (núcleo §1.1, Spec v0.9).
- Datos (Daniel, 23-oct): partición train ≤ 2012 / val 2013 / test 2014 (2015 fuera: 93 % sin desenlace); 15 variables de
  originación; `grade` e `int_rate` solo en un análisis de sensibilidad aparte (B10, `sensibilidad_lc.parquet`).
- Git: rama personal `dev/<nombre>` con push libre; a `main` solo por PR, **solo Daniel aprueba y fusiona** (CODEOWNERS + ruleset).
- Metodología: Spec Driven + Subagent Driven Development, como en el Trabajo 1.
- Spec por partes (22-oct): núcleo único con los contratos + una spec por parte; los contratos **no se copian** a las partes.
- Para no bloquearse: contratos congelados el jue 22 + datos sintéticos con el mismo esquema (núcleo §5.6).
- [PROPUESTO] Notebooks por parte (`notebooks/NN_*.py` en formato percent → `.ipynb` ejecutado con `run.py notebooks`):
  celda de parámetros con las mismas claves que `--set`, cero lógica propia; el CLI se mantiene.
- [PROPUESTO] Python 3.12 + PyTorch + scikit-learn + Optuna TPE (PD 40 ensayos, LGD 30) + SHAP; blog MkDocs con el hook del
  Trabajo 1; app Next.js estática en Vercel con inferencia en el navegador; video con Remotion; Design System `apple-design`.

## ⏳ Hitos (núcleo §11)
jue 22 E0 · vie 23 2 p. m. datos reales · vie 23 noche búsquedas · sáb 24 mediodía PD final · sáb 24 tarde grabación ·
sáb 24 6 p. m. LGD final · sáb 24 noche JSON real · dom 25 8 p. m. blog de cada parte · lun 26 clon limpio y blog publicado ·
mar 27 6 p. m. entrega.

## ⏳ Pendientes (decisiones)
0. **Al abrir el PR de `dev/andres`** (B1 + B2), declarar: (a) añade `src/common/torch_utils.py` y `src/common/metricas.py`
   (compartidos); (b) Spec v0.10 corrige B-pd §3.2 (L2 = (λ/2)‖W‖², H-08): avisar a Daniel y pedirle que confirme el origen
   de H-08; (c) `iterar_minilotes(n, batch, rng, device)` recibe `device`: avisar a D (Camila y José).
1. Reparto de tareas dentro de cada pareja (D y E; B+C ya repartió).
4. Modelo que va a la app (sáb 24).

## 🎯 Cacería (resumen)
Confirmados: 2 (H-03 y H-04, Código, Parte B) · Candidatos: 6 (H-01, H-05, H-06, H-07 Código; H-02 Datos; H-08 Matemáticas) · Afirmaciones por verificar: V-01…V-11 · Meta ≥ 5 en ≥ 3 categorías, ≥ 1 por parte.

## 📓 Bitácora
- **2026-10-25** — B2 (Andrés): `src/pd/mlp.py` (red Linear→φ→Dropout con salida logit, `MLPPD` según `ModeloPD`),
  `src/pd/entrenar.py` (bucle genérico, minilotes reproducibles, parada temprana con restauración, L2 por grupos) y
  `src/common/torch_utils.py`; `tests/test_mlp.py` con 91 pruebas que matan 47 mutantes. Percent format, una celda por
  función, un comentario por línea. revisor-spec APROBADO; revisor-calidad RECHAZADO 3 veces y APROBADO en la 4.ª;
  verificador-matematico APROBADO. Candidatos H-05, H-06 (míos), H-07 (implementador) y H-08 (Spec: λ‖W‖² ≠ weight_decay;
  Spec v0.10). Sin push: Andrés revisa el código antes.
- **2026-10-24** — B1 (Andrés): `src/common/metricas.py` en percent format (fórmula antes de cada función; AUC, Gini, KS,
  Brier, ROC con TPR/FPR aparte, calibración por cuantiles; dependencias inyectadas) + `test_metricas.py` (63 pruebas de caja
  negra contra sklearn/scipy). TDD con 2 rondas: revisor-spec, revisor-calidad y verificador-matematico APROBADO (27
  mutantes mueren). H-03 (convención de borde de `calibration_curve`) y H-04 (AUC errónea con ±inf), confirmados por Andrés. Pendiente:
  TODO(cita) de Murphy (1973); el PR debe declarar que toca `src/common/` (compartido) y esperar a que se fusione el PR #2.
- **2026-10-24** — Spec v0.9: Andrés y Juan José reparten la pareja B+C por parte completa (Andrés B, Juan José C); PR
  `dev/andres` → `main`.
- **2026-10-24** — PR de la Parte A (`dev/daniel` → `main`) abierto: 17 commits.
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
