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
| Fase | En `main`: Parte A (PR #1), reparto B+C (PR #2) y Parte E parcial con `montos.parquet` y contratos v0.10 (PR #3, 24-oct 10:26 p. m.). Spec v0.11 |
| Próxima tarea | Todos: `git pull origin main` en su rama. B (Andrés): B1–B4 (Optuna de noche). C (Juan José): C1–C2. D (Camila y José): D1–D3. Camila: E1 (app). Daniel: blog 09, notebook 05, confirmar H-01 y H-02 |
| Entrega | **mar 27-oct-2026, 23:59** (meta interna 6 p. m.; días restantes: 3) |
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

## ⏳ Hitos (núcleo §11)
jue 22 E0 · vie 23 2 p. m. datos reales · vie 23 noche búsquedas · sáb 24 mediodía PD final · sáb 24 tarde grabación ·
sáb 24 6 p. m. LGD final · sáb 24 noche JSON real · dom 25 8 p. m. blog de cada parte · lun 26 clon limpio y blog publicado ·
mar 27 6 p. m. entrega.

## ⏳ Pendientes (decisiones)
1. Reparto de tareas dentro de cada pareja (D y E; B+C ya repartió).
4. Modelo que va a la app (sáb 24).

## 🎯 Cacería (resumen)
Confirmados: 0 · Candidatos: 2 (H-01 Código, H-02 Datos del mundo real) · Afirmaciones por verificar: V-01…V-11 · Meta ≥ 5 en ≥ 3 categorías, ≥ 1 por parte.

## 📓 Bitácora
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
