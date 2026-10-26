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
| Última actualización | 2026-10-26 |
| Fase | En `main`: Parte A (PR #1), reparto B+C (PR #2), Parte E parcial con `montos.parquet` y contratos v0.10 (PR #3), diccionario y etiquetas en español (PR #4–#5), Parte B: B1–B3 (PR #6) y B4–B5 MLP de PD con 30 semillas (PR #10), Parte D: D1–D8 (PR #7), Parte C: C1–C2 (PR #8) y C3 scorecard de la logística + C5 los dos modelos de PD en la app (PR #12, 26-oct 2:46 p. m.), Parte E: E5 E[L] por banda de puntaje, notebook 05 y blog 09 (PR #9). Spec v0.18 |
| Próxima tarea | Todos: `git pull origin main` en su rama. B (Andrés): B6 (demo) y B7; marcar ✅ B1 en `B-pd.md`; arreglar las 2 pruebas de permisos de `test_hpo.py` que fallan en Windows. C (Juan José): C4 y C6; PR de Spec con C §3.1 [DECIDIDO] y `export.yaml` y `src/scorecard/puntos.py` en el núcleo §4. D (José): PR de `dev/jose` (D9, D10, D4b, D7b; Spec v0.19) para revisión de Daniel; luego Daniel rehace la E[L] (semilla LGD 22). Camila: E1–E4 con los dos JSON reales (`app/public/model/pd_model_{logistica,mlp}.json`) y selector de modelo. Daniel: activar Pages (Settings → Pages → GitHub Actions); blog 10, video; blog 00 y 13 al día |
| Entrega | **mar 27-oct-2026, 23:59** (meta interna 6 p. m.; días restantes: 1) |
| Sustentación | Posterior al 27-oct, por confirmar |

## 👥 Equipo
| Parte | Spec | Quién |
|---|---|---|
| A — Datos y caso de uso | `docs/spec/partes/A-datos.md` | Daniel |
| B — PD: modelos y evaluación | `docs/spec/partes/B-pd.md` | Andrés |
| C — Scorecard y variables de riesgo | `docs/spec/partes/C-scorecard.md` | Juan José |
| D — LGD | `docs/spec/partes/D-lgd.md` | Camila y José |
| E — Integración y producto (puente) | `docs/spec/partes/E-producto.md` | Daniel, Camila y José (José: app desde el 26-oct) |

Ramas personales: `dev/daniel`, `dev/camila`, `dev/andres`, `dev/juanjose`, `dev/jose`.

Integrantes: Edinson Andrés Ariza Mendoza, Maria Camila Agudelo Espinosa, Daniel Gonzalez Henao, Jose Miguel Pulgarin Agudelo,
Juan José González Morales.

## ✅ Decisiones vigentes
- App (Daniel, 26-oct; Spec v0.17): **los dos modelos de PD** (logística y MLP, semilla 20) con selector, por defecto el MLP;
  `pd_model_{logistica,mlp}.json` y `casos_paridad_{logistica,mlp}.json`, paridad para cada uno (núcleo §5.5).
- Scorecard (Juan José, C1): base 600 en odds 50:1 bueno:malo, PDO 20; `{base, odds_base, pdo, factor, offset}` en float;
  puntaje desde el logit, sin redondear. C §3.1 sigue [PROPUESTO] en el Spec: pasarlo a [DECIDIDO] en un PR de Spec.
- Puntos (Juan José, C3): `scorecard.json` = §5.4 + modelo, puntos_base, aditividad, evidencia; tabla por valor crudo en p1…p99
  de train; rango p1–p99 (num, = recorte del preprocesador) o cat con ≥ 1 %; bandas = deciles del puntaje de TRAIN (`cortes_bandas`, fijos en val/test).
- Export (Juan José, C2/C5): `run.py export` → `pd_model_{logistica,mlp}.json` + `casos_paridad_{…}.json` (todo o nada; MLP =
  semilla 20), MISMA muestra de 1 000 casos para los dos. min/max = recorte HACIA AFUERA al paso; defecto = mediana / moda de
  train; cat sin OTRA en orden de `categorias_es`, con `ayuda_es`; casos [{id, entradas crudas, pd, score}].
- LGD (José, 26-oct; v0.19, D §3.3): L2 (λ/2)‖W‖², `evidencia` §6, summary por modelo y semilla por MAE de **val** (como B).
- LGD (Camila, 24-oct): población = Charged Off (2 estados); EAD ≤ 0 o NA → excluir y contar; diferencia entre particiones
  (LGD = 1: train 26 %, test 63 %) se acepta y documenta ("consistente con censura"); `preprocesador_lgd.json` = el de A
  ajustado solo con el train de la población LGD (D §3.2).
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
- B4–B5 (Andrés, 25/26-oct; Spec v0.15–v0.16, B-pd §3.3 y §3.5): TPE de Optuna 5 por defecto (multivariado), log-loss de
  val, presupuesto = Σ fit, `mlp_final.yaml` heredado; semilla mediana = (logloss_val, seed) en (n−1)//2; `busqueda` optuna o
  `sin_busqueda` (presupuesto null); pre-chequeo antes de la logística; tolerancia entre máquinas 3·√2·EE.
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
5. Avisar a D: `iterar_minilotes(n, batch, rng, device)` recibe `device`; `evidencia` (§5.4, v0.14) va en los JSON de la LGD.
6. **PR de `dev/andres`** (B4+B5, v0.16): avisar a D (TPE multivariado, `FloatingPointError`), C y E (`pd_mlp.pt` = semilla
   20; PD media de test 0,204 vs tasa 0,262). Luego refactor `TODO(refactor-post-B5)`; blog 06: `mejor_epoca` = 1 en la
   mediana (val ruidosa) como limitación.
7. Para B6: la logística con todo train tarda ~65–90 s; la demo necesita `datos.muestra` o C fijo (`--set logistica.rejilla_C=[...]`).

## 🎯 Cacería (resumen)
Confirmados: 21 de 21 (Daniel, 26-oct): Código 13, Matemáticas 4 (H-08, H-10, H-14, H-15), Datos del mundo real 2 (H-02, H-13), Conceptos 2 (H-16, H-21), Bibliografía 0; por parte A 2, B 11, C 3, D 1, E 4 · Afirmaciones por verificar: V-01…V-12 · Meta del enunciado (≥ 5 en ≥ 3 categorías) y nuestra (≥ 1 por parte): **cumplidas**. Blog 12 al día.

## 📓 Bitácora
- **2026-10-26** — D9/D10/D4b/D7b (José, `dev/jose`; Spec v0.19): D4 rehecho. Mejor ensayo #29 [32,64,64,32] leaky_relu, λ 8,9e-4; MLP
  MAE test 0,07686 ± 0,00037 (9 953 parám.), lineal 0,07712, mediana 0,06986 (gana); semilla 22; búsqueda casi plana.
- **2026-10-26** — Blog 10 (discusión): 2 rondas; verificador-matematico y cazador APROBADO. Corregidas la 2.4 y la 09; H-21. Blog 12 (cacería): 21 hallazgos,
  todos confirmados por Daniel (H-15 pasa a Matemáticas); cazador APROBADO en la 2.ª ronda.
- **2026-10-26** — Spec v0.18: §5.4 de la E[L] y `blog.yml` (Pages). Antes, PR #11 (1:12 p. m.): Spec v0.17, dos modelos en la app.
- **2026-10-26** — PR #12 fusionado (2:46 p. m.; revisado por Daniel): C3 y C5 en `main`; los dos JSON de la app verificados aparte.
- **2026-10-26** — C5 (Juan José): los 4 JSON de la app (logística y MLP) en `app/public/model/`; receta vs modelo |ΔPD| ≤ 4,5e-8;
  118 pruebas. Spec APROBADO; calidad APROBADO en la 2.ª (todo o nada al escribir).
- **2026-10-26** — C3 (Juan José): `puntos.py` + `run.py scorecard` → `scorecard.json`; aditividad 1,1e-13; candidato H-20. Aprobado.
- **2026-10-26** — PR #10 fusionado (12:38 p. m., CI en verde; revisado por Daniel): B4 y B5 en `main`. MLP de PD: AUC 0,6716 ±
  0,0031 frente a 0,6697 de la logística; Brier y log-loss mejores en 30/30 semillas. 2 pruebas de permisos fallan en Windows.
- **2026-10-26** — PR #9 fusionado (10:26 a. m.): E5, E11 y blog 09 en `main`. Daniel confirma los 15 hallazgos (meta cumplida).
- **2026-10-26** — E5, E11 y blog 09 (Daniel, en `dev/daniel`): E[L] por bandas de puntaje (scorecard de C) con la logística y la
  MLP de LGD: 196,7 M USD en test; contraste solo con los resueltos: 177,9 M frente a 168,2 M realizados (razón 1,057); 5 460
  en mora con 61,3 M provisionales (LGD = 1 supuesta). La razón ≈ 1 resulta de tres efectos: PD 0,925 · LGD 0,970 · EAD 1,179. H-13–H-15.
- **2026-10-26** — PR #8 fusionado (7:46 a. m., CI en verde; revisado por Daniel): Parte C C1–C2 en `main`. JSON de la app verificado
  aparte: contrato §5.5, 600 puntos en 50:1, paridad |ΔPD| ≤ 1,3e-8 en los 1 000 casos e igual al joblib de B en todo el test.
- **2026-10-26** — B5 (Andrés): `run.py pd`; 812 pruebas; 3 revisores + analista OK. Test: MLP AUC 0,67163 ± 0,00309 (logística
  0,66965; 24/30), Brier 0,18324 ± 0,00108 (0,18652; 30/30); mediana = semilla 20; semilla 0 = ensayo #11 bit a bit. H-18, H-19.
- **2026-10-25** — PR #7 fusionado (1:26 p. m.; revisado por Daniel): Parte D D1–D8 en `main`. Primera E[L] real (logística de B +
  MLP de D) en todo el test: 196,7 M USD. Detalles para Camila en el PR; figuras de A y notebooks 00/01 restaurados.
- **2026-10-25** — PR #6 fusionado (12:12 p. m.; revisado por Daniel): Parte B B1–B3 en `main`, cifras reproducidas aparte. Daniel
  confirma H-08 (λ‖W‖² lo escribió Claude en el Spec). La PD media de test (0,187) < tasa observada (0,262): la tasa sube por año.
- **2026-10-25** — C1 y C2 (Juan José): `scorecard.py` y `run.py export` (logística real, |ΔPD| 1,5e-8); candidatos H-11 y H-12. Aprobados.
- **2026-10-25** — `tests/test_no_network.py` corregido (Parte E, con permiso de Daniel): `pytest` ya no reescribe `lgd-*.png`; redirige
  `SALIDA` de todo `src/viz/figs_*.py` y compara SHA-256. Un `figs_*.py` nuevo necesita `SALIDA` y conectarse en `cli._figures`.
- **2026-10-25** — B4 (Andrés): `run.py pd-hpo`, 40/40 ensayos, 702,4 s; mejor #11 [32, 16, 32] relu adam, log-loss val
  0,50748, 2 673 parám.; reproducible. Candidatos H-16 (TPE multivariado) y H-17.
- **2026-10-25** — D7 (Camila): blog 08 escrito y verificado (matemático y cazador); cita recortada de `loan_amnt` corregida. Revisión de autora hecha (lr en el borde como limitación).
- **2026-10-25** — D8 (Camila): notebook `04_lgd` (4,4 s; §3.2b + demo + oficiales); tablas en `src/lgd/tablas.py`. Rechazado 1.ª ronda (faltaba §3.2b).
- **2026-10-25** — D6 (Camila): 7 figuras `lgd-*.png` (`src/viz/figs_lgd.py`, en `run.py figures`). D1b: `descriptivo.json`; hallazgo para A: con 2 grupos
  "monótona" siempre (caso `pub_rec`). D5 ✅ (auditor, clon sin red).
- **2026-10-25** — D4 (Camila): TPE 30/30 + 30 semillas. Test: MLP MAE 0,07708 ± 0,00059 (14 593 parám.), lineal 0,07712 (49);
  la mediana constante gana (0,06986). Rechazado 1.ª ronda (`mejor_config.json` pisaba `--set`); arreglado con `mlp_final.yaml`.
- **2026-10-25** — Revisión de lo que B, C, D y E consumen de A desde un `main` limpio: todo funciona. Spec v0.12: `categorias_es`
  y `etiquetas_categorias` en el JSON de la app. Siddiqi: 2.ª ed. (PR #5, fusionado 8:27 a. m.).
- **2026-10-25** — Spec v0.11: `data/processed/diccionario_loanstats.csv` (copia literal del diccionario, hoja LoanStats) para
  que D cite los campos de la LGD sin el crudo; lo pidió Camila. Prueba: las citas de `features.yaml` coinciden letra por letra.
- **2026-10-25** — B3 (Andrés): `LogisticaPD` + `correr_logistica`; C 3,16e-3, 49 parám., 13 config.; AUC test 0,670, Brier
  0,187 (PD media 0,187 vs tasa 0,262; malos: train 0,163, val 0,230, test 0,262). 3 revisores APROBADO. Candidato H-09.
- **2026-10-25** — B2 (Andrés): `src/pd/mlp.py`, `entrenar.py`, `torch_utils.py`; 91 pruebas, 47 mutantes; revisor-calidad
  APROBADO en la 4.ª ronda. Candidatos H-05…H-08 (H-08: λ‖W‖² ≠ weight_decay, Spec v0.13).
- **2026-10-24** — PR #3 fusionado (10:26 p. m.): `montos.parquet` y contratos v0.10 en `main`, que Camila necesitaba para D1/D2;
  E5, blog 00 y 11–13, notebook 00 y el proyecto del video.
- **2026-10-24** — D2 (Camila): `loans_lgd.parquet` (43 236), `preprocesador_lgd.json` y `resumen_objetivo.json` (`python -m src.lgd.preparar`).
- **2026-10-24** — D3 (Camila): `LinealLGD` y `MLPLGD` (sigmoide + BCE fraccional); rechazado 1.ª ronda (NaN en silencio); candidato H-10.
- **2026-10-24** — PR #2 fusionado: Spec v0.9, Andrés y Juan José reparten la pareja B+C por parte completa (Andrés B,
  Juan José C). Los contratos de la E[L] de `dev/daniel` pasan a ser la v0.10.
- **2026-10-24** — Blog 00 (resumen) y notebook 00 de la sustentación (demos de `run.py` en carpetas temporales; la celda de
  modificación exige escribir la predicción antes). `configs/sustentacion.yaml`: cada parte registra ahí su demo.
- **2026-10-24** — E6: proyecto Remotion en `video/` con las 9 escenas del guion (clips como marcadores). E7 parcial: blog 11–13.
  V-05 (licencia de Remotion) verificada.
- **2026-10-24** — E5: `src/el/` + `run.py el` con pruebas. Spec v0.10: `montos.parquet` (solo LGD y E[L]), columnas `pred_<modelo>`
  y `lgd/pred_pd_test.parquet`. Al castigo seguía pendiente en promedio el 84 % del monto (castigados de test, E5 del 26-oct).
- **2026-10-24** — PR #1 (Parte A, 17 commits) fusionado en `main`; CI `pruebas` en verde en Linux; ruleset activo.
  B, C y D ya pueden usar `loans_pd.parquet` y `preprocesador_pd.json` desde `main`.
- **2026-10-24** — B1 (Andrés): `src/common/metricas.py` + 63 pruebas contra sklearn/scipy; 3 revisores APROBADO; H-03 y
  H-04. Pendiente: TODO(cita) de Murphy (1973).
- **2026-10-23** — Parte A: crudo de Kaggle (887 379 × 74, SHA-256 en `data/fuentes.md`); objetivo, 15 variables, partición
  ≤ 2012 / 2013 / 2014 (2015: 93 % sin resolver) y `loans_pd.parquet` (237 695), aprobados por Daniel (Spec v0.8, con la
  sensibilidad `grade`/`int_rate`). A6/A6b: 11 hipótesis y PSI (H-01). revisor-spec RECHAZADA → APROBADA (Spec v0.7, H-02).
  `.venv` 3.12, `requirements.txt` fijado y CI; blog portado del Trabajo 1 con 01–03.
