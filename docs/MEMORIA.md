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
| Fase | En `main`: Parte A (PR #1), reparto B+C (PR #2), Parte E parcial con `montos.parquet` y contratos v0.10 (PR #3), diccionario y etiquetas en español (PR #4–#5), Parte B: B1–B3 (PR #6) y B4–B5 MLP de PD con 30 semillas (PR #10), Parte D: D1–D8 (PR #7), Parte C: C1–C2 (PR #8) y C3 scorecard de la logística + C5 los dos modelos de PD en la app (PR #12, 26-oct 2:46 p. m.), Parte C: C4 SHAP del MLP (PR #18), Parte E: E5 E[L] por banda de puntaje, notebook 05 y blog 09 (PR #9), E[L] con la LGD de D4b (PR #15), Parte D: D9, D10, D4b y D7b (PR #14, José). Parte B: B8–B10 (PR #19). Parte B: B6 demo y B7 figuras (PR #16). Parte E: app E1–E3c (PR #17, José). Spec v0.24 |
| Próxima tarea | Todos: `git pull origin main` en su rama. B (Andrés): B6 🟨 (falta medir la demo en otra máquina: Daniel o Juan José); B7–B10 ✅; marcar ✅ B1 en `B-pd.md`; arreglar las 2 pruebas de permisos de `test_hpo.py` que fallan en Windows. C (Juan José): C6, C8 y C7; PR de Spec de la Parte C (pendiente 9). D: fila V-09. José: Vercel, E4 (paridad en CI) y E9 de la app (la app pasa de Camila a José, confirmado por Daniel 27-oct). Daniel: confirmar H-22…H-26 (blog 12); README con torch CPU en Linux (PR aparte); video; clon limpio |
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
- App: arquitectura y UX (José, 26-oct): `app/lib/` es dominio puro (`contrato`, `preprocesar`, `modelo`, `escenarios`), hooks como
  adaptadores, componentes presentacionales y una sola página contenedora; los casos de paridad solo en pruebas. Un solo panel:
  perfil base → escenario (+Δ) con deslizadores y el top-3 de cambios; densidad SVG desde los 101 percentiles; entradas en la URL;
  CSS Modules + `apple-design`. El aviso "no es una oferta de crédito" va como una línea discreta en el pie.
- B10 (Andrés, 27-oct; Spec v0.23, B-pd §3.8): modos reusar (misma arquitectura) y optuna (buscada con 17 variables), los dos se corren; carpetas `pd_sensibilidad`; tabla 6.6 en B10.
- B9 (Andrés, 26-oct; Spec v0.22, B-pd §3.7): notebook 02 = moldes (logística, MLP, comunes) + comando de terminal en texto; 1 demo en vivo (carpeta temporal) · 2 datos de A · 3 búsquedas · 4 modelo final; cifras desde los JSON; `src/pd/tablas.py`.
- Ahorro de tokens (Daniel, 26-oct): blog con 1 verificador (cazador, `model: sonnet`); informes de subagente ≤ 15 líneas; sesiones cortas.
- App (Daniel, 26-oct; Spec v0.17): **los dos modelos de PD** (logística y MLP, semilla 20) con selector, por defecto el MLP;
  `pd_model_{logistica,mlp}.json` y `casos_paridad_{logistica,mlp}.json`, paridad para cada uno (núcleo §5.5).
- Scorecard (Juan José, C1): base 600 en odds 50:1, PDO 20; `{base, odds_base, pdo, factor, offset}` en float; sin redondear.
- Puntos (Juan José, C3): `scorecard.json` = §5.4 + modelo, puntos_base, aditividad, evidencia; rango p1–p99; bandas = deciles de TRAIN.
- Export (Juan José, C2/C5): `run.py export` → `pd_model_{logistica,mlp}.json` + `casos_paridad_{…}.json` (todo o nada); MISMA
  muestra de 1 000 casos para los dos; min/max = recorte HACIA AFUERA al paso; cat sin OTRA, con `ayuda_es`.
- SHAP (Juan José, C4): `configs/scorecard/mlp.yaml` → `scorecard_mlp.json` + `shap_mlp.json`; Shapley intervencional del LOGIT
  (forward float64 = app), 15 jugadores, fondo 100 de train, 1 000 de test, max_evals 500; jugadores codificados (H-24).
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
6. B (tras PR #10): refactor `TODO(refactor-post-B5)` (incluye rechazar `datos.muestra` en `pd-hpo`). (Blog 06: `mejor_epoca` = 1 ya va en §6.7, B8.)
7. B6: medir la demo en otra máquina y marcar ✅; `demo: pd` aceptada (Daniel, 26-oct); README: instalar torch CPU en Linux (PR aparte).
9. PR de Spec de la Parte C: §5.4 (`scorecard_mlp.json`, `shap_mlp.json`), §4 (`puntos.py`, `export.yaml`, `mlp.yaml`), C §3.1 [DECIDIDO].
8. `run.py figures` cambia bytes entre máquinas (sin las fuentes de `paleta.FUENTES` usa DejaVu Sans): avisar a A, D y E (E8).

## 🎯 Cacería (resumen)
Confirmados: 21 de 21 (Daniel, 26-oct): Código 13, Matemáticas 4 (H-08, H-10, H-14, H-15), Datos del mundo real 2 (H-02, H-13), Conceptos 2 (H-16, H-21), Bibliografía 0; por parte A 2, B 11, C 3, D 1, E 4 · Afirmaciones por verificar: V-01…V-12 · Meta del enunciado (≥ 5 en ≥ 3 categorías) y nuestra (≥ 1 por parte): **cumplidas**. Blog 12 al día. Candidatos de B7 sin confirmar (de `dev/andres`): H-22 Conceptos (diagonal al revés), H-23 Datos (redondeo). Candidatos de C sin confirmar: H-24 y H-25 (C4), Código. De B10: H-26 Código (guarda de rutas; registrado primero como H-24).

## 📓 Bitácora
- **2026-10-27** — PR #17 fusionado (app E1–E3c de José, Spec v0.24): sin conflictos, CI verde; 104 pruebas y build en Windows; revisada en el navegador.
- **2026-10-27** — PR #18 (C4) y #19 (B8–B10) fusionados por Daniel; en el #19, Daniel resolvió los conflictos con el #18 y el H-24 de B10 pasó a H-26.
- **2026-10-27** — B8 ✅ (Andrés): blog 04 (fundamentos) y 05 (métricas) escritos y 06 completo (tabla `pd-comparacion`, §6.7); verificador-matematico y cazador APROBADO; quedan 2 marcas (H-22 y época 1).
- **2026-10-27** — B10 ✅ (Andrés, Spec v0.23): con `grade` e `int_rate`, ΔAUC test logística +0,0167, MLP +0,0159 (misma arquitectura) / +0,0170 (buscada; no se distingue del ruido); PD media 0,199–0,218 vs 0,262; 2 041 s; analista y cazador OK; H-26 (guarda de rutas); blog 6.6.
- **2026-10-26** — B9 ✅ (Andrés, Spec v0.22): notebook 02 (moldes + comando, demo 15 s, datos, búsquedas, modelo final), `src/pd/tablas.py` y `espacio.py`; 1 525 pruebas; spec OK, calidad OK en la 2.ª.
- **2026-10-26** — PR #15 y #16 fusionados (B6, B7; Spec v0.21). Daniel acepta `demo: pd`: notebook 00 con la demo de PD (12,7 s);
  figuras `pd-*` regeneradas en Windows con las fuentes de la paleta (en Linux salían con DejaVu).
- **2026-10-26** — B6 y B7 (Andrés, PR #16; Spec v0.20–v0.21): demo de PD 13,8–16,0 s en CPU sin red; 7 figuras `pd-*` + JSON de la escena 7 del video; 1 426 pruebas; revisores OK; H-22, H-23; falta medir la demo en otra máquina.
- **2026-10-26** — App E1–E3c (José, `dev/jose`, sin PR): Next 16 estático; paridad 1000/1000 por modelo (|ΔPD| 2,2e-16); panel
  único base → escenario, top-3 "¿Y si…?", tarjeta con anillo, 25 barras, tooltips, avatar de José; 104 pruebas; revisores APROBADO.
  Falta: Vercel (cuenta de José), E4 (paridad en CI), E9, URL del video, decidir ±25 % en el top-3.
- **2026-10-26** — PR #14 fusionado (Daniel; D4b, LGD semilla 22). E[L] rehecha en `dev/daniel`: 197,03 M USD (antes 196,71), resueltos 1,059 (antes 1,057; PD 0,926 · LGD 0,971 · EAD 1,179); MLP de PD 208,20 / 1,123. Blogs 00, 09, 10, 13; cazador APROBADO.
- **2026-10-26** — D9/D10/D4b/D7b (José, `dev/jose`; Spec v0.19): D4 rehecho. Mejor ensayo #29 [32,64,64,32] leaky_relu, λ 8,9e-4; MLP MAE test 0,07686 ± 0,00037 (9 953 parám.), lineal 0,07712, mediana 0,06986 (gana); semilla 22; búsqueda casi plana.
- **2026-10-26** — Blog 10 (discusión): 2 rondas; verificador-matematico y cazador APROBADO. Corregidas la 2.4 y la 09; H-21. Blog 12 (cacería): 21 hallazgos, todos confirmados por Daniel (H-15 pasa a Matemáticas). Blog 00 y 13 al día; Pages activado y publicado.
- **2026-10-26** — Spec v0.18: §5.4 de la E[L] y `blog.yml` (Pages). Antes, PR #11 (1:12 p. m.): Spec v0.17, dos modelos en la app.
- **2026-10-26** — PR #12 fusionado (2:46 p. m.; revisado por Daniel): C3 y C5 en `main`; los dos JSON de la app verificados aparte.
- **2026-10-26** — C4 (Juan José): `shap_pd.py`; eficiencia 6,2e-15, logística = forma cerrada (1,6e-15), estabilidad ρ 0,986; 83 s.
  Top SHAP: term, revol_util, annual_inc. Candidatos H-24 (np.isclose del masker) y H-25. Spec, calidad y verificador APROBADO.
- **2026-10-26** — C3 y C5 (Juan José): `scorecard.json` (aditividad 1,1e-13; H-20) y los 4 JSON de la app (|ΔPD| ≤ 4,5e-8). Aprobados.
- **2026-10-26** — PR #10 fusionado (12:38 p. m., CI en verde; revisado por Daniel): B4 y B5 en `main`. MLP de PD: AUC 0,6716 ±
  0,0031 frente a 0,6697 de la logística; Brier y log-loss mejores en 30/30 semillas. 2 pruebas de permisos fallan en Windows.
- **2026-10-26** — PR #9 fusionado (10:26 a. m.): E5, E11 y blog 09 en `main`. Daniel confirma los 15 hallazgos (meta cumplida).
- **2026-10-26** — E5, E11 y blog 09 (Daniel, en `dev/daniel`): E[L] por bandas de puntaje (scorecard de C) con la logística y la
  MLP de LGD: 196,7 M USD en test; contraste solo con los resueltos: 177,9 M frente a 168,2 M realizados (razón 1,057); 5 460
  en mora con 61,3 M provisionales (LGD = 1 supuesta). La razón ≈ 1 resulta de tres efectos: PD 0,925 · LGD 0,970 · EAD 1,179. H-13–H-15.
- **2026-10-26** — PR #8 fusionado (7:46 a. m., CI en verde; revisado por Daniel): Parte C C1–C2 en `main`. JSON de la app verificado aparte: contrato §5.5, 600 puntos en 50:1, paridad |ΔPD| ≤ 1,3e-8 en los 1 000 casos e igual al joblib de B en todo el test.
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
- **2026-10-25** — Revisión de lo que B, C, D y E consumen de A desde un `main` limpio: todo funciona. Spec v0.12: `categorias_es` y `etiquetas_categorias` en el JSON de la app. Siddiqi: 2.ª ed. (PR #5, fusionado 8:27 a. m.).
- **2026-10-25** — Spec v0.11: `data/processed/diccionario_loanstats.csv` (copia literal del diccionario, hoja LoanStats) para que D cite los campos de la LGD sin el crudo; lo pidió Camila. Prueba: las citas de `features.yaml` coinciden letra por letra.
- **2026-10-25** — B3 (Andrés): `LogisticaPD` + `correr_logistica`; C 3,16e-3, 49 parám., 13 config.; AUC test 0,670, Brier 0,187 (PD media 0,187 vs tasa 0,262; malos: train 0,163, val 0,230, test 0,262). 3 revisores APROBADO. Candidato H-09.
- **2026-10-25** — B2 (Andrés): `src/pd/mlp.py`, `entrenar.py`, `torch_utils.py`; 91 pruebas, 47 mutantes; revisor-calidad APROBADO en la 4.ª ronda. Candidatos H-05…H-08 (H-08: λ‖W‖² ≠ weight_decay, Spec v0.13).
- **2026-10-24** — PR #3 fusionado (10:26 p. m.): `montos.parquet` y contratos v0.10 en `main`, que Camila necesitaba para D1/D2; E5, blog 00 y 11–13, notebook 00 y el proyecto del video.
- **2026-10-24** — D2 (Camila): `loans_lgd.parquet` (43 236), `preprocesador_lgd.json` y `resumen_objetivo.json` (`python -m src.lgd.preparar`).
- **2026-10-24** — D3 (Camila): `LinealLGD` y `MLPLGD` (sigmoide + BCE fraccional); rechazado 1.ª ronda (NaN en silencio); candidato H-10.
- **2026-10-24** — PR #2 fusionado: Spec v0.9, Andrés hace la Parte B y Juan José la C; contratos de la E[L] = v0.10.
- **2026-10-24** — Blog 00 (resumen) y notebook 00 de la sustentación (demos de `run.py` en carpetas temporales; la celda de modificación exige escribir la predicción antes). `configs/sustentacion.yaml`: cada parte registra ahí su demo.
- **2026-10-24** — E6: proyecto Remotion en `video/` con las 9 escenas del guion (clips como marcadores). E7 parcial: blog 11–13; V-05.
- **2026-10-24** — E5: `src/el/` + `run.py el` con pruebas. Spec v0.10: `montos.parquet` (solo LGD y E[L]), columnas `pred_<modelo>` y `lgd/pred_pd_test.parquet`. Al castigo seguía pendiente en promedio el 84 % del monto (castigados de test, E5 del 26-oct).
- **2026-10-24** — PR #1 (Parte A, 17 commits) fusionado en `main`; CI `pruebas` en verde en Linux; ruleset activo. B, C y D ya pueden usar `loans_pd.parquet` y `preprocesador_pd.json` desde `main`.
- **2026-10-24** — B1 (Andrés): `src/common/metricas.py` + 63 pruebas; 3 revisores APROBADO; H-03 y H-04; TODO(cita) Murphy (1973).
- **2026-10-23** — Parte A: crudo de Kaggle, objetivo, 15 variables, partición ≤ 2012 / 2013 / 2014 y `loans_pd.parquet` (237 695), aprobados por Daniel (Spec v0.7–v0.8, H-01, H-02); `.venv` 3.12, CI y blog portado del Trabajo 1 con 01–03.
