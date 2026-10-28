# SPEC (núcleo) — Trabajo 2: Redes neuronales para datos tabulares (¿me prestas?)

| | |
|---|---|
| Curso | Redes neuronales y algoritmos bioinspirados — UNAL, Facultad de Minas, 2026-02 |
| Versión | **0.27** — 2026-10-28 (E-producto: E4 pasa a Daniel y la app entra al check `pruebas` del CI (`npm --prefix app ci`, `test` con la paridad y `build`); R-APP-5 y E9 sin la prueba de usabilidad con 2 personas [DECIDIDO, Daniel]; Daniel) · 0.26 — 2026-10-28 (Entrega extendida: del mar 27-oct al **vie 30-oct, 23:59**; fila Entrega y §11; Daniel) · 0.25 — 2026-10-27 (Parte C: el Spec queda igual a lo implementado en C1–C8. §3 Interpretabilidad pasa a [DECIDIDO] con el método de C4; árbol §4 con `configs/scorecard/{base,export,mlp}.yaml` y `src/scorecard/puntos.py`; §5.4 fija las claves de `scorecard.json` y `hipotesis.json` y añade `scorecard_mlp.json` y `shap_mlp.json`; C-scorecard §3.1–§3.4 pasan a [DECIDIDO], §5 lista las pruebas reales y §6 corrige los puntos calientes (H-24, H-28); afecta a E (consume `scorecard.json`, `hipotesis.json` y los JSON de la app en el blog, la E[L] y la app); Juan José) · 0.24 — 2026-10-26 (§1.1 y E-producto: la app (E1–E4, E9) pasa de Camila a José, de común acuerdo; José; numerada 0.20 en `dev/jose`, se renumera al integrar `main`) · 0.23 — 2026-10-27 (B10: B-pd §3.8 [DECIDIDO] fija el análisis de sensibilidad con `grade` e `int_rate`: unión con `sensibilidad_lc.parquet` y preprocesador propio ajustado solo con train; modos `reusar` (misma arquitectura que el modelo principal) y `optuna` (arquitectura buscada con las 17 variables, 40 ensayos); mismo pipeline con `RutasPD` inyectadas; `run.py pd-sensibilidad`, `configs/pd/sensibilidad.yaml`, `results/data/pd_sensibilidad/` y `models/pd_sensibilidad/` en §4 y §5.4; afecta a E (`src/common/cli.py`: un subcomando, se declara en el PR); Andrés) · 0.22 — 2026-10-26 (B9: B-pd §3.7 [DECIDIDO] fija la estructura del notebook `02_pd`: moldes de los parámetros de la demo (logística, MLP y comunes) y del comando de terminal en celdas de texto, demo en vivo como sección 1, y luego datos que consume la PD, búsqueda de hiperparámetros y modelo final; añade `src/pd/tablas.py` (tablas del notebook) y `src/pd/espacio.py` (espacio de búsqueda sin Optuna, lo re-exporta `hpo.py`) a las carpetas de B y al árbol §4; cumple el núcleo §5.8 sin cambiarlo; no toca contratos; Andrés) · 0.21 — 2026-10-26 (B7: B-pd §3.6 [DECIDIDO] fija las 7 figuras de PD de `src/viz/figs_pd.py` (`pd-roc`, `pd-ks`, `pd-calibracion`, `pd-semillas`, `pd-aprendizaje`, `pd-busqueda-mlp`, `pd-busqueda-logistica`) y `blog/docs/figures/pd-calibracion.json` con los puntos de la curva de calibración, que consume la escena 7 del video; `run.py figures` las regenera desde `results/data/pd/`; afecta a E (video y `run.py figures`, E8); Andrés) · 0.20 — 2026-10-26 (B6: B-pd §3.5 fija la demo de PD [DECIDIDO]: `configs/pd/demo.yaml` completa sin `hereda`, `datos.muestra` con la regla de la LGD y salidas en `results/data/pd_demo` y `models/pd_demo`; `configs/pd/base.yaml` declara `datos: {muestra: null}`; `run.py pd` sale con código 2 y una línea en stderr ante un error esperado; B-pd §3.3: `pd-hpo` ignora `datos.muestra`; para que la demo corra en < 2 min y `--set datos.muestra=20000` (§4.1) funcione; afecta a E (`configs/sustentacion.yaml`: `salida_temporal: salida` en la demo `pd`; se propone `demo: pd`); Andrés) · 0.19 — 2026-10-26 (§4: el árbol de `src/lgd/` lista los módulos que ya existen (`perdidas`, `metricas`, `validacion`, `procedencia`, `preparar`, `descriptivo`, `tablas`) y `configs/lgd/` añade `mlp_final.yaml`; solo documenta lo que está en `main` desde el PR #7; D-lgd §3.3: semilla representativa del MLP de LGD por MAE de val mediano (antes test), `summary.json` de la LGD por modelo como B y L2 = (λ/2)‖W‖²; afecta a E (cambian `pred_test` y `pred_pd_test` de la LGD); José) · 0.18 — 2026-10-26 (E5: §5.4 fija los bloques de `results/data/el/summary.json`, su `evidencia` con las huellas de las dos predicciones (la E[L] no usa preprocesador) y que `perdida_observada` de `el_test.parquet` es la pérdida realizada (NaN en mora, provisional aparte); árbol §4 con `src/el/{experimento,tablas}.py` y `.github/workflows/blog.yml` (Pages); Daniel) · 0.17 — 2026-10-26 (los dos modelos de PD van a la app: §5.5 pasa a `pd_model_{modelo}.json` y `casos_paridad_{modelo}.json` para `logistica` y `mlp`, paridad para cada uno; §13 pendiente 5 decidido; afecta a C (C5: exportar los dos) y E (app con selector, R-APP-2 a R-APP-4); Daniel) · 0.16 — 2026-10-26 (B5: §5.4 fija los campos extra de `semillas/mlp_seed{k}.json` y de la entrada `mlp` de `results/data/pd/summary.json` (presupuesto de la búsqueda, semilla mediana por log-loss de val, `config_de_busqueda`); B-pd §3.5 pasa a [DECIDIDO] con la regla de la semilla mediana y las claves `semillas`, `salida` y `modelos` de `configs/pd/base.yaml`; B-pd §8 fija la tolerancia de reproducción; para que C y E consuman `pd_mlp.pt` y `pred_mlp` sabiendo de qué semilla salen; afecta a C y E (`n_configuraciones_busqueda` puede ser null si la config no es la de la búsqueda); Andrés) · 0.15 — 2026-10-25 (B4: §4 añade `configs/pd/mlp_final.yaml` (lo genera `run.py pd-hpo`; lo hereda `configs/pd/base.yaml`, que declara `base_seed`); §5.4 fija los campos de `results/data/pd/hpo_ensayos.json`; B-pd §3.3 pasa a [DECIDIDO]: TPE de Optuna 5 con valores por defecto (multivariado), NopPruner, criterio, desempate, presupuesto, fallos y regla de `en_borde`; para que B5 herede la configuración ganadora y el presupuesto sea verificable; avisa a D (Optuna 5 usa TPE multivariado por defecto; `FloatingPointError` no es `RuntimeError`); Andrés) · 0.14 — 2026-10-25 (B3: §5.4 añade `busqueda_logistica.json` y el campo `evidencia` en los JSON de resultados; §6 define las dos huellas SHA-256 de la columna "Evidencia"; B-pd §3.1 fija rejilla, criterio, desempate, sin refit y desviación muestral; para que B5 reutilice el mismo esquema y la comparación sea verificable; afecta a D (campo `evidencia` en los JSON de la LGD); Andrés; numerada 0.11 en `dev/andres`, se renumera al integrar `main`) · 0.13 — 2026-10-25 (B-pd §3.2: L2 = (λ/2)‖W‖² ≡ `weight_decay = λ`, sin L2 en sesgos; corrige λ‖W‖², hallazgo H-08; Andrés; numerada 0.10 en `dev/andres`) · 0.12 — 2026-10-25 (`categorias_es` en `features.yaml` (§5.1) y `etiquetas_categorias` en las entradas del JSON de la app (§5.5): la app mostraba los códigos en inglés; afecta a C (exportador) y E (app); Siddiqi 2.ª ed. 2017 verificada; Daniel) · 0.11 — 2026-10-25 (`data/processed/diccionario_loanstats.csv` en §5.1: copia literal versionada del diccionario para citar variables sin el crudo; la pidió Camila para D1; Daniel) · 0.10 — 2026-10-24 (contratos para la E[L]: `montos.parquet` en §5.1; columnas `pred_<modelo>` de `pred_test.parquet` y LGD predicha para todo el test en §5.4; afecta a B y D; Daniel; se numera 0.10 porque el reparto B+C de Andrés ya era la 0.9) · 0.9 — 2026-10-24 (reparto de la pareja B+C: Andrés la Parte B, Juan José la Parte C, §1.1 y §13; Andrés) · 0.8 — 2026-10-23 (Daniel decide: partición ≤ 2012 / 2013 / 2014 y las 15 variables; `grade` e `int_rate` como análisis de sensibilidad aparte, archivo `sensibilidad_lc.parquet` en §5.1 y tarea B10) · 0.7 — 2026-10-23 (correcciones de revisor-spec a la Parte A: `run.py eda` en §4.1, limpieza de categóricas declarada en §5.2, campos adicionales del resumen en §5.1; Daniel) · 0.6 — 2026-10-23 (regla de push en §7.2: a main solo por PR; push libre a la rama personal; Daniel) · 0.5 — 2026-10-23 (Parte A entrega datos reales a las 9 a. m.: §5.6 deja de ser necesario; partición propuesta ≤ 2012 / 2013 / 2014 en A §3.5; Daniel) · 0.4 — 2026-10-22 (notebooks de demostración como capa delgada sobre `src/`, §5.8; análisis descriptivo después del preprocesamiento en las Partes A y D; propuesto por Daniel a partir del consejo del profesor en el Trabajo 1) · 0.3 — 2026-10-22 (equipo asignado a las partes; enunciado literal dentro del Spec, en el Anexo; cualquier integrante puede cambiar el Spec, §0.1) · 0.2 — 2026-10-22 (núcleo + una spec por parte en `docs/spec/partes/`; cronograma comprimido al 27-oct) · 0.1 — 2026-10-19 (borrador inicial) |
| Repositorio | https://github.com/danielgonzalez2709/Trabajo2_Redes_Neuronales |
| Entrega | **Viernes 30-oct-2026, 23:59** (extendida desde el martes 27-oct; meta interna por definir) · Sustentación: posterior, fecha por confirmar |
| Enunciado | **Anexo al final de este archivo** (transcripción literal del PDF del curso). **Si el Spec lo contradice, gana el enunciado.** |
| Metodología | Spec Driven + Subagent Driven Development (§10) |
| Estado de cada punto | **[DECIDIDO]** acordado por el equipo · **[PROPUESTO]** propuesta pendiente de acuerdo · **[PENDIENTE]** lo decide el equipo con evidencia · **[VERIFICAR]** confirmar con fuente primaria antes de usar |

### Cómo se lee este Spec

| Documento | Qué contiene | Quién lo lee |
|---|---|---|
| `docs/spec/SPEC.md` (este) | Reglas, decisiones, estructura, comandos, **contratos**, comparación justa, Git, cacería, hitos | **Todos** los humanos y subagentes |
| `docs/spec/partes/A-datos.md` | Parte A: datos, caso de uso, EDA, partición, preprocesador | Responsables de A y sus subagentes |
| `docs/spec/partes/B-pd.md` | Parte B: logística, MLP, búsqueda, métricas, 30 semillas | Responsables de B y sus subagentes |
| `docs/spec/partes/C-scorecard.md` | Parte C: scorecard, SHAP, hipótesis, exportación a la app | Responsables de C y sus subagentes |
| `docs/spec/partes/D-lgd.md` | Parte D: LGD, EAD, lineal, MLP, 30 semillas | Responsables de D y sus subagentes |
| `docs/spec/partes/E-producto.md` | Parte E: infraestructura, E[L], app, blog, video | Responsables de E y sus subagentes |
| `docs/spec/PLAN.md` | Tablero global: una línea por tarea, dependencias, estado | Puente (y quien quiera ver el avance) |
| **Anexo** de este archivo | **Enunciado literal del trabajo**: requisitos, entregables, sustentación y rúbrica | **Todos**; es el contexto de base de cualquier sesión |

**Regla de oro:** los **contratos (§5) viven solo aquí**. Las specs de parte los referencian (`núcleo §5.x`), nunca los copian.
Si una parte necesita cambiar un contrato, abre un PR que modifique este archivo y sube la versión.

---

## 0. Reglas (humanos y subagentes)

1. **Cualquier integrante puede cambiar el Spec (núcleo o partes) bajo su criterio**, incluido lo marcado [DECIDIDO]:
   - el cambio va en un PR (como todo lo que entra a `main`), sube la versión y deja una línea en el historial de la cabecera
     con qué cambió, por qué y quién lo cambió;
   - si toca un contrato (§5), un hito (§11) o algo que otra parte consume, el PR **nombra a las partes afectadas** y se les avisa
     antes de fusionar;
   - el Anexo (enunciado) **no se edita**: es la transcripción del PDF; solo se corrige si la transcripción tiene un error.
2. **El Spec es la fuente de verdad** (después del enunciado). Si el código o un subagente lo contradice, gana el Spec, o se
   cambia el Spec explícitamente según la regla 1.
3. Los **subagentes** no cambian el Spec ni toman decisiones marcadas [DECIDIDO], ni resuelven por su cuenta las
   [PENDIENTE]/[PROPUESTO]: **escalan** al humano que los dirige. Solo los humanos cambian el Spec.
4. **Prohibido inventar datos, constantes, significados de variables, referencias o resultados.** Si falta algo:
   `TODO(dato-faltante)` / `TODO(cita)` y se reporta. Un dato inventado es peor que uno faltante.
5. El significado de cada variable se cita del **diccionario de Kaggle** (`LCDataDictionary`), nunca de una IA. Copia literal
   versionada de la hoja LoanStats: `data/processed/diccionario_loanstats.csv` (§5.1).
6. Todo prompt enviado a una IA (incluidos los de los subagentes) se guarda **literal** en `alucinaciones/prompts/` (§9).
7. Toda cifra del blog sale de un archivo de `results/` o `data/processed/` y se puede regenerar con un comando.
8. **Los datos sintéticos (§5.6) son solo para desarrollar.** Ninguna cifra, figura o conclusión del blog sale de ellos.

---

## 1. Contexto, equipo y partes

Un banco quiere estimar, **antes de prestar**, la probabilidad de incumplimiento (PD, clasificación binaria) y la pérdida dado
incumplimiento (LGD, regresión), y combinarlas en E[L] = PD · LGD · EAD. En cada tarea se compara un modelo de baja complejidad
con un perceptrón multicapa bajo una **comparación justa** (§6). El modelo de PD se representa con una **scorecard** y se
publica en una **app web**; además: video promocional, reporte-blog, videos individuales y cacería de la alucinación.

Pesos (enunciado): reporte 20 %, repositorio 5 %, app 10 %, video promocional 5 %, ejecución y modificación en vivo 10 %,
cacería 15 %, video individual 5 % (individual), sustentación 30 % (individual).

### 1.1 Partes y responsables [DECIDIDO, equipo, 22-oct]

| Parte | Alcance (enunciado) | Spec | Responsables |
|---|---|---|---|
| **A — Datos y caso de uso** | §Datos, P1.1, P1.2, partición temporal de P1.5, preprocesador común; después: bibliografía verificada y coordinación de la grabación del video | `partes/A-datos.md` | **Daniel** |
| **B — PD** | P1.3, P1.4, P1.5 (métricas), 30 semillas de PD | `partes/B-pd.md` | **Andrés** |
| **C — Scorecard y variables de riesgo** | P1.6, P1.7, cierre de hipótesis, exportación a la app | `partes/C-scorecard.md` | **Juan José** |
| **D — LGD** | P2.1–P2.4, 30 semillas de LGD | `partes/D-lgd.md` | **Camila y José** |
| **E — Integración y producto (puente)** | Infraestructura, P2.5 (E[L]), app, despliegue, blog, video, CI, revisión de PR, clon limpio | `partes/E-producto.md` | **Daniel, Camila y José** (José: app, desde el 26-oct) |

Integrantes: Edinson **Andrés** Ariza Mendoza, Maria **Camila** Agudelo Espinosa, **Daniel** Gonzalez Henao, **Jose** Miguel
Pulgarin Agudelo, **Juan José** González Morales.

| Persona | Partes | Rama personal |
|---|---|---|
| Daniel | A, E (infraestructura, E[L], blog, video, revisión de PR) | `dev/daniel` |
| Camila | D, E | `dev/camila` |
| Andrés | B | `dev/andres` |
| Juan José | C | `dev/juanjose` |
| José | D, E (app, desde el 26-oct) | `dev/jose` |

**Parejas:** cada pareja reparte entre sus dos integrantes las tareas de su spec (columna "Quién" de la tabla de tareas) **hoy,
jueves 22**, de modo que dos personas no editen los mismos archivos a la vez. Lo que no se reparta es de los dos.
La pareja B+C repartió por parte completa (24-oct): **Andrés la Parte B y Juan José la Parte C**.

**Obligaciones de todo responsable de parte:** su sección del blog con los fundamentos matemáticos de su parte, ≥ 1 hallazgo genuino
de la cacería, su video individual, y estar en capacidad de explicar **todo** el trabajo (fichas del `tutor-sustentacion`).

### 1.2 Mapa de dependencias (quién le entrega qué a quién)

```
             contratos §5 + datos sintéticos §5.6 (E, jue 22)  ──► todos arrancan sin esperar
A ──datos reales (vie 23, 2 p. m.)──► B, C, D        D también puede construir su objetivo desde el CSV crudo
B ──modelo PD final (sáb 24, mediodía)──► C, E
C ──pd_model_{logistica,mlp}.json reales (sáb 24, noche)──► E (app)
B + D ──predicciones de test (sáb 24)──► E (E[L])
todos ──secciones del blog (dom 25, 8 p. m.)──► E (publicación)
```
Las flechas son **puntos de integración, no bloqueos**: antes de cada entrega, quien la recibe trabaja con los sintéticos
(§5.6) o con un modelo de juguete en el formato del contrato. Cambiar a lo real es cambiar una ruta en la config.

---

## 2. Alcance

**Incluido:** todo lo exigido por el enunciado.
**Opcional (solo si sobra tiempo, en este orden):** (1) LightGBM como tercer punto de comparación en PD; (2) LGD y E[L] en la
app; (3) base de datos para registrar consultas anónimas de la app.
**Fuera de alcance:** redes no-MLP (TabNet, transformers tabulares), datos externos al conjunto de Kaggle, entrenamiento en la nube.

---

## 3. Decisiones

| Tema | Decisión |
|---|---|
| Git [DECIDIDO, Daniel, 19-oct] | `main` protegida; cada integrante en su **rama personal** (`dev/<nombre>`, §1.1) y PR; **solo Daniel aprueba y fusiona** (§7.2). Commits con la cuenta de cada autor; PR pequeños, al menos uno por día por persona. |
| Metodología [DECIDIDO, Daniel, 19-oct] | Spec Driven + Subagent Driven Development, como en el Trabajo 1 (§10). |
| Spec por partes [DECIDIDO, equipo, 22-oct] | Núcleo único con los contratos + una spec por parte, para que cada subagente lea solo lo que necesita. |
| Lenguaje y entorno [PROPUESTO] | **Python 3.12**, `requirements.txt` con versiones fijas (`pip freeze`). Node 22 LTS para `app/` y `video/`, cada uno con su `package-lock.json`. |
| Redes [PROPUESTO] | **PyTorch**, con bucle de entrenamiento propio: semillas, parada temprana, regularización y exportación de pesos bajo control. |
| Referencias y métricas [PROPUESTO] | **scikit-learn** para logística y lineal; las métricas se implementan en `src/common/metricas.py` y se contrastan con sklearn/scipy en pruebas. |
| Búsqueda [PROPUESTO] | **Optuna, TPE con semilla fija**, número de ensayos fijo y declarado: PD 40, LGD 30 (reducido por tiempo; §6). |
| Interpretabilidad [DECIDIDO, Juan José, 26-oct] | **SHAP intervencional del logit** del MLP: `shap.explainers.Permutation` con masker `Independent` (fondo de train), 15 jugadores (las categóricas enteras) y explicados de test; importancia global = media de \|φ\|. Contraste: importancia por permutación en test (caída de AUC y aumento de log-loss) para el MLP y la logística. Los tamaños (fondo de 100 filas de train, 1 000 explicados de test, 16 permutaciones por fila con `max_evals` = 500, 5 repeticiones de la permutación) son valores por defecto de `configs/scorecard/mlp.yaml`. Detalle en C §3.3. |
| GPU [PROPUESTO] | Código agnóstico al dispositivo (`device: auto | cpu | cuda`). La **demo corre en CPU** (< 2 min). Los resultados registran el dispositivo. |
| Reutilización del Trabajo 1 [PROPUESTO] | Se adaptan de `Trabajo1_Redes_Neuronales` (mismo equipo): `src/common/config.py`, `cli.py`, `seeds.py`, `results.py` y sus pruebas, `blog/hooks/reporte.py` con sus pruebas, `blog/apa.csl`. |
| Blog [PROPUESTO] | **Material for MkDocs 1.6.x + hook `reporte.py`** del Trabajo 1, en GitHub Pages. |
| App [PROPUESTO] | **Next.js (TypeScript) exportado como sitio estático** en `app/`, inferencia **en el navegador** desde `app/public/model/*.json`, en **Vercel Hobby**. Sin base de datos. |
| Video [PROPUESTO] | **Remotion** en `video/` (render local) + clips grabados de los 5; publicado en YouTube como no listado. |
| Design System [PROPUESTO] | Skill **`apple-design`**, como en el Trabajo 1 (no se copia al repo). Se aplica a app, figuras, blog y video. |
| Datos crudos | No se versionan; URL, fecha de descarga y SHA-256 en `data/fuentes.md`. Se versionan los **procesados** en Parquet (< 100 MB por archivo). |
| Notebooks [PROPUESTO, Daniel, 22-oct] | Consejo del profesor en el Trabajo 1: le resulta más fácil correr y modificar en vivo con **celdas de Jupyter** que con comandos. Se agrega una carpeta `notebooks/` como **capa delgada**: cada notebook solo carga la config, llama funciones de `src/` y muestra resultados; **cero lógica propia**. El CLI se mantiene (es lo que exige el enunciado y lo que usa la CI). Contrato en §5.8. |
| Arquitectura | Los modelos **no dibujan**: escriben en `results/`; las figuras se construyen con `run.py figures`. Los experimentos **no usan internet**. |

---

## 4. Estructura del repositorio y comandos

```
Trabajo2_Redes_Neuronales/
├── README.md · CLAUDE.md · requirements.txt · run.py
├── .github/        CODEOWNERS, pull_request_template.md, workflows/ci.yml, workflows/blog.yml (Pages, E)
├── configs/        datos.yaml, features.yaml, pd/{base,demo,hpo,mlp_final,sensibilidad}.yaml, lgd/{base,demo,hpo,mlp_final}.yaml,
│                   scorecard/{base,export,mlp}.yaml, el/base.yaml
├── src/
│   ├── common/     config.py, cli.py, seeds.py, results.py, preprocesamiento.py, metricas.py, torch_utils.py   (E y A)
│   ├── datos/      cargar.py, objetivo.py, particion.py, eda.py, sintetico.py                                   (A; sintetico: E)
│   ├── pd/         logistica.py, mlp.py, entrenar.py, hpo.py, espacio.py, experimento.py, tablas.py, sensibilidad.py (B)
│   ├── scorecard/  scorecard.py, puntos.py, shap_pd.py, hipotesis.py                                            (C)
│   ├── export/     exportar_app.py                                                                              (C)
│   ├── lgd/        objetivo.py, lineal.py, mlp.py, perdidas.py, metricas.py, experimento.py, validacion.py,      (D)
│   │               procedencia.py, preparar.py, descriptivo.py, tablas.py
│   ├── el/         perdida_esperada.py, experimento.py, tablas.py                                               (E)
│   └── viz/        theme.py, style.py (E) · figs_datos.py (A) · figs_pd.py (B) · figs_scorecard.py (C) · figs_lgd.py (D) · figs_el.py (E)
├── scripts/        preparar_datos.py (A)
├── data/           raw/ (no versionado) · processed/ (versionado) · fuentes.md
├── models/         modelos entrenados versionados
├── results/        data/, tables/
├── blog/           mkdocs.yml, secciones/, docs/figures/, hooks/, references.bib, apa.csl
├── app/            Next.js (E); app/public/model/*.json lo genera `run.py export`
├── video/          Remotion (E)
├── notebooks/      00_sustentacion, 01_datos, 02_pd, 03_scorecard, 04_lgd, 05_perdida_esperada (.py fuente + .ipynb generado, §5.8)
├── tests/
├── alucinaciones/  registro.md, prompts/, evidencias/
├── docs/           MEMORIA.md, spec/{SPEC.md, PLAN.md, partes/}, sustentacion/
└── .claude/agents/
```
Cada parte toca **solo sus carpetas**. `src/common/`, `run.py`, `requirements.txt`, `configs/features.yaml` y el README son
compartidos: se cambian solo con un PR dedicado (§7.2).

### 4.1 Comandos (contrato con el usuario final) [PROPUESTO]

```bash
python scripts/preparar_datos.py                                  # crudo → data/processed (una vez)
python run.py eda --config configs/datos.yaml                     # Parte A: EDA, hipótesis y descriptivo posterior (~5 s)
python run.py pd --config configs/pd/demo.yaml                    # DEMO PD (< 2 min, CPU)
python run.py pd --config configs/pd/base.yaml                    # PD completo: logística + MLP final con 30 semillas
python run.py pd-hpo --config configs/pd/hpo.yaml                 # búsqueda Optuna del MLP de PD
python run.py pd-sensibilidad --config configs/pd/sensibilidad.yaml # B10: PD con grade e int_rate (aparte)
python run.py lgd --config configs/lgd/demo.yaml                  # DEMO LGD (< 2 min, CPU)
python run.py lgd --config configs/lgd/base.yaml                  # LGD completo con 30 semillas
python run.py lgd-hpo --config configs/lgd/hpo.yaml
python run.py scorecard --config configs/scorecard/base.yaml --set scorecard.pdo=40
python run.py el --config configs/el/base.yaml                    # pérdida esperada
python run.py export                                              # modelos → app/public/model/*.json + casos de paridad
python run.py figures                                             # regenera todas las figuras desde results/
npm --prefix app run dev                                          # app local (un solo comando)
mkdocs serve -f blog/mkdocs.yml                                   # blog local
python run.py notebooks                                           # .py → .ipynb y los ejecuta en modo demo (§5.8)
jupyter lab notebooks/                                            # abrir los notebooks (sustentación)
```
- Overrides con `--set clave.subclave=valor` (repetible), mismo cargador del Trabajo 1: YAML 1.2 (`1e-3` es float), clave
  inexistente = error, un bloque no se reemplaza por un escalar, herencia con `hereda: base.yaml`.
- Todo comando imprime la configuración resuelta antes de ejecutar (sirve para la predicción en vivo) y el tiempo total.
- Modificaciones en vivo que deben funcionar sin tocar código: `--set mlp.capas=[64,32]`, `--set mlp.dropout=0.3`,
  `--set mlp.lr=1e-2`, `--set mlp.max_epocas=5`, `--set scorecard.pdo=40`, `--set scorecard.base=500`,
  `--set datos.muestra=20000`, `--set semillas.n=5`.
- Semillas: `base_seed + run_id`; un único helper `src/common/seeds.py::fijar_semillas(seed)` (numpy, torch, random).
- `datos.ruta` apunta a `data/processed/` (reales) o a `data/sintetico/` (§5.6). Es lo único que cambia al pasar a datos reales.

---

## 5. Contratos (interfaces) — **congelados el jueves 22-oct**

### 5.1 Datos procesados (produce: A · consumen: B, C, D, E)

| Archivo | Contenido |
|---|---|
| `data/processed/loans_pd.parquet` | Una fila por crédito con objetivo definido (sin NA): `id` (str), `issue_d` (fecha), `anio_origen` (int), `target` (int 0/1), `split` ∈ {`train`, `val`, `test`}, y las variables crudas de `configs/features.yaml` **antes** de transformar |
| `data/processed/sensibilidad_lc.parquet` | **Solo para el análisis de sensibilidad (B10):** `id`, `grade`, `int_rate`, mismas filas y mismo orden que `loans_pd.parquet` (unir por `id`). Variables definidas en `configs/features_sensibilidad.yaml` (`adicionales`). Nunca entran al modelo principal ni a la app |
| `data/processed/diccionario_loanstats.csv` | **Diccionario de variables** (A, v0.11): copia literal de la hoja LoanStats de `LCDataDictionary.xlsx`, columnas `variable` y `descripcion` (UTF-8). Es la fuente de toda `cita_diccionario`; una prueba comprueba que las citas de `features.yaml` coinciden letra por letra |
| `data/processed/montos.parquet` | **Montos posteriores al desembolso** (A, v0.10): `id`, `loan_status`, `funded_amnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee`; mismas filas y orden que `loans_pd.parquet`. Solo para construir la LGD (D) y para la EAD y el contraste de la E[L] (E). **Nunca como variables de un modelo** (§5.7) |
| `data/processed/loans_lgd.parquet` | Subconjunto elegible para LGD (definido en la Parte D): `id`, `issue_d`, `split`, variables de `features.yaml`, campos de la fórmula de LGD [VERIFICAR nombres en el diccionario], `ead` (float), `lgd` (float en [0, 1]), `lgd_sin_recorte` (float) |
| `data/processed/resumen_datos.json` | `{archivo_crudo, sha256, fecha_descarga, n_registros_crudo, n_variables_crudo, periodo {min, max}, conteo_por_loan_status, n_por_split, tasa_malos_por_split, tasa_malos_por_anio, fraccion_na_por_anio}` + campos adicionales que usa el blog: `url`, `n_objetivo_na`, `n_resueltos_posteriores_a_test`, `particion` (cortes), `n_modelado`, `n_por_anio`, `variables`, `faltantes_por_variable_train` |
| `configs/features.yaml` | Lista de variables: `{nombre, tipo: num|cat, disponible_en_originacion: true, justificacion, cita_diccionario, transformacion, en_app: bool, etiqueta_es, ayuda_es}`; las categóricas añaden `categorias_es: {código: etiqueta}` (v0.12) |

`loans_lgd.parquet` lo escribe D (con el preprocesador de A); el resto, A.

### 5.2 Preprocesador común (produce: A · consumen: B, C, D, E)

```python
# src/common/preprocesamiento.py
class Preprocesador:
    """Se ajusta SOLO con train. Mismo objeto para todos los modelos de una tarea (regla de comparación justa)."""
    def fit(self, df: pd.DataFrame, features: list[dict]) -> "Preprocesador": ...
    def transform(self, df: pd.DataFrame) -> np.ndarray: ...      # float32, columnas en orden fijo
    def nombres_salida(self) -> list[str]: ...
    def to_dict(self) -> dict: ...                               # serializable a JSON (lo consume la app)
    @classmethod
    def from_dict(cls, d: dict) -> "Preprocesador": ...
```
Operaciones permitidas (todas reimplementables en TypeScript): imputación por mediana (numéricas) o categoría `"DESCONOCIDO"`
(categóricas), recorte a percentiles de train, `log1p` en variables declaradas, estandarización (x − media)/desv de train,
one-hot con categorías fijas de train + `"OTRA"` para no vistas. Antes del one-hot, las categóricas se limpian: se quitan
los espacios al inicio y al final (el CSV trae `" 36 months"`) y el texto vacío cuenta como vacío. `to_dict()` declara esta
limpieza y el orden de las operaciones numéricas para que la app las repita. **Nada más** sin cambiar este contrato.
Mientras A lo termina, E deja en el esqueleto una versión mínima (mediana + estandarización + one-hot) con esta misma firma.

### 5.3 Modelos (producen: B y D · consumen: C, E)

```python
class ModeloPD(Protocol):                              # src/pd/*.py (logística y MLP)
    def fit(self, X_tr, y_tr, X_val, y_val, cfg: dict, seed: int) -> "ModeloPD": ...
    def predict_proba(self, X) -> np.ndarray: ...      # P(target = 1) = P(malo), shape (n,)
    def predict_logit(self, X) -> np.ndarray: ...      # log(p / (1 − p)); lo usa la scorecard
    def n_parametros(self) -> int: ...                 # entrenables
    def guardar(self, ruta) -> None: ...
    @classmethod
    def cargar(cls, ruta) -> "ModeloPD": ...

class ModeloLGD(Protocol):                             # src/lgd/*.py (lineal y MLP)
    def fit(self, X_tr, y_tr, X_val, y_val, cfg: dict, seed: int) -> "ModeloLGD": ...
    def predict(self, X) -> np.ndarray: ...            # LGD estimada, shape (n,)
    def n_parametros(self) -> int: ...
    def guardar(self, ruta) -> None: ...
    @classmethod
    def cargar(cls, ruta) -> "ModeloLGD": ...
```
Bloque de configuración del MLP (igual en PD y LGD): `mlp: {capas: [int], activacion: relu|leaky_relu|tanh, dropout: float,
l2: float, optimizador: adam|sgd, lr: float, momento: float, batch: int, max_epocas: int, paciencia: int,
device: auto|cpu|cuda}`; PD añade `pos_weight`, LGD añade `perdida`.
El MLP de PD es una pila `Linear → activación → Dropout` con salida `Linear(·, 1)` que devuelve el **logit** (lo exige §5.5).

### 5.4 Resultados (producen: B, C, D, E · consumen: figuras y blog)

```text
results/data/{pd|lgd}/semillas/{modelo}_seed{k}.json   # {modelo, seed, config, metricas{...}, n_parametros, tiempo_s, device, epocas, evidencia}
                                        #   PD mlp (v0.16), además: mejor_epoca, mejor_perdida_val, historial [{epoca,
                                        #   perdida_train, perdida_val}], hilos_torch, version_torch, calibracion_global
                                        #   {pd_media_val, tasa_val, pd_media_test, tasa_test}, n {train, val, test}
results/data/{pd|lgd}/summary.json      # por modelo: n_parametros, n_configuraciones_busqueda, tiempo_total_s,
                                        #   metricas {nombre: {media, desv, mejor, peor, n}}, evidencia
                                        #   PD mlp (v0.16), además: tiempo_busqueda_s, tiempo_semillas_s (Σ de los fit),
                                        #   semillas, semilla_mediana, criterio_semilla_mediana, config, config_de_busqueda,
                                        #   busqueda (bloque de mlp_final.yaml), hilos_torch, version_torch; tiempo_total_s =
                                        #   tiempo_busqueda_s + tiempo_semillas_s. Si config_de_busqueda es false:
                                        #   n_configuraciones_busqueda, tiempo_busqueda_s y busqueda = null y
                                        #   tiempo_total_s = tiempo_semillas_s
results/data/pd/busqueda_logistica.json # búsqueda de C (B3): criterio, filas [{C, logloss_val, auc_val, n_iter, tiempo_s}],
                                        #   C_elegido, en_borde, tiempo_total_s, evidencia
results/data/{pd|lgd}/hpo_ensayos.json  # cada ensayo: parámetros, métrica de validación, tiempo, estado
                                        #   PD (v0.15): criterio, muestreador {nombre, semilla, n_startup_trials,
                                        #   multivariate}, podador, semilla_ensayos, versiones {optuna, torch}, hilos_torch,
                                        #   espacio, mlp_base (solo claves no buscadas), n_ensayos (todos), estados,
                                        #   tiempo_total_s (Σ de los fit, incluidos FAIL), tiempo_reloj_s, mejor {numero,
                                        #   parametros, logloss_val, auc_val, n_parametros}, en_borde {lr, l2, dropout},
                                        #   ensayos [{numero, estado, parametros, logloss_val, auc_val, perdida_val_entrenar,
                                        #   epocas, mejor_epoca, n_parametros, tiempo_s, error si FAIL}], evidencia
results/data/{pd|lgd}/pred_test.parquet # columnas: id (str), y, pred_<modelo> (float; p. ej. pred_logistica, pred_mlp)
                                        #   con la semilla de la métrica mediana. PD: todo el test. LGD: el test de la LGD.
results/data/pd_sensibilidad/           # B10 (v0.23): loans_pd_sensibilidad.parquet; reusar/ y optuna/ con el esquema
                                        #   de results/data/pd/; comparacion.json (B-pd §3.8). Modelos y preprocesador en
                                        #   models/pd_sensibilidad/. No lo consume nadie más (no va a la app)
results/data/lgd/pred_pd_test.parquet   # LGD predicha para TODOS los créditos de test de loans_pd (no solo los
                                        #   incumplidos): id, pred_<modelo>. La necesita la E[L] (Parte E).
results/data/scorecard/scorecard.json   # {base, odds_base, pdo, factor, offset, modelo, puntos_base, aditividad
                                        #   {error_max, tolerancia}, puntos_por_variable, distribucion_scores, evidencia}
                                        #   (v0.25): puntos_por_variable [{nombre, tipo, etiqueta_es, rango_puntos, …}]:
                                        #   num con beta, puntos_por_unidad_estandarizada, transformacion, tabla [{percentil,
                                        #   valor, puntos}]; cat con categorias [{categoria, etiqueta, beta, puntos,
                                        #   fraccion_train}] y otra. distribucion_scores {cortes_bandas, n_bandas, ancho_bin,
                                        #   train|val|test {n, media, desv, percentiles, histograma, bandas [{banda, desde,
                                        #   hasta, n, puntaje_medio, pd_media, tasa_malos}]}}
results/data/scorecard/scorecard_mlp.json # (v0.25) mismas claves de raíz con "modelo": "mlp"; puntos_por_variable con tramos
                                        #   y n_sin_valor (num) o categorias con n y puntos_medios (cat): media de −factor·φ
                                        #   de los explicados, en vez de beta y tabla
results/data/scorecard/shap_mlp.json    # (v0.25) {modelo, metodo, phi0, eficiencia, verificacion_logistica,
                                        #   diferencia_mlppd, importancia_global [{nombre, media_abs_phi, puesto}], direccion,
                                        #   estabilidad, permutacion {k, semilla, conjunto, logistica, mlp}, explicados {ids,
                                        #   phi, logit}, evidencia, tiempos_s}
results/data/scorecard/hipotesis.json   # (v0.25) {criterio {reglas, senales, contraste_sin_voto, umbrales, sensibilidad, …},
                                        #   hipotesis [{id, enunciado, variable, mas_riesgosa, referencia, eda, shap,
                                        #   logistica, test, psi_test, veredicto, motivo, matiz, sensibilidad,
                                        #   cambia_con_sensibilidad}], evidencia}
results/data/el/summary.json            # E[L] total, por banda de puntaje, contraste con la pérdida observada
                                        #   (v0.18): bloques total, resueltos (E[L], pérdida realizada, razón), en_mora
                                        #   (pérdida provisional), anomalia_pagados y por_banda con los tres; scorecard,
                                        #   bandas, n_bandas_pedidas/obtenidas; evidencia = {sha256_particion,
                                        #   sha256_pred_pd, sha256_pred_lgd} (la E[L] no usa preprocesador)
results/data/el/el_test.parquet         # por crédito de test: id, target, pd, lgd, ead, el, perdida_observada
                                        #   (v0.18): perdida_observada = pérdida REALIZADA: max(0, (funded − rec_prncp) −
                                        #   (recoveries − fee)) en castigados, 0 en pagados, NaN en mora (cobranza
                                        #   abierta); además estado, perdida_provisional (mora, LGD = 1 supuesta),
                                        #   principal_no_registrado (pagados), observacion_completa, score y banda
```
JSON en UTF-8, sangría 2, sin NaN/inf (`save_json` del Trabajo 1). `evidencia` = `{sha256_particion, sha256_preprocesador}` (§6). En `summary.json`, `desv` es la desviación muestral s = √(Σ(xᵢ − x̄)²/(n − 1)); con n = 1 no está definida y se escribe 0 por convención; `mejor`/`peor` según el sentido de la métrica (AUC, Gini, KS: máx; Brier, log-loss: mín). Nombres de métricas: `{auc,gini,ks,brier,logloss}_{val,test}`. Modelos en `models/`: `pd_logistica.joblib`, `pd_mlp.pt`,
`lgd_lineal.joblib`, `lgd_mlp.pt`, `preprocesador_pd.json`, `preprocesador_lgd.json`.

### 5.5 Exportación a la app (produce: C · consume: E)

**Los dos modelos de PD van a la app** [DECIDIDO, Daniel, 26-oct; v0.17]: un archivo por modelo,
`app/public/model/pd_model_{modelo}.json` y `app/public/model/casos_paridad_{modelo}.json` con `{modelo}` ∈
{`logistica`, `mlp`} (el MLP es la semilla mediana guardada en `models/pd_mlp.pt`). Cada archivo sigue el esquema de abajo,
con su propia `red`, sus `percentiles_score` (los puntajes de test de **ese** modelo) y sus `metricas`; `entradas`,
`preprocesamiento` y `scorecard` son iguales en los dos.

`app/public/model/pd_model_{modelo}.json`:
```jsonc
{
  "version_contrato": 1,
  "generado": "<ISO 8601>", "commit": "<sha>", "modelo": "mlp | logistica",
  "entradas": [            // orden del formulario; solo las de features.yaml con en_app: true
    {"nombre": "<variable>", "tipo": "num", "etiqueta_es": "...", "ayuda_es": "...", "min": 0, "max": 0, "paso": 1,
     "valor_defecto": 0},
    {"nombre": "<variable>", "tipo": "cat", "etiqueta_es": "...", "categorias": ["..."], "valor_defecto": "...",
     "etiquetas_categorias": {"<código>": "<etiqueta en español>"}}   // de categorias_es (v0.12); OTRA no se ofrece
  ],
  "preprocesamiento": { /* Preprocesador.to_dict() */ },
  "red": { "capas": [ {"W": [[/* salida × entrada, igual que torch.nn.Linear.weight */]], "b": [], "activacion": "relu"} ],
           "salida": "sigmoid" },
  "scorecard": { "base": 0, "odds_base": 0, "pdo": 0, "factor": 0.0, "offset": 0.0 },
  "poblacion": { "percentiles_score": [/* 101 valores: p0 … p100 */], "referencia": "test" },
  "metricas": { "auc_test": 0.0, "ks_test": 0.0, "brier_test": 0.0 }
}
```
(Los números del ejemplo son marcadores, no valores.) En `pd_model_logistica.json`, `red.capas` tiene una sola capa con
activación `"identidad"`; en `pd_model_mlp.json`, una capa por cada capa oculta y la última con `"identidad"` (el logit).
`app/public/model/casos_paridad_{modelo}.json`: 1 000 filas de test con entradas crudas, `pd` y `score` calculados en Python.
**Prueba de paridad** en CI, para **cada** modelo: |PD_TS − PD_Py| < 1e-6 y |score_TS − score_Py| < 1e-3 en las 1 000.

### 5.6 Datos sintéticos — **ya no hacen falta** (v0.5): los datos reales están en `data/processed/` desde el vie 23, 9 a. m.

`python -m src.datos.sintetico` escribe `data/sintetico/loans_pd.parquet`, `loans_lgd.parquet` y `features.yaml` con **las
mismas columnas y tipos** de §5.1, una relación simple entre variables y objetivo (para que los modelos aprendan algo), los tres
`split` y una semilla fija. Prueba: el esquema coincide con §5.1. No se usan para ninguna cifra del blog (§0.7).

### 5.7 Política de variables (fuga de información)

Solo entran variables **disponibles en la originación**, según el caso de uso (Parte A). Prohibido en cualquier modelo: pagos y
saldos posteriores (`total_pymnt*`, `total_rec_*`, `out_prncp*`, `last_pymnt_*`, `next_pymnt_d`), recuperaciones
(`recoveries`, `collection_recovery_fee`, que solo se usan para construir la LGD), `last_credit_pull_d`, `loan_status`,
identificadores y texto libre [VERIFICAR la lista completa contra el diccionario en A3]. `tests/test_fuga.py` falla si alguna
aparece en `configs/features.yaml`.

### 5.8 Notebooks (cada parte el suyo · infraestructura: E)

| Notebook | Parte | Qué muestra |
|---|---|---|
| `00_sustentacion` | E | Guion de los 10 minutos: demo en una celda, celda de modificación en vivo con la predicción escrita antes de ejecutar, enlaces a los demás |
| `01_datos` | A | Conteos, objetivo, partición, análisis descriptivo antes y **después** del preprocesamiento, hipótesis |
| `02_pd` | B | Logística y MLP en modo demo, métricas, curvas ROC/KS/calibración, resumen de las 30 semillas (leído de `results/`) |
| `03_scorecard` | C | Probabilidad → puntaje, puntos por variable, SHAP, hipótesis confirmadas/refutadas |
| `04_lgd` | D | Distribución de la LGD observada, lineal vs. MLP en modo demo, MAE/RMSE, distribuciones |
| `05_perdida_esperada` | E | E[L] por crédito y por banda de puntaje |

Reglas:
- **Fuente:** `notebooks/NN_nombre.py` en formato *percent* de Jupytext (celdas `# %%`; se corren celda a celda en VS Code y se
  revisan bien en un PR). `python run.py notebooks` los convierte a `.ipynb` y los **ejecuta en modo demo**; los `.ipynb` con
  salidas se versionan para que se vean en GitHub sin correr nada. **Solo se edita el `.py`**; el `.ipynb` es generado.
- **Primera celda de código = parámetros** (etiqueta `parameters`):
  ```python
  CONFIG = "configs/pd/demo.yaml"
  SET = {"mlp.dropout": 0.3, "scorecard.pdo": 40}    # mismas claves que --set; una clave inexistente es un error
  ```
  La segunda celda resuelve la config con el mismo cargador del CLI y la **imprime** (sirve para la predicción en vivo).
- **Cero lógica en el notebook:** nada de entrenar, transformar o calcular métricas "a mano"; todo es una llamada a `src/`.
  Si algo hace falta, se agrega a `src/` con su prueba.
- Cada notebook corre de principio a fin en **< 2 min en CPU** con su config demo, o lee resultados ya guardados.
- `tests/test_notebooks.py` (marcada lenta) ejecuta todos en modo demo.

---

## 6. Regla de comparación justa (criterios verificables)

| Regla (enunciado) | Cómo se cumple | Evidencia |
|---|---|---|
| Mismos datos | Un `Preprocesador` y una partición por tarea, compartidos por todos los modelos | Hash de la partición y del preprocesador en cada JSON de resultados (campo `evidencia`): `sha256_particion` = SHA-256 (hex) del texto UTF-8 de las líneas `"{id},{split}\n"` en el orden del parquet; `sha256_preprocesador` = SHA-256 (hex) de los bytes del JSON del preprocesador (`src/pd/experimento.py`: `huella_particion`, `huella_archivo`) |
| Presupuesto explícito | `n_configuraciones_busqueda` y `tiempo_total_s` por familia en `summary.json` (también para la logística/lineal) | Tabla del blog |
| ≥ 30 semillas | Configuración final de cada red con `base_seed + 0…29` | 30 archivos en `semillas/` por red |
| Complejidad | `n_parametros()` de cada modelo | Tabla del blog |

---

## 7. Requisitos no funcionales y flujo de trabajo

### 7.1 Reproducibilidad
| Requisito | Criterio verificable |
|---|---|
| Versiones fijas | `pip install -r requirements.txt` en venv limpio (Python 3.12); `npm ci` en `app/` |
| Semillas | Misma semilla y mismo dispositivo → mismas métricas; `torch.use_deterministic_algorithms(True)` en CPU |
| GPU | Opcional; los JSON registran `device`; entre CPU y GPU se acepta una tolerancia documentada |
| Sin internet | Experimentos con la red bloqueada (`test_no_network.py` del Trabajo 1) |
| Un comando | Cada línea de §4.1 funciona desde un clon limpio |
| Parámetros sin editar código | Todo lo de §5.3 y la scorecard se cambia con `--set` |
| Demo < 2 min | `pd` y `lgd` demo medidos en CPU en una máquina distinta a la de desarrollo |
| Modelos guardados | La app y `run.py figures` funcionan sin reentrenar |
| Figuras regenerables | `python run.py figures` las reconstruye todas desde `results/` |
| Notebooks | `python run.py notebooks` los ejecuta todos sin error, cada uno en < 2 min en CPU |
| Historial real | Commits frecuentes de las 5 cuentas |

### 7.2 Git y PR [DECIDIDO, Daniel, 19-oct; regla de push 23-oct]
- **Nunca se hace push directo a `main`. Todo cambio llega a `main` por PR**, que Daniel autoriza y fusiona. Vale para
  todas las ramas y todo el contenido (código, datos, Spec, blog).
- **A la rama personal (`dev/<nombre>`) cada quien hace push libremente**, sin PR: es su espacio de trabajo.
- Flujo: commits frecuentes → push a `dev/<nombre>` → se verifica (pruebas, revisores) → PR hacia `main` → Daniel fusiona
  → los demás actualizan su rama desde `main`.
- Las sesiones de Claude hacen los commits que haga falta y el push a la rama personal **cuando el integrante lo indique**
  (p. ej. si no está en su computador para autenticarse).
- Ruleset en `main`: PR obligatorio, 1 aprobación, *Require review from Code Owners*, sin force push ni borrado.
  `.github/CODEOWNERS`: `* @danielgonzalez2709`.
- Cada quien trabaja en su rama personal (`dev/<nombre>`, §1.1); después de cada merge, actualiza su rama desde `main`.
  Dentro de una pareja, cada uno toca solo los archivos de las tareas que le tocaron.
- Archivos compartidos (`requirements.txt`, `src/common/`, `configs/features.yaml`, `run.py`, README, este núcleo): solo por
  PR dedicado.
- Plantilla de PR: tarea, qué cambia, cómo se probó (salida real), revisiones de subagentes, prompts guardados, candidatos a la
  cacería y **la explicación del autor** (el puente no fusiona lo que el autor no sabe explicar).
- El puente revisa PR **al menos tres veces al día** (mañana, tarde, noche) para no bloquear a nadie.

---

## 8. Pruebas mínimas (cada parte agrega las suyas en su spec)
`test_config.py`, `test_no_network.py`, `test_sintetico.py` (esquema = §5.1), `test_fuga.py`, `test_preprocesamiento.py`,
`test_metricas.py`, `test_export.py` (contrato §5.5), `test_notebooks.py` (lenta), `app/tests/paridad.test.ts`. CI las corre en cada PR (las lentas se
marcan y se corren a mano).

---

## 9. Cacería de la alucinación (15 %)
- Registro: `alucinaciones/registro.md` (plantilla) + `evidencias/` + `prompts/`.
- **Guardar prompts literales siempre**: conversaciones con IA, este Spec y cada prompt del orquestador a un subagente.
- Meta: ≥ 5 hallazgos **genuinos** en ≥ 3 categorías; **cada parte aporta ≥ 1**. **Nunca inventar.**
- Cada spec de parte trae su lista de "puntos calientes" (dónde una IA suele equivocarse en esa parte).
- Las afirmaciones de la asesoría inicial están listadas para verificar (V-01…V-11 en el registro).

---

## 10. Spec Driven + Subagent Driven Development

### 10.1 Flujo por tarea
```
Responsable de la tarea (orquesta su sesión de Claude Code)
  → implementador (prueba primero) → revisor-spec → revisor-calidad → verificador (si la tarea lo indica)
  → PR del responsable → puente revisa y fusiona → MEMORIA.md actualizada
```

### 10.2 Contexto de cada sesión y subagente
**Sesión principal de cada integrante:** lee el núcleo completo, **incluido el Anexo (enunciado)**, y la spec de su parte.
**Subagente (mínimo y suficiente):**
1. Este núcleo: §0, §3, §4.1 y **las secciones de §5 que la tarea consume o produce**.
2. **Los apartados del Anexo (enunciado) que cubre su parte** (cada spec de parte los lista en "Lee del núcleo").
3. La spec de su parte (`docs/spec/partes/X.md`), sección de la tarea.
4. El texto de la tarea y su prueba de aceptación.
Nada de otras partes. Si un subagente necesita algo de otra parte, eso es un contrato: está en §5 o falta y se escala.
El prompt literal se guarda en `alucinaciones/prompts/<ID>.md` **antes** de enviarlo.

### 10.3 Agentes (`.claude/agents/`)
`implementador`, `implementador-web`, `revisor-spec`, `revisor-calidad`, `verificador-matematico`, `curador-datos`,
`analista-experimentos`, `cazador-alucinaciones`, `auditor-reproducibilidad`, `redactor-tecnico`, `tutor-sustentacion`.

---

## 11. Hitos de integración

| Cuándo | Hito | De → a |
|---|---|---|
| **Jue 22, noche** | Repo en `main` con ruleset, código común del Trabajo 1, contratos congelados, datos sintéticos, CI | E → todos |
| **Vie 23, 2 p. m.** | Datos reales: `loans_pd.parquet`, preprocesador, `features.yaml`, partición | A → B, C, D |
| **Vie 23, noche** | Búsquedas de Optuna de PD y LGD corriendo con datos reales | B, D |
| **Sáb 24, mediodía** | Modelo final de PD (30 semillas) + `pred_test` | B → C, E |
| **Sáb 24, tarde** | Grabación del video con los 5 (≈ 30 min, todos juntos) | todos |
| **Sáb 24, 6 p. m.** | LGD final (30 semillas) + `pred_test` | D → E |
| **Sáb 24, noche** | `pd_model_{logistica,mlp}.json` reales + casos de paridad | C → E |
| **Dom 25, 8 p. m.** | Secciones del blog de cada parte + ≥ 1 hallazgo por parte | todos → E |
| **Lun 26** | Clon limpio en otra máquina, blog publicado, app final, videos individuales | E + todos |
| **Vie 30, 23:59** | Entrega (límite oficial; antes era el mar 27) | todos |

Si un hito se atrasa, quien lo recibe sigue con los sintéticos y avisa al puente; nadie se queda quieto esperando.

---

## 12. Riesgos y mitigaciones
| Riesgo | Mitigación |
|---|---|
| El puente es cuello de botella con los PR | PR pequeños, CI antes de la revisión, revisión tres veces al día (§7.2) |
| Los datos reales se atrasan | Sintéticos con el mismo esquema (§5.6); D puede partir del CSV crudo |
| Fuga de información infla el AUC | §5.7 + `test_fuga.py` + revisión del `curador-datos` |
| La app calcula distinto que Python | Contrato §5.5 + prueba de paridad en CI |
| La búsqueda y las 30 semillas no caben en el tiempo | Presupuesto reducido (§3), corridas de noche, GPU; demo en CPU |
| El MLP no supera a la logística | Es un resultado válido y se reporta con honestidad (el enunciado lo anticipa) |
| Alguien no entiende otra parte | `tutor-sustentacion` por PR importante + simulacro antes de la sustentación |
| Contratos copiados que divergen | Los contratos viven solo en este núcleo (regla de oro) |

---

## 13. Pendientes
1. Reparto de tareas dentro de cada pareja (D: Camila y José; E: Daniel y Camila) — **hoy, jue 22**. B+C ya repartió (24-oct, §1.1).
2. Aprobación de los [PROPUESTO] de §3 — hoy.
3. Variables del modelo y de la app — Parte A, vie 23 en la mañana.
4. Cortes de la partición temporal — Parte A, vie 23 al mediodía.
5. ~~Modelo que va a la app~~ — [DECIDIDO, Daniel, 26-oct; v0.17]: los dos (logística y MLP), con selector en la app (§5.5).
6. Fecha de la sustentación.

---

## Anexo — Enunciado literal del trabajo

> Transcripción del PDF del curso (texto extraído con PyMuPDF el 19-oct-2026 y limpiado solo en formato: viñetas, tablas y
> la fórmula de E[L]). **Si el Spec contradice este texto, gana este texto y se corrige el Spec.** Este anexo no se edita (núcleo §0.1).

Profesor: Juan David Ospina Arango · Monitor: Andrés Mauricio Zapata Rincón
Universidad Nacional de Colombia · Facultad de Minas · Departamento de Ciencias de la Computación y de la Decisión
Semestre 2026-02 · Curso: Redes neuronales y algoritmos bioinspirados
Modalidad: trabajo en equipo, con sustentación oral y evaluación individual

### Propósito

Un banco quiere saber dos cosas antes de prestarle dinero a alguien: qué tan probable es que no le paguen y, si no le pagan,
cuánto va a perder. En este trabajo van a responder ambas preguntas con redes neuronales artificiales: la primera es un problema
de clasificación binaria y la segunda un problema de regresión. Además, van a convertir su modelo en una herramienta que
cualquier persona pueda usar y van a salir a venderla.

Las redes neuronales no siempre son la mejor herramienta para datos tabulares. Parte del reto es averiguar, con evidencia y
comparaciones justas, qué aportan en este problema y qué no.

Pueden (y se espera) que usen herramientas de inteligencia artificial. Pero la IA se equivoca, a veces con total seguridad.
Parte del trabajo consiste en demostrar que ustedes entienden lo suficiente como para darse cuenta. Lo que se evalúa no es solo
lo que entregan, sino lo que son capaces de explicar, modificar y defender en vivo.

### Datos

Usarán el conjunto Credit Risk Analysis de Kaggle, con créditos de consumo otorgados por la plataforma Lending Club:
https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset

Reporten en el texto el número de registros y variables con que trabajaron, el periodo que cubren y la fecha de descarga.
Consulten el diccionario de variables incluido en el conjunto y citen su significado desde esa fuente, no desde lo que una IA
les diga que significan.

### Parte 1: Probabilidad de incumplimiento (clasificación binaria)

#### Variable objetivo

Modelarán la probabilidad de que un cliente sea "malo", es decir, que incumpla con el pago de su crédito. Construyan una
variable binaria a partir de `loan_status` con la siguiente codificación:

| Categoría original | Codificación | Justificación |
|---|---|---|
| Fully Paid | 0 | Buen pagador confirmado. |
| Does not meet the credit policy. Status:Fully Paid | 0 | No cumple requisitos, pero pagó: buen pagador confirmado. |
| Charged Off | 1 | Crédito reconocido como incobrable. |
| Default | 1 | Impago confirmado. |
| Late (31-120 days) | 1 | Se considera mal pagador. |
| Does not meet the credit policy. Status:Charged Off | 1 | No cumple requisitos y no pagó. |
| Current | NA | Está al día; no hay certeza sobre su comportamiento final. |
| Issued | NA | Aprobado, pero sin historia de pago. |
| In Grace Period | NA | Aún no está obligado a pagar. |
| Late (16-30 days) | NA | No hay certeza sobre su comportamiento final. |

Los casos NA no se incluyen en el entrenamiento. Los modelos de riesgo de crédito estiman la probabilidad de ser malo
(valor 1), no la de ser bueno.

#### Tareas

1. **Caso de uso.** Describan quién usa el modelo, para qué decisión, en qué momento y con qué información disponible en ese
   momento. Todas las decisiones de modelamiento posteriores deben ser coherentes con este caso de uso.
2. **Análisis descriptivo.** Exploren los datos y formulen hipótesis explícitas sobre qué hace más riesgosa a una persona. Al
   final del trabajo, digan cuáles se confirmaron y cuáles no.
3. **Modelo de referencia.** Ajusten una regresión logística como modelo de baja complejidad.
4. **Red neuronal.** Construyan un perceptrón multicapa para estimar la probabilidad de incumplimiento y optimicen su
   arquitectura e hiperparámetros: número de capas, neuronas por capa, funciones de activación, regularización (por ejemplo,
   dropout, penalización L2, parada temprana), optimizador y tasa de aprendizaje. El método de búsqueda es libre, pero debe
   estar justificado.
5. **Evaluación.** Definan y justifiquen sus conjuntos de entrenamiento, validación y prueba, incluyendo al menos una
   partición temporal según la fecha de originación. Evalúen:
   - discriminación: AUC, índice de Gini y estadístico KS;
   - calibración: puntaje de Brier y curva de calibración.
6. **Scorecard.** Representen el modelo con una scorecard. Definan el puntaje base, las odds de referencia y los puntos para
   duplicar las odds (PDO), y deriven la transformación de probabilidad a puntaje. Expliquen cómo obtienen los puntos a partir
   de su modelo. Las scorecards pueden construirse con variables continuas.
7. **Variables de riesgo.** Analicen qué variables hacen más riesgosa a una persona, con un método de interpretabilidad
   justificado y citado.

### Parte 2: Pérdida dado incumplimiento (regresión)

Cuando un cliente incumple, el banco suele recuperar parte del dinero. La pérdida dado incumplimiento (LGD, por sus siglas en
inglés) es la fracción de la exposición que finalmente se pierde.

1. **Variable objetivo.** Para los créditos que incumplieron, construyan la LGD a partir de las variables del conjunto.
   Escriban la fórmula, justifiquen cada campo que usan y discutan sus limitaciones.
2. **Modelo de referencia.** Ajusten un modelo de baja complejidad, por ejemplo una regresión lineal.
3. **Red neuronal.** Construyan y optimicen un perceptrón multicapa para estimar la LGD. Justifiquen la elección de la función
   de pérdida y de la activación de la capa de salida a partir de la naturaleza de la variable objetivo.
4. **Evaluación.** Reporten MAE y RMSE, y comparen la distribución de las predicciones con la distribución observada.
5. **Pérdida esperada.** Combinen ambos modelos para estimar la pérdida esperada de cada crédito,
   E[L] = PD · LGD · EAD,
   definiendo y justificando la exposición al incumplimiento (EAD) que usan.

### Regla de comparación justa

Para comparar modelos deben hacerlo bajo condiciones explícitas y equivalentes:

- **Mismos datos:** todos los modelos de una misma tarea usan el mismo preprocesamiento y las mismas particiones.
- **Presupuesto de búsqueda explícito:** reporten cuántas configuraciones entrenaron para cada familia de modelos y el tiempo
  de cómputo total. Una arquitectura que "gana" tras 200 intentos contra un modelo de referencia ajustado una sola vez no es una
  comparación justa sin esa aclaración.
- **Varias corridas independientes:** entrenen la configuración final de cada red con al menos 30 semillas distintas y reporten
  media, desviación estándar, mejor y peor valor de cada métrica.
- **Complejidad:** reporten el número de parámetros entrenables de cada modelo.

### Discusión

¿Qué aportó la red neuronal frente al modelo de referencia en cada tarea? Respondan con base en las métricas, su variabilidad
entre semillas, el costo de cómputo y la interpretabilidad. Incluyan un listado de aprendizajes sobre el problema que surgieron
del proceso de modelamiento.

### Entregables

#### 1. Reporte técnico publicado como entrada de blog

El reporte debe contener el entendimiento desarrollado en el trabajo, la metodología seguida y justificada, los resultados y la
discusión. Se exige:

- **Bibliografía en formato APA (7.ª edición).** Todas las referencias deben existir, ser verificables y estar citadas en el
  texto. Una referencia inexistente se penaliza.
- **Figuras y tablas numeradas**, con título y fuente, y todas citadas explícitamente en el texto (por ejemplo, "como se observa
  en la Figura 3..."). No se aceptan figuras huérfanas.
- **Fundamentación matemática:**
  - definición de neurona artificial, perceptrón multicapa y cada función de activación utilizada;
  - derivación de cada función de pérdida a partir de un modelo probabilístico (máxima verosimilitud);
  - el algoritmo de retropropagación y las ecuaciones de actualización del optimizador utilizado;
  - la formulación de cada técnica de regularización;
  - la definición de cada métrica de evaluación;
  - la derivación de la scorecard.
- **Sección de uso de IA:** los principales prompts utilizados y una discusión de su impacto en el resultado final.
- **Sección "Cacería de la alucinación"** (ver abajo).
- **Enlaces** al repositorio Git, a la aplicación web y al video promocional.

#### 2. Repositorio Git reproducible

El repositorio debe permitir que cualquier persona reproduzca los resultados. Requisitos mínimos:

- README con instrucciones de instalación y ejecución.
- Entorno especificado con versiones fijas (`requirements.txt`, `environment.yml` o similar).
- Semillas aleatorias fijas y documentadas.
- Los datos procesados que usan los modelos guardados en el repositorio, junto con el código que los genera desde el archivo
  original, para que la ejecución no dependa de conexión a internet ni de APIs externas. Si el archivo original excede los
  límites de tamaño del repositorio, guarden una versión procesada y comprimida, y documenten cómo se obtuvo.
- Los modelos entrenados guardados, de modo que la aplicación y las figuras no requieran reentrenar.
- Las figuras deben poder regenerarse desde el código.
- Historial de commits que refleje el trabajo real del equipo. Un único commit al final no es aceptable.

#### 3. Todo debe correr en vivo

Durante la sustentación el equipo ejecutará su código frente al profesor. Por eso:

- Cada experimento principal debe lanzarse con un solo comando.
- Los parámetros (arquitectura, tasa de aprendizaje, regularización, número de épocas, PDO, puntaje base, etc.) deben poder
  cambiarse desde la línea de comandos o un archivo de configuración, sin editar el código fuente.
- Una corrida de demostración debe tardar menos de dos minutos. Si el entrenamiento completo tarda más, preparen un modo de
  demostración (por ejemplo, con una muestra de los datos) que se lance también con un solo comando.
- La aplicación web debe poder levantarse localmente con un solo comando.
- Antes de la sustentación, prueben todo en un clon limpio del repositorio en una máquina distinta a la de desarrollo.

#### 4. Aplicación web

Una aplicación publicada en la que una persona ingrese sus características y obtenga:

- su puntaje en la scorecard y su probabilidad de incumplimiento;
- cómo se ubica frente a la población (por ejemplo, su percentil sobre la distribución de puntajes);
- cómo cambiaría su puntaje si cambiara alguna de sus características.

La aplicación debe ser intuitiva para un usuario nuevo y contener los enlaces al reporte técnico y al video promocional.

#### 5. Video promocional

Un video de 2 a 3 minutos que presente la aplicación como la solución a un problema y genere entusiasmo por usarla. Todos los
integrantes del equipo deben aparecer en el video promocionando el trabajo. Saber vender lo que uno construye también se
aprende.

#### 6. Videos de contribución individual

Cada integrante graba un video de 2 a 3 minutos hablando en primera persona sobre sus aportes específicos. El video debe
mostrar evidencia concreta: un fragmento de código que escribió, un commit, una figura que elaboró o una sección del reporte
que redactó, y explicar brevemente cómo funciona.

Lo que cada persona declare en su video podrá ser preguntado en la sustentación.

#### 7. Cacería de la alucinación 🎯

Las herramientas de IA producen errores que suenan convincentes. Su misión es cazarlos. Documenten al menos cinco hallazgos en
los que una IA generó algo incorrecto, impreciso, desactualizado o subóptimo durante el desarrollo del trabajo. Deben cubrir al
menos tres de estas categorías:

- **Matemáticas:** derivadas, funciones de pérdida, fórmulas de métricas o de la scorecard.
- **Código:** errores de implementación, preprocesamiento defectuoso, métricas mal calculadas.
- **Datos del mundo real:** significado de las variables, definiciones del negocio de crédito, normativa.
- **Bibliografía:** referencias inexistentes, autores o años equivocados, DOI que no resuelven.
- **Conceptos:** afirmaciones falsas sobre las redes neuronales, su entrenamiento o su desempeño en datos tabulares.

Para cada hallazgo, reporten:

1. El prompt utilizado (literal).
2. La parte relevante de la respuesta de la IA.
3. Cómo sospecharon que había un error.
4. La evidencia que lo demuestra: una derivación, un experimento, una fuente primaria.
5. La corrección.
6. La lección aprendida: ¿qué verificación habría evitado el error?

Si no logran encontrar cinco hallazgos genuinos, documenten las verificaciones que hicieron y sus resultados. Un hallazgo
inventado o sin evidencia se penaliza más que un hallazgo faltante.

Al final del curso se premiará la mejor alucinación cazada: la más sutil, la mejor documentada o la más peligrosa si hubiera
pasado desapercibida.

### Sustentación

Cada equipo tiene 10 minutos:

| Momento | Duración | Contenido |
|---|---|---|
| Demostración | 2 min | Ejecución en vivo de un experimento, con un solo comando. |
| Modificación en vivo | 3 min | El profesor pide un cambio de parámetro o de configuración. Antes de ejecutarlo, el equipo predice qué ocurrirá y por qué. |
| Preguntas | 5 min | El profesor pregunta a cualquier integrante, escogido al azar, sobre cualquier parte del trabajo, incluyendo lo declarado en los videos individuales. |

Todos los integrantes deben estar en capacidad de explicar todo el trabajo, no solo su parte.

### Evaluación

| Componente | Peso | Tipo |
|---|---|---|
| Reporte técnico (rigor, APA, figuras y tablas, discusión) | 20 % | Grupal |
| Repositorio y reproducibilidad | 5 % | Grupal |
| Aplicación web | 10 % | Grupal |
| Video promocional | 5 % | Grupal |
| Ejecución y modificación en vivo | 10 % | Grupal |
| Cacería de la alucinación | 15 % | Grupal |
| Video de contribución individual | 5 % | Individual |
| Respuestas en la sustentación | 30 % | Individual |

La nota individual de la sustentación puede ajustar la nota grupal si un integrante no logra explicar un componente que
declaró como propio.
