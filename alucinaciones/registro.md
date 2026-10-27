# 🎯 Registro de la Cacería de la alucinación

> Meta: ≥ 5 hallazgos **genuinos** en ≥ 3 categorías, al menos 1 por parte. **Nunca inventar.** Un candidato sin prompt
> literal o sin evidencia no cuenta. Evidencias en `alucinaciones/evidencias/`. Prompts literales en `alucinaciones/prompts/`.

## Cobertura actual
| Categoría | Candidatos | Confirmados |
|---|---|---|
| Matemáticas | 3 (H-08, H-10, H-14) | 3 (H-08, H-10, H-14) |
| Código | 15 (H-01, H-03, H-04, H-05, H-06, H-07, H-09, H-11, H-12, H-17, H-18, H-19, H-20, H-24, H-25) | 9 (H-01, H-03, H-04, H-05, H-06, H-07, H-09, H-11, H-12) |
| Datos del mundo real | 2 (H-02, H-13) | 2 (H-02, H-13) |
| Bibliografía | 0 | 0 |
| Conceptos | 2 (H-15, H-16) | 1 (H-15) |

## Afirmaciones pendientes de verificar (de la asesoría inicial, `prompts/asesoria_puente_claude.md`)
Afirmaciones que hizo Claude al planear el trabajo (19-oct-2026). **No son hallazgos**: hay que verificarlas, y solo si resultan
falsas se registran como candidatos.

| # | Afirmación | Cómo verificarla | Resultado |
|---|---|---|---|
| V-01 | Las funciones serverless de Vercel tienen un límite de ~250 MB y PyTorch no cabe | Documentación oficial de límites de Vercel | ⬜ |
| V-02 | El plan Hobby de Vercel es gratis y para uso personal/no comercial | Términos y precios oficiales de Vercel | ⬜ |
| V-03 | Neon Postgres tiene un plan gratis en el Marketplace de Vercel | Documentación de Vercel/Neon | ⬜ |
| V-04 | Render (plan gratis) duerme los servicios tras un periodo sin uso | Documentación de Render | ⬜ |
| V-05 | Remotion es gratis para individuos y equipos pequeños | Licencia oficial de Remotion | ✅ 24-oct: el LICENSE.md oficial da la licencia gratis a "an individual", "a for-profit organization with up to 3 employees" y "a non-profit or not-for-profit organization"; un video de curso hecho por estudiantes queda cubierto |
| V-06 | Los créditos recientes ya resueltos sesgan la tasa de malos hacia arriba | Experimento A4 (tasa de malos y fracción de NA por año) | ✅ 23-oct: consistente con los datos. Sin desenlace: 7 % (2012), 46 % (2013), 69 % (2014), 93 % (2015); tasa de malos de los resueltos: 16,7 % → 23,0 % → 26,2 % → 25,5 % (`resumen_datos.json`). Además, al incumplir seguía pendiente el 80 % del monto en test frente al 64 % en train (incumplimientos más tempranos) |
| V-07 | `AdamW` aplica weight decay desacoplado, que no equivale a L2 con Adam | Loshchilov & Hutter + documentación de PyTorch | ⬜ |
| V-08 | El conjunto cubre créditos de 2007 a 2015 (afirmación implícita en la partición propuesta) | `resumen_datos.json` (A1) | ✅ 23-oct: `issue_d` de 2007-06 a 2015-12 |
| V-09 | La EAD al castigo = `funded_amnt` − `total_rec_prncp`, y LGD = 1 − (`recoveries` − `collection_recovery_fee`)/EAD | Diccionario de Kaggle + casos reales (D1) || ✅ 25-oct: los 5 campos citados literalmente de `diccionario_loanstats.csv` en `src/lgd/objetivo.py` (prueba en `tests/test_lgd.py`); `recoveries` = "post charge off gross recovery" justifica restar la tarifa. Datos coherentes: 0 EAD ≤ 0, 0 LGD > 1, 32 LGD < 0 (intereses fuera de la EAD). Supuesto no verificable sin fechas: lo cobrado tras el castigo no entra en `total_rec_prncp`. Evidencia: `evidencias/D1_verificacion_lgd.*` |
| V-10 | Material for MkDocs está en mantenimiento con correcciones hasta el 5-nov-2026 (dato del Trabajo 1) | Anuncio oficial del proyecto | ⬜ |
| V-11 | Referencias propuestas: Bergstra et al. (2011), Akiba et al. (2019), Lundberg & Lee (2017), Papke & Wooldridge (1996), Loshchilov & Hutter (2019), Grinsztajn et al. (2022), Siddiqi (2017) | Crossref / DOI de cada una | ✅ 23-oct: las 14 candidatas (estas + Rumelhart, Kingma & Ba, Srivastava, Brier, Hanley & McNeil, Ferrari & Cribari-Neto, Yurdakul & Naranjo) existen y sus DOI/URL resuelven al trabajo esperado (`evidencias/A8_verificacion_bib.*`). Matices, no alucinaciones: Kingma & Ba y Loshchilov & Hutter tienen DOI de arXiv con año 2014 y 2017 (los años 2015 y 2019 son de ICLR, sin DOI); Siddiqi: Crossref 2016 en línea, 2017 impreso. Autores de Bergstra y Srivastava confirmados en las páginas oficiales. 25-oct: Siddiqi es la 2.ª edición (enero de 2017) según la página de Wiley; Crossref la registra como edición 1, error de metadatos de Crossref y no de la IA |

---

## Plantilla
```markdown
### H-XX — <título corto>
- **Estado:** candidato / confirmado / descartado
- **Categoría:** Matemáticas | Código | Datos | Bibliografía | Conceptos
- **Parte:** A | B | C | D | E
- **Herramienta y modelo:** · **Fecha:** · **Quién lo cazó:**
1. **Prompt (literal):**
2. **Respuesta relevante de la IA (cita textual o captura):**
3. **Cómo sospechamos:**
4. **Evidencia:** (derivación / experimento / fuente primaria con URL y fecha)
5. **Corrección:**
6. **Lección aprendida — ¿qué verificación lo habría evitado?:**
```

---

## Candidatos

### H-01 — PSI que da 0 en silencio para variables discretas
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Código
- **Parte:** A
- **Herramienta y modelo:** Claude Code, Claude Opus 5.5 · **Fecha:** 23-oct-2026 · **Quién lo cazó:** la misma sesión de
  Claude al revisar los resultados, antes del commit (tarea A6b)
1. **Prompt (literal):** `prompts/asesoria_puente_claude.md`, Prompt I ("Empecemos con lo mio, primero lo mas importante, que
   es la parte A que de eso dependen los demas") con la spec A §3.8, punto 3 ("índice de estabilidad poblacional (PSI) por
   variable").
2. **Respuesta relevante de la IA:** `src/datos/eda.py`, función `psi` original: los bins se definen con
   `np.unique(np.quantile(referencia, deciles))` y se cuentan con `np.searchsorted(..., side="right")`.
3. **Cómo sospechamos:** en `post_preprocesamiento.json` el PSI de `pub_rec` era **exactamente 0,0000** en val y en test,
   mientras las demás variables daban entre 0,003 y 0,27. Un 0 exacto en datos reales es sospechoso.
4. **Evidencia:** `evidencias/H-01_psi_discreto.py` y su salida `.txt`. Tras el recorte al p99, `pub_rec` solo toma dos
   valores; los nueve cuantiles coinciden en un único corte igual al mínimo, y con `side="right"` ambos valores caen en el
   mismo bin: e = a = 1 y PSI = 0. Pero la proporción con registros públicos pasa de 4,07 % en train a 12,75 % en val y
   19,50 % en test (2,8 % en 2012 y 19,5 % en 2014 en los datos crudos): el PSI correcto es 0,107 (val) y 0,269 (test).
5. **Corrección:** si la referencia tiene ≤ n_bins valores distintos, cada valor es su propio bin (cortes en los puntos
   medios entre valores consecutivos). Prueba de regresión `tests/test_eda.py::test_psi_variable_discreta_no_colapsa`.
6. **Lección aprendida:** probar las métricas con casos calculados a mano que incluyan variables discretas y desconfiar de
   resultados exactamente 0 o 1 en datos reales. Una prueba con solo datos continuos no habría fallado.

### H-02 — "En `mths_since_last_delinq`, un vacío significa que nunca hubo mora"
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Datos del mundo real (significado de una variable)
- **Parte:** A
- **Herramienta y modelo:** Claude Code, Claude Opus 5.5 · **Fecha:** 23-oct-2026 · **Quién lo cazó:** el subagente
  `revisor-spec` (primera ronda de la revisión de la Parte A, hallazgo B1), antes de fusionar
1. **Prompt (literal):** `prompts/asesoria_puente_claude.md`, Prompt I ("Empecemos con lo mio, primero lo mas importante, que
   es la parte A que de eso dependen los demas"), con la spec A §3.4 y la regla del núcleo §0.5 ("el significado de cada
   variable se cita del diccionario de Kaggle, nunca de una IA").
2. **Respuesta relevante de la IA:** `blog/secciones/02-datos.md` (commit da3deb3): "variables con más de la mitad de valores
   vacíos, como `mths_since_last_delinq`, donde un vacío significa «nunca ocurrió» y no un dato faltante".
3. **Cómo sospechamos:** el revisor contrastó cada afirmación sobre variables con `LCDataDictionary.xlsx` y la de los vacíos
   no aparece ahí (prompt literal de la revisión en `prompts/A-revision-spec.md`).
4. **Evidencia:** `evidencias/H-02_mths_since_last_delinq.py` y su salida `.txt`. El diccionario solo dice "The number of
   months since the borrower's last delinquency." y ninguna definición habla de vacíos (missing, blank, null, never, empty).
   Además, los datos contradicen la interpretación: entre los créditos con el campo vacío (51,2 %), el **0,46 % tiene
   `delinq_2yrs` > 0**, es decir, moras en los últimos 2 años. Si vacío significara "nunca hubo mora", esa proporción sería 0.
5. **Corrección:** el blog dice ahora que el diccionario no explica el vacío y que por eso no la imputamos (commit e9243c0).
   La variable sigue fuera del modelo.
6. **Lección aprendida:** una explicación "de sentido común" de un vacío es una afirmación sobre los datos: se cita del
   diccionario o se prueba con los datos, como hace el script de evidencia. Revisar cada frase "X significa Y" contra la fuente.


### H-03 — "calibration_curve asigna al bin superior el dato que cae en un borde (`side="right"`)"
- **Estado:** confirmado (Andrés, 24-oct-2026)
- **Categoría:** Código
- **Parte:** B
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 24-oct-2026 · **Quién lo
  cazó:** el subagente `verificador-matematico` al revisar las pruebas de B1 antes de implementar; el orquestador lo
  comprobó en el código fuente de sklearn
1. **Prompt (literal):** `prompts/B1.md`, sección "1. implementador — fase roja".
2. **Respuesta relevante de la IA:** "Puntos en un borde interno. La interfaz tampoco dice a qué bin van. Las pruebas a mano lo
   evitan y el contraste con sklearn fija su convención: `searchsorted(bordes[1:-1], p, side="right")`."
3. **Cómo sospechamos:** el verificador probó una implementación mutante con `side="right"` y pasaba las 52 pruebas: el
   contraste con sklearn solo usaba datos sin empates, donde ningún dato cae en un borde, así que la prueba no fijaba nada.
4. **Evidencia:** `sklearn/calibration.py:1332` (scikit-learn 1.9.1): `binids = np.searchsorted(bins[1:-1], y_prob)`, es decir,
   lado por defecto `"left"`: el dato que cae en un borde va al bin **inferior**. Con datos con empates (p a 2 decimales)
   8 de los 9 bordes internos coinciden con datos; `side="left"` reproduce `calibration_curve` y `side="right"` no
   (`evidencias/H-03_calibracion_borde.py` y `.txt`).
5. **Corrección:** la prueba de contraste con `calibration_curve` se corre también con datos con empates y la implementación
   usa `side="left"` (ronda verde de B1).
6. **Lección aprendida:** una afirmación del tipo "la referencia hace X" se comprueba en el código fuente de la referencia;
   y una prueba de contraste solo fija una convención si sus datos ejercitan el caso (aquí, empates en los bordes).

### H-04 — AUC errónea en silencio cuando el puntaje trae ±inf
- **Estado:** confirmado (Andrés, 24-oct-2026)
- **Categoría:** Código
- **Parte:** B
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 24-oct-2026 · **Quién lo
  cazó:** el subagente `revisor-calidad`; el orquestador lo reprodujo
1. **Prompt (literal):** `prompts/B1.md`, sección "3. implementador — fase verde".
2. **Respuesta relevante de la IA:** `src/common/metricas.py` (commit a11a56c) declara en la celda de la ROC que $p_i \in
   \mathbb{R}$ y que el umbral `+inf` es "nadie clasificado como malo", pero `validar_entradas` solo rechaza NaN
   (`np.isnan`), no ±inf.
3. **Cómo sospechamos:** al revisar casos borde: el logit de una PD saturada en 0 o 1 es ±inf, y la prueba de invariancia
   recortaba p a [1e-6, 1 − 1e-6], justo para evitar ese caso.
4. **Evidencia:** con y = [0,1,0,1,0,1] y p = [−inf, −inf, 0.2, 0.5, inf, inf], `auc` devuelve 0.444; por pares
   (Mann-Whitney, empates ½) la AUC es 5/9 = 0.556. Causa: `np.diff` entre infinitos iguales da NaN (≠ 0), así que los
   empates en ±inf no se agrupan, y el umbral inicial `+inf` choca con datos que valen `+inf`. `roc_auc_score` rechaza esa
   entrada. `evidencias/H-04_auc_infinitos.py` y `.txt`.
5. **Corrección:** `validar_entradas` exige valores finitos (`np.isfinite`) y hay una prueba con `inf` (ronda 2 de B1).
6. **Lección aprendida:** validar "sin NaN" no es validar "número real": ±inf también rompe las diferencias y los
   umbrales. Las pruebas de propiedades no deben recortar las entradas justo donde la función puede fallar.

### H-05 — Prueba de L2 "con lr y gradiente de la pérdida en 0", que no verifica nada
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Código
- **Parte:** B
- **Herramienta y modelo:** Claude Code, sesión principal (orquestador, Claude Opus 5.5) en modo plan · **Fecha:**
  24-oct-2026 · **Quién lo cazó:** Andrés, al revisar el plan de B2
1. **Prompt (literal):** `prompts/B2.md`, sección "Prompt de Andrés a la sesión principal".
2. **Respuesta relevante de la IA:** en el plan de B2, la prueba 10 decía: "Paso a mano de L2: con lr y gradiente de la
   pérdida en 0 (pérdida constante), un paso de SGD sin momento deja `W ← W(1 − lr·l2)` y el sesgo intacto (confirma
   (λ/2)‖W‖² y que los sesgos no se penalizan)."
3. **Cómo sospechamos:** Andrés: "si el lr está en 0, no se supone que no se va a mover por el peso y la prueba no haría
   nada?". Con lr = 0, la actualización θ ← θ − lr·(∇L + λθ) deja θ igual, así que `W(1 − lr·l2) = W` y la prueba pasaría
   aunque el L2 no existiera o se aplicara a los sesgos.
4. **Evidencia:** la regla de actualización de SGD con `weight_decay`, θ ← θ − η(∇L(θ) + λθ). Con η = 0 da θ ← θ para
   cualquier λ, así que la aserción no distingue un L2 correcto de uno ausente. Lo que se quería decir era η > 0 con
   ∇L = 0, que da θ ← θ(1 − ηλ). La frase "con lr y gradiente de la pérdida en 0" pone los dos en 0.
5. **Corrección:** la prueba se eliminó del plan. Además de estar mal escrita, probaba la matemática de `weight_decay`
   de PyTorch y no nuestro código (criterio de Andrés). El L2 se cubre con la prueba de los grupos de parámetros (pesos con
   `weight_decay = l2`, sesgos con 0) y con la de "l2 > 0 da pesos de menor norma".
6. **Lección aprendida:** toda prueba debe poder fallar. Antes de aceptarla, preguntarse qué implementación incorrecta la
   haría fallar; si ninguna, la prueba no verifica nada.

### H-06 — "La prueba de determinismo y la de dropout al predecir ya cubren los minilotes y `evaluar_perdida`"
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Código
- **Parte:** B
- **Herramienta y modelo:** Claude Code, sesión principal (orquestador, Claude Opus 5.5) · **Fecha:** 24/25-oct-2026 ·
  **Quién lo cazó:** el subagente `implementador`, con pruebas de mutantes (fase verde de B2)
1. **Prompt (literal):** `prompts/B2.md`, "Corrección de Andrés al primer plan" (pide quitar las pruebas redundantes, p. ej.
   la 11 y la 14) y "2. implementador — fase verde".
2. **Respuesta relevante de la IA:** el plan corregido de B2 decía: "Pruebas quitadas: … minilotes (15), …
   `evaluar_perdida` en eval() (14)". En la fase verde, el orquestador repitió "el orden reproducible lo cubre 15" y
   "lo cubre 14".
3. **Cómo sospechamos:** el implementador repitió sus mutantes sobre el código real y 3 sobrevivieron.
4. **Evidencia:** los tres mutantes pasan las 71 pruebas: (a) `range(0, n − batch + 1, batch)`, que con 600 filas y
   batch 512 descarta 88 filas por época; (b) minilotes sin barajar (`np.arange`): la prueba 15 no lo ve porque otra semilla
   cambia igual la inicialización; (c) `evaluar_perdida` sin `eval()`: la prueba 14 no lo ve porque `predict_proba` llama a
   `eval()` por su cuenta, así que la validación se mediría con dropout activo. "Cubre" se afirmó sin preguntarse qué
   mutante mata cada prueba.
5. **Corrección:** Andrés decide volver a poner tres pruebas (minilotes: cobertura con lote incompleto; minilotes: orden
   barajado y reproducible; `evaluar_perdida` con dropout 0.5 da dos medidas iguales).
6. **Lección aprendida:** "la prueba X cubre el fallo Y" es una afirmación verificable: se comprueba con un mutante de Y.
   Una prueba de integración no cubre un fallo si otra vía (otra semilla, otra llamada a `eval()`) lo enmascara.

### H-07 — "`@dataclass(frozen=True)` hace inmutable el historial de entrenamiento"
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Código
- **Parte:** B
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 25-oct-2026 · **Quién lo
  cazó:** el subagente `revisor-calidad` (3.ª revisión de B2)
1. **Prompt (literal):** `prompts/B2.md`, sección "8. implementador — ronda 4".
2. **Respuesta relevante de la IA:** comentarios en el commit af1456a: `src/pd/entrenar.py:316` "inmutable: quien lo reciba
   (B4, B5, D) no puede alterar el historial"; `src/pd/mlp.py:418` "inmutable: se puede compartir sin copiar";
   `entrenar.py:34` "resultado inmutable del entrenamiento".
3. **Cómo sospechamos:** `frozen=True` solo prohíbe reasignar atributos (`r.historial = ...`), y `historial` es una `list`.
4. **Evidencia:** el revisor ejecutó `r.historial.append(1)` sobre un `ResultadoEntrenamiento` y funcionó sin error: la lista
   se puede modificar. Es el comportamiento documentado de `dataclasses` (la congelación es superficial).
5. **Corrección:** comentarios reescritos (commit siguiente a af1456a): "frozen: no se reasignan los campos; la lista del
   historial sí es mutable". Alternativa no aplicada: guardar `tuple(historial)`.
6. **Lección aprendida:** "inmutable" de un contenedor congelado no se extiende a sus campos mutables; un comentario
   también es una afirmación que se verifica.

### H-08 — "Penalizar λ‖W‖² en la pérdida equivale al `weight_decay` de Adam/SGD"
- **Estado:** confirmado (Daniel, 25-oct-2026)
- **Categoría:** Matemáticas
- **Parte:** B
- **Herramienta y modelo:** Claude Code (app de escritorio), Claude Opus 5.5, en la sesión del puente (Daniel) que escribió el
  Spec; la línea entra al historial en la v0.4 (commit b562895, 22-oct). Daniel confirma el origen el 25-oct · **Fecha de detección:** 25-oct-2026 · **Quién lo cazó:** el subagente
  `verificador-matematico` (B2)
1. **Prompt (literal):** "Dame el spec yo lo reviso" (`prompts/asesoria_puente_claude.md`), el pedido que produjo la spec por partes
   (v0.2), donde se creó `B-pd.md`. La versión exacta en que entró la línea no quedó registrada: el primer commit es la v0.4. Verificación: `prompts/B2.md`, sección
   "9. verificador-matematico".
2. **Respuesta relevante de la IA:** `docs/spec/partes/B-pd.md` §3.2 (hasta v0.9): "el código debe hacer L2 (término
   λ‖W‖² en la pérdida, o `weight_decay` de `Adam`/`SGD`, que lo suma al gradiente)".
3. **Cómo sospechamos:** al verificar la celda de L2 de `src/pd/entrenar.py`, que usa (λ/2)‖W‖².
4. **Evidencia:** ∇(λ‖W‖²) = 2λW, mientras que `weight_decay = λ` suma λW al gradiente: lo equivalente es (λ/2)‖W‖².
   Numéricamente (5 pasos, scripts en el scratchpad de la sesión): `weight_decay = λ` y "+(λ/2)‖W‖²" dan los mismos pesos
   (dif. máx. 2,8e-17 con SGD y 5,6e-17 con Adam); con "+λ‖W‖²" difieren (6,1e-3 con SGD y 5,5e-2 con Adam).
5. **Corrección:** Spec v0.13 (25-oct, Andrés; numerada 0.10 en `dev/andres`): B-pd §3.2 dice (λ/2)‖W‖² ≡ `weight_decay = λ`, sin L2 en los sesgos.
   El código ya seguía esta convención (decisión de Andrés, 24-oct). El blog 04 (B8) debe escribir (λ/2)‖W‖².
6. **Lección aprendida:** "equivalente" entre una penalización y un hiperparámetro de la librería se verifica derivando el
   gradiente y con un experimento de pocos pasos; el factor ½ cambia la escala de λ.

### H-09 — "La prueba compara `warnings.filters` antes y después de `fit`, así que detecta un filtro global"
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Código
- **Parte:** B
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 25-oct-2026 · **Quién lo
  cazó:** el propio `implementador`, con el mutante M30 de la fase verde de B3
1. **Prompt (literal):** `prompts/B3.md`, sección "1. implementador — fase roja" (prueba 11: "warnings.filters es igual
   antes y después de fit").
2. **Respuesta relevante de la IA:** `tests/test_logistica.py` (commit 7d72a68), `test_11_fit_no_altera_los_filtros_globales`,
   presentada como la prueba que atrapa un `simplefilter("error")` global en lugar de un `catch_warnings` local.
3. **Cómo sospechamos:** el mutante M30 (quitar el `catch_warnings` local de `ajustar_sin_avisos`) sobrevivía.
4. **Evidencia:** el `pytestmark` del módulo ya instala `error::ConvergenceWarning` y `error::FutureWarning`; el filtro
   global del mutante coincidía con uno ya presente y la lista `warnings.filters` no cambiaba. Tras partir de un entorno con
   `simplefilter("ignore")`, M30 muere (commit dc4cc9d).
5. **Corrección:** prueba reforzada en la fase verde de B3 (commit dc4cc9d).
6. **Lección aprendida:** una prueba de "no efectos secundarios" debe partir de un estado distinto del que el efecto
   produciría; verificarla con un mutante, no por lectura.

### H-10 — "El estimador es consistente para la media condicional", dicho del MLP de LGD
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Matemáticas
- **Parte:** D
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 24-oct-2026 · **Quién lo
  cazó:** el subagente `verificador-matematico`; el orquestador guardó la evidencia
- **Nota de numeración:** registrado primero como H-05 en `dev/camila`; renumerado a H-10 al integrar `main` (PR #6), porque H-03…H-09 son de la Parte B.
1. **Prompt (literal):** `prompts/D3.md`, ronda 1 (punto 2 de "Qué entregar": "por qué se minimiza en pᵢ = yᵢ también para
   yᵢ fraccional").
2. **Respuesta relevante de la IA:** docstring de `src/lgd/perdidas.py` (commit 922f2f1): "la condición de primer orden solo
   exige que la media condicional esté bien especificada; no exige que y siga una Bernoulli. Por eso el estimador es
   consistente para la media condicional con objetivos fraccionales", en el módulo de la pérdida del **MLP**.
3. **Cómo sospechamos:** el argumento (E[μ − y | x] = 0 cuando μ = E[y | x]) prueba que E[y | x] es el **minimizador
   poblacional** de la pérdida esperada; la **consistencia** de Papke & Wooldridge (1996) es un resultado para un modelo
   paramétrico con la media bien especificada y condiciones de regularidad, que no se traslada sin más a un MLP no
   identificable entrenado con L2 y parada temprana.
4. **Evidencia:** `evidencias/H-10_consistencia_qmle.py` y `.txt`: (i) para y fija, el argmin de la BCE está en μ = y;
   (ii) con y ~ Beta y E[y|x] = σ(−0,5 + 1,2x), n = 200 000, el QMLE **paramétrico** da (−0,4997; 1,2023): la consistencia
   vale para ese modelo. Fuente del resultado: Oberhofer & Pfaffermayr (2009), réplica de Papke & Wooldridge, ec. (2) y
   "the proposed QLME is consistent as long as the conditional mean function is correctly specified",
   https://www2.uibk.ac.at/downloads/c4041030/wpaper/2009-02.pdf (consultada el 24-oct-2026). El artículo original (Journal
   of Applied Econometrics, 11(6), 619–632) no se pudo leer: el PDF disponible es un escaneo sin texto.
5. **Corrección:** ronda 2 de D3: el docstring separa el minimizador poblacional E[y | x] (lo que aplica al MLP) de la
   consistencia del QMLE paramétrico (Papke & Wooldridge, 1996), con `TODO(cita)` sobre la fuente secundaria.
6. **Lección aprendida:** un resultado asintótico tiene hipótesis (modelo paramétrico, media bien especificada); citarlo para
   un modelo que no las cumple es una exageración. Pedir al verificador que contraste cada "es consistente / es insesgado"
   con las hipótesis del teorema citado.

### H-11 — "Las funciones usan factor y offset tal como vienen", probado solo en dos de las cuatro funciones
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Código
- **Parte:** C
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 25-oct-2026 · **Quién lo
  cazó:** el subagente `verificador-matematico`, con mutantes sobre las pruebas de la fase roja de C1; Juan José había
  pedido revisar el caso del bool en `params` antes de la verificación
1. **Prompt (literal):** `prompts/C1.md`, sección "1. implementador — fase roja" (interfaz: "params debe tener EXACTAMENTE
   las claves de CLAVES_PARAMETROS …, factor real finito > 0 y offset real finito. No se comprueba que factor y offset
   coincidan con base/odds_base/pdo").
2. **Respuesta relevante de la IA:** `tests/test_scorecard.py` (fase roja, sin commit),
   `test_03e_usa_factor_y_offset_recibidos`, con el docstring "Las funciones usan factor y offset tal como vienen (no los
   recalculan desde base/odds_base/pdo)", pero solo llama a `score_desde_logit` y `logit_desde_score`. Además, la lista de
   `params` inválidos no incluye ningún bool, aunque `validar_cfg_scorecard` sí los rechaza.
3. **Cómo sospechamos:** el docstring afirma algo de "las funciones" (las cuatro) y la prueba cubre dos; el verificador
   escribió mutantes a propósito para esa afirmación.
4. **Evidencia:** `evidencias/H-11_mutantes_sobreviven.txt` (con `H-11_referencia_con_mutantes.py` y
   `H-11_pruebas_propuestas.py`): el mutante `recalcula_solo_pd` (recalcula factor y offset dentro de `score_desde_pd` y
   `pd_desde_score`) y el mutante `bool_params` (`float(True) = 1.0` aceptado como factor) pasan las 199 pruebas; con las
   pruebas propuestas mueren (1 y 24 fallos).
5. **Corrección:** antes de la fase verde de C1 se añaden `test_03f` (las funciones de PD usan el factor y el offset
   recibidos: σ(2) → 80 con factor 10 y offset 100) y `test_09o` (bool en factor u offset → ValueError en las 4 funciones).
6. **Lección aprendida:** cuando un docstring de prueba dice "las funciones…", la prueba debe recorrer todas (parametrizar
   por función); y una regla de validación (rechazar bool) se prueba en cada lugar donde se aplica, no solo en el primero.

### H-12 — "Texto u objetos no numéricos fallan aquí", dicho de `np.asarray(x, dtype=np.float64)`
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Código
- **Parte:** C
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 25-oct-2026 · **Quién lo
  cazó:** el subagente `revisor-calidad` (bloqueante B1 de la revisión de C1); el orquestador lo reprodujo
1. **Prompt (literal):** `prompts/C1.md`, sección "3. implementador — fase verde" ("entrada convertida con
   np.asarray(..., dtype=np.float64)"; estilo de validación como `_es_real`, que rechaza bool y texto).
2. **Respuesta relevante de la IA:** `src/scorecard/scorecard.py` (fase verde, sin commit), `_a_float64`:
   `try:  # texto u objetos no numéricos fallan aquí` / `return np.asarray(x, dtype=np.float64)`, presentado como el
   filtro que rechaza entradas no numéricas en las cuatro transformaciones.
3. **Cómo sospechamos:** el mismo módulo rechaza bool y texto en la config y en `params` porque `float(True) = 1.0` "se
   colaría"; el revisor probó esas mismas entradas en las transformaciones.
4. **Evidencia:** `evidencias/H-12_entrada_texto_bool.txt`: `score_desde_logit("1.5")` = 443.84, `score_desde_pd("0.3")` =
   511.57 y `score_desde_logit(True)` = 458.27 sin error; solo falla el texto que no parece número (`"abc"`). NumPy convierte
   con `float()` los textos numéricos y los bool al pedir `dtype=float64`.
5. **Corrección:** ronda 2 de C1: `_a_float64` rechaza todo dtype que no sea entero o real (`dtype.kind not in "iuf"`: bool,
   texto, objeto, complejo) antes de convertir, con pruebas parametrizadas sobre las cuatro funciones.
6. **Lección aprendida:** "esto falla con X" sobre una conversión de NumPy se comprueba ejecutándola con X; una regla de
   validación del módulo (no bool, no texto) debe aplicarse y probarse en todas sus entradas, no solo en la config.

### H-13 — "La pérdida observada es la misma construcción de la LGD de D" y "en mora es parcial"
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Datos del mundo real
- **Parte:** E
- **Herramienta y modelo:** Claude Code (app de escritorio), Claude Opus 5.5, la sesión orquestadora de Daniel, que escribió
  la 1.ª versión de `src/el/` sin subagente · **Fecha:** 24-oct-2026 · **Quién lo cazó:** el subagente
  `verificador-matematico` en la revisión de E5 (26-oct); el revisor-spec, en paralelo, señaló el contraste por banda
1. **Prompt (literal):** "Vamos haciendo lo de la parte E" (Daniel, 24-oct, en la sesión del puente); la verificación está en
   `prompts/E5.md`, sección 3.
2. **Respuesta relevante de la IA:** docstring de `src/el/perdida_esperada.py` (commit de E5 del 24-oct): la pérdida
   observada `max(0, (funded_amnt − total_rec_prncp) − (recoveries − collection_recovery_fee))` "es la misma construcción de
   la LGD de la Parte D (D §3.2) multiplicada por la EAD al incumplimiento. En los créditos en mora […] su pérdida observada
   es parcial y se marca", aplicada a todos los estados de test.
3. **Cómo sospechamos:** el verificador recalculó la pérdida por estado y la contrastó con la población de la LGD de D
   (solo castigados).
4. **Evidencia:** `evidencias/H-13_perdida_observada.py` y `.txt`: de los 231,2 M USD "observados", 168,2 M son pérdida
   realizada de castigados; 1 284 créditos `Fully Paid` suman 1,67 M (197 con más de 1 USD) porque su principal recibido
   no llega al desembolsado; y los 5 460 en mora suman 61,3 M (26,5 %) con `recoveries` = 0 en todos: no es una pérdida
   "parcial" (cota inferior) sino LGD = 1 supuesta, una cifra provisional y pesimista. Con eso, la razón E[L]/observada
   salía 0,85 ("el modelo subestima") cuando en los créditos resueltos es 1,06.
5. **Corrección:** ronda 2 de E5 (26-oct): pérdida realizada solo en castigados, 0 en pagados (la diferencia se reporta
   como anomalía de los datos), pérdida provisional aparte en mora y contraste solo con los resueltos, total y por banda.
6. **Lección aprendida:** una pérdida "observada" se define por estado del crédito y con la misma población que la LGD; antes
   de llamar "parcial" a una cifra, se mira si de verdad es una cota (aquí las recuperaciones en mora eran 0).

### H-14 — "Montos desembolsados parecidos (136,15 y 91,31 millones). La diferencia la produce la PD"
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Matemáticas
- **Parte:** E
- **Herramienta y modelo:** Claude Code, subagente `redactor-tecnico` (Claude Opus 5.5) · **Fecha:** 26-oct-2026 · **Quién
  lo cazó:** el subagente `cazador-alucinaciones` y, en paralelo, el `verificador-matematico`, en la revisión del blog 09
1. **Prompt (literal):** `prompts/E7.md`, sección 1 (punto 5: "explica con cuidado […] SIN afirmar más de lo que muestran
   los datos"); la verificación, en la sección 2.
2. **Respuesta relevante de la IA:** `blog/secciones/09-perdida-esperada.md` (borrador sin commit, línea 95): la E[L] de la
   banda 1 es casi diez veces la de la banda 10 "con montos desembolsados parecidos (136,15 y 91,31 millones). La diferencia
   la produce la PD (de 0,416 a 0,063)".
3. **Cómo sospechamos:** la frase llama "parecidos" a dos montos que ella misma cita y que difieren en un 49 %.
4. **Evidencia:** `evidencias/H-14_bandas_1_y_10.py` y `.txt`: la razón 9,92 entre las E[L] de las bandas 1 y 10 se
   reparte en PD × 6,64, EAD × 1,49 y LGD × 1,01; en escala logarítmica la PD explica el 82,5 % y la EAD el 17,4 %. La PD
   pesa más, pero no es "la" diferencia. Además, las bandas son deciles del puntaje, que es una transformación monótona de
   la misma PD (correlación de rangos −1): parte de la conclusión es por construcción.
5. **Corrección:** ronda 2 del blog 09: "sobre todo la PD", con el factor de la EAD, y aclaración de que las bandas
   ordenan por la PD de la logística.
6. **Lección aprendida:** un adjetivo cualitativo ("parecidos", "la causa") junto a cifras se comprueba con las cifras
   mismas; una descomposición multiplicativa dice cuánto aporta cada factor.

### H-15 — "Lo que hace alta la razón de la banda 10 es el factor EAD, el mayor de todos"
- **Estado:** confirmado (Daniel, 26-oct-2026)
- **Categoría:** Conceptos
- **Parte:** E
- **Herramienta y modelo:** Claude Code, subagente `cazador-alucinaciones` (Claude Opus 5.5), en su propia ronda 1 de
  verificación del blog 09 · **Fecha:** 26-oct-2026 · **Quién lo cazó:** el mismo subagente, en la ronda 2 (lo reportó
  como error propio)
1. **Prompt (literal):** `prompts/E7.md`, sección 2 (verificación del blog 09); la frase pasó al encargo de la ronda 2 del
   redactor (sección 3, punto 5), que la copió en el texto.
2. **Respuesta relevante de la IA:** informe de la ronda 1 del cazador: "Lo que hace alta la razón de la banda 10 es el
   factor EAD, el mayor de todos (1,275)".
3. **Cómo sospechamos:** en la ronda 2, el cazador comparó los factores de las bandas 8 a 10 y los del orden alternativo de
   la descomposición.
4. **Evidencia:** `evidencias/H-15_factor_ead_banda_10.py` y `.txt`: la banda 8 tiene casi el mismo factor EAD (1,2742
   frente a 1,2747) y razón 1,002; la 9, 1,2718 y razón 1,107. Lo que separa a la banda 10 es su factor PD (0,981 frente a
   0,818 y 0,903). Con el orden alternativo, el factor EAD máximo es el de la banda 8 (1,2706 frente a 1,2696).
5. **Corrección:** el orquestador aplicó al blog 09 la redacción que propuso el cazador: la razón de la banda 10 combina un
   factor PD cercano a 1 con el factor EAD más alto, comparada con las bandas 2, 8 y 9.
6. **Lección aprendida:** un verificador también puede atribuir una causa única donde hay dos que se combinan; antes de
   explicar un extremo por un factor, se mira si otra banda con ese factor igual tiene el mismo resultado.

### H-16 — "Optuna TPE" justificado solo con Bergstra et al. (2011), cuando Optuna 5 usa TPE multivariado por defecto
- **Estado:** candidato (falta el texto literal de la respuesta: la asesoría inicial solo guardó un resumen)
- **Categoría:** Conceptos
- **Parte:** B
- **Nota de numeración:** registrado como H-11 en `dev/andres`; renumerado a H-16 al integrar `main` (26-oct, después del PR #9), porque H-11…H-15 ya eran de las Partes C y E.
- **Herramienta y modelo:** Claude (asesoría inicial del puente y Spec v0.1) · **Fecha:** 25-oct-2026 · **Quién lo cazó:**
  la sesión orquestadora de Andrés (Claude Opus 5.5), al planear B4, leyendo el código fuente de Optuna instalado
1. **Prompt (literal):** `prompts/asesoria_puente_claude.md` (prompt A); la respuesta guardada es un resumen ("Optuna TPE").
   El texto que se usó está en la Spec: B-pd §3.3 hasta v0.14, "Justificación citada: TPE (Bergstra et al., 2011)".
2. **Respuesta relevante de la IA:** B-pd §3.3 (v0.14): "Optuna TPE, 40 ensayos … Justificación citada: TPE (Bergstra et
   al., 2011) y Optuna (Akiba et al., 2019)".
3. **Cómo sospechamos:** `TPESampler` tiene `multivariate=None` en la firma, no `False`; había que saber qué hace `None`.
4. **Evidencia:** `evidencias/H-16_tpe_multivariado.py` y `.txt`: con Optuna 5.0.0 (versión fijada), `_is_multivariate`
   devuelve True en estudios de un objetivo ("By default, the multivariate TPE is used for single-objective optimization").
   El TPE de Bergstra et al. (2011) modela cada hiperparámetro por separado.
5. **Corrección:** Spec v0.15 (B-pd §3.3): se mantienen los valores por defecto (decisión de Andrés), `hpo_ensayos.json`
   registra `multivariate: true` y el blog cita además el TPE multivariado: TODO(cita) Falkner et al. (2018) [VERIFICAR].
   Aviso a D, que también usa los valores por defecto.
6. **Lección aprendida:** los valores por defecto cambian entre versiones mayores; antes de citar el método de una
   biblioteca, leer la firma y el código de la versión fijada en `requirements.txt`, no la documentación de memoria.

### H-17 — Caso de prueba "dropout = 0.475 → borde máximo" falso en coma flotante
- **Estado:** candidato
- **Categoría:** Código
- **Parte:** B
- **Nota de numeración:** registrado como H-12 en `dev/andres`; renumerado a H-17 al integrar `main` (26-oct, después del PR #9), porque H-11…H-15 ya eran de las Partes C y E.
- **Herramienta y modelo:** Claude Code, subagente `revisor-calidad` y sesión orquestadora (Claude Opus 5.5) · **Fecha:**
  25-oct-2026 · **Quién lo cazó:** el subagente `implementador` al escribir la prueba (ronda 2 de B4)
1. **Prompt (literal):** `prompts/B4.md`, sección 5 (ronda 2), recomendado 1: "test_04: caso dropout = 0.475 → "max" y otro
   justo fuera (simétricos a dropout_justo/dropout_fuera)".
2. **Respuesta relevante de la IA:** el revisor-calidad propuso el caso y el orquestador lo pasó sin verificarlo.
3. **Cómo sospechamos:** la prueba falló contra el código correcto.
4. **Evidencia:** `evidencias/H-17_borde_coma_flotante.py` y `.txt`: 0.5 − 0.475 = 0.025000000000000022 > 0.05 · 0.5 =
   0.025, así que la regla `distancia ≤ tolerancia` da False; con valores exactos en binario (f = 0.25, 0.375) da True.
5. **Corrección:** `test_04_en_borde_limites_exactos_simetricos` usa `fraccion = 0.25` y valores representables exactamente
   (0.125, 0.375 y sus vecinos).
6. **Lección aprendida:** una prueba de límite con `≤` necesita valores exactos en binario (potencias de 2); antes de
   proponer un caso, calcular la distancia en el intérprete.

### H-18 — "La prueba 12 (reloj de paso fijo) atrapa que el tiempo incluya la carga o la medición"
- **Estado:** candidato
- **Categoría:** Código
- **Parte:** B
- **Nota de numeración:** registrado como H-13 en `dev/andres`; renumerado a H-18 al integrar `main` (26-oct, después del PR #9), porque H-11…H-15 ya eran de las Partes C y E.
- **Herramienta y modelo:** Claude Code, sesión orquestadora (Claude Opus 5.5) · **Fecha:** 26-oct-2026 · **Quién lo
  cazó:** el subagente `implementador` (ronda 1 de B5), con un mutante
1. **Prompt (literal):** `prompts/B5.md`, prompt de Andrés ("Quiero el plan con: … (d) riesgos y qué prueba los atrapa").
2. **Respuesta relevante de la IA:** tabla de riesgos del plan de B5: "Tiempo incluye carga/medición | 12", con la prueba 12
   = "reloj inyectado: `tiempo_s` = 1.0 en cada semilla (solo el fit)".
3. **Cómo sospechamos:** el implementador metió la medición dentro de la zona cronometrada y la prueba 12 siguió en verde.
4. **Evidencia:** `evidencias/H-18_reloj_paso_fijo.py` y `.txt`: con un reloj que avanza un paso por lectura, cronometrar
   solo el fit o el fit más trabajo extra da lo mismo (1.0 y 1.0): el reloj cuenta lecturas, no trabajo.
5. **Corrección:** prueba 12b (`tests/test_experimento_pd.py`): un reloj espía registra los eventos y comprueba que entre
   los dos tics de cada semilla hay exactamente un `fit` y ninguna predicción.
6. **Lección aprendida:** antes de afirmar "esta prueba atrapa X", escribir el mutante X y verlo fallar.

### H-19 — "Las pruebas 17–18 comprueban que `pd_mlp.pt` y `pred_mlp` son los de la semilla mediana"
- **Estado:** candidato
- **Categoría:** Código
- **Parte:** B
- **Nota de numeración:** registrado como H-14 en `dev/andres`; renumerado a H-19 al integrar `main` (26-oct, después del PR #9), porque H-11…H-15 ya eran de las Partes C y E.
- **Herramienta y modelo:** Claude Code, sesión orquestadora y subagente `implementador` (Claude Opus 5.5) · **Fecha:**
  26-oct-2026 · **Quién lo cazó:** el subagente `revisor-calidad` (ronda 2 de B5), con mutantes
1. **Prompt (literal):** `prompts/B5.md`, prompt de Andrés (prueba 5: "MLPPD.cargar(...).predict_proba(X_test) ==
   pred_mlp, y es la red de summary.semilla_mediana") y sección 5 (revisor-calidad).
2. **Respuesta relevante de la IA:** el plan (pruebas 17–18) y el implementador dieron por cubierta la regla "se guarda la
   red de la semilla mediana" con un escenario de 3 semillas reales.
3. **Cómo sospechamos:** el revisor-calidad aplicó el mutante "guardar la red de la primera semilla" y sobrevivió (el de
   "la última" sí moría).
4. **Evidencia:** en ese escenario la semilla mediana coincidía con la primera, así que el mutante y el código correcto
   escribían el mismo archivo; el script de mutantes (`mutar.py` del revisor, en el scratchpad de la sesión) lo reportó
   como "sobrevive". En la ronda 3, `test_36` usa un modelo falso con log-loss de val monótona en la semilla: con {7, 8, 9} la
   mediana es 8 y el mutante muere.
5. **Corrección:** `test_36_red_guardada_es_la_mediana_del_medio` (`tests/test_experimento_pd.py`).
6. **Lección aprendida:** una prueba de "elige el elemento k" necesita un caso donde k no sea el primero ni el último.

### H-20 — Prueba del rango de puntos que no distingue `percentiles_rango` de los extremos de `percentiles_tabla`
- **Estado:** candidato
- **Categoría:** Código
- **Parte:** C
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 26-oct-2026 · **Quién lo
  cazó:** el subagente `verificador-matematico`, con mutantes sobre C3; el orquestador lo reprodujo
- **Nota de numeración:** registrado primero como H-13 en `dev/juanjose`; renumerado a H-20 porque en `main` ya existen
  H-13…H-19 (Partes B y E).
1. **Prompt (literal):** `prompts/C3.md`, sección "1. implementador — fase roja" (entrada_numerica: "rango_puntos =
   |puntos(p_b) − puntos(p_a)| con percentiles_rango"; prueba 1: "entrada_numerica (tabla y rango)").
2. **Respuesta relevante de la IA:** `tests/test_puntos.py` (sin commit): el modelo de juguete usa `percentiles_tabla =
   [25, 50, 75]` y `percentiles_rango = [25, 75]`, justo los extremos de la tabla, y ninguna prueba con el modelo real fija
   `rango_puntos`; la prueba se presentaba como la que comprueba el rango.
3. **Cómo sospechamos:** el verificador aplicó el mutante "el rango usa los extremos de la tabla en vez de
   `percentiles_rango`".
4. **Evidencia:** `evidencias/H-20_rango_percentiles_mutante.txt`: con el mutante pasan las 74 pruebas (igual que el
   original). Con datos reales el mutante cambiaría el rango de `pub_rec` de 0 (p5–p95) a 9,889 (p1–p99), y el top 5 de
   variables que se imprime en la sustentación.
5. **Corrección:** ronda 3 de C3: el juguete usa un `percentiles_rango` distinto de los extremos de la tabla y se añaden
   pruebas para el umbral exacto de `fraccion_min_categoria` y para el error de aditividad rota.
6. **Lección aprendida:** en un caso de prueba, los parámetros que deben distinguirse no pueden coincidir por casualidad
   (aquí, rango = extremos de la tabla); verificar cada prueba con el mutante que debería matar.

### H-24 — "La eficiencia del explainer de permutaciones de shap es exacta con cualquier número de permutaciones"
- **Estado:** candidato
- **Categoría:** Código
- **Parte:** C
- **Herramienta y modelo:** Claude Code, sesión orquestadora de Juan José y subagente `implementador` (Claude Opus 5.5) ·
  **Fecha:** 26-oct-2026 · **Quién lo cazó:** el subagente `revisor-calidad` (bloqueante B1 de C4); el orquestador lo reprodujo
- **Nota de numeración:** registrado primero como H-21 en `dev/juanjose`; renumerado a H-24 porque en `main` ya
  existen H-21…H-23 (Partes B y E).
1. **Prompt (literal):** `prompts/C4.md`: el plan del orquestador ("la eficiencia es exacta sin importar cuántas
   permutaciones se usen") y la fase roja ("Eficiencia exacta por telescopio (verificado en el código de shap)"); el
   implementador lo repitió en la celda markdown de `src/scorecard/shap_pd.py`.
2. **Respuesta relevante de la IA:** `src/scorecard/shap_pd.py` (fase verde, sin commit), celda del explainer: "(suma
   telescópica), así que la eficiencia es **exacta con cualquier número de permutaciones**".
3. **Cómo sospechamos:** el revisor leyó el masker de shap: decide qué variable "no cambia" frente al fondo con
   `np.isclose` (rtol 1e-5), no con igualdad, y probó semillas distintas.
4. **Evidencia:** `evidencias/H-24_isclose_eficiencia.txt` y `.py`: (1) `run.py scorecard --config
   configs/scorecard/mlp.yaml --set shap.base_seed=34` aborta con `Eficiencia rota: 2.702e-09 > 1e-09`; (2) dos valores
   transformados de `revol_bal` que difieren en 7,7e-6 (float32 del preprocesador) son "iguales" para `np.isclose`: shap no
   reevalúa esa variable, φ = 0 y la eficiencia falla por 7,7 con una f sensible. La telescopia es exacta solo si el
   masker reevalúa todas las variables que difieren.
5. **Corrección:** ronda 2 de C4: cada jugador se codifica con un índice entero en la tabla de valores únicos (fondo y
   explicados) antes de pasarlo a shap, de modo que dos valores distintos nunca son "iguales" para `np.isclose`; prueba
   nueva con un valor que difiere en 1e-6 relativo.
6. **Lección aprendida:** "verificado en el código" exige leer también lo que el algoritmo **delega** (aquí, el masker y
   su `np.isclose`) y probar con varias semillas; una garantía matemática solo vale si la implementación evalúa lo que la
   prueba supone.

### H-25 — Pruebas de SHAP y de la importancia por permutación que no distinguen signo, desviación ni agrupación
- **Estado:** candidato
- **Categoría:** Código
- **Parte:** C
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 26-oct-2026 · **Quién lo
  cazó:** el subagente `verificador-matematico`, con mutantes sobre C4; el orquestador guardó la evidencia
- **Nota de numeración:** registrado primero como H-22 en `dev/juanjose`; renumerado a H-25 porque en `main` ya
  existen H-21…H-23 (Partes B y E).
1. **Prompt (literal):** `prompts/C4.md`, "1. implementador — fase roja" (punto 7, permutación: "caída de AUC y aumento de
   log-loss (media y desviación muestral)"; jugadores = 15 variables con las categóricas enteras).
2. **Respuesta relevante de la IA:** `tests/test_shap_pd.py` (ronda 2, sin commit), presentado como el conjunto que prueba
   la permutación (07a–07c), la agrupación de las categóricas y la fuga.
3. **Cómo sospechamos:** el verificador aplicó mutantes con sentido sobre una copia del módulo.
4. **Evidencia:** `evidencias/H-25_mutantes_sobreviven.txt` y `evidencias/H-25_mutantes/`: pasan las 62 pruebas con
   (a) SHAP con las 48 columnas como jugadores (el 64 % de las filas evaluadas no son one-hot válidas), (b) desviación
   poblacional (ddof = 0), (c) caída de AUC con el signo cambiado, (d) aumento de log-loss con el signo cambiado; y (e) el
   fondo sacado de test solo lo mata una prueba marcada `lento`, que la CI no corre.
5. **Corrección:** ronda 3 de C4: caso a mano de la permutación (media, s con ddof = 1, signos), comparación de
   `explicados.phi` con `valores_shap` recalculado por la ruta de jugadores agrupados (o registro de las entradas evaluadas)
   y una prueba no lenta de que el fondo sale de train.
6. **Lección aprendida:** una prueba de formato ("las claves y los tipos son correctos") no prueba una fórmula; cada
   decisión (signo, ddof, agrupación, procedencia de los datos) necesita un caso que falle si se invierte.
