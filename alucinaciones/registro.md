# 🎯 Registro de la Cacería de la alucinación

> Meta: ≥ 5 hallazgos **genuinos** en ≥ 3 categorías, al menos 1 por parte. **Nunca inventar.** Un candidato sin prompt
> literal o sin evidencia no cuenta. Evidencias en `alucinaciones/evidencias/`. Prompts literales en `alucinaciones/prompts/`.

## Cobertura actual
| Categoría | Candidatos | Confirmados |
|---|---|---|
| Matemáticas | 1 (H-05) | 0 |
| Código | 1 (H-01) | 0 |
| Datos del mundo real | 1 (H-02) | 0 |
| Bibliografía | 0 | 0 |
| Conceptos | 0 | 0 |

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
| V-09 | La EAD al castigo = `funded_amnt` − `total_rec_prncp`, y LGD = 1 − (`recoveries` − `collection_recovery_fee`)/EAD | Diccionario de Kaggle + casos reales (D1) | ⬜ |
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
- **Estado:** candidato (falta que Daniel lo confirme)
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
- **Estado:** candidato (falta que Daniel lo confirme)
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

### H-05 — "El estimador es consistente para la media condicional", dicho del MLP de LGD
- **Estado:** candidato
- **Categoría:** Matemáticas
- **Parte:** D
- **Herramienta y modelo:** Claude Code, subagente `implementador` (Claude Opus 5.5) · **Fecha:** 24-oct-2026 · **Quién lo
  cazó:** el subagente `verificador-matematico`; el orquestador guardó la evidencia
- **Nota de numeración:** H-03 y H-04 ya están en `dev/andres` (Parte B), aún sin fusionar.
1. **Prompt (literal):** `prompts/D3.md`, ronda 1 (punto 2 de "Qué entregar": "por qué se minimiza en pᵢ = yᵢ también para
   yᵢ fraccional").
2. **Respuesta relevante de la IA:** docstring de `src/lgd/perdidas.py` (commit 922f2f1): "la condición de primer orden solo
   exige que la media condicional esté bien especificada; no exige que y siga una Bernoulli. Por eso el estimador es
   consistente para la media condicional con objetivos fraccionales", en el módulo de la pérdida del **MLP**.
3. **Cómo sospechamos:** el argumento (E[μ − y | x] = 0 cuando μ = E[y | x]) prueba que E[y | x] es el **minimizador
   poblacional** de la pérdida esperada; la **consistencia** de Papke & Wooldridge (1996) es un resultado para un modelo
   paramétrico con la media bien especificada y condiciones de regularidad, que no se traslada sin más a un MLP no
   identificable entrenado con L2 y parada temprana.
4. **Evidencia:** `evidencias/H-05_consistencia_qmle.py` y `.txt`: (i) para y fija, el argmin de la BCE está en μ = y;
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
