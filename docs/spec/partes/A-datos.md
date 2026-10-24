# Parte A — Datos y caso de uso

| | |
|---|---|
| Responsables | **Daniel** |
| Ramas personales | `dev/daniel` |
| Carpetas propias | `src/datos/` (menos `sintetico.py`), `scripts/preparar_datos.py`, `configs/datos.yaml`, `configs/features.yaml`, `src/common/preprocesamiento.py`, `src/viz/figs_datos.py`, `data/`, `notebooks/01_datos.py`, `blog/secciones/01–03` |
| Lee del núcleo | §0, §3, §4.1, §5.1, §5.2, §5.7, §6 · **Anexo (enunciado):** Datos; Parte 1, "Variable objetivo" y tareas 1, 2 y 5; Entregables 2 y 3 |
| Es la ruta crítica | Sí: B, C y D pasan a datos reales con lo que entregues el **viernes 23 a las 2 p. m.** |

## 1. Lo que pide el enunciado
- Reportar **número de registros y de variables, periodo y fecha de descarga**; citar el significado de las variables desde el
  **diccionario de Kaggle**, no desde una IA.
- Variable objetivo con la **tabla exacta** de el Anexo del núcleo (Parte 1, "Variable objetivo"); los NA no entran al entrenamiento.
- P1.1 **Caso de uso**: quién usa el modelo, para qué decisión, en qué momento y con qué información disponible.
- P1.2 **Análisis descriptivo** con **hipótesis explícitas** sobre qué hace más riesgosa a una persona.
- P1.5 (parte): conjuntos de entrenamiento, validación y prueba, con **al menos una partición temporal por fecha de originación**.
- Repositorio: datos procesados versionados + el código que los genera desde el archivo original.

## 2. Consumo / Entrego
| Consumo | De | Cuándo |
|---|---|---|
| Esqueleto del repo, `src/common/`, versión mínima del preprocesador | E | jue 22, noche |

| Entrego | A | Cuándo |
|---|---|---|
| `configs/features.yaml` (variables del modelo y de la app) | B, C, D, E | **vie 23, 10 a. m.** (checkpoint del equipo) |
| `loans_pd.parquet`, `resumen_datos.json`, `Preprocesador` completo, partición | B, C, D | **vie 23, 2 p. m.** |
| Hipótesis H1…Hn con su evidencia del EDA | C (las cierra) | sáb 24 |
| Análisis descriptivo después del preprocesamiento (`post_preprocesamiento.json`): B y C lo usan para interpretar; D reutiliza las funciones en la LGD | B, C, D | sáb 24 |
| Secciones 01–03 del blog | E | dom 25, 8 p. m. |
| Después de los datos: `blog/references.bib` verificado (DOI contra Crossref) | E | dom 25 |
| Después de los datos: coordinar la grabación del video (lugar, hora, guion de cada quien) | E | sáb 24 |

## 3. Especificación

### 3.1 Fuente y descarga
- Kaggle `ranadeep/credit-risk-dataset`, descarga **manual** en el navegador (no se versiona la clave de la API).
  Archivos esperados: CSV de préstamos y diccionario de variables [VERIFICAR nombres exactos y tamaños al descargar].
- `data/fuentes.md`: archivo, URL, fecha de descarga, SHA-256, tamaño, quién descargó.
- Todas las cifras del conjunto se **miden** y quedan en `resumen_datos.json` (núcleo §5.1). Ninguna se escribe antes de medirla.

### 3.2 Caso de uso [PROPUESTO]
**Quién:** analista de originación de un banco (o su motor de decisión). **Decisión:** aprobar o rechazar una solicitud de
crédito de consumo y fijar condiciones. **Momento:** al recibir la solicitud, **antes** del desembolso. **Información:** lo que
declara el solicitante y su historial de buró en ese momento. Consecuencia: solo variables de la originación (núcleo §5.7).

### 3.3 Variable objetivo [DECIDIDO por el enunciado]
`src/datos/objetivo.py` con un diccionario literal de la tabla. **Un `loan_status` que no esté en la tabla es un error**, no un
NA silencioso. Verificar los textos exactos del CSV (espacios, mayúsculas, puntos) [VERIFICAR].

### 3.4 Selección de variables [DECIDIDO, Daniel, 23-oct: las 15 de `configs/features.yaml`]
- Excluir todo lo de núcleo §5.7 y revisar, **una por una contra el diccionario**, cualquier otra variable que pueda
  actualizarse después del desembolso.
- `grade`, `sub_grade`, `int_rate`, `installment`: las asigna Lending Club **después** de evaluar el riesgo; usarlas es usar el
  puntaje de otro. [DECIDIDO, Daniel, 23-oct] Fuera del modelo principal y de la app; **`grade` e `int_rate` se usan en un
  análisis de sensibilidad aparte** (tarea B10): A entrega `data/processed/sensibilidad_lc.parquet` y
  `configs/features_sensibilidad.yaml`. `sub_grade` (redundante con `grade`) e `installment` (depende de la tasa) no entran.
- Fuera: `id`, `member_id`, `url`, `desc`, `emp_title`, `title`. Sensibles (`zip_code`, `addr_state`): discutir equidad;
  [PROPUESTO] fuera de la app.
- Derivadas permitidas solo con información de la originación (p. ej. antigüedad crediticia en meses = `issue_d` −
  `earliest_cr_line`).
- Meta: **12–18 variables** que una persona pueda llenar en un formulario (`en_app: true`), con `etiqueta_es` y `ayuda_es`.

### 3.5 Partición temporal [decisión vie 23, mediodía]
- ~~Propuesta inicial: train ≤ 2013, val = 2014, test = 2015.~~ Descartada con los datos (23-oct): en 2015 el 93 % de los
  créditos sigue sin resolverse (`Current`, etc.); los 30 835 resueltos son una muestra muy sesgada para un test.
- **[DECIDIDO, Daniel, 23-oct] train = originados ≤ 2012, val = 2013, test = 2014; los resueltos de 2015
  quedan fuera** (y se cuentan en `resumen_datos.json`). Tamaños: 90 347 / 73 194 / 74 154; tasa de malos 16,3 % / 23,0 % /
  26,2 %. Los cortes están en `configs/datos.yaml` (`particion.*`) y se cambian sin tocar código.

  | Año | Créditos | Resueltos | Sin resolver | Tasa de malos (resueltos) |
  |---|---|---|---|---|
  | 2007–2010 | 20 814 | 20 805 | ~0 % | 13,7–26,2 % |
  | 2011 | 21 721 | 19 729 | 9 % | 16,5 % |
  | 2012 | 53 367 | 49 813 | 7 % | 16,7 % |
  | 2013 | 134 755 | 73 194 | 46 % | 23,0 % |
  | 2014 | 235 628 | 74 154 | 69 % | 26,2 % |
  | 2015 | 421 094 | 30 835 | 93 % | 25,5 % |
- **Trampa a cuantificar antes de decidir:** en los años recientes, los créditos que ya están resueltos tienden a ser los que
  incumplieron pronto (los demás siguen en `Current` → NA). Medir la tasa de malos y la fracción de NA por año
  (`resumen_datos.json` + figura) y decidir los cortes con esa figura; la discusión va al blog.
- Modo demo: `datos.muestra` con muestreo estratificado por `split` y `target`, semilla fija.

### 3.6 Preprocesador
Completar `src/common/preprocesamiento.py` según núcleo §5.2 (E deja una versión mínima con la misma firma). Ajuste **solo con
train**; `to_dict → from_dict` reproduce la salida bit a bit. `scripts/preparar_datos.py`: crudo → `loans_pd.parquet` +
`resumen_datos.json`, sin internet.

### 3.7 Análisis descriptivo inicial e hipótesis (datos con objetivo, antes de transformar)
`src/datos/eda.py` → `results/data/eda/` (tasa de malos por bins y categorías, faltantes, valores atípicos). Formular H1…Hn
**antes** de ver los modelos (p. ej. "a mayor `dti`, mayor riesgo", "plazo de 60 meses más riesgoso", "más consultas recientes →
más riesgo"), cada una con su figura. C las cierra con SHAP.
Criterio de "relación monótona": |Spearman| entre el orden del grupo y su tasa ≥ `eda.umbral_monotonia` (0,9 en
`configs/datos.yaml`); con datos reales la monotonía estricta se rompe por oscilaciones de milésimas entre deciles vecinos.
Una hipótesis numérica con apoyo pero relación no monótona se reporta como "la apoya, pero no es monótona".

### 3.8 Análisis descriptivo después del preprocesamiento
Se hace sobre la matriz que **realmente ven los modelos** (la salida del `Preprocesador` ajustado en train), por `split`.
Responde: ¿qué le hizo el preprocesamiento a los datos, y llegan sanos y comparables a los modelos?
1. **Efecto de cada transformación:** % imputado por variable, % recortado por los percentiles, % de categorías llevadas a
   `"OTRA"`/`"DESCONOCIDO"`, distribución antes vs. después (`log1p`, recorte).
2. **Verificación de la estandarización:** media ≈ 0 y desviación ≈ 1 en train; en val y test se miden. No tienen por qué
   ser exactas: la diferencia ya es una señal de cambio en el tiempo.
3. **Estabilidad temporal entre train, val y test:** tasa de malos por split y distribución de cada variable por split, con el
   **índice de estabilidad poblacional (PSI)** por variable [VERIFICAR fórmula y umbrales con una fuente citada]. Conecta con
   la trampa de la partición (§3.5).
4. **Correlaciones entre variables transformadas** (multicolinealidad): afectan la lectura de los coeficientes de la
   logística y de los puntos de la scorecard.
5. **Relación con el objetivo después de transformar:** tasa de malos por decil de cada variable transformada. Si la
   relación es claramente no lineal o no monótona, es un argumento a favor de que el MLP aporte algo frente a la logística
   (se retoma en la Discusión).

Salida: `results/data/eda/post_preprocesamiento.json` + figuras; subsección propia en el blog 03 y en el notebook 01.
La Parte D reutiliza las mismas funciones en el subconjunto de LGD.

## 4. Tareas
| ID | Tarea | Depende | Aceptación | + Verificador | Fecha | Quién |
|---|---|---|---|---|---|---|
| A1 | Descarga, `fuentes.md`, carga y conteos | — | Registros, variables y periodo medidos | curador-datos | jue 22 | Daniel |
| A2 | Variable objetivo | A1 | `test_objetivo.py`: mapeo literal, estado desconocido = error | revisor-spec | jue 22 | Daniel |
| A3 | Caso de uso + `features.yaml` con cita del diccionario por variable | A1 | `test_fuga.py` pasa; **checkpoint del equipo** | curador-datos + cazador-alucinaciones | vie 23, 10 a. m. | Daniel |
| A4 | Tasa de malos y fracción de NA por año + partición | A2 | `test_particion.py` (sin solapamiento, cortes de la config); figura | analista-experimentos | vie 23, mediodía | Daniel |
| A5 | Preprocesador completo + `preparar_datos.py` → parquet | A3, A4 | `test_preprocesamiento.py`; parquet versionado | revisor-spec | **vie 23, 2 p. m.** | Daniel |
| A6 | EDA inicial + hipótesis con figuras (§3.7) | A5 | Cada H-k con figura y tabla | analista-experimentos | sáb 24 | Daniel |
| A6b | Descriptivo después del preprocesamiento (§3.8) | A5 | `post_preprocesamiento.json` con los 5 puntos; `test_eda_post.py` (media ≈ 0 y desv ≈ 1 en train) | analista-experimentos | sáb 24 | Daniel |
| A7 | Blog 01–03 | A6 | Cifras desde `resumen_datos.json`; `mkdocs build --strict` | redactor-tecnico + cazador | dom 25 | Daniel |
| A8 | `references.bib` verificado (todas las partes) | — | Todo DOI resuelve y coincide con Crossref | cazador-alucinaciones | dom 25 | Daniel |
| A9 | Notebook `01_datos` (núcleo §5.8) | A6, A6b | Corre en < 2 min; cero lógica propia | revisor-spec | dom 25 | Daniel |

## 5. Pruebas de la parte
`test_objetivo.py`, `test_fuga.py`, `test_particion.py`, `test_preprocesamiento.py`, `test_preparar_datos.py` (esquema del
parquet = núcleo §5.1), `test_eda_post.py`.

## 6. Puntos calientes de alucinación (dónde verificar)
- Significado de variables (`dti`, `revol_util`, `inq_last_6mths`, `earliest_cr_line`, `policy_code`…): una IA los "explica"
  con seguridad; la verdad está en el diccionario.
- Qué significa "Does not meet the credit policy" y cuándo se conoce cada campo.
- Formato real de los campos (`term` con espacios, `emp_length` como texto, `revol_util` con o sin %).
- Periodo y número de registros "conocidos" del conjunto: se miden, no se recuerdan.
- Fórmula y umbrales del PSI (las IA suelen dar umbrales "estándar" sin fuente).

## 7. Blog y fundamentos
Secciones 01 (introducción y caso de uso), 02 (datos: fuente, registros, variables, periodo, fecha de descarga, objetivo,
partición y su justificación) y 03 (análisis descriptivo inicial e hipótesis, y análisis después del preprocesamiento),
con figuras numeradas y citadas.

## 8. Definición de terminado
Parquet y preprocesador en `main`; B, C y D cambiaron `datos.ruta` a `data/processed/` sin tocar código; las cifras del blog
salen de `resumen_datos.json`; ≥ 1 hallazgo o verificación documentada en la cacería.
