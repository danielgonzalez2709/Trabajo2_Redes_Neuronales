<!-- Responsable: Daniel + todos. Enunciado (Anexo del núcleo), Entregable 7: ≥ 5 hallazgos en ≥ 3 categorías, cada uno con
prompt literal, respuesta, sospecha, evidencia, corrección y lección. Fuente: alucinaciones/registro.md (no copiar a mano
cifras que no estén ahí). Cada parte agrega aquí sus hallazgos cuando estén confirmados. Guía: blog/README.md -->
## 12. Cacería de la alucinación

Registramos cada error de la IA en `alucinaciones/registro.md`, con el prompt literal (en `alucinaciones/prompts/`), su
evidencia (derivación, experimento o fuente; los scripts y salidas en `alucinaciones/evidencias/` cuando existen) y la
corrección. Un hallazgo queda **confirmado** cuando un integrante lo revisa (el autor de la parte o Daniel, que coordina la
cacería). Los veintinueve hallazgos registrados están confirmados: la [[tab:caceria]] los resume y la [[tab:caceria-cobertura]]
los cuenta por categoría.

::: tabla caceria
Título: Hallazgos de la cacería de la alucinación
Fuente: Elaboración propia a partir de alucinaciones/registro.md y alucinaciones/evidencias/.
| Id | Categoría | Parte | Estado | Qué dijo o hizo la IA | Qué mostró la evidencia |
|---|---|---|---|---|---|
| H-01 | Código | A | Confirmado | Calculó el índice de estabilidad poblacional con cortes por cuantiles | En variables discretas los cortes colapsan y el índice da 0 aunque la población cambie |
| H-02 | Datos del mundo real | A | Confirmado | Sobre `mths_since_last_delinq`: "un vacío significa «nunca ocurrió» y no un dato faltante" | El diccionario no lo dice y el 0,46 % de esos vacíos sí tiene moras recientes |
| H-03 | Código | B | Confirmado | Afirmó que `calibration_curve` manda al intervalo superior el dato que cae en un borde | El código de scikit-learn lo manda al inferior; la prueba no lo notaba porque sus datos no tenían empates |
| H-04 | Código | B | Confirmado | Validó las entradas de la AUC rechazando solo los NaN | Con puntajes ±inf la AUC da 0,444 en lugar de 5/9 = 0,556, sin error |
| H-05 | Código | B | Confirmado | Propuso una prueba de L2 "con lr y gradiente de la pérdida en 0" | Con lr = 0 los pesos no cambian para ningún λ: la prueba no puede fallar |
| H-06 | Código | B | Confirmado | Quitó las pruebas de los minilotes y de `evaluar_perdida` porque, según el plan, las pruebas 14 y 15 ya las cubrían | Tres mutantes de esos fallos pasan las 71 pruebas |
| H-07 | Código | B | Confirmado | Comentó un `@dataclass(frozen=True)`: "inmutable: quien lo reciba (B4, B5, D) no puede alterar el historial" | `r.historial.append(1)` funciona: la congelación es superficial |
| H-08 | Matemáticas | B | Confirmado | La especificación pedía L2 con el "término λ‖W‖² en la pérdida, o `weight_decay` de `Adam`/`SGD`, que lo suma al gradiente" | Lo equivalente a `weight_decay` es (λ/2)‖W‖²; con λ‖W‖² los pesos difieren |
| H-09 | Código | B | Confirmado | Presentó como la prueba que detecta un filtro global de avisos una que compara `warnings.filters` antes y después de `fit` | El mutante sobrevivía: el módulo de pruebas ya tenía ese filtro |
| H-10 | Matemáticas | D | Confirmado | "el estimador es consistente para la media condicional", dicho del MLP de la LGD | La consistencia es del QMLE paramétrico; al MLP solo le aplica el minimizador poblacional |
| H-11 | Código | C | Confirmado | "Las funciones usan factor y offset tal como vienen", en una prueba que llama a dos de las cuatro | Dos mutantes pasan las 199 pruebas |
| H-12 | Código | C | Confirmado | "texto u objetos no numéricos fallan aquí", sobre `np.asarray(x, dtype=np.float64)` | `"1.5"` y `True` se convierten sin error |
| H-13 | Datos del mundo real | E | Confirmado | La pérdida observada "es la misma construcción de la LGD de la Parte D" y, en los créditos en mora, "su pérdida observada es parcial" | 61,3 de los 231,2 M USD eran créditos en mora con recuperaciones 0 (LGD = 1 supuesta) y 1,67 M eran créditos pagados |
| H-14 | Matemáticas | E | Confirmado | "con montos desembolsados parecidos (136,15 y 91,31 millones). La diferencia la produce la PD" | Difieren en un 49 %; la EAD aporta un factor 1,49 y la PD uno de 6,64 |
| H-15 | Matemáticas | E | Confirmado | "Lo que hace alta la razón de la banda 10 es el factor EAD, el mayor de todos" | La banda 8 tiene casi el mismo factor EAD y razón 1,002; lo que separa a la 10 es su factor PD |
| H-16 | Conceptos | B | Confirmado | "Optuna TPE", justificado solo con @bergstra2011 | Optuna 5.0.0 usa por defecto el TPE multivariado en estudios de un objetivo |
| H-17 | Código | B | Confirmado | Propuso el caso de prueba `dropout = 0.475 → "max"` (borde máximo) | En coma flotante la distancia (0,025000000000000022) supera la tolerancia (0,025) |
| H-18 | Código | B | Confirmado | En el plan, la prueba 12 (reloj de paso fijo) atrapa el riesgo "Tiempo incluye carga/medición" | Con ese reloj, cronometrar solo el ajuste o el ajuste con trabajo extra da lo mismo |
| H-19 | Código | B | Confirmado | Dio por comprobado con las pruebas 17–18 que se guarda la red de la semilla mediana | En su escenario la mediana era la primera semilla y el mutante que guarda la primera sobrevivía |
| H-20 | Código | C | Confirmado | Presentó como la prueba del rango de puntos una con `percentiles_rango` igual a los extremos de la tabla | El mutante que usa los extremos de la tabla pasa las 74 pruebas |
| H-21 | Conceptos | E | Confirmado | La PD media de prueba queda baja porque "los dos aprendieron la tasa de las cosechas hasta 2012" | En entrenamiento la logística reproduce la tasa (0,1630) y el MLP queda por encima (0,1884) |
| H-22 | Conceptos | B | Confirmado | En la calibración de prueba, "los 10 bins de ambos modelos quedan por debajo de la diagonal" | La tasa observada supera a la PD media en los 20 bins: los puntos quedan por encima |
| H-23 | Datos del mundo real | B | Confirmado | "Dos ensayos posteriores, el 19 y el 27, quedaron a menos de 0,001" del mejor | Son tres: el 23 queda a 0,000987 y se perdió al redondear antes de comparar |
| H-24 | Código | C | Confirmado | La eficiencia de SHAP con el explainer de permutaciones "es exacta con cualquier número de permutaciones" | El masker compara con `np.isclose`: con otra semilla la suma se rompe (2,7e-9 > 1e-9) |
| H-25 | Código | C | Confirmado | Presentó las pruebas de SHAP y de la permutación como la cobertura de su fórmula | Pasan las 62 pruebas con 48 jugadores, ddof = 0 o los signos de AUC y log-loss invertidos |
| H-26 | Código | B | Confirmado | Dio por cubierto que las rutas que pisarían las del modelo principal "dan `ValueError`" | Era una lista negra incompleta: una prueba sobrescribió `results/data/pd/pred_test.parquet` |
| H-27 | Código | C | Confirmado | Presentó las pruebas de la regla del veredicto como un caso por cada rama | Pasan con "≥ 1 en contra" en vez de "≥ 2" y con las desigualdades de borde cambiadas |
| H-28 | Matemáticas | C | Confirmado | "Usar log10 en vez de ln cambia el factor: duplicar las odds ya no sumaría pdo" | Con log10 en las dos partes el puntaje es idéntico; el error es mezclar bases (8,69 puntos) |
| H-29 | Datos del mundo real | E | Confirmado | Propuso no desplegar PyTorch en funciones de Vercel "por el límite de ~250 MB" | Para Python el límite ya era de 500 MB; la decisión se sostiene porque torch para CPU ocupa 715,5 MB |
:::

### Cobertura

Los veintinueve hallazgos están confirmados. La [[tab:caceria-cobertura]] los reparte por categoría. Por parte, son dos de la
A (H-01, H-02), catorce de la B (H-03 a H-09, H-16 a H-19, H-22, H-23 y H-26), siete de la C (H-11, H-12, H-20, H-24, H-25,
H-27 y H-28), uno de la D (H-10) y cinco de la E (H-13 a H-15, H-21 y H-29). Se cumple la meta del enunciado (≥ 5 hallazgos en
≥ 3 categorías; aquí, 29 en cuatro de las cinco) y la nuestra de al menos uno por parte (núcleo §9). No encontramos errores de bibliografía: las referencias que
propuso la IA resultaron correctas (ver "Verificaciones que salieron correctas" al final).

::: tabla caceria-cobertura
Título: Hallazgos confirmados por categoría
Fuente: Elaboración propia a partir de alucinaciones/registro.md (sección "Cobertura actual").
| Categoría | Hallazgos | Total |
|---|---|---|
| Matemáticas | H-08, H-10, H-14, H-15, H-28 | 5 |
| Código | H-01, H-03, H-04, H-05, H-06, H-07, H-09, H-11, H-12, H-17, H-18, H-19, H-20, H-24, H-25, H-26, H-27 | 17 |
| Datos del mundo real | H-02, H-13, H-23, H-29 | 4 |
| Bibliografía | — | 0 |
| Conceptos | H-16, H-21, H-22 | 3 |
| **Total** | | **29** |
:::

Dos patrones se repiten. El primero: once de los diecisiete errores de código no están en el código sino en **las pruebas**
o en afirmaciones sobre ellas ("esta prueba cubre X"): H-03, H-05, H-06, H-09, H-11, H-17, H-18, H-19, H-20, H-25 y H-27.
Nueve de ellos se cazaron con mutantes, es decir, versiones del código con el error a propósito que la prueba debería
rechazar (H-03, H-06, H-09, H-11, H-18, H-19, H-20, H-25 y H-27).

El segundo: los errores vienen de todos los papeles, no solo de quien implementa. Según la herramienta que el registro anota
en cada hallazgo, once los cometió una sesión principal, la que dirige cada integrante o la que planeó el trabajo y escribió
la especificación (H-01, H-02, H-05, H-06, H-08, H-13, H-16, H-18, H-22, H-23 y H-29); doce, el subagente implementador (H-03,
H-04, H-07, H-09, H-10, H-11, H-12, H-20, H-25, H-26, H-27 y H-28); dos, el redactor técnico (H-14 y H-21); uno, el
cazador de alucinaciones, que verifica (H-15); y tres, una sesión principal junto con otro subagente (H-17, con el revisor
de calidad, y H-19 y H-24, con el implementador). Solo cuatro (H-01, H-09, H-15 y H-22) los cazó el mismo agente que
cometió el error; los demás los encontró otro agente (un subagente u otra sesión) o un integrante.

### H-01 — Un índice que da 0 en silencio (Código)

*Parte A · confirmado (Daniel, 26-oct) · lo cazó la misma sesión de Claude al revisar los resultados, antes del commit.*

- **Prompt:** `prompts/asesoria_puente_claude.md`, prompt I (ver la sección 11.2), con la spec A §3.8, punto 3 ("índice de
  estabilidad poblacional (PSI) por variable").
- **Respuesta de la IA:** una función que define los intervalos con los cuantiles de entrenamiento.
- **Cómo sospechamos:** el PSI de los registros públicos negativos era exactamente 0,0000 en validación y en prueba, mientras
  las demás variables daban entre 0,003 y 0,27. Un cero exacto en datos reales no es creíble.
- **Evidencia:** tras el recorte, la variable solo toma dos valores; los nueve cuantiles coinciden en un único corte y los
  dos valores caen en el mismo intervalo, así que el índice compara 100 % contra 100 %. Pero la proporción de solicitantes con
  al menos un registro pasa de 4,1 % en entrenamiento a 12,8 % en validación y 19,5 % en prueba
  (`evidencias/H-01_psi_discreto.py`).
- **Corrección:** en variables con pocos valores distintos, cada valor es su propio intervalo; el PSI pasa a 0,107 y 0,269.
  Una prueba con un caso calculado a mano impide que vuelva a pasar.
- **Lección:** probar las métricas también con variables discretas y desconfiar de los ceros exactos.

### H-02 — El significado de un vacío (Datos del mundo real)

*Parte A · confirmado (Daniel, 26-oct) · lo cazó el subagente revisor-spec en la primera ronda de la Parte A.*

- **Prompt:** `prompts/asesoria_puente_claude.md`, prompt I (ver la sección 11.2), con la spec A §3.4 y la regla del núcleo
  §0.5 ("el significado de cada variable se cita del diccionario de Kaggle, nunca de una IA").
- **Respuesta de la IA:** en el borrador de la sección 2, "variables con más de la mitad de valores vacíos, como
  `mths_since_last_delinq`, donde un vacío significa «nunca ocurrió» y no un dato faltante".
- **Cómo sospechamos:** el subagente revisor contrastó cada afirmación sobre variables con el diccionario y esta no estaba.
- **Evidencia:** el diccionario solo dice "The number of months since the borrower's last delinquency." y ninguna definición
  habla de vacíos. Además, entre los créditos con el campo vacío, el 0,46 % tiene moras en los últimos dos años
  (`evidencias/H-02_mths_since_last_delinq.py`): si el vacío significara "nunca hubo mora", esa proporción sería cero.
- **Corrección:** la sección 2.3 dice ahora que el diccionario no explica el vacío y que por eso no imputamos la variable.
- **Lección:** una explicación "de sentido común" sobre los datos es una afirmación que se cita o se comprueba.

### H-03 — El borde de la curva de calibración (Código)

*Parte B · confirmado (Andrés, 24-oct) · lo cazó el subagente verificador-matematico antes de implementar.*

- **Prompt:** `prompts/B1.md`, sección "1. implementador — fase roja".
- **Respuesta de la IA:** "el contraste con sklearn fija su convención: `searchsorted(bordes[1:-1], p, side="right")`", es
  decir, que el dato que cae en un borde va al intervalo superior.
- **Cómo sospechamos:** una implementación con `side="right"` pasaba las 52 pruebas, porque el contraste con scikit-learn
  solo usaba datos sin empates, donde ningún dato cae en un borde.
- **Evidencia:** en el código fuente de scikit-learn 1.9.1 (`sklearn/calibration.py:1332`) la búsqueda usa el lado por
  defecto, `"left"`: el dato del borde va al intervalo **inferior**. Con probabilidades redondeadas a dos decimales, 8 de
  los 9 bordes internos coinciden con datos y solo `side="left"` reproduce `calibration_curve`
  (`evidencias/H-03_calibracion_borde.py`).
- **Corrección:** la implementación usa `side="left"` y el contraste se corre también con datos con empates.
- **Lección:** "la referencia hace X" se comprueba en el código de la referencia, y una prueba de contraste solo fija una
  convención si sus datos ejercitan el caso.

### H-04 — Una AUC errónea con puntajes infinitos (Código)

*Parte B · confirmado (Andrés, 24-oct) · lo cazó el subagente revisor-calidad.*

- **Prompt:** `prompts/B1.md`, sección "3. implementador — fase verde".
- **Respuesta de la IA:** `src/common/metricas.py` declara que el puntaje es un número real, pero la validación de entradas
  solo rechaza NaN (`np.isnan`), no ±inf.
- **Cómo sospechamos:** el logit de una PD saturada en 0 o en 1 es ±inf, y la prueba de invariancia recortaba las
  probabilidades a [1e-6, 1 − 1e-6] justo para evitar ese caso.
- **Evidencia:** con y = [0, 1, 0, 1, 0, 1] y puntajes [−inf, −inf, 0,2, 0,5, inf, inf], la función devuelve 0,444; contando
  pares (Mann-Whitney, empates ½) la AUC es 5/9 = 0,556. Las diferencias entre infinitos iguales dan NaN, así que esos
  empates no se agrupan (`evidencias/H-04_auc_infinitos.py`).
- **Corrección:** la validación exige valores finitos (`np.isfinite`), con una prueba que usa `inf`.
- **Lección:** "sin NaN" no es "número real"; las pruebas de propiedades no deben recortar las entradas justo donde la
  función puede fallar.

### H-05 — Una prueba de L2 que no puede fallar (Código)

*Parte B · confirmado (Daniel, 26-oct) · lo cazó Andrés al revisar el plan de B2.*

- **Prompt:** `prompts/B2.md`, sección "Prompt de Andrés a la sesión principal".
- **Respuesta de la IA:** en el plan, "Paso a mano de L2: con lr y gradiente de la pérdida en 0 (pérdida constante), un paso
  de SGD sin momento deja `W ← W(1 − lr·l2)` y el sesgo intacto".
- **Cómo sospechamos:** Andrés preguntó si, con lr en 0, los pesos no se moverían de todos modos.
- **Evidencia:** la actualización de SGD con `weight_decay` es $\theta \leftarrow \theta - \eta(\nabla L(\theta) + \lambda\theta)$.
  Con $\eta = 0$ queda $\theta \leftarrow \theta$ para cualquier $\lambda$, así que la aserción no distingue un L2 correcto de
  uno ausente o aplicado a los sesgos. Lo que se quería era $\eta > 0$ con $\nabla L = 0$. La evidencia es esta derivación;
  no hay script en `evidencias/`.
- **Corrección:** la prueba se eliminó (además probaba PyTorch y no nuestro código); el L2 lo cubren la prueba de los grupos
  de parámetros (pesos con `weight_decay`, sesgos sin él) y la de que l2 > 0 da pesos de menor norma.
- **Lección:** toda prueba debe poder fallar: antes de aceptarla, preguntarse qué implementación incorrecta la rompería.

### H-06 — "Esa prueba ya lo cubre" (Código)

*Parte B · confirmado (Daniel, 26-oct) · lo cazó el subagente implementador con pruebas de mutantes.*

- **Prompt:** `prompts/B2.md`, "Corrección de Andrés al primer plan" (quitar pruebas redundantes) y "2. implementador — fase
  verde".
- **Respuesta de la IA:** el plan corregido quitó las pruebas de los minilotes y de `evaluar_perdida` en modo de evaluación
  porque "lo cubre 15" y "lo cubre 14".
- **Cómo sospechamos:** al repetir los mutantes sobre el código real, tres sobrevivieron.
- **Evidencia:** pasan las 71 pruebas (a) un recorrido que descarta el último lote incompleto (88 de 600 filas por época con
  lotes de 512), (b) minilotes sin barajar, que la prueba 15 no ve porque otra semilla cambia igual la inicialización, y
  (c) `evaluar_perdida` sin `eval()`, que la 14 no ve porque `predict_proba` llama a `eval()` por su cuenta: la pérdida de
  validación se habría medido con *dropout* activo. La evidencia es la corrida de mutantes del implementador en la fase
  verde de B2, descrita en el registro; no quedó un script en `evidencias/`.
- **Corrección:** Andrés repuso tres pruebas (lote incompleto, orden barajado y reproducible, y `evaluar_perdida` con
  *dropout* 0,5 da dos medidas iguales).
- **Lección:** "la prueba X cubre el fallo Y" se comprueba con un mutante de Y.

### H-07 — Un historial "inmutable" que se puede modificar (Código)

*Parte B · confirmado (Daniel, 26-oct) · lo cazó el subagente revisor-calidad en la tercera revisión de B2.*

- **Prompt:** `prompts/B2.md`, sección "8. implementador — ronda 4".
- **Respuesta de la IA:** comentarios del commit af1456a en `src/pd/entrenar.py` y `src/pd/mlp.py`: "inmutable: quien lo
  reciba (B4, B5, D) no puede alterar el historial".
- **Cómo sospechamos:** `frozen=True` solo prohíbe reasignar atributos, y el historial es una lista.
- **Evidencia:** `r.historial.append(1)` sobre un resultado de entrenamiento funcionó sin error; es el comportamiento
  documentado de `dataclasses` (la congelación es superficial). Se reproduce con `evidencias/H-07_frozen_superficial.py`:
  la lista se modifica con `append`, y solo reasignar el campo da `FrozenInstanceError`.
- **Corrección:** comentarios reescritos: "frozen: no se reasignan los campos; la lista del historial sí es mutable".
- **Lección:** un comentario también es una afirmación que se verifica.

### H-08 — El factor ½ del L2 (Matemáticas)

*Parte B · confirmado (Daniel, 25-oct) · lo cazó el subagente verificador-matematico en B2; el error venía de la
especificación que escribió la IA en la sesión del puente.*

- **Prompt:** "Dame el spec yo lo reviso" (`prompts/asesoria_puente_claude.md`), el pedido que produjo la especificación por
  partes; la verificación está en `prompts/B2.md`, sección "9. verificador-matematico".
- **Respuesta de la IA:** la especificación de la Parte B decía "el código debe hacer L2 (término λ‖W‖² en la pérdida, o
  `weight_decay` de `Adam`/`SGD`, que lo suma al gradiente)".
- **Cómo sospechamos:** al verificar el código de entrenamiento, que usa $(\lambda/2)\lVert W\rVert^2$.
- **Evidencia:** $\nabla(\lambda\lVert W\rVert^2) = 2\lambda W$, mientras que `weight_decay = λ` suma $\lambda W$ al
  gradiente: lo equivalente es $(\lambda/2)\lVert W\rVert^2$. En un experimento de 5 pasos, `weight_decay = λ` y
  $(\lambda/2)\lVert W\rVert^2$ dan los mismos pesos (diferencia máxima 2,8e-17 con SGD y 5,6e-17 con Adam), y con
  $\lambda\lVert W\rVert^2$ difieren (6,1e-3 y 5,5e-2). Esos scripts quedaron en el área temporal de la sesión, no en
  `evidencias/`; la derivación basta para reproducirlo.
- **Corrección:** especificación v0.13: $(\lambda/2)\lVert W\rVert^2 \equiv$ `weight_decay = λ`, sin L2 en los sesgos. El
  código ya seguía esta convención.
- **Lección:** una "equivalencia" con un hiperparámetro de la librería se verifica derivando el gradiente; el factor ½
  cambia la escala de λ.

### H-09 — Una prueba de efectos secundarios que parte del estado equivocado (Código)

*Parte B · confirmado (Daniel, 26-oct) · lo cazó el propio implementador con un mutante en la fase verde de B3.*

- **Prompt:** `prompts/B3.md`, sección "1. implementador — fase roja" (prueba 11: "warnings.filters es igual antes y
  después de fit").
- **Respuesta de la IA:** `test_11_fit_no_altera_los_filtros_globales` (commit 7d72a68), presentada como la prueba que atrapa
  un filtro global de avisos en lugar de uno local.
- **Cómo sospechamos:** el mutante que quita el `catch_warnings` local sobrevivía.
- **Evidencia:** el módulo de pruebas ya instalaba los filtros `error::ConvergenceWarning` y `error::FutureWarning`; el filtro
  global del mutante coincidía con uno presente y la lista de filtros no cambiaba. No hay script en `evidencias/`: la
  evidencia es la prueba reforzada del commit dc4cc9d, con la que el mutante muere.
- **Corrección:** la prueba parte de un entorno con `simplefilter("ignore")` y el mutante muere (commit dc4cc9d).
- **Lección:** una prueba de "no efectos secundarios" debe partir de un estado distinto del que el efecto produciría.

### H-10 — Consistencia atribuida a un MLP (Matemáticas)

*Parte D · confirmado (Daniel, 26-oct) · lo cazó el subagente verificador-matematico en D3.*

- **Prompt:** `prompts/D3.md`, ronda 1 ("por qué se minimiza en pᵢ = yᵢ también para yᵢ fraccional").
- **Respuesta de la IA:** el docstring de `src/lgd/perdidas.py`, en el módulo del MLP, concluía: "Por eso el estimador es
  consistente para la media condicional con objetivos fraccionales".
- **Cómo sospechamos:** el argumento ($E[\mu - y \mid x] = 0$ cuando $\mu = E[y \mid x]$) prueba que $E[y \mid x]$ minimiza
  la pérdida esperada; la consistencia de @papke1996 es un resultado para un modelo paramétrico con la media bien
  especificada, que no se traslada sin más a un MLP no identificable entrenado con L2 y parada temprana.
- **Evidencia:** `evidencias/H-10_consistencia_qmle.py`: para $y$ fija, el mínimo de la entropía cruzada está en $\mu = y$; y
  con $y$ con distribución Beta y $E[y \mid x] = \sigma(-0{,}5 + 1{,}2x)$, $n = 200\,000$, el estimador **paramétrico** da
  $(-0{,}4997;\ 1{,}2023)$: la consistencia vale para ese modelo. El enunciado del resultado se tomó de una fuente secundaria
  anotada en el registro, porque el PDF original disponible es un escaneo sin texto.
- **Corrección:** el docstring separa el minimizador poblacional (lo que aplica al MLP) de la consistencia del estimador
  paramétrico.
- **Lección:** un resultado asintótico tiene hipótesis; antes de escribir "es consistente", se contrastan con el modelo.

### H-11 — "Las funciones" eran dos de cuatro (Código)

*Parte C · confirmado (Daniel, 26-oct) · lo cazó el subagente verificador-matematico con mutantes sobre la fase roja de C1.*

- **Prompt:** `prompts/C1.md`, sección "1. implementador — fase roja" ("factor real finito > 0 y offset real finito").
- **Respuesta de la IA:** `test_03e_usa_factor_y_offset_recibidos`, con el docstring "Las funciones usan factor y offset tal
  como vienen", pero solo llamaba a dos de las cuatro transformaciones; y ningún caso inválido usaba un bool, aunque la
  validación de la config sí los rechaza.
- **Cómo sospechamos:** el docstring habla de todas las funciones y la prueba cubría dos.
- **Evidencia:** un mutante que recalcula factor y offset dentro de las dos funciones de PD y otro que acepta `True` como
  factor (`float(True) = 1.0`) pasan las 199 pruebas; con las pruebas propuestas mueren
  (`evidencias/H-11_mutantes_sobreviven.txt`).
- **Corrección:** antes de la fase verde se añadieron `test_03f` (σ(2) → 80 con factor 10 y offset 100) y `test_09o` (bool
  en factor u offset → error en las cuatro funciones).
- **Lección:** si la prueba dice "las funciones…", se parametriza por función.

### H-12 — Una conversión que no rechaza lo que decía rechazar (Código)

*Parte C · confirmado (Daniel, 26-oct) · lo cazó el subagente revisor-calidad en la revisión de C1.*

- **Prompt:** `prompts/C1.md`, sección "3. implementador — fase verde" (entrada convertida con
  `np.asarray(..., dtype=np.float64)`, rechazando bool y texto).
- **Respuesta de la IA:** en `src/scorecard/scorecard.py`, `try:  # texto u objetos no numéricos fallan aquí` sobre
  `np.asarray(x, dtype=np.float64)`.
- **Cómo sospechamos:** el mismo módulo rechaza bool y texto en la config porque `float(True) = 1.0` "se colaría"; el revisor
  probó esas entradas en las transformaciones.
- **Evidencia:** `score_desde_logit("1.5")` = 443,84, `score_desde_pd("0.3")` = 511,57 y `score_desde_logit(True)` = 458,27,
  sin error; solo falla `"abc"` (`evidencias/H-12_entrada_texto_bool.txt`).
- **Corrección:** se rechaza todo tipo que no sea entero o real (`dtype.kind not in "iuf"`) antes de convertir, con pruebas
  sobre las cuatro funciones.
- **Lección:** "esto falla con X" sobre una conversión de NumPy se comprueba ejecutándola con X.

### H-13 — Una pérdida "observada" que no lo era (Datos del mundo real)

*Parte E · confirmado (Daniel, 26-oct) · lo cazó el subagente verificador-matematico en la revisión de E5; la primera versión
la escribió la sesión principal de Daniel sin subagente.*

- **Prompt:** "Vamos haciendo lo de la parte E" (Daniel, 24-oct; `prompts/asesoria_puente_claude.md`, prompt L); la
  verificación está en `prompts/E5.md`, sección 3.
- **Respuesta de la IA:** el docstring de `src/el/perdida_esperada.py` decía que la pérdida observada
  `max(0, (funded_amnt − total_rec_prncp) − (recoveries − collection_recovery_fee))` "es la misma construcción de la LGD de
  la Parte D" y que en los créditos en mora "su pérdida observada es parcial", y la aplicaba a todos los estados de prueba.
- **Cómo sospechamos:** el verificador recalculó la pérdida por estado y la contrastó con la población de la LGD de la
  Parte D, que solo incluye créditos castigados.
- **Evidencia:** de los 231,2 M USD "observados", 168,2 M son pérdida realizada de castigados; 1 284 créditos pagados suman
  1,67 M porque su principal recibido no llega al desembolsado; y los 5 460 en mora suman 61,3 M (26,5 %) con recuperaciones
  0 en todos: no es una cota inferior sino una LGD = 1 supuesta (`evidencias/H-13_perdida_observada.py`). Con esa pérdida,
  la razón entre la E[L] y la pérdida observada daba 0,85 ("el modelo subestima"), según el registro; en los créditos
  resueltos es 1,06 (`results/data/el/summary.json`).
- **Corrección:** pérdida realizada solo en castigados, 0 en pagados (la diferencia se reporta como anomalía de los datos),
  pérdida provisional aparte para la mora y contraste solo con los créditos resueltos (sección 9).
- **Lección:** una pérdida observada se define por estado del crédito y con la misma población que la LGD.

### H-14 — Dos montos "parecidos" que difieren en un 49 % (Matemáticas)

*Parte E · confirmado (Daniel, 26-oct) · lo cazaron los subagentes cazador-alucinaciones y verificador-matematico en la
revisión del borrador de la sección 9.*

- **Prompt:** `prompts/E7.md`, sección 1 ("explica con cuidado […] SIN afirmar más de lo que muestran los datos").
- **Respuesta de la IA:** la E[L] de la banda 1 es casi diez veces la de la banda 10 "con montos desembolsados parecidos
  (136,15 y 91,31 millones). La diferencia la produce la PD (de 0,416 a 0,063)".
- **Cómo sospechamos:** la frase llama "parecidos" a dos montos que ella misma cita.
- **Evidencia:** la razón 9,92 entre las E[L] de las bandas 1 y 10 se reparte en PD × 6,64, EAD × 1,49 y LGD × 1,01; en
  escala logarítmica la PD explica el 82,5 % y la EAD el 17,4 %. Además, las bandas son deciles de un puntaje que es una
  transformación monótona de la misma PD (correlación de rangos −1), así que parte de la conclusión es por construcción
  (`evidencias/H-14_bandas_1_y_10.py`).
- **Corrección:** la sección 9 dice "sobre todo la PD", con el factor de la EAD, y aclara que las bandas ordenan por la PD
  de la regresión logística.
- **Lección:** un adjetivo cualitativo junto a cifras se comprueba con esas cifras; una descomposición multiplicativa dice
  cuánto aporta cada factor.

### H-15 — Una causa única donde había dos (Matemáticas)

*Parte E · confirmado (Daniel, 26-oct) · lo cazó el subagente cazador-alucinaciones sobre su propio informe anterior.*

- **Prompt:** `prompts/E7.md`, sección 2 (verificación de la sección 9); la frase pasó al encargo de la segunda ronda de
  redacción (sección 3, punto 5).
- **Respuesta de la IA:** "Lo que hace alta la razón de la banda 10 es el factor EAD, el mayor de todos (1,275)".
- **Cómo sospechamos:** en la segunda ronda, el cazador comparó los factores de las bandas 8 a 10 y los del orden
  alternativo de la descomposición.
- **Evidencia:** la banda 8 tiene casi el mismo factor EAD (1,2742 frente a 1,2747) y razón 1,002, y la 9, 1,2718 y razón
  1,107. Lo que separa a la banda 10 es su factor PD (0,981 frente a 0,818 y 0,903). Con el orden alternativo, el factor
  EAD máximo es el de la banda 8 (`evidencias/H-15_factor_ead_banda_10.py`).
- **Corrección:** la sección 9 dice que la razón de la banda 10 combina un factor PD cercano a 1 con el factor EAD más alto,
  y la compara con las bandas 2, 8 y 9.
- **Lección:** antes de explicar un extremo por un factor, se mira si otra banda con ese factor igual tiene el mismo
  resultado. También el verificador se equivoca.

### H-16 — El TPE que usa Optuna 5 (Conceptos)

*Parte B · confirmado (Daniel, 26-oct) · lo cazó la sesión principal de Andrés al planear B4, leyendo el código de Optuna
instalado.*

- **Prompt:** `prompts/asesoria_puente_claude.md` (prompt A); de la respuesta solo se guardó un resumen ("Optuna TPE"), así
  que el texto que se usó es el que pasó a la especificación.
- **Respuesta de la IA:** la especificación de la Parte B (hasta v0.14): "Optuna TPE, 40 ensayos … Justificación citada: TPE
  (Bergstra et al., 2011) y Optuna (Akiba et al., 2019)".
- **Cómo sospechamos:** `TPESampler` tiene `multivariate=None` en la firma, no `False`.
- **Evidencia:** en Optuna 5.0.0, la versión fijada, `_is_multivariate` devuelve `True` en estudios de un objetivo ("By
  default, the multivariate TPE is used for single-objective optimization"), mientras que el TPE de @bergstra2011 modela
  cada hiperparámetro por separado (`evidencias/H-16_tpe_multivariado.py`).
- **Corrección:** especificación v0.15: se mantienen los valores por defecto, `hpo_ensayos.json` registra
  `multivariate: true` y, para el TPE multivariado, la sección 4 cita la referencia a la que remite la documentación de
  `TPESampler`, @falkner2018, verificada en las actas de ICML 2018 (PMLR, volumen 80).
- **Lección:** los valores por defecto cambian entre versiones mayores; se lee el código de la versión fijada.

### H-17 — Un caso de borde falso en coma flotante (Código)

*Parte B · confirmado (Daniel, 26-oct) · lo cazó el subagente implementador al escribir la prueba (ronda 2 de B4).*

- **Prompt:** `prompts/B4.md`, sección 5, recomendado 1: "test_04: caso dropout = 0.475 → "max" y otro justo fuera".
- **Respuesta de la IA:** el revisor-calidad propuso el caso y la sesión principal lo pasó sin verificarlo.
- **Cómo sospechamos:** la prueba falló contra el código correcto.
- **Evidencia:** 0,5 − 0,475 = 0,025000000000000022 > 0,05 · 0,5 = 0,025, así que la regla `distancia ≤ tolerancia` da
  falso; con valores exactos en binario da verdadero (`evidencias/H-17_borde_coma_flotante.py`).
- **Corrección:** la prueba usa una fracción de 0,25 y valores representables exactamente (0,125, 0,375 y sus vecinos).
- **Lección:** una prueba de límite con `≤` necesita valores exactos en binario; la distancia se calcula antes en el
  intérprete.

### H-18 — Un reloj que cuenta lecturas, no trabajo (Código)

*Parte B · confirmado (Daniel, 26-oct) · lo cazó el subagente implementador con un mutante (ronda 1 de B5).*

- **Prompt:** `prompts/B5.md`, prompt de Andrés ("(d) riesgos y qué prueba los atrapa").
- **Respuesta de la IA:** en la tabla de riesgos del plan, "Tiempo incluye carga/medición | 12", con la prueba 12 = "reloj
  inyectado: `tiempo_s` = 1.0 en cada semilla".
- **Cómo sospechamos:** el implementador metió la medición dentro de la zona cronometrada y la prueba siguió en verde.
- **Evidencia:** con un reloj que avanza un paso por lectura, cronometrar solo el ajuste o el ajuste más trabajo extra da lo
  mismo (1,0 y 1,0) (`evidencias/H-18_reloj_paso_fijo.py`).
- **Corrección:** la prueba 12b usa un reloj espía que registra los eventos y comprueba que entre los dos tics de cada
  semilla hay exactamente un ajuste y ninguna predicción.
- **Lección:** antes de afirmar "esta prueba atrapa X", se escribe el mutante X y se ve fallar.

### H-19 — La semilla mediana que era la primera (Código)

*Parte B · confirmado (Daniel, 26-oct) · lo cazó el subagente revisor-calidad con mutantes (ronda 2 de B5).*

- **Prompt:** `prompts/B5.md`, prompt de Andrés (prueba 5: "es la red de summary.semilla_mediana") y sección 5.
- **Respuesta de la IA:** el plan (pruebas 17–18) y el implementador dieron por cubierta la regla "se guarda la red de la
  semilla mediana" con un escenario de 3 semillas.
- **Cómo sospechamos:** el mutante "guardar la red de la primera semilla" sobrevivió (el de "la última" sí moría).
- **Evidencia:** en ese escenario la semilla mediana coincidía con la primera, así que el mutante y el código correcto
  escribían el mismo archivo. El script de mutantes del revisor quedó en el área temporal de la sesión; la prueba que lo
  reemplaza es reproducible.
- **Corrección:** `test_36_red_guardada_es_la_mediana_del_medio` usa un modelo falso con log-loss de validación monótona en la
  semilla: con {7, 8, 9} la mediana es 8 y el mutante muere.
- **Lección:** una prueba de "elige el elemento k" necesita un caso donde k no sea el primero ni el último.

### H-20 — Dos parámetros que coincidían por casualidad (Código)

*Parte C · confirmado (Daniel, 26-oct) · lo cazó el subagente verificador-matematico con mutantes sobre C3.*

- **Prompt:** `prompts/C3.md`, sección "1. implementador — fase roja" ("rango_puntos = |puntos(p_b) − puntos(p_a)| con
  percentiles_rango").
- **Respuesta de la IA:** en `tests/test_puntos.py`, el modelo de juguete usa `percentiles_tabla = [25, 50, 75]` y
  `percentiles_rango = [25, 75]`, justo los extremos de la tabla, y se presentaba como la prueba del rango.
- **Cómo sospechamos:** el verificador aplicó el mutante "el rango usa los extremos de la tabla".
- **Evidencia:** con el mutante pasan las 74 pruebas, igual que con el original, y con datos reales el rango de `pub_rec`
  pasaría de 0 (p5–p95) a 9,889 (p1–p99) (`evidencias/H-20_rango_percentiles_mutante.txt`). Según el registro, eso
  cambiaría también el top 5 de variables que se muestra en la sustentación.
- **Corrección:** el juguete usa un `percentiles_rango` distinto de los extremos de la tabla, con pruebas adicionales para el
  umbral exacto de `fraccion_min_categoria` y para el error de aditividad rota.
- **Lección:** en un caso de prueba, los parámetros que deben distinguirse no pueden coincidir.

### H-21 — Dos modelos que no aprendieron la misma tasa (Conceptos)

*Parte E · confirmado (Daniel, 26-oct) · lo cazaron los subagentes verificador-matematico y cazador-alucinaciones en la
revisión del borrador de la sección 10.*

- **Prompt:** `prompts/E7.md`, sección 4 (redacción de la sección 10); la verificación, en la sección 7.
- **Respuesta de la IA:** el borrador de `blog/secciones/10-discusion.md` explicaba que la PD media de prueba queda por debajo
  de la tasa observada porque "los dos aprendieron la tasa de las cosechas hasta 2012".
- **Cómo sospechamos:** la PD media del MLP en prueba (0,204) era distinta de la de la logística (0,187); si los dos hubieran
  aprendido la misma tasa de entrenamiento, deberían coincidir en nivel.
- **Evidencia:** con los modelos guardados, la PD media en entrenamiento es 0,1630 en la logística, igual a la tasa, como
  corresponde a la máxima verosimilitud con intercepto, y 0,1884 en el MLP, por encima de la tasa; en prueba, 0,1868 y
  0,2041 frente a una tasa de 0,2622. Además, la PD del MLP está más comprimida: en su decil de mayor riesgo predice 0,382
  frente a una tasa de 0,478 (la logística, 0,416 frente a 0,476) (`evidencias/H-21_pd_media_por_conjunto.py`).
- **Corrección:** la sección 10 da las PD medias por conjunto de cada modelo, sin afirmar la causa del nivel del MLP.
- **Lección:** "los dos modelos hacen X" se comprueba con cada modelo por separado; una red con parada temprana en otra
  cosecha no tiene por qué reproducir la tasa de entrenamiento como la logística.

### H-22 — La diagonal del lado equivocado (Conceptos)

*Parte B · confirmado (Daniel, 28-oct) · lo cazó la sesión principal que orquestaba B7, al mirar `pd-calibracion.png`.*

- **Prompt:** `prompts/B7.md`, "Prompt de Andrés a la sesión principal" (pide las figuras de B7 y sugerir la curva para el
  video).
- **Respuesta de la IA:** el plan aprobado decía que en prueba "los 10 bins de ambos modelos quedan **por debajo de la
  diagonal**", y el implementador lo repitió en su entrega.
- **Cómo sospechamos:** la figura pone la tasa observada en el eje y y la PD media en el eje x, y los puntos se veían arriba
  de y = x. Si el modelo subestima (PD < tasa), los puntos deben quedar por encima.
- **Evidencia:** en los 20 bins (10 por modelo) la tasa observada supera a la PD media, con y − x entre +0,012 y +0,109; la
  PD media es 0,187 y 0,204 frente a una tasa de 0,262 (`evidencias/H-22_calibracion_diagonal.txt`).
- **Corrección:** el blog 06 (§6.4) y la especificación dicen "por encima de la diagonal".
- **Lección:** una frase sobre una figura se verifica contra la figura y sus ejes, no contra la conclusión: "subestima" era
  correcto, la posición respecto de la diagonal no.

### H-23 — El ensayo que se perdió al redondear (Datos del mundo real)

*Parte B · confirmado (Daniel, 28-oct) · lo cazó el subagente revisor-spec de B7 al recalcular cada cifra del blog 06.*

- **Prompt:** `prompts/B7.md`, "Prompt de Andrés a la sesión principal" y la respuesta "Bloques + una frase por figura".
- **Respuesta de la IA:** el primer borrador del blog 06 decía "Dos ensayos posteriores, el 19 y el 27, quedaron a menos de
  0,001 de ese valor". La frase se decidió mirando valores redondeados a 4 decimales.
- **Cómo sospechamos:** el revisor recalculó las diferencias sin redondear desde `hpo_ensayos.json`.
- **Evidencia:** el ensayo 23 queda a 0,000987 del mejor, por debajo del umbral; redondeado se ve igual que el 39, que sí
  queda fuera (0,001013) (`evidencias/H-23_ensayos_cercanos.txt`).
- **Corrección:** "Tres ensayos posteriores (el 19, el 23 y el 27) quedaron a menos de 0,001 de ese valor".
- **Lección:** una comparación con un umbral se hace con los valores sin redondear; redondear antes cambia el conteo justo
  en el borde.

### H-24 — Una eficiencia "exacta" que dependía de `np.isclose` (Código)

*Parte C · confirmado (Daniel, 28-oct) · lo cazó el subagente revisor-calidad de C4; la sesión principal de Juan José lo
reprodujo.*

- **Prompt:** `prompts/C4.md`: el plan, que daba la eficiencia por exacta para cualquier número de permutaciones, y la fase roja
  ("verificado en el código de shap").
- **Respuesta de la IA:** en `src/scorecard/shap_pd.py`, la celda del explainer decía que, por la suma telescópica, "la
  eficiencia es **exacta con cualquier número de permutaciones**".
- **Cómo sospechamos:** el revisor leyó el masker de shap, que decide si una variable "no cambia" frente al fondo con
  `np.isclose` (tolerancia relativa 1e-5) y no con igualdad, y probó otras semillas.
- **Evidencia:** con `--set shap.base_seed=34`, `run.py scorecard --config configs/scorecard/mlp.yaml` se detiene con
  "Eficiencia rota: 2.702e-09 > 1e-09"; dos valores transformados de `revol_bal` que difieren en 7,7e-6 son "iguales" para
  `np.isclose`, así que shap no reevalúa esa variable (`evidencias/H-24_isclose_eficiencia.txt` y `.py`).
- **Corrección:** cada jugador se codifica con un índice entero en la tabla de valores únicos antes de pasarlo a shap, con
  una prueba nueva de un valor que difiere en 1e-6 relativo.
- **Lección:** "verificado en el código" exige leer también lo que el algoritmo delega (aquí, el masker) y probar con varias
  semillas; la garantía matemática solo vale si la implementación evalúa lo que la prueba supone.

### H-25 — Pruebas de SHAP que no distinguían signo, desviación ni agrupación (Código)

*Parte C · confirmado (Daniel, 28-oct) · lo cazó el subagente verificador-matematico con mutantes sobre C4.*

- **Prompt:** `prompts/C4.md`, "1. implementador — fase roja" (permutación: "caída de AUC y aumento de log-loss (media y
  desviación muestral)"; jugadores = 15 variables con las categóricas enteras).
- **Respuesta de la IA:** `tests/test_shap_pd.py`, presentado como el conjunto que prueba la permutación, la agrupación de
  las categóricas y la fuga.
- **Cómo sospechamos:** el verificador aplicó mutantes con sentido sobre una copia del módulo.
- **Evidencia:** pasaban las 62 pruebas con SHAP sobre las 48 columnas en vez de las 15 variables, con la desviación
  poblacional (ddof = 0) y con los signos de la caída de AUC o del aumento de log-loss cambiados; el fondo sacado de prueba
  solo lo detectaba una prueba lenta que el CI no corre (`evidencias/H-25_mutantes_sobreviven.txt` y
  `evidencias/H-25_mutantes/`).
- **Corrección:** un caso a mano de la permutación (media, desviación con ddof = 1 y signos), la comparación de los valores
  de SHAP con la ruta de jugadores agrupados y una prueba no lenta de que el fondo sale de entrenamiento.
- **Lección:** una prueba de formato no prueba una fórmula; cada decisión (signo, ddof, agrupación, origen de los datos)
  necesita un caso que falle si se invierte.

### H-26 — Una guarda que no guardaba (Código)

*Parte B · confirmado (Daniel, 28-oct) · lo cazaron los subagentes revisor-spec (ronda 1) y revisor-calidad (rondas 2 y 3)
de B10, intentando romper la guarda.*

- **Prompt:** `prompts/B10.md`, "1. implementador" ("NO toques `results/data/pd/`, `models/pd_*`,
  `configs/pd/mlp_final.yaml`…").
- **Respuesta de la IA:** el informe de la ronda 1 daba como caso cubierto que las rutas que pisarían las del modelo
  principal "dan `ValueError`"; la guarda era una lista negra que no revisaba dos de las rutas configurables, y las dos
  correcciones siguientes repitieron el patrón.
- **Cómo sospechamos:** los revisores llamaron a la validación con la configuración real y un `--set` por caso, y después
  desde `notebooks/` con `../`.
- **Evidencia:** en una ronda, una prueba que llamaba al CLI sobrescribió `results/data/pd/pred_test.parquet` (se restauró y
  su SHA-256 volvió a coincidir con el versionado) (`evidencias/H-26_guarda_rutas_sensibilidad.txt`).
- **Corrección:** una lista blanca evaluada contra la raíz fija del repositorio: dentro del repo solo se escribe en
  `results/data/pd_sensibilidad*` y `models/pd_sensibilidad*`, y las pruebas de la guarda solo validan.
- **Lección:** una protección de escritura se define por lo permitido, no por lo prohibido, y contra una raíz que no dependa
  del directorio desde donde se ejecuta.

### H-27 — Un caso por rama no bastaba (Código)

*Parte C · confirmado (Daniel, 28-oct) · lo cazó el subagente revisor-calidad de C6 con mutantes.*

- **Prompt:** `prompts/C6.md`, "1. implementador — fases roja y verde" (pruebas: "juguetes de cada rama").
- **Respuesta de la IA:** `tests/test_hipotesis.py` (34 pruebas), presentado como cobertura de cada rama de la regla del
  veredicto (confirmada, refutada, no concluyente).
- **Cómo sospechamos:** el revisor buscó, para cada desigualdad de `src/scorecard/hipotesis.py`, un caso que fallara al
  invertirla.
- **Evidencia:** sobrevivían los mutantes que refutan con una sola fuente en contra (`≥ 1` en vez de `≥ 2`) y los que mueven
  las desigualdades de borde (`< −u` en vez de `≤ −u`, `a ≥ 0` en vez de `a > 0`) (`evidencias/H-27_mutantes_regla_c6.txt`).
- **Corrección:** una prueba con una fuente en contra y dos nulas (no concluyente) y los casos de borde ρ = −0,5 y a = 0; los
  tres mutantes ya fallan.
- **Lección:** cada umbral necesita su caso en el borde y cada conteo ("≥ 2") el caso justo por debajo.

### H-28 — El error no era la base del logaritmo, sino mezclarlas (Matemáticas)

*Parte C · confirmado (Daniel, 28-oct) · lo cazó el subagente verificador-matematico en C7.*

- **Prompt:** `prompts/C1.md`, sección 3 "implementador — fase verde", en la nota de puntos calientes ("log10 cambia el
  factor").
- **Respuesta de la IA:** un comentario de `src/scorecard/scorecard.py` (commit dad6f00) decía que "usar $\log_{10}$ en vez
  de $\ln$ cambia el factor: duplicar las odds ya no sumaría pdo".
- **Cómo sospechamos:** el verificador recalculó la fórmula con las dos bases en lugar de aceptar la nota.
- **Evidencia:** con ln, duplicar las odds suma 20,0 puntos; con log10 en las dos partes (factor = PDO / log10 2 = 66,44),
  también suma 20,0 y el puntaje es el mismo (527,12 con p = 0,2). Solo al mezclar bases (factor = PDO / ln 2 con log10 en
  el puntaje) duplicar suma 8,6859 (`evidencias/C7_log10.py`, sin datos).
- **Corrección:** el comentario de `scorecard.py` y la sección 7 dicen que el error es mezclar bases.
- **Lección:** una nota de "punto caliente" también se verifica numéricamente; aquí el riesgo real era la inconsistencia, no
  la base elegida.

### H-29 — Un límite de servicio que ya había cambiado (Datos del mundo real)

*Parte E · confirmado (Daniel, 28-oct) · lo cazó el subagente cazador-alucinaciones al verificar V-01.*

- **Prompt:** `prompts/asesoria_puente_claude.md`, prompt A ("Propongo para el despliegue de la aplicacion vercel,
  necesito que me confirmes eso…"). De la respuesta solo se guardó un resumen, como en H-16.
- **Respuesta de la IA:** no desplegar PyTorch en funciones de Vercel "por el límite de ~250 MB".
- **Cómo sospechamos:** era una de las afirmaciones de la planeación pendientes de verificar (V-01); se contrastó con la
  página de límites de Vercel.
- **Evidencia:** la documentación de Vercel (consultada el 28 de octubre) da "250 MB, or 500 MB for Python", y una función con
  PyTorch es de Python. La decisión sí se sostiene: la rueda de torch para CPU en Linux ocupa 715,5 MB sin comprimir y no
  cabe ni en 500 MB (`alucinaciones/registro.md`, H-29).
- **Corrección:** el límite vigente para Python es de 500 MB; la inferencia en el navegador no cambia.
- **Lección:** las cifras de límites y planes de servicios en la nube cambian; se citan con la página oficial y su fecha, no de
  memoria.

### Verificaciones que salieron correctas

No todo lo que dijo la IA estaba mal. El enunciado contempla documentar las verificaciones hechas y sus resultados, y aquí
lo hacemos aunque los hallazgos superen la meta. De las once afirmaciones que hizo la IA al planear el trabajo, nueve resultaron
correctas (las cuatro últimas, sobre planes de despliegue y el optimizador, se consultaron el 28 de octubre):

- **Referencias.** Las 14 que propuso existen y sus DOI o direcciones oficiales llevan al trabajo esperado
  (`evidencias/A8_verificacion_bib.*`). Los matices de año (la versión de arXiv frente a la del congreso en Kingma y Ba y en
  Loshchilov y Hutter, y la 2.ª edición de Siddiqi, que Crossref registra como edición 1) son de los registros y no de la IA,
  y quedaron anotados en `blog/references.bib`.
- **Sesgo de los créditos recientes.** Su advertencia de que los créditos recientes ya resueltos sobrerrepresentan los
  incumplimientos tempranos es consistente con los datos (sección 2.4).
- **Cobertura temporal.** El conjunto cubre créditos originados de junio de 2007 a diciembre de 2015, como suponía la
  partición propuesta.
- **Definición de la LGD.** La EAD al castigo como `funded_amnt − total_rec_prncp` y la LGD como
  `1 − (recoveries − collection_recovery_fee) / EAD`: los cinco campos se citan literalmente del diccionario y los datos son
  coherentes (ninguna EAD ≤ 0, ninguna LGD > 1 y 32 LGD < 0, por intereses fuera de la EAD). Queda un supuesto que no se
  puede verificar sin fechas: que lo cobrado tras el castigo no entra en `total_rec_prncp` (`evidencias/D1_verificacion_lgd.*`).
- **Licencia de Remotion.** La licencia oficial es gratis para individuos, organizaciones con fines de lucro de hasta tres
  empleados y organizaciones sin ánimo de lucro; un video de curso hecho por estudiantes queda cubierto.
- **Plan Hobby de Vercel.** Es gratis y se limita a uso personal y no comercial: "The Hobby plan is free and aimed at developers
  with personal projects" y "Hobby teams are restricted to non-commercial personal use only" (documentación de Vercel).
- **Neon en el Marketplace de Vercel.** La ficha de Neon en el Marketplace ofrece un plan gratis ("Free plan with generous
  quotas"); según Neon, el plan Free cuesta 0 USD al mes e incluye 1 GB de almacenamiento por proyecto.
- **Render.** Su plan gratis duerme el servicio tras inactividad: "Render spins down a Free web service that goes 15 minutes
  without receiving any inbound traffic", y despertarlo toma cerca de un minuto (documentación de Render).
- **`AdamW`.** Aplica el *weight decay* desacoplado, que no equivale a L2 con Adam: Loshchilov y Hutter muestran que "this is not
  the case for adaptive gradient algorithms, such as Adam", y la documentación de PyTorch dice que en `AdamW` el decaimiento "does
  not accumulate in the momentum nor variance".

Las otras dos afirmaciones resultaron correctas solo en parte. La del límite de tamaño de las funciones de Vercel es el
hallazgo H-29: la cifra ya estaba desactualizada para Python, aunque la decisión se sostiene. La del mantenimiento de Material for
MkDocs daba como fin el 5 de noviembre de 2026, que era la fecha oficial al planear; el proyecto la movió al 5 de mayo de 2027
el 24 de octubre. Esa no es una alucinación sino un dato que cambió después, y no tiene prompt de este trabajo, así que se
descartó (`alucinaciones/registro.md`, H-30).
