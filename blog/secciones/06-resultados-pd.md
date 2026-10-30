<!-- Responsable: Andrés (B). Enunciado (Anexo del núcleo): resultados de PD y comparación justa (30 semillas, presupuesto, parámetros). Guía de sintaxis: blog/README.md -->
<!-- B7 (26-oct): bloques de figura con una frase que cita cada una; cifras de results/data/pd/{summary,hpo_ensayos,
busqueda_logistica}.json, results/data/pd/semillas/ y results/data/pd/pred_test.parquet (python run.py pd --config
configs/pd/base.yaml) y data/processed/resumen_datos.json (tasas por split). Figuras: python run.py figures (src/viz/figs_pd.py). El texto completo de la sección es B8. -->
## 6. Resultados de la PD

Esta sección compara la regresión logística (el modelo de referencia) con el MLP, con los mismos datos, la misma partición y
el mismo preprocesador. Primero se muestran la búsqueda de hiperparámetros y el entrenamiento; después, la discriminación, la
calibración y la variabilidad entre 30 semillas, con la tabla de la comparación justa ([[tab:pd-comparacion]]). Siguen la
sensibilidad a las variables de Lending Club y las limitaciones, y cierra la sección 6.8, que convierte la PD en
una decisión de aprobar o rechazar.

### 6.1 Búsqueda de hiperparámetros

La logística eligió su penalización con una rejilla de 13 valores de C ([[fig:pd-busqueda-logistica]]). La log-loss de
validación baja hasta 0,5178 en C = 0,0031623 y después sube un poco y se aplana; el valor elegido no está en el borde de la
rejilla.

::: figura pd-busqueda-logistica
Título: Log-loss y AUC de validación de la regresión logística según la penalización C
Fuente: Elaboración propia a partir de results/data/pd/busqueda_logistica.json.
![Dos paneles con eje logarítmico de C: la log-loss de validación tiene su mínimo en C = 0,0031623 y la AUC de validación crece hasta aplanarse](figures/pd-busqueda-logistica.png)
:::

Para el MLP no se usó una rejilla: el espacio tiene 8 hiperparámetros, tres de ellos continuos (tasa de aprendizaje, L2 y
dropout), y solo su parte discreta ya suma 18 720 combinaciones (sección 4.8); discretizar además los tres continuos en
3 valores cada uno las multiplicaría por 27, hasta 505 440. El TPE, en cambio, usa los ensayos ya hechos para
proponer los siguientes en las zonas que dieron menor pérdida (la teoría está en la sección 4.8).

La búsqueda TPE del MLP probó 40 configuraciones; los 10 primeros ensayos son aleatorios y los demás los propone el TPE
([[fig:pd-busqueda-mlp]]). La mejor fue el ensayo 11, con una log-loss de validación de 0,5075, frente a 0,5178 de la logística.
Tres ensayos posteriores (el 19, el 23 y el 27) quedaron a menos de 0,001 de ese valor.

::: figura pd-busqueda-mlp
Título: Log-loss de validación de los 40 ensayos de la búsqueda TPE del MLP
Fuente: Elaboración propia a partir de results/data/pd/hpo_ensayos.json y results/data/pd/summary.json.
![Puntos por ensayo entre 0,507 y 0,542, con el mejor valor acumulado en escalón, el ensayo 11 marcado y la línea de la logística en 0,5178](figures/pd-busqueda-mlp.png)
:::

### 6.2 Entrenamiento del MLP final

En la semilla mediana (semilla 20) la pérdida de validación es mínima en la primera época (0,5099) y no mejora en las 10
épocas de paciencia, así que se restauran los pesos de la época 1 ([[fig:pd-aprendizaje]]). La diferencia de nivel entre las
dos curvas no es solo sobreajuste: la tasa de malos es 16,3 % en entrenamiento y 22,9 % en validación, y la entropía de esas
tasas (la BCE de predecir siempre la tasa) es 0,445 y 0,539. Entre las 30 semillas, la mejor época va de 1 a 20.

::: figura pd-aprendizaje
Título: Pérdida BCE de entrenamiento y de validación por época del MLP (semilla mediana)
Fuente: Elaboración propia a partir de results/data/pd/semillas/mlp_seed20.json.
![Curva de entrenamiento que baja de 0,436 a 0,408 y curva de validación que oscila entre 0,510 y 0,519, con la mejor época en la 1](figures/pd-aprendizaje.png)
:::

### 6.3 Discriminación

En prueba, las curvas ROC de los dos modelos casi se superponen ([[fig:pd-roc]]): la AUC es 0,6696 en la logística y 0,6716
en el MLP de la semilla mediana (Gini 0,339 y 0,343). El KS es 0,247 y 0,249 ([[fig:pd-ks]]); la máxima separación entre malos
y buenos ocurre en una PD de 0,153 en la logística y de 0,196 en el MLP.

::: figura pd-roc
Título: Curvas ROC de la regresión logística y del MLP en prueba
Fuente: Elaboración propia a partir de results/data/pd/pred_test.parquet (MLP: semilla mediana).
![Dos curvas ROC casi superpuestas por encima de la diagonal de azar, con AUC de 0,670 y 0,672](figures/pd-roc.png)
:::

::: figura pd-ks
Título: Distribuciones acumuladas de la PD en malos y en buenos, y estadístico KS, en prueba
Fuente: Elaboración propia a partir de results/data/pd/pred_test.parquet (MLP: semilla mediana).
![Dos paneles con las acumuladas de malos y buenos y el segmento de máxima separación: KS de 0,247 en la logística y de 0,249 en el MLP](figures/pd-ks.png)
:::

### 6.4 Calibración

Los dos modelos ordenan bien el riesgo, pero lo subestiman en prueba: en los 10 bins la tasa observada de malos queda por
encima de la PD media predicha, es decir, todos los puntos están sobre la diagonal ([[fig:pd-calibracion]]). La PD media es
0,187 en la logística y 0,204 en el MLP, frente a una tasa observada de 0,262; la tasa de malos sube de 16,3 % en
entrenamiento a 26,2 % en prueba.

::: figura pd-calibracion
Título: Curva de calibración de la regresión logística y del MLP en prueba (10 bins por cuantiles)
Fuente: Elaboración propia a partir de results/data/pd/pred_test.parquet (MLP: semilla mediana).
![Puntos de la PD media frente a la tasa observada por bin, todos por encima de la diagonal y = x, para los dos modelos](figures/pd-calibracion.png)
:::

### 6.5 Variabilidad entre semillas

Con 30 semillas, el MLP supera a la logística en Brier de prueba en las 30, en AUC en 24 y en KS en 23
([[fig:pd-semillas]]). La AUC de prueba del MLP va de 0,6630 a 0,6766, y la logística (una sola corrida, desviación 0)
queda en 0,6696.

::: figura pd-semillas
Título: AUC, KS y Brier de prueba de las 30 semillas del MLP frente a la regresión logística
Fuente: Elaboración propia a partir de results/data/pd/semillas/ y results/data/pd/summary.json.
![Tres paneles con cajas y puntos de las 30 semillas, la semilla mediana marcada y la logística como línea horizontal](figures/pd-semillas.png)
:::

La [[tab:pd-comparacion]] reúne la comparación justa: lo que cuesta cada modelo (parámetros, configuraciones probadas y
tiempo) y sus métricas de prueba con la variabilidad entre semillas. La logística tiene una sola corrida porque su
optimizador, `lbfgs`, es determinista: no usa azar, así que con otra semilla da exactamente el mismo modelo y su desviación
es 0. El MLP gana en las cinco métricas en promedio, pero con un margen pequeño en discriminación (AUC +0,0020, menos que
una desviación entre semillas, 0,0031) y con un costo mucho mayor: unas 55 veces más parámetros y unas 15 veces más tiempo.

::: tabla pd-comparacion
Título: Comparación justa de la regresión logística y el MLP: costo y métricas de prueba
Fuente: Elaboración propia a partir de results/data/pd/summary.json. MLP: 30 semillas (media ± desviación muestral, mejor y peor); logística: una corrida (n = 1), con desviación 0 y mejor = peor = media. En Brier y log-loss, "mejor" es el valor más bajo. Tiempo en CPU: en la logística, el reloj del ajuste completo de la rejilla de 13 valores de C; en el MLP, la suma de los ajustes de los 40 ensayos de la búsqueda más la de las 30 semillas.
| Concepto | Logística | MLP: media ± desv. | MLP: mejor | MLP: peor |
|---|---:|---:|---:|---:|
| Parámetros | 49 | 2 673 | — | — |
| Configuraciones probadas | 13 | 40 | — | — |
| Tiempo total (s) | 60,8 | 940,4 (búsqueda 702,4 + semillas 238,0) | — | — |
| AUC | 0,6696 | 0,6716 ± 0,0031 | 0,6766 | 0,6630 |
| Gini | 0,3393 | 0,3433 ± 0,0062 | 0,3533 | 0,3261 |
| KS | 0,2465 | 0,2500 ± 0,0054 | 0,2581 | 0,2340 |
| Brier | 0,1865 | 0,1832 ± 0,0011 | 0,1808 | 0,1857 |
| Log-loss | 0,5600 | 0,5490 ± 0,0033 | 0,5417 | 0,5561 |
:::

### 6.6 Sensibilidad a la evaluación de Lending Club

<!-- B10 (27-oct): cifras de results/data/pd_sensibilidad/comparacion.json (python run.py pd-sensibilidad --config
configs/pd/sensibilidad.yaml); MLP = media ± desv. de 30 semillas; PD media = semilla mediana (calibracion_global).
Composición por partición: data/processed/sensibilidad_lc.parquet unido con data/processed/loans_pd.parquet. -->

Para medir cuánto ganaría la PD si viera la evaluación de riesgo del propio Lending Club, repetimos los dos modelos con las
15 variables de originación más `grade` ("LC assigned loan grade") e `int_rate` ("Interest Rate on the loan"), con la misma
partición y un preprocesador propio ajustado solo con entrenamiento. Hay dos modos: la **misma arquitectura que el modelo
principal** y una **arquitectura buscada con las 17 variables** (TPE, 40 ensayos); la logística usa en ambos la misma
rejilla de C y da el mismo resultado. Estas dos variables quedaron fuera del modelo principal porque son la evaluación de
riesgo de Lending Club (A §3.4), así que este análisis no va a la app ni lo reemplaza ([[tab:pd-sensibilidad]]).

::: tabla pd-sensibilidad
Título: Métricas de prueba de la PD con y sin `grade` e `int_rate`
Fuente: Elaboración propia a partir de results/data/pd_sensibilidad/comparacion.json y results/data/pd/summary.json. MLP: media ± desviación de 30 semillas; logística: una corrida. ΔAUC frente al modelo principal. PD media de prueba: semilla mediana en el MLP; la tasa observada de prueba es 0,2622.
| Modelo | Variables | Parámetros | AUC | KS | Brier | ΔAUC | PD media |
|---|---|---:|---:|---:|---:|---:|---:|
| Logística | 15 (principal) | 49 | 0,6696 | 0,2465 | 0,1865 | — | 0,1868 |
| Logística | 17 (misma arquitectura y buscada) | 58 | 0,6863 | 0,2722 | 0,1821 | +0,0167 | 0,1988 |
| MLP | 15 (principal) | 2 673 | 0,6716 ± 0,0031 | 0,2500 ± 0,0054 | 0,1832 ± 0,0011 | — | 0,2041 |
| MLP | 17, misma arquitectura | 2 961 | 0,6876 ± 0,0030 | 0,2737 ± 0,0046 | 0,1797 ± 0,0009 | +0,0159 | 0,2128 |
| MLP | 17, arquitectura buscada | 16 193 | 0,6886 ± 0,0032 | 0,2753 ± 0,0059 | 0,1795 ± 0,0008 | +0,0170 | 0,2179 |
:::

Con la nota de Lending Club, la AUC de prueba sube unas 0,016–0,017 en los dos modelos, unas 5 veces la desviación entre
semillas del MLP (0,0030–0,0032); en la logística la Δ viene de una sola corrida, sin variabilidad medida. Ver esa
evaluación mejora la discriminación, pero la AUC sigue cerca de 0,69. Buscar una arquitectura nueva con las 17 variables no
se distingue del ruido: gana 0,0010 de AUC frente a la misma arquitectura, con 16 193 parámetros en vez de 2 961.

La PD media de prueba sube (logística 0,1868 → 0,1988; MLP 0,2041 → 0,2128 con la misma arquitectura y 0,2179 con la
buscada), pero sigue por debajo de la tasa observada de 0,2622 (§6.4). Es consistente con la deriva de composición entre
entrenamiento y prueba: la participación del grado A baja de 23,2 % en entrenamiento a 12,2 % en validación y 12,9 % en
prueba, y la `int_rate` media sube de 12,77 % a 14,69 % y 14,33 %. No basta para afirmar una causa; también puede influir
el sesgo de selección de los créditos recientes (sección 2.4).

### 6.7 Limitaciones

- **Parada muy temprana.** En la semilla mediana la mejor época es la 1 (§6.2): la pérdida de validación no vuelve a bajar
  después de la primera pasada por los datos. No es un caso aislado, pero tampoco la regla: la mejor época es 1 en 2 de las
  30 semillas y va de 1 a 20 entre todas. Una explicación posible es que la validación tiene más malos que el entrenamiento
  (22,9 % frente a 16,3 %), así que lo que el MLP sigue aprendiendo de train no mejora la pérdida en val; no lo probamos.
- **Subestimación en prueba.** Los dos modelos predicen menos malos de los que hay en prueba (§6.4): la PD media es 0,187 y
  0,204 frente a una tasa de 0,262. Ordenan bien el riesgo, pero si se usa la PD como probabilidad (por ejemplo, en la
  pérdida esperada) hace falta recalibrarla o vigilar la deriva.
- **Búsqueda casi plana.** Tres ensayos (el 19, el 23 y el 27) quedaron a menos de 0,001 de log-loss de validación del mejor
  (§6.1). Esa diferencia es menor que la desviación entre semillas de la configuración final (0,0018 en la log-loss de
  validación), así que el ensayo 11 no se distingue con claridad de esos tres; se eligió por la regla fijada de antemano (la
  menor log-loss de validación), no porque sea claramente mejor.

### 6.8 De la PD a la decisión: elección del umbral

<!-- B11 (30-oct): cifras de results/data/pd_umbral/umbral.json (python scripts/umbral_pd.py --config
configs/pd/umbral.yaml; 30 semillas, 867 s; la semilla mediana reentrenada coincide bit a bit con models/pd_mlp.pt).
MLP = semilla mediana (20) salvo donde se indica media ± desv. de las 30 semillas. Definiciones en la sección 5.8. -->

Las secciones anteriores evalúan la PD como probabilidad. Un banco, en cambio, tiene que decidir: aprobar o rechazar. Esta
sección ilustra cómo cambia esa decisión según el corte y según cuánto más cuesta prestarle a un malo (FN) que rechazar a
un bueno (FP), con las definiciones de la sección 5.8. Es un complemento: no cambia ninguna cifra anterior ni fija una
política de corte, que depende del apetito de riesgo del banco. En todo lo que sigue, el MLP es la semilla mediana (la de
la app), salvo donde se indica la variación de las 30 semillas.

#### 6.8.1 Frente al modelo trivial

El punto de partida es un modelo que no mira al solicitante: le da a todos la misma PD, la tasa de malos de entrenamiento
(0,16303). Su Brier sigue la fórmula de la sección 5.8. Como referencia se agrega el **piso teórico** de una PD constante en
prueba: la tasa de prueba (0,26217), que solo se conoce mirando las etiquetas de prueba y por eso no es un modelo
utilizable ([[tab:pd-umbral-trivial]]).

::: tabla pd-umbral-trivial
Título: Puntaje de Brier del modelo trivial (PD constante) y de los dos modelos, en validación y prueba
Fuente: Elaboración propia a partir de results/data/pd_umbral/umbral.json (python scripts/umbral_pd.py --config configs/pd/umbral.yaml). MLP: semilla mediana. La fila de la tasa de prueba usa las etiquetas de prueba: es un piso de comparación, no un modelo. Tasas de malos: entrenamiento 0,16303, validación 0,22950, prueba 0,26217.
| Modelo | PD | Brier val | Brier test |
|---|---|---:|---:|
| PD constante = tasa de entrenamiento | 0,16303 | 0,18125 | 0,20327 |
| PD constante = tasa de prueba (piso teórico en prueba) | 0,26217 | 0,17790 | 0,19344 |
| Regresión logística | Por solicitante | 0,16908 | 0,18652 |
| MLP | Por solicitante | 0,16696 | 0,18398 |
:::

Los dos modelos bajan el Brier de prueba del trivial de 0,20327 a 0,18652 (logística) y 0,18398 (MLP), y quedan también por
debajo del piso teórico (0,19344), que ninguna PD constante puede mejorar en prueba. El trivial, además, no ordena: su AUC
es 0,5 y, como decisión, aprueba a todos o rechaza a todos según si su PD queda por debajo o por encima del umbral.

#### 6.8.2 Matriz de confusión con el umbral 0,5

El corte "intuitivo" es rechazar si la PD es al menos 0,5 ([[tab:pd-umbral-matriz]]).

::: tabla pd-umbral-matriz
Título: Matriz de confusión con el umbral 0,5 (se rechaza si PD ≥ 0,5), en validación y prueba
Fuente: Elaboración propia a partir de results/data/pd_umbral/umbral.json (python scripts/umbral_pd.py --config configs/pd/umbral.yaml). MLP: semilla mediana. VP = malos rechazados, FN = malos aprobados, FP = buenos rechazados, VN = buenos aprobados.
| Modelo | Conjunto | VP | FN | FP | VN | TPR | TNR | Tasa de rechazo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logística | Validación | 288 | 16 510 | 268 | 56 128 | 0,0171 | 0,9952 | 0,0076 |
| Logística | Prueba | 380 | 19 061 | 305 | 54 408 | 0,0195 | 0,9944 | 0,0092 |
| MLP | Validación | 0 | 16 798 | 0 | 56 396 | 0,0000 | 1,0000 | 0,0000 |
| MLP | Prueba | 0 | 19 441 | 0 | 54 713 | 0,0000 | 1,0000 | 0,0000 |
:::

Con 0,5 casi nadie es rechazado. La logística rechaza menos del 1 % y atrapa el 2 % de los malos de prueba (380 de 19 441).
El MLP no rechaza a nadie, porque ninguna de sus PD llega a 0,5: aprueba a los 19 441 malos de prueba. No es un defecto del
modelo sino del corte: con tasas de malos entre 0,16 y 0,26, casi ninguna PD llega a 0,5. El 0,5 solo sería el corte correcto si los dos errores costaran lo mismo ($L/G = 1$, sección 5.8).

#### 6.8.3 Sensibilidad al umbral

La [[fig:pd-umbral-sensibilidad]] recorre los umbrales de 0 a 1 en pasos de 0,01. Al bajar el umbral se rechaza a más
gente: la TPR sube (menos FN, menos malos aprobados) y la TNR baja (más FP, más buenos rechazados). Los dos modelos dibujan
curvas parecidas; las del MLP llegan a cero antes, porque su PD más alta está entre 0,40 y 0,41 (en validación y en
prueba, el último umbral de la rejilla con algún rechazo es 0,40). Por ejemplo, en prueba con el umbral 0,20 la logística atrapa el 51 % de
los malos (TPR 0,5147) y aprueba el 71 % de los buenos (TNR 0,7127), rechazando al 35 % de las solicitudes.

::: figura pd-umbral-sensibilidad
Título: TPR, TNR y tasa de rechazo según el umbral de PD, en validación y prueba
Fuente: Elaboración propia a partir de results/data/pd_umbral/umbral.json (python scripts/umbral_pd.py --config configs/pd/umbral.yaml). MLP: semilla mediana. Rejilla de 101 umbrales (pasos de 0,01); la línea vertical marca el umbral 0,5.
![Dos paneles (validación y prueba) con la TPR que baja, la TNR que sube y la tasa de rechazo que baja al subir el umbral, para la logística y el MLP; las curvas del MLP llegan a 0 y a 1 cerca de 0,4](figures/pd-umbral-sensibilidad.png)
:::

El umbral donde las dos poblaciones se separan más es el del KS (sección 5.8): en la rejilla, el máximo de
TPR + TNR − 1 del MLP en prueba es 0,2483, con el umbral 0,18, apenas por debajo de su KS de prueba (0,2488), que se alcanza
en una PD de 0,196 (sección 6.3). Como se explicó en la sección 5.8, la rejilla da una cota inferior del KS. Ese umbral
separa bien, pero no tiene en cuenta los costos: trata igual un FN que un FP.

#### 6.8.4 Cuota por costo y metas de sensibilidad

Para incluir los costos, para cada razón $L/G$ se elige la **cuota** $r^*$ (fracción de solicitudes que se rechazan, las de
mayor PD) que minimiza el costo $\frac{L}{G}\,\mathrm{FN} + \mathrm{FP}$ **observado en validación** (2013), en una rejilla de
cuotas con pasos de 0,01 (empate: la menor). Luego esa cuota se aplica tal cual en prueba (2014): se rechaza el $r^*$ de la
cohorte de prueba con mayor PD. La [[tab:pd-umbral-escenarios]] muestra tres escenarios.

::: tabla pd-umbral-escenarios
Título: Cuota de rechazo que minimiza el costo en validación y su efecto en prueba, para L/G = 2, 5 y 10
Fuente: Elaboración propia a partir de results/data/pd_umbral/umbral.json (python scripts/umbral_pd.py --config configs/pd/umbral.yaml). r* = cuota elegida en validación; las demás columnas son de prueba. Logística: una corrida; MLP: semilla mediana; "MLP, 30 semillas": media ± desviación muestral (cuota: rango de las r* elegidas). p* = 1/(1 + L/G).
| L/G | p* | Modelo | r* | Malos atrapados (VP) | Buenos rechazados (FP) | TPR | TNR | Tasa de aprobación |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | 0,333 | Logística | 0,16 | 5 352 | 6 513 | 0,2753 | 0,8810 | 0,8400 |
| 2 | 0,333 | MLP | 0,17 | 5 613 | 6 994 | 0,2887 | 0,8722 | 0,8300 |
| 2 | 0,333 | MLP, 30 semillas | 0,13–0,21 | — | — | 0,2836 ± 0,0324 | 0,8758 ± 0,0186 | 0,8340 ± 0,0222 |
| 5 | 0,167 | Logística | 0,70 | 16 526 | 35 382 | 0,8501 | 0,3533 | 0,3000 |
| 5 | 0,167 | MLP | 0,68 | 16 293 | 34 132 | 0,8381 | 0,3762 | 0,3200 |
| 5 | 0,167 | MLP, 30 semillas | 0,65–0,72 | — | — | 0,8428 ± 0,0125 | 0,3674 ± 0,0210 | 0,3123 ± 0,0187 |
| 10 | 0,091 | Logística | 0,93 | 19 043 | 49 921 | 0,9795 | 0,0876 | 0,0700 |
| 10 | 0,091 | MLP | 0,91 | 18 874 | 48 607 | 0,9708 | 0,1116 | 0,0900 |
| 10 | 0,091 | MLP, 30 semillas | 0,90–0,94 | — | — | 0,9764 ± 0,0041 | 0,0982 ± 0,0117 | 0,0787 ± 0,0097 |
:::

La razón de costos domina la decisión. Con $L/G = 2$ conviene rechazar solo el 16–17 % de mayor PD, y se atrapa algo menos
de un tercio de los malos; con $L/G = 5$ la cuota salta a 0,68–0,70 y se atrapa el 84–85 % de los malos a costa de rechazar
a casi dos tercios de los buenos; con $L/G = 10$ se rechaza más del 90 % de las solicitudes. Que la cuota óptima suba tanto
entre 2 y 5 es consistente con que los modelos discriminan de forma moderada (AUC cerca de 0,67): para atrapar más malos hay que
rechazar muchos buenos. La [[fig:pd-umbral-costo]] (panel izquierdo) muestra la cuota para toda la gama de $L/G$ entre 1 y 20.

::: figura pd-umbral-costo
Título: Cuota que minimiza el costo según L/G y umbral observado en validación frente a p*
Fuente: Elaboración propia a partir de results/data/pd_umbral/umbral.json (python scripts/umbral_pd.py --config configs/pd/umbral.yaml). MLP: semilla mediana. Rejilla logarítmica de 25 valores de L/G entre 1 y 20. Izquierda: tasa de rechazo en prueba con la cuota de validación (continua) y con el umbral de PD de validación (discontinua). Derecha: menor PD rechazada por la cuota en validación frente a p* = 1/(1 + L/G).
![Izquierda: la tasa de rechazo en prueba sube con L/G de casi 0 a casi 1, igual para los dos modelos. Derecha: el umbral observado de los dos modelos baja con L/G y queda por debajo de la curva punteada de p* en casi toda la gama](figures/pd-umbral-costo.png)
:::

**Variación entre semillas.** Con $L/G = 2$, la cuota óptima del MLP va de 0,13 a 0,21 según la semilla, y la TPR de
prueba tiene una desviación de 0,0324: la elección del corte es sensible al azar del entrenamiento cuando el costo es bajo.
Con $L/G = 5$ y $10$ la desviación de la TPR baja a 0,0125 y 0,0041. En los tres escenarios la logística queda dentro de una
desviación entre semillas de la media del MLP en TPR, TNR y tasa de aprobación: **en estas cifras, la logística y el MLP no
se distinguen del ruido**.

**Metas de sensibilidad.** Otra forma de fijar el corte, sin $L/G$, es pedir una TPR mínima: para cada meta se toma la menor
cuota con la que la TPR de validación la alcanza ([[tab:pd-umbral-metas]]).

::: tabla pd-umbral-metas
Título: Menor cuota que alcanza cada meta de sensibilidad en validación y su efecto en prueba
Fuente: Elaboración propia a partir de results/data/pd_umbral/umbral.json (python scripts/umbral_pd.py --config configs/pd/umbral.yaml). Cuota y TPR de validación; las demás columnas son de prueba. Logística: una corrida; MLP: semilla mediana; "MLP, 30 semillas": media ± desviación muestral (cuota: rango).
| Meta de TPR | Modelo | Cuota | TPR val | TPR test | TNR test | Tasa de aprobación test |
|---:|---|---:|---:|---:|---:|---:|
| 0,5 | Logística | 0,33 | 0,5054 | 0,4953 | 0,7287 | 0,6700 |
| 0,5 | MLP | 0,33 | 0,5080 | 0,4996 | 0,7302 | 0,6700 |
| 0,5 | MLP, 30 semillas | 0,32–0,34 | — | 0,4989 ± 0,0045 | 0,7304 ± 0,0037 | 0,6703 ± 0,0032 |
| 0,7 | Logística | 0,52 | 0,7025 | 0,7016 | 0,5445 | 0,4800 |
| 0,7 | MLP | 0,52 | 0,7050 | 0,7021 | 0,5447 | 0,4800 |
| 0,7 | MLP, 30 semillas | 0,52–0,53 | — | 0,7040 ± 0,0033 | 0,5431 ± 0,0049 | 0,4783 ± 0,0038 |
| 0,8 | Logística | 0,64 | 0,8028 | 0,8049 | 0,4186 | 0,3600 |
| 0,8 | MLP | 0,64 | 0,8073 | 0,8078 | 0,4196 | 0,3600 |
| 0,8 | MLP, 30 semillas | 0,63–0,64 | — | 0,8029 ± 0,0037 | 0,4246 ± 0,0062 | 0,3650 ± 0,0051 |
| 0,9 | Logística | 0,78 | 0,9027 | 0,9041 | 0,2641 | 0,2200 |
| 0,9 | MLP | 0,78 | 0,9048 | 0,9043 | 0,2641 | 0,2200 |
| 0,9 | MLP, 30 semillas | 0,77–0,78 | — | 0,9009 ± 0,0028 | 0,2684 ± 0,0060 | 0,2240 ± 0,0050 |
:::

Las metas se sostienen en prueba: la TPR de prueba queda a menos de 0,011 de la de validación en todos los casos. El precio
crece rápido: atrapar la mitad de los malos exige rechazar un tercio de las solicitudes, y atrapar el 90 %, el 78 %. Aquí
también los dos modelos dan cifras casi iguales, con diferencias del orden de una desviación entre semillas.

#### 6.8.5 PD absoluta frente a cuota, y p* frente al corte observado

La cuota se aplicó en prueba como **fracción** de la cohorte. La alternativa es fijar el **umbral de PD** que esa cuota
implicó en validación (la menor PD rechazada en 2013) y aplicarlo tal cual a las solicitudes de 2014. Con $L/G = 2$, ese
umbral (0,2781 en la logística y 0,2870 en el MLP) rechaza en prueba al 19,5 % y al 20,1 %, no al 16 % y 17 % previstos; con
$L/G = 5$, al 71,0 % y 69,1 % en vez de 70 % y 68 %; con las metas de sensibilidad pasa lo mismo (por ejemplo, la cuota de
0,33 se vuelve un rechazo de 0,3766 en la logística). Solo con $L/G = 10$, donde ya se rechaza más del 90 %, la tasa baja
levemente (0,9284 y 0,9077 frente a 0,93 y 0,91). Es decir, una fracción mayor de las PD de 2014 supera esos umbrales y un umbral fijo en PD
rechaza más: el corte en PD absoluta se mueve con la deriva entre años (la tasa de malos sube de 0,22950 en validación a
0,26217 en prueba), mientras que la cuota mantiene fija la fracción rechazada. El costo por crédito de prueba de las dos
reglas es casi el mismo (con $L/G = 2$, 0,4678 con la cuota y 0,4648 con el umbral en la logística).

El panel derecho de la [[fig:pd-umbral-costo]] compara el umbral observado en validación con el teórico $p^* = 1/(1 + L/G)$
de la sección 5.8 ([[tab:pd-umbral-pstar]]).

::: tabla pd-umbral-pstar
Título: Umbral teórico p* y umbral de PD implicado por la cuota óptima en validación
Fuente: Elaboración propia a partir de results/data/pd_umbral/umbral.json (python scripts/umbral_pd.py --config configs/pd/umbral.yaml). Umbral observado = menor PD rechazada por la cuota r* en validación; MLP: semilla mediana.
| L/G | p* | Umbral observado (logística / MLP) |
|---:|---:|---:|
| 2 | 0,333 | 0,278 / 0,287 |
| 5 | 0,167 | 0,114 / 0,145 |
| 10 | 0,091 | 0,071 / 0,093 |
:::

En casi toda la gama el umbral que minimiza el costo observado queda **por debajo** de $p^*$. Las excepciones están en el
extremo de costos altos, donde la cuota ya rechaza más del 90 %: el MLP con $L/G = 10$ (0,0930 frente a 0,0909) y en los
puntos de la rejilla de $L/G$ desde 10,7 hasta 20, y la logística con $L/G = 20$ (0,0531 frente a 0,0476). Que el corte
observado quede por debajo de $p^*$ es lo esperable si la PD subestima el riesgo (sección 5.8), y va en la misma dirección
que la subestimación de la sección 6.4 (PD media de prueba 0,187 y 0,204 frente a una tasa de 0,262). Estas cifras no
bastan para atribuirlo a una causa: muestran que, sin recalibrar, $p^*$ no es un buen corte para estas PD.

#### 6.8.6 Conclusión y próximos pasos

La PD por sí sola no decide: el corte depende de cuánto cuesta un malo frente a un bueno, y con $L/G$ entre 2 y 10 la
fracción rechazada va del 16 % a más del 90 %. Estas cifras **ilustran** cómo cambia la decisión; no fijan una política,
porque $L/G$ depende del negocio y del apetito de riesgo. La cuota es una política **por lotes o de cuota** (se rechaza una
fracción de una cohorte ya reunida), no la decisión individual en tiempo real que muestra la app, que solo calcula la PD de
un solicitante. En estas cifras la logística y el MLP no se distinguen del ruido entre semillas.

Próximos pasos: **recalibrar la PD** (para que $p^*$ sea un corte usable y el umbral en PD no se mueva con la deriva);
**inferencia de rechazados**, porque solo vemos créditos que Lending Club aprobó y el costo de rechazar se mide en esa
población; **monitorear la deriva con el PSI** de la PD y de las variables, para saber cuándo revisar el corte; y **fijar
$L/G$ con datos del negocio** (pérdida dado el incumplimiento y margen de un buen cliente). Por último, como el puntaje de la
scorecard es lineal en el logaritmo de las odds, un corte en PD equivale a un corte en puntaje (sección 7).
