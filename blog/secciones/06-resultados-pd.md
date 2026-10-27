<!-- Responsable: Andrés (B). Enunciado (Anexo del núcleo): resultados de PD y comparación justa (30 semillas, presupuesto, parámetros). Guía de sintaxis: blog/README.md -->
<!-- B7 (26-oct): bloques de figura con una frase que cita cada una; cifras de results/data/pd/{summary,hpo_ensayos,
busqueda_logistica}.json, results/data/pd/semillas/ y results/data/pd/pred_test.parquet (python run.py pd --config
configs/pd/base.yaml) y data/processed/resumen_datos.json (tasas por split). Figuras: python run.py figures (src/viz/figs_pd.py). El texto completo de la sección es B8. -->
## 6. Resultados de la PD

Esta sección compara la regresión logística (el modelo de referencia) con el MLP, con los mismos datos, la misma partición y
el mismo preprocesador. Primero se muestran la búsqueda de hiperparámetros y el entrenamiento; después, la discriminación, la
calibración y la variabilidad entre 30 semillas, con la tabla de la comparación justa ([[tab:pd-comparacion]]). Cierran la
sensibilidad a las variables de Lending Club y las limitaciones.

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
  <!-- revisar: autor -->
- **Subestimación en prueba.** Los dos modelos predicen menos malos de los que hay en prueba (§6.4): la PD media es 0,187 y
  0,204 frente a una tasa de 0,262. Ordenan bien el riesgo, pero si se usa la PD como probabilidad (por ejemplo, en la
  pérdida esperada) hace falta recalibrarla o vigilar la deriva.
- **Búsqueda casi plana.** Tres ensayos (el 19, el 23 y el 27) quedaron a menos de 0,001 de log-loss de validación del mejor
  (§6.1). Esa diferencia es menor que la desviación entre semillas de la configuración final (0,0018 en la log-loss de
  validación), así que el ensayo 11 no se distingue con claridad de esos tres; se eligió por la regla fijada de antemano (la
  menor log-loss de validación), no porque sea claramente mejor.
