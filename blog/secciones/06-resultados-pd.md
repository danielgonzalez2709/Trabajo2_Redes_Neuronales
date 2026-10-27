<!-- Responsable: Andrés (B). Enunciado (Anexo del núcleo): resultados de PD y comparación justa (30 semillas, presupuesto, parámetros). Guía de sintaxis: blog/README.md -->
<!-- B7 (26-oct): bloques de figura con una frase que cita cada una; cifras de results/data/pd/{summary,hpo_ensayos,
busqueda_logistica}.json, results/data/pd/semillas/ y results/data/pd/pred_test.parquet (python run.py pd --config
configs/pd/base.yaml) y data/processed/resumen_datos.json (tasas por split). Figuras: python run.py figures (src/viz/figs_pd.py). El texto completo de la sección es B8. -->
## 6. Resultados de la PD

### 6.1 Búsqueda de hiperparámetros

La logística eligió su penalización con una rejilla de 13 valores de C ([[fig:pd-busqueda-logistica]]). La log-loss de
validación baja hasta 0,5178 en C = 0,0031623 y después sube un poco y se aplana; el valor elegido no está en el borde de la
rejilla.

::: figura pd-busqueda-logistica
Título: Log-loss y AUC de validación de la regresión logística según la penalización C
Fuente: Elaboración propia a partir de results/data/pd/busqueda_logistica.json.
![Dos paneles con eje logarítmico de C: la log-loss de validación tiene su mínimo en C = 0,0031623 y la AUC de validación crece hasta aplanarse](figures/pd-busqueda-logistica.png)
:::

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
<!-- revisar: autor -->

La PD media de prueba sube (logística 0,1868 → 0,1988; MLP 0,2041 → 0,2128 con la misma arquitectura y 0,2179 con la
buscada), pero sigue por debajo de la tasa observada de 0,2622 (§6.4). Es consistente con la deriva de composición entre
entrenamiento y prueba: la participación del grado A baja de 23,2 % en entrenamiento a 12,2 % en validación y 12,9 % en
prueba, y la `int_rate` media sube de 12,77 % a 14,69 % y 14,33 %. No basta para afirmar una causa; también puede influir
el sesgo de selección de los créditos recientes (sección 2.4).
