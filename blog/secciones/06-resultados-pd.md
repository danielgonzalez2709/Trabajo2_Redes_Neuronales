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
Dos ensayos posteriores, el 19 y el 27, quedaron a menos de 0,001 de ese valor.

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
