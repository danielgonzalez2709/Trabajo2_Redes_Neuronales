<!-- Responsable: Todos (consolida Daniel). Enunciado (Anexo del núcleo): qué aportó la red frente al modelo de referencia; aprendizajes. Guía de sintaxis: blog/README.md -->
<!-- Cifras: results/data/pd/{summary,hpo_ensayos,busqueda_logistica}.json, results/data/pd/semillas/,
results/data/pd/pred_test.parquet; results/data/lgd/summary.json y results/data/lgd/semillas/; results/data/el/summary.json;
results/data/scorecard/scorecard.json; data/processed/resumen_datos.json. La E[L] con el MLP de PD sale de
python run.py el --config configs/el/base.yaml --set pd.modelo=mlp --set salida=<carpeta temporal> (no está en results/). -->
## 10. Discusión

El enunciado pregunta qué aportó la red neuronal frente al modelo de referencia en cada tarea, con base en las métricas, su
variabilidad entre semillas, el costo de cómputo y la interpretabilidad. Las dos comparaciones se hicieron bajo la regla de
comparación justa: en cada tarea los dos modelos usan la misma partición y el mismo preprocesador (en la PD, los dos registros
de `results/data/pd/summary.json` tienen las mismas huellas SHA-256 de la partición y del preprocesador), el presupuesto de
búsqueda está declarado, cada red se entrenó con 30 semillas y se reporta el número de parámetros. La respuesta corta es que
la red **mejoró poco la PD y no mejoró la LGD**, a un costo de cómputo y de interpretabilidad mucho mayor. En la PD casi
no discrimina mejor, y la mayor parte de su ventaja en Brier y log-loss se explica por el nivel de su PD media, más cercano
al observado en prueba (diagnóstico de la sección 10.1, con la semilla mediana).

### 10.1 Probabilidad de incumplimiento: logística frente al MLP

La [[tab:discusion-pd]] compara la regresión logística (determinista, una corrida) con las 30 semillas del MLP en el
conjunto de prueba (créditos de 2014). La última columna cuenta, semilla por semilla, cuántas corridas del MLP superan a la
logística en cada métrica.

::: tabla discusion-pd
Título: Métricas de prueba de la PD: regresión logística frente a las 30 semillas del MLP
Fuente: Elaboración propia a partir de results/data/pd/summary.json (media, desviación estándar muestral, mejor y peor) y de results/data/pd/semillas/ (conteo de semillas). Logística: n = 1; MLP: n = 30 (semillas 0 a 29). En AUC, Gini y KS "mejor" es el mayor; en Brier y log-loss, el menor.
| Métrica (prueba) | Logística | MLP: media ± desv. | MLP: mejor | MLP: peor | MLP − logística | Semillas del MLP mejores que la logística |
|---|---:|---:|---:|---:|---:|---:|
| AUC | 0,6696 | 0,6716 ± 0,0031 | 0,6766 | 0,6630 | +0,0020 | 24 de 30 |
| Gini | 0,3393 | 0,3433 ± 0,0062 | 0,3533 | 0,3261 | +0,0040 | 24 de 30 |
| KS | 0,2465 | 0,2500 ± 0,0054 | 0,2581 | 0,2340 | +0,0035 | 23 de 30 |
| Brier | 0,1865 | 0,1832 ± 0,0011 | 0,1808 | 0,1857 | −0,0033 | 30 de 30 |
| Log-loss | 0,5600 | 0,5490 ± 0,0033 | 0,5417 | 0,5561 | −0,0110 | 30 de 30 |
:::

**Discriminación: una ganancia menor que la variabilidad entre semillas.** En AUC, Gini y KS la media del MLP supera a la
logística por 0,0020, 0,0040 y 0,0035, cerca de dos tercios de una desviación estándar entre semillas en cada caso (0,0031,
0,0062 y 0,0054). La logística queda dentro del rango de las 30 semillas en las tres métricas, y entre 6 y 7 semillas
discriminan peor que ella. En validación el patrón es el mismo: 25 de 30 semillas superan su AUC y 20 de 30 su KS (Fuente:
`results/data/pd/semillas/`).

**Brier y log-loss: una ganancia consistente.** Aquí las 30 semillas mejoran a la logística, en prueba y en validación, y
hasta la peor semilla del MLP (Brier 0,1857, log-loss 0,5561) queda por debajo de la logística (0,1865 y 0,5600). La
diferencia de medias equivale a unas tres desviaciones entre semillas (0,0033 frente a 0,0011 en Brier; 0,0110 frente a
0,0033 en log-loss). Dos cautelas para leer estas cifras. Primera, la desviación entre semillas solo mide lo que cambia con la
inicialización, el *dropout* y el orden de los lotes; el conjunto de prueba es el mismo en todas las corridas, así que no
mide la incertidumbre por la muestra de créditos. No hicimos una prueba estadística pareada sobre los créditos de prueba y por
eso no llamamos "significativa" a ninguna diferencia. Segunda, Brier y log-loss premian tanto el orden como el nivel de la PD,
y la red predice un nivel más alto (ver la calibración, abajo); el diagnóstico del párrafo siguiente estima cuánto pesa el
nivel.

**Calibración: las dos se quedan cortas.** La tasa de malos sube de un conjunto al siguiente: 16,3 % en entrenamiento, 23,0 %
en validación y 26,2 % en prueba ([[tab:particion]]). La PD media de la logística es 0,1630 en entrenamiento (igual a la tasa
de malos de ese conjunto), 0,1775 en validación y 0,1868 en prueba; la del MLP (semilla mediana) es 0,1884, 0,1978 y 0,2041,
es decir, queda por encima de la tasa de entrenamiento. Ninguna de las dos llega a la tasa de prueba, 0,2622 (Fuente:
`models/pd_logistica.joblib` y `models/pd_mlp.pt` aplicados a `data/processed/loans_pd.parquet` con
`models/preprocesador_pd.json`; las predicciones de prueba coinciden con `results/data/pd/pred_test.parquet`). No sabemos por
qué el nivel del MLP queda por encima de la tasa de entrenamiento; posiblemente influye la parada temprana, que elige la
época por la pérdida de validación (un conjunto con tasa de malos mayor), pero no lo probamos. Con el MLP la PD media también queda por debajo de la
tasa de malos en las diez bandas de puntaje del conjunto de prueba, como con la logística ([[fig:el-pd-vs-tasa]]; para el
MLP, la salida de `run.py el` de la sección 10.4).

**Diagnóstico: cuánto de la ventaja en Brier y log-loss es nivel.** Para separarlo, desplazamos el intercepto de cada modelo
(sumamos una constante $c$ a su logit, $p_i' = \sigma\big(\operatorname{logit}(p_i) + c\big)$) hasta que su PD media en
prueba iguala la tasa de malos de prueba, y volvemos a medir. Con la semilla 20 del MLP, la brecha de log-loss entre la
logística y el MLP baja de 0,0101 a 0,0018 y la de Brier de 0,0025 a 0,0008; es decir, la mayor parte de la ventaja viene de
que el MLP predice un nivel más alto, y una parte menor se mantiene con el nivel igualado. El ajuste usa las
etiquetas de prueba, así que **es solo un diagnóstico**: no es un modelo que se pueda usar ni una métrica de prueba válida.
<!-- revisar: autor (diagnóstico calculado fuera de src/ con models/pd_logistica.joblib y models/pd_mlp.pt; el script está
en el scratchpad de la sesión. Si se mantiene, conviene llevarlo a src/pd/ con su prueba) -->

**Forma de la distribución.** La PD de la logística va de 0,015 a 0,736 (desviación 0,106) y la del MLP de 0,018 a 0,410
(desviación 0,092): la del MLP está más comprimida (Fuente: `results/data/pd/pred_test.parquet`). Por eso, aunque su media
es más alta, subestima más el decil de mayor riesgo: en la banda 1 su PD media es 0,382 frente a una tasa de malos de 0,478,
mientras que la de la logística es 0,416 frente a 0,476 (Fuente: `results/data/el/summary.json` y la salida de `run.py el`
con el MLP, bloque `por_banda`; cada modelo con sus propias bandas).

**Búsqueda y entrenamiento.** La búsqueda del MLP (TPE, 40 ensayos, criterio log-loss de validación) eligió una red de tres
capas ocultas de 32, 16 y 32 neuronas con ReLU, *dropout* 0,094, Adam y lotes de 512 (ensayo 11, 2 673 parámetros). De los
tres hiperparámetros continuos que se revisan contra el borde (tasa de aprendizaje, L2 y *dropout*), ninguno quedó en el
borde del espacio (Fuente: `results/data/pd/hpo_ensayos.json`, bloques `mejor` y `en_borde`); la revisión no cubre los
discretos, y la segunda capa tiene 16 neuronas, el mínimo de la lista {16, 32, 64, 128, 256}. La
búsqueda no fue difícil de ganar: 35 de los 40 ensayos superaron la log-loss de validación de la logística (0,5178); las
redes probadas tenían de 1 601 a 62 081 parámetros. Las corridas son cortas: se detuvieron entre las épocas 11 y 30, y la mejor época de
validación fue la 5 o una anterior en 16 de las 30 semillas; en la semilla mediana (la 20, que es la que usan la pérdida
esperada y la aplicación) fue la primera (Fuente: `results/data/pd/semillas/`). La logística probó 13 valores de $C$ y eligió
$C = 3{,}16 \times 10^{-3}$ (Fuente: `results/data/pd/busqueda_logistica.json`).
<!-- revisar: autor (Andrés: ¿la mejor época temprana se discute en la sección 6? Si allí hay una explicación, citarla) -->

### 10.2 Costo de cómputo y complejidad

::: tabla discusion-presupuesto
Título: Parámetros entrenables, configuraciones evaluadas y tiempo total por modelo en las dos tareas
Fuente: Elaboración propia a partir de results/data/pd/summary.json y results/data/lgd/summary.json (n_parametros, n_configuraciones_busqueda, tiempo_total_s, tiempo_busqueda_s y tiempo_semillas_s). Tiempo en CPU; el de la red suma la búsqueda y las 30 semillas.
| Tarea | Modelo | Parámetros | Configuraciones de la búsqueda | Tiempo total |
|---|---|---:|---:|---:|
| PD | Regresión logística | 49 | 13 | 60,8 s |
| PD | MLP (32-16-32, ReLU) | 2 673 | 40 | 940,4 s (702,4 + 238,0) |
| LGD | Regresión lineal | 49 | 1 | 0,03 s |
| LGD | MLP (32-64-64-32, leaky ReLU) | 9 953 | 30 | 165,3 s (126,1 + 39,2) |
:::
<!-- revisar: autor (los tiempos de PD y de LGD se midieron en máquinas posiblemente distintas; hilos_torch = 4 en la PD.
Confirmar con Andrés y Camila antes de compararlos entre tareas) -->

La [[tab:discusion-presupuesto]] resume el costo. En la PD la red tiene unas 55 veces los parámetros de la logística y su
ciclo completo tardó unas 15 veces más (940,4 s frente a 60,8 s); en la LGD tiene unas 200 veces los parámetros de la
lineal. En términos absolutos ninguno de los costos es alto: el ciclo completo de cada red cabe en minutos en una CPU. El
costo también se nota en la aplicación web, donde los dos modelos de PD se ejecutan en el navegador: el archivo del MLP pesa
107 567 bytes y el de la logística 16 491 (Fuente: `app/public/model/pd_model_mlp.json` y `pd_model_logistica.json`).

### 10.3 Pérdida dado el incumplimiento: la red no aportó

La sección 8 ya hizo esta comparación ([[tab:lgd-resultados]] y [[fig:lgd-semillas]]); aquí la resumimos con los mismos
criterios. En MAE de prueba la lineal obtiene 0,07712 y el MLP 0,07686 ± 0,00037: la diferencia (0,00026) es menor que la
desviación entre semillas (unas 0,7 desviaciones), y 23 de las 30 semillas mejoran a la lineal y 7 no; en MAE de validación
el MLP es algo peor (0,0761 frente a 0,0759; unas 3,4 desviaciones; ninguna de las 30 semillas mejora a la lineal). En RMSE
la red sí queda de forma consistente por delante: las 30 semillas mejoran a la lineal en validación (0,09244 frente a
0,09276, unas 6,3 desviaciones) y en prueba (0,0855 frente a 0,0866, unas 2,7 desviaciones). Las diferencias absolutas son
mínimas en todos los casos: a lo sumo 0,0011 en RMSE y 0,0003 en MAE (Fuente: `results/data/lgd/summary.json` y
`results/data/lgd/semillas/`). Ninguno de los dos se separa de predecir una constante: la media de entrenamiento queda a menos
de una milésima en ambas métricas de prueba, y la mediana de entrenamiento (0,943) obtiene el menor MAE y el menor RMSE de
prueba de todos (0,0699 y 0,0811). La sección 8.5 explica esa ventaja con dos razones que se suman: la mediana es la
constante que minimiza el error absoluto, mientras que los modelos apuntan a la media; y el conjunto de prueba está
desplazado hacia LGD = 1, de modo que 0,943 queda más cerca de esa población que la media de entrenamiento. En validación la
misma constante es la peor opción en MAE y en RMSE. Las predicciones de los dos modelos casi no varían entre créditos
(sección 8.5), de modo que en la pérdida esperada la LGD actúa como una constante cercana a 0,917 (sección 9.1).

Lo que sí aportó el diseño de la red no es precisión sino forma: la salida sigmoide garantiza predicciones en $(0, 1)$ y la
entropía cruzada fraccional apunta a la media condicional, que es lo que necesita la pérdida esperada (sección 8.3). En estos
datos la lineal tampoco salió del intervalo (0 predicciones fuera de $[0, 1]$ en validación, en prueba y en los 74 154
créditos de prueba de la PD; Fuente: `results/data/lgd/summary.json`), así que la garantía no se tradujo en una diferencia
observable.

### 10.4 Pérdida esperada con cada modelo de PD

La sección 9 calculó la pérdida esperada con la logística. Para ver qué cambia con la red, la repetimos con la PD del MLP
(semilla mediana) y la misma LGD y EAD ([[tab:discusion-el]]). Las bandas se recalculan con el puntaje de cada modelo, así
que no son los mismos créditos en una banda dada.

::: tabla discusion-el
Título: Pérdida esperada del conjunto de prueba con la PD de la logística y con la del MLP, y descomposición de la razón en los créditos resueltos
Fuente: Elaboración propia. Logística: results/data/el/summary.json y results/data/el/el_test.parquet. MLP: salida de python run.py el --config configs/el/base.yaml --set pd.modelo=mlp --set salida=<carpeta temporal> (summary.json y el_test.parquet; no se guarda en results/). LGD: MLP de la Parte D (semilla 22); EAD: funded_amnt. Factores: la descomposición de la sección 9.5, calculada con el_test.parquet de cada corrida. Montos en millones de USD.
| | PD de la logística | PD del MLP (semilla 20) |
|---|---:|---:|
| E[L], todo el conjunto de prueba (74 154 créditos) | 197,03 | 208,20 |
| PD media / tasa de malos, todo el conjunto | 0,187 / 0,262 | 0,204 / 0,262 |
| E[L], resueltos (68 694 créditos) | 178,16 | 188,81 |
| Pérdida realizada, resueltos | 168,20 | 168,20 |
| **E[L] / realizada, resueltos** | **1,059** | **1,123** |
| PD media / tasa de malos, resueltos | 0,183 / 0,204 | 0,201 / 0,204 |
| Factor PD | 0,926 | 0,981 |
| Factor LGD | 0,971 | 0,971 |
| Factor EAD | 1,179 | 1,179 |
| E[L] / realizada por banda (mínimo y máximo) | 0,927 (banda 5) – 1,209 (banda 10) | 1,031 (banda 1) – 1,360 (banda 10) |
:::
<!-- revisar: autor (la E[L] con el MLP y su descomposición no están en results/: decidir si se guarda la corrida, p. ej. en
results/data/el_mlp/, y llevar la descomposición a src/el/tablas.py, como pide la nota de la sección 9.5) -->

Con el MLP, la pérdida esperada de todo el conjunto de prueba sube de 197,03 a 208,20 millones de USD, y la razón entre
pérdida esperada y realizada en los resueltos pasa de 1,059 a 1,123. Por banda, con el MLP la razón supera a 1 en las diez
bandas, mientras que con la logística quedaba por debajo de 1 en las bandas 5 y 6. La tasa de malos baja de forma monótona de
la banda 1 a la 10 con los dos modelos (de 0,476 a 0,087 con la logística y de 0,478 a 0,088 con el MLP), así que los dos
ordenan bien el riesgo por grupos.

**Por qué una razón más lejana de 1 no quiere decir una PD peor.** La descomposición de la sección 9.5 lo explica. Los
factores de LGD y de EAD no dependen del modelo de PD y son iguales en las dos columnas: la LGD se queda corta un 2,9 % y la
EAD de originación agrega un 17,9 %. Lo único que cambia es el factor PD, que pasa de 0,926 a 0,981: la PD del MLP está más
cerca del nivel de los resueltos (0,201 frente a una tasa de 0,204). Con la logística, la subestimación de la PD compensaba
en parte la sobreestimación de la EAD y la razón quedaba cerca de 1; con el MLP esa compensación casi desaparece y queda a la
vista el sesgo conservador de la EAD. Es decir, **la razón agregada no mide la calibración de la PD**: una PD más cercana a
la tasa observada produjo una razón más lejana de 1. Además, la tasa de los resueltos (0,204) no es la de todo el conjunto
(0,262): en todo el conjunto de prueba la PD media del MLP sigue por debajo de la tasa de malos en las diez bandas.

### 10.5 Interpretabilidad

La logística se lee como una **scorecard**: el puntaje de cada crédito es una suma de puntos por variable más unos puntos
base (525,96), y esa suma reproduce el puntaje con un error máximo de $1{,}1 \times 10^{-13}$ (Fuente:
`results/data/scorecard/scorecard.json`, bloque `aditividad`). La [[tab:discusion-scorecard]] ordena las variables por su
rango de puntos.

::: tabla discusion-scorecard
Título: Rango de puntos de cada variable en la scorecard de la regresión logística
Fuente: Elaboración propia a partir de results/data/scorecard/scorecard.json (puntos_por_variable, rango_puntos) y configs/scorecard/base.yaml. Numéricas: diferencia de puntos entre los percentiles 1 y 99 del valor crudo en entrenamiento. Categóricas: diferencia entre la categoría de más y la de menos puntos, entre las que tienen al menos el 1 % de entrenamiento. Los dos tipos de rango no son comparables entre sí.
| Variable | Etiqueta en la aplicación | Tipo | Rango de puntos |
|---|---|---|---:|
| `annual_inc` | Ingreso anual (USD) | Numérica | 45,9 |
| `revol_util` | Uso de tus cupos rotativos (%) | Numérica | 36,4 |
| `inq_last_6mths` | Consultas de crédito en los últimos 6 meses | Numérica | 29,3 |
| `term` | Plazo | Categórica | 28,9 |
| `revol_bal` | Saldo en tarjetas de crédito (USD) | Numérica | 28,8 |
| `loan_amnt` | Monto solicitado (USD) | Numérica | 26,2 |
| `open_acc` | Líneas de crédito abiertas | Numérica | 22,5 |
| `purpose` | Destino del préstamo | Categórica | 22,2 |
| `total_acc` | Total de líneas de crédito en tu historial | Numérica | 12,7 |
| `delinq_2yrs` | Moras de 30+ días en los últimos 2 años | Numérica | 10,2 |
| `pub_rec` | Registros públicos negativos | Numérica | 9,9 |
| `emp_length` | Antigüedad en el empleo | Categórica | 7,9 |
| `dti` | Deuda mensual sobre ingreso mensual (%) | Numérica | 6,7 |
| `home_ownership` | Vivienda | Categórica | 3,0 |
| `antiguedad_credito_anios` | Años de historial crediticio | Numérica | 1,5 |
:::

Con la logística se puede decir, por ejemplo, que el ingreso anual mueve el puntaje hasta 45,9 puntos entre los percentiles 1
y 99 de entrenamiento, y que la antigüedad del historial crediticio casi no lo mueve (1,5 puntos). El rango es una medida
simple del peso de la variable en la población de entrenamiento: no es una medida causal y depende de la escala y la
dispersión de cada variable (`src/scorecard/puntos.py`). El MLP no admite esa
lectura: su puntaje se obtiene de la PD con la misma transformación, pero no se descompone en puntos por variable. La
explicación de la red con SHAP [@lundberg2017] está planeada (tarea C4) y **todavía no existe**, así que hoy no podemos decir
qué variables pesan en el MLP ni si coinciden con las de la scorecard.
<!-- TODO(dato-faltante): resultados de SHAP del MLP de PD (C4, Parte C); actualizar este párrafo cuando existan. -->

En la LGD la diferencia es la misma: la lineal se lee coeficiente por coeficiente y el MLP necesitaría herramientas
adicionales (sección 8.6).

La aplicación web lleva los dos modelos de PD: la especificación (v0.17, 26 de octubre) decidió que vayan los dos, con un
selector entre la red y la logística, y los dos modelos ya están exportados en `app/public/model/`.
<!-- revisar: autor (confirmar con Camila que el selector ya está en la app publicada antes de la entrega) -->

### 10.6 Qué aportó la red, en resumen

- **PD.** El MLP discrimina igual o apenas mejor que la logística (AUC de prueba 0,6716 ± 0,0031 frente a 0,6696, una
  diferencia menor que la variabilidad entre semillas) y mejora de forma consistente Brier y log-loss (las 30 semillas); según
  el diagnóstico de la sección 10.1 (con la semilla mediana), la mayor parte de esa mejora viene de que su PD media está más cerca de la tasa de malos
  de 2014. Cuesta unas 55 veces más parámetros, unas 15 veces más tiempo y, por ahora, no tiene explicación por variable.
- **LGD.** El MLP no aportó en la práctica: empata con la lineal en MAE de prueba dentro de la variabilidad entre semillas, y su
  ventaja en RMSE, consistente entre semillas, es de alrededor de una milésima. Tiene unas 200 veces más parámetros, y ninguno de
  los dos supera con claridad a una constante.
- **Pérdida esperada.** Cambiar de modelo de PD cambia el total en 11,17 millones de USD (de 197,03 a 208,20), pero la
  razón con la pérdida realizada no sirve para elegir entre ellos, porque depende sobre todo de cómo se compensan los sesgos
  de la PD y de la EAD.

En este problema el límite no parece estar en la capacidad del modelo, sino en la información de las 15 variables de la
originación y en el cambio de la población entre cosechas (es una lectura de los resultados, no algo que hayamos probado).
Es consistente con la idea de que en datos tabulares una red no gana por defecto [@grinsztajn2022], con dos matices: su
conclusión es para conjuntos de tamaño mediano (del orden de 10 000 muestras), y aquí el entrenamiento de la PD tiene 90 347
créditos; y nosotros no comparamos con modelos basados en árboles.
<!-- revisar: autor (la cita de Grinsztajn et al. solo se apoya en su tesis general y en el tamaño de datos que estudian;
si se prefiere, quitarla) -->

### 10.7 Aprendizajes sobre el problema

1. **El cambio entre cosechas pesa más que la diferencia entre modelos.** La tasa de malos pasa de 16,3 % en entrenamiento a
   26,2 % en prueba ([[tab:particion]]). La logística reproduce la tasa de entrenamiento (PD media 0,1630) y el MLP queda
   más arriba (0,1884 en entrenamiento, 0,2041 en prueba), pero en prueba los dos se quedan por debajo de 0,2622. Con la
   partición temporal, la distancia entre las dos PD medias de prueba (0,017) es menor que la que separa a cualquiera de
   ellas de la tasa observada (al menos 0,058).
2. **Los créditos recientes resueltos son una muestra sesgada.** En 2014 el 68,5 % de los créditos no tiene desenlace y en
   2015 el 92,7 % (Fuente: `data/processed/resumen_datos.json`, `fraccion_na_por_anio`). Entre los que ya terminaron están
   sobrerrepresentados los incumplimientos tempranos (sección 2.4), aunque no son mayoría: de los créditos resueltos de prueba
   solo el 20,4 % fue castigado (13 981 de 68 694; Fuente: `results/data/el/el_test.parquet`). Eso explica, al menos en
   parte, que la tasa de malos de prueba sea alta, y es la razón por la que dejamos 2015 fuera.
3. **La población cambia también en las variables.** La proporción con al menos un registro público pasa de 4,07 % en
   entrenamiento a 19,50 % en prueba, con un PSI de 0,269 (hallazgo H-01 de la sección 12). Ese cambio estuvo oculto al
   principio porque el PSI daba 0 en silencio para variables que, tras el recorte, solo toman dos valores: medir el cambio de
   población exige tratar aparte las variables discretas.
4. **La LGD de estos datos está casi toda cerca de 1, y en prueba más.** El 43,6 % de los castigados tiene LGD = 1 (25,9 %
   en entrenamiento y 62,7 % en prueba), un patrón consistente con cobranzas que todavía no habían terminado (sección 8.2).
   Con esa forma, el conjunto de evaluación decide el ganador: la mediana constante de entrenamiento gana a los dos modelos
   en prueba, en MAE y en RMSE (0,0699 frente a 0,0769 y 0,0771 en MAE), y es la peor opción en validación en las dos métricas.
5. **La EAD es una decisión con un sesgo de signo conocido.** Al castigo seguía pendiente, en promedio, el 83,9 % del monto
   desembolsado (Fuente: `results/data/el/summary.json`); usar el monto completo agrega un 17,9 % a la razón entre la
   pérdida esperada y la realizada de los resueltos (factor EAD 1,179), con cualquiera de los dos modelos de PD.
6. **Una razón cercana a 1 puede salir de errores que se compensan.** Con la logística la razón entre pérdida esperada y
   realizada fue 1,059 y con el MLP 1,123, aunque la PD del MLP está más cerca de la tasa observada (sección 10.4). Para
   validar la pérdida esperada hay que mirar cada factor por separado.
7. **La pérdida "observada" se define por el estado del crédito.** Sumar como pérdida observada los créditos en mora (con
   LGD = 1 supuesta, porque no registran recuperaciones) y los pagados con principal sin registrar daba 231,2 millones de USD
   y una razón de 0,85, que se leía como "el modelo subestima"; con solo los resueltos la razón es 1,06 (hallazgo H-13 de la
   sección 12).
8. **El significado de un vacío no se supone.** En `mths_since_last_delinq` un vacío no quiere decir "nunca hubo mora": el
   0,46 % de los créditos con el campo vacío tiene `delinq_2yrs` > 0 (hallazgo H-02 de la sección 12). La variable ya estaba
   fuera del modelo por tener más de la mitad de valores vacíos (sección 2.3); el hallazgo cambió el motivo para no imputarla:
   el diccionario no dice qué significa un vacío, y suponerlo le daría un significado que no podemos respaldar.
