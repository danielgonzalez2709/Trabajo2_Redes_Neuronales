<!-- Responsable: Daniel (A). Enunciado (Anexo del núcleo): Parte 1, tarea 2 (análisis descriptivo e hipótesis).
Cifras: results/data/eda/*.json (python run.py eda --config configs/datos.yaml). Guía: blog/README.md -->
## 3. Análisis descriptivo

### 3.1 Hipótesis

Antes de entrenar cualquier modelo escribimos once hipótesis sobre qué hace más riesgosa a una persona
(`configs/hipotesis.yaml`). Para no mirar los conjuntos de validación y prueba al formularlas, todo el análisis de esta
sección usa solo el conjunto de entrenamiento. La [[fig:eda-tasa-variables]] muestra la tasa de malos por decil (o por
categoría) de cada variable, y la [[tab:hipotesis]] resume la primera evidencia. El veredicto definitivo lo da el análisis de
interpretabilidad del modelo (sección 7).

::: figura eda-tasa-variables
Título: Tasa de malos por decil o categoría de cada variable (conjunto de entrenamiento)
Fuente: Elaboración propia a partir de results/data/eda/eda_inicial.json. Línea discontinua: tasa de malos de entrenamiento (16,3 %). Las variables de conteo con muchos ceros se agrupan por valor.
![Quince paneles con la tasa de malos por decil o categoría de cada variable](figures/eda-tasa-variables.png)
:::

::: tabla hipotesis
Título: Hipótesis y primera evidencia descriptiva (conjunto de entrenamiento)
Fuente: Elaboración propia a partir de results/data/eda/hipotesis_eda.json. Tasas de malos del primer y del último grupo ordenado (o de las dos categorías comparadas); ρ: correlación de Spearman entre el orden del grupo y su tasa.
| Id | Hipótesis | Evidencia | Primera lectura |
|---|---|---|---|
| H1 | Los créditos a 60 meses son más riesgosos que los de 36 | 31,9 % vs. 13,1 % | La apoya |
| H2 | A mayor razón deuda/ingreso, mayor riesgo | 13,1 % → 20,4 %; ρ = 1,00 | La apoya |
| H3 | A mayor ingreso, menor riesgo | 19,1 % → 12,5 %; ρ = −0,98 | La apoya |
| H4 | Más consultas recientes, más riesgo | 14,8 % → 26,1 %; ρ = 1,00 | La apoya |
| H5 | A mayor uso de los cupos rotativos, mayor riesgo | 10,9 % → 22,0 %; ρ = 1,00 | La apoya |
| H6 | Un historial crediticio más largo reduce el riesgo | 17,7 % → 14,8 %; ρ = −0,73 | La apoya, pero no es monótona |
| H7 | Quien arrienda es más riesgoso que quien paga hipoteca | 17,1 % vs. 15,4 % | La apoya (diferencia pequeña) |
| H8 | Pequeña empresa es más riesgosa que consolidación de deudas | 28,0 % vs. 17,3 % | La apoya |
| H9 | Los registros públicos negativos aumentan el riesgo | 16,0 % → 23,1 % | La apoya |
| H10 | A mayor monto, mayor riesgo | 14,7 % → 22,2 %; ρ = 0,76 | La apoya, pero no es monótona |
| H11 | Las moras recientes aumentan el riesgo | 16,1 % → 18,5 %; ρ = 1,00 | La apoya |
:::

Todas las hipótesis encuentran apoyo descriptivo, pero no con la misma forma. En H10 el riesgo es casi plano en los siete
primeros deciles del monto y salta alrededor de 6 puntos en los tres últimos; en H6 el primer decil (historiales muy cortos)
es el más riesgoso y el resto baja con oscilaciones. Una regresión logística con las variables continuas solo puede
representar tendencias monótonas en el logaritmo de las odds, así que estas formas son el primer lugar donde una red neuronal
podría aportar algo; lo retomamos en la discusión.

### 3.2 Correlaciones

La [[fig:post-correlacion]] muestra las correlaciones entre las variables numéricas ya transformadas. La más alta es la de
líneas abiertas con el total de líneas del historial (0,68), seguida de saldo rotativo con uso del cupo (0,48) y monto con
ingreso (0,44). Ninguna es tan alta como para duplicar información, pero conviene recordarlas al interpretar los coeficientes
de la regresión logística y los puntos de la scorecard, que reparten el efecto entre variables correlacionadas.

::: figura post-correlacion
Título: Correlaciones de Pearson entre las variables numéricas transformadas (conjunto de entrenamiento)
Fuente: Elaboración propia a partir de results/data/eda/post_preprocesamiento.json. El color codifica el valor absoluto; el número, el signo.
![Matriz de correlaciones de once variables numéricas](figures/post-correlacion.png)
:::

### 3.3 Los datos después del preprocesamiento

Repetimos el análisis sobre la matriz que realmente reciben los modelos, por conjunto:

- **Imputación y recortes.** Los vacíos son escasos: el 3,2 % de las personas no reporta antigüedad laboral (categoría
  `DESCONOCIDO`), el 0,15 % no tiene uso del cupo rotativo y el resto de variables tiene menos del 0,05 % de vacíos. El recorte
  a los percentiles 1 y 99 toca hasta el 1 % de cada cola; en las variables de conteo y en el monto (cuyo máximo, 35 000
  USD, coincide con su percentil 99) la cola superior se recorta menos o nada.
- **Distribución antes y después.** La [[fig:post-transformaciones]] compara los histogramas de entrenamiento antes y
  después de transformar. El logaritmo cumple su función con el ingreso, cuya asimetría pasa de 38,7 a 0,0, y la reduce en el
  monto (de 1,0 a −0,5). En el saldo rotativo deja un efecto que conviene conocer: el 1,5 % de las personas tiene saldo 0, y
  como $\log(1+0) = 0$ queda lejos del resto, ese grupo aparece aislado a la izquierda (la asimetría pasa de 11,8 a −3,1).

::: figura post-transformaciones
Título: Distribución de cuatro variables en entrenamiento, antes y después del preprocesamiento
Fuente: Elaboración propia a partir de results/data/eda/post_preprocesamiento.json (histogramas de 30 intervalos).
![Ocho histogramas: monto, ingreso, saldo rotativo y razón deuda/ingreso, crudos arriba y transformados abajo](figures/post-transformaciones.png)
:::
- **Estandarización.** En entrenamiento todas las variables numéricas quedan con media 0 y desviación 1. En validación y
  prueba no: por ejemplo, la razón deuda/ingreso tiene media 0,25 en validación y 0,32 en prueba, y los registros públicos
  0,44 y 0,78. Esa diferencia ya es una medida de cuánto cambiaron los solicitantes con el tiempo.
- **Estabilidad entre conjuntos.** Para cuantificar ese cambio usamos el índice de estabilidad poblacional (PSI), cuyas
  propiedades estadísticas estudian @yurdakul2020. Lo calculamos con los cortes de los deciles de entrenamiento; con $e_i$ y
  $a_i$ las proporciones de entrenamiento y del conjunto comparado en el intervalo $i$:

$$
\mathrm{PSI} = \sum_{i} (a_i - e_i)\,\ln\frac{a_i}{e_i}
= \sum_i a_i \ln\frac{a_i}{e_i} + \sum_i e_i \ln\frac{e_i}{a_i}
= D_{\mathrm{KL}}(a\,\|\,e) + D_{\mathrm{KL}}(e\,\|\,a),
$$

  es decir, la divergencia de Kullback–Leibler en los dos sentidos: vale 0 si las dos distribuciones son iguales y crece con
  la diferencia. No usamos umbrales de interpretación porque no encontramos una fuente que pudiéramos verificar.

  En las variables discretas cada valor es su propio intervalo (con cuantiles, los cortes colapsan y el índice da 0 aunque la
  población cambie; ver la sección 12, hallazgo H-01). La [[fig:post-psi]] muestra que los dos cambios más grandes son el
  destino del crédito (PSI de 0,27 en prueba) y los registros públicos negativos (0,27): la proporción de solicitantes con al
  menos un registro pasa de 4,1 % en entrenamiento a 12,8 % en validación y 19,5 % en prueba.

::: figura post-psi
Título: Índice de estabilidad poblacional de cada variable frente al conjunto de entrenamiento
Fuente: Elaboración propia a partir de results/data/eda/post_preprocesamiento.json.
![Barras horizontales del PSI de validación y de prueba para cada variable](figures/post-psi.png)
:::

- **Relación con el objetivo.** Las transformaciones son monótonas, así que no cambian el orden de los deciles ni la forma
  de la [[fig:eda-tasa-variables]]: las relaciones no lineales del monto y del historial crediticio siguen ahí después de
  transformar.

En conjunto, los modelos se entrenan con una población (2007–2012) que no es igual a la que evaluamos (2013 y 2014), y la
diferencia está medida variable por variable. Es la situación real de un banco que usa un modelo entrenado con el pasado.
