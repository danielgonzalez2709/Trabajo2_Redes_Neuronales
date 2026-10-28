<!-- Responsable: Camila y José (D). Enunciado (Anexo del núcleo): Parte 2, puntos 1–4; regla de comparación justa;
Entregable 1 (derivación de la pérdida). Cifras: results/data/lgd/{resumen_objetivo,descriptivo,summary,hpo_ensayos}.json,
results/data/lgd/semillas/ y results/data/lgd/pred_test.parquet (python run.py lgd --config configs/lgd/base.yaml).
Guía: blog/README.md -->
## 8. Pérdida dado el incumplimiento (LGD)

La probabilidad de incumplimiento (secciones 6 y 7) dice *si* una persona deja de pagar. Para estimar cuánto se pierde hace
falta una segunda pieza: la **pérdida dado el incumplimiento** (LGD, por *loss given default*), la fracción de lo adeudado que
el banco no recupera. En esta sección la construimos a partir de los campos del conjunto, la describimos, la modelamos con una
regresión lineal y con un perceptrón multicapa (MLP), y comparamos tanto los errores como la forma de las predicciones.

### 8.1 Variable objetivo

**Población.** Solo tiene sentido medir la pérdida cuando la cobranza ya terminó. Por eso la LGD se calcula sobre los créditos
castigados: *Charged Off* (42 475) y *Does not meet the credit policy. Status:Charged Off* (761), en total **43 236
créditos** de los 237 695 con desenlace conocido de la sección 2. Los otros dos estados que la sección 2.2 codifica como
malos, *Default* (832) y *Late (31-120 days)* (6 900), quedan fuera: su cobranza no ha terminado y, en estos datos, no
registran recuperaciones (Fuente: `results/data/lgd/resumen_objetivo.json`).

**Fórmula.** Para cada crédito castigado $i$ definimos la exposición en el incumplimiento (EAD) como el principal que quedaba
pendiente al castigo, y la LGD como la fracción de esa exposición que no se recuperó, neta del costo de cobranza:

$$
\mathrm{EAD}_i = F_i - P_i,
$$

$$
\mathrm{LGD}_i = \min\left\{1,\ \max\left\{0,\ 1 - \frac{R_i - C_i}{\mathrm{EAD}_i}\right\}\right\},
$$

donde $F_i$ es `funded_amnt`, $P_i$ es `total_rec_prncp`, $R_i$ es `recoveries` y $C_i$ es `collection_recovery_fee`
([[tab:lgd-campos]]). Si la EAD no es positiva o falta alguno de los cuatro campos, el crédito se excluye y se cuenta; en
estos datos no hubo ningún caso de cada tipo. Tampoco hubo créditos con la tarifa de cobranza mayor que la recuperación bruta,
ni con tarifa positiva y recuperación nula.

::: tabla lgd-campos
Título: Campos usados para construir la EAD y la LGD
Fuente: Elaboración propia; definiciones textuales de LCDataDictionary.xlsx (hoja LoanStats), copiadas en data/processed/diccionario_loanstats.csv.
| Campo | Definición del diccionario (textual) | Papel en la fórmula y por qué |
|---|---|---|
| `loan_status` | "Current status of the loan" | Define la población: solo los dos estados *Charged Off* |
| `funded_amnt` | "The total amount committed to that loan at that point in time." | $F_i$: el monto comprometido en el crédito; es la base de la exposición |
| `total_rec_prncp` | "Principal received to date" | $P_i$: principal ya pagado; restado de $F_i$ da el principal pendiente (EAD) |
| `recoveries` | "post charge off gross recovery" | $R_i$: lo recuperado después del castigo; el diccionario dice que es **bruto** |
| `collection_recovery_fee` | "post charge off collection fee" | $C_i$: costo de la cobranza; como $R_i$ es bruto, se resta para obtener la recuperación neta |
:::

Dos decisiones merecen explicación. Usamos `funded_amnt` porque su definición literal es "The total amount committed to
that loan at that point in time", es decir, el monto comprometido en el crédito, que es lo que queda expuesto. El diccionario
define `loan_amnt` como "The listed amount of the loan applied for by the borrower. If at some point in time, the credit
department reduces the loan amount, then it will be reflected in this value." En la población LGD los dos difieren: `loan_amnt`
es mayor que `funded_amnt` en 376 de los 43 236 créditos y nunca menor (Fuente: `data/processed/loans_lgd.parquet` unido con
`data/processed/montos.parquet`). Y restamos el principal recibido (`total_rec_prncp`) y no el total
de pagos (`total_pymnt`, "Payments received to date for total amount funded"), porque este último no separa cuánto de lo
pagado redujo el principal, que es lo que define la exposición.



**Limitaciones.** La fórmula es una aproximación contable y tiene límites que conviene declarar:

- **Sin descuento del dinero en el tiempo.** Un peso recuperado años después del castigo vale menos que uno recuperado al
  día siguiente; el conjunto no trae fechas de las recuperaciones, así que no podemos descontarlas.
- **Solo el costo de cobranza registrado.** `collection_recovery_fee` es el único costo disponible; los costos internos de
  la gestión no aparecen y la LGD real puede ser mayor.
- **Recuperaciones abiertas (censura).** En los castigos recientes la cobranza puede seguir en curso, de modo que una
  recuperación todavía baja se lee como una pérdida alta (sección 8.2).
- **Intereses no incluidos.** La EAD solo cuenta principal. Si una recuperación neta cubre además intereses o cargos, puede
  superar la EAD y la LGD sin recortar queda por debajo de 0. Es una explicación posible (no verificada campo a campo) de
  los **32 casos** con LGD negativa antes del recorte, que el recorte lleva a 0. Ningún caso quedó por encima de 1.
- **Masa de probabilidad en los extremos.** El 43,6 % de los créditos tiene LGD exactamente igual a 1 (sección 8.2), lo que
  condiciona la elección de la pérdida y de la activación de salida (sección 8.3).
- **Un supuesto no verificable.** La fórmula supone que lo cobrado después del castigo se registra en `recoveries` y no
  también en `total_rec_prncp`; el diccionario no lo aclara y no podemos comprobarlo porque no usamos fechas de pago (el
  archivo crudo trae `last_pymnt_d`, pero no está en los datos procesados).

### 8.2 La LGD observada

**Forma de la distribución.** La [[fig:lgd-histograma]] muestra que la LGD está muy concentrada cerca de 1: el 96,4 % de los
43 236 créditos tiene LGD de 0,8 o más y solo el 0,8 % está por debajo de 0,5. La media es 0,926 y la mediana 0,955. La masa en
los extremos es muy desigual: **18 865 créditos (43,6 %) tienen LGD = 1** y solo 36 (0,08 %) tienen LGD = 0, de los cuales 32
llegan ahí por el recorte. En estos datos LGD = 1 coincide exactamente con los créditos sin ninguna recuperación
(`alucinaciones/evidencias/D1_verificacion_lgd.txt`).

::: figura lgd-histograma
Título: Distribución de la LGD observada por conjunto y masa exacta en 0 y en 1
Fuente: Elaboración propia a partir de results/data/lgd/descriptivo.json (python run.py figures). Histograma de 20 intervalos; el último incluye LGD = 1.
![Histograma de la LGD por conjunto, concentrado entre 0,8 y 1, y barras con el porcentaje de créditos con LGD igual a 0 y a 1](figures/lgd-histograma.png)
:::

**Diferencias entre conjuntos.** La fracción con LGD = 1 crece de un conjunto al siguiente: 25,9 % en entrenamiento, 42,9 % en
validación y 62,7 % en prueba. La media se mueve menos (0,917, 0,918 y 0,945) y la mediana pasa de 0,943 en entrenamiento a
0,884 en validación y a 1,0 en prueba ([[tab:lgd-splits]]). Por año de originación, la fracción con LGD = 1 tiene forma de U
([[fig:lgd-por-anio]]): baja de 44,2 % en 2008 a 22,7 % en 2011 y sube a 42,9 % en 2013 y 62,7 % en 2014. En el panel de la LGD media el eje vertical no empieza en 0, así que las diferencias
entre años se ven más grandes de lo que son (de 0,911 en 2012 a 0,945 en 2014).

::: tabla lgd-splits
Título: LGD y EAD observadas por conjunto
Fuente: Elaboración propia a partir de results/data/lgd/descriptivo.json (observado.por_split) y results/data/lgd/resumen_objetivo.json. EAD en dólares.
| Conjunto | Años | Créditos | LGD media | LGD mediana | LGD = 1 | EAD mediana |
|---|---|---:|---:|---:|---:|---:|
| Entrenamiento | 2007–2012 | 14 419 | 0,917 | 0,943 | 25,9 % | 6 836 |
| Validación | 2013 | 14 836 | 0,918 | 0,884 | 42,9 % | 9 477 |
| Prueba | 2014 | 13 981 | 0,945 | 1,000 | 62,7 % | 11 180 |
| **Total** | | **43 236** | 0,926 | 0,955 | 43,6 % | |
:::

::: figura lgd-por-anio
Título: Fracción de créditos con LGD = 1 y LGD media por año de originación
Fuente: Elaboración propia a partir de results/data/lgd/descriptivo.json (observado.por_anio y fraccion_lgd_1_por_anio).
![Dos paneles de barras por año de 2007 a 2014, coloreadas según el conjunto: porcentaje con LGD igual a 1 y LGD media](figures/lgd-por-anio.png)
:::

Mantuvimos la partición temporal de la sección 2.4 y **aceptamos esta diferencia como una limitación documentada**. En 2013 y
2014 el patrón es **consistente con censura**: dentro de cada cosecha, los créditos que alcanzaron a pagar más principal antes
del castigo (posiblemente castigados más tarde; los datos que usamos no traen la fecha del castigo) tienen con más frecuencia recuperación nula, como esperaríamos si su
cobranza todavía no hubiera terminado cuando se publicó el conjunto (`alucinaciones/evidencias/D1_verificacion_lgd.txt`). No es
una causa probada: no descartamos un cambio en la composición de los créditos ni en el canal de cobranza, y el lado izquierdo
de la U (2007–2009) necesita otra explicación, porque esas cosechas tuvieron más tiempo para cerrar su cobranza. Sea cual sea la causa, el conjunto de prueba está desplazado hacia LGD = 1 frente
al de entrenamiento, y eso pesa en la lectura de la sección 8.5.

**EAD y plazo.** La [[fig:lgd-ead-plazo]] muestra que la exposición depende mucho del plazo y poco la LGD: la EAD mediana es
6 544 dólares en los créditos a 36 meses (28 104 créditos) y 15 705 en los de 60 meses (15 132), mientras que la LGD media es
casi igual (0,927 y 0,925). La EAD mediana también tiende a crecer con el año de originación (con una excepción: la de 2010 es menor
que la de 2009), y pasa de 6 836 dólares en entrenamiento a 11 180 en prueba.

::: figura lgd-ead-plazo
Título: EAD por plazo y por año de originación, y LGD media por plazo
Fuente: Elaboración propia a partir de results/data/lgd/descriptivo.json (observado.por_plazo y observado.por_anio). Barras: mediana; líneas: percentiles 25 a 75.
![Tres paneles: EAD mediana por plazo, LGD media por plazo y EAD mediana por año de originación](figures/lgd-ead-plazo.png)
:::

**Después del preprocesamiento.** El preprocesador de la LGD es el de la sección 2.5 (mismas variables, percentiles y
transformaciones) ajustado **solo con el entrenamiento de la población LGD**; lo comparten la regresión lineal y el MLP.
Repetimos sobre él el análisis de la sección 3.3 (Fuente: `results/data/lgd/descriptivo.json`, bloque `preprocesado`):

- **Transformaciones.** Como en la PD, el logaritmo reduce la asimetría del ingreso (de 4,71 a 0,01) y deja aislado el grupo
  con saldo rotativo 0 (la asimetría del saldo pasa de 17,95 a −3,20). La antigüedad laboral desconocida sube de 4,2 % en
  entrenamiento a 5,5 % en validación y 6,6 % en prueba.
- **Estabilidad.** Con el PSI de la sección 3.3, los mayores cambios del conjunto de **prueba** frente a entrenamiento son el
  destino del crédito (0,32; 0,21 en validación), la razón deuda/ingreso (0,23; 0,15 en validación), los registros públicos
  (0,16 en prueba, pero solo 0,04 en validación, donde no están entre los mayores) y las consultas recientes (0,13 en
  ambos). La proporción con al menos un registro público pasa de 5,8 % a 11,6 % y 18,2 %.
- **Relación con la LGD.** Es débil: la LGD media por grupo (deciles, o valores en las variables de conteo) de cada variable
  numérica se mueve en un rango estrecho, de 0,899 a 0,935 entre todos los grupos de todas las variables. Las tendencias más
  claras son las moras de los últimos dos años (de 0,918 a 0,899 en cuatro grupos; ρ de Spearman = −0,80), el ingreso (de
  0,920 en el decil más bajo a 0,900 en el más alto; ρ = −0,77) y las consultas recientes (de 0,915 a 0,932 en cuatro grupos; ρ = 1,00). Dos
  cautelas: en los registros públicos, que tras el recorte solo toman dos valores, la relación sale "monótona" por
  construcción (con dos grupos ρ es solo el signo de la diferencia), y no es evidencia; y el análisis no cubre las variables
  categóricas antigüedad laboral, vivienda y destino (el plazo sí, con la [[fig:lgd-ead-plazo]]).

Esta debilidad anticipa el resultado: con las variables de la originación hay poca señal para distinguir una LGD de otra.

### 8.3 Modelos

**Referencia lineal.** Una regresión lineal por mínimos cuadrados (`LinearRegression` de scikit-learn) sobre la misma matriz
preprocesada: 48 columnas de entrada y 49 parámetros. No restringe su salida a [0, 1] y no la recortamos; contamos cuántas
predicciones salen del intervalo y fueron **0** en validación, en prueba y en los 74 154 créditos del conjunto de prueba de la
PD (Fuente: `results/data/lgd/summary.json`).

**MLP con salida sigmoide.** La red produce un logit $z(\mathbf{x})$ y la LGD estimada es su sigmoide,

$$
\mu(\mathbf{x}) = \sigma\big(z(\mathbf{x})\big) = \frac{1}{1 + e^{-z(\mathbf{x})}}.
$$

La sigmoide tiene rango $(0, 1)$ (en aritmética exacta), el mismo de la LGD, así que toda predicción es una LGD admisible sin recortes. Además es la
**inversa del enlace logit**, $\operatorname{logit}(\mu) = \ln\frac{\mu}{1-\mu} = z$, que es el enlace de la cuasi-verosimilitud
que usamos como pérdida.

**La pérdida, por máxima (cuasi) verosimilitud.** La LGD es una fracción, $y \in [0, 1]$, no un 0 o un 1. Aun así se puede
usar la log-verosimilitud de Bernoulli como *cuasi*-verosimilitud: lo que se modela es la media condicional
$\mu(\mathbf{x}) = \mathbb{E}[y \mid \mathbf{x}]$, no la distribución completa de $y$. Para una observación,

$$
\ell(\mu; y) = y \ln \mu + (1 - y) \ln (1 - \mu), \qquad y \in [0, 1],
$$

y la pérdida es la cuasi-log-verosimilitud media cambiada de signo, es decir, la entropía cruzada con objetivo fraccional:

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \ln \mu_i + (1 - y_i) \ln (1 - \mu_i) \Big], \qquad \mu_i = \sigma(z_i).
$$

Con $\frac{d\sigma}{dz} = \mu(1-\mu)$, el gradiente respecto al logit queda

$$
\frac{\partial \ell}{\partial \mu} = \frac{y}{\mu} - \frac{1 - y}{1 - \mu} = \frac{y - \mu}{\mu(1 - \mu)},
\qquad
\frac{\partial (-\ell)}{\partial z} = -\frac{\partial \ell}{\partial \mu}\,\mu(1 - \mu) = \mu - y .
$$

Su esperanza condicional, $\mathbb{E}[\mu - y \mid \mathbf{x}] = \mu - \mathbb{E}[y \mid \mathbf{x}]$, se anula en
$\mu = \mathbb{E}[y \mid \mathbf{x}]$ (si $0 < \mathbb{E}[y \mid \mathbf{x}] < 1$), y como $-\ell$ es convexa en $\mu$ para $y \in [0, 1]$, ese es el minimizador
poblacional de la pérdida esperada. Es decir, con esta pérdida el MLP apunta a la **media condicional** de la LGD sin suponer
que $y$ sea Bernoulli. Para un modelo paramétrico con la media bien especificada, @papke1996 muestran que el estimador de
cuasi-máxima verosimilitud de Bernoulli es consistente; por eso la entropía cruzada es una pérdida legítima para una
respuesta fraccional, y no solo para objetivos binarios. El gradiente $\mu - y$ es el mismo de la regresión logística, solo
que ahora $y$ puede tomar cualquier valor del intervalo.

Hay un caso de borde que en estos datos no es marginal: cuando una LGD es exactamente 1 (el 43,6 % de los casos) o 0, el logit
óptimo para esa observación diverge ($z \to +\infty$ o $-\infty$). Lo contienen la penalización L2 y la parada temprana. La L2
se suma a la pérdida como

$$
L_{\lambda} = L + \frac{\lambda}{2} \sum_{l} \lVert W_l \rVert^2
$$

sobre las matrices de pesos $W_l$ (sin los sesgos). Su gradiente es $\lambda W_l$, así que equivale a un `weight_decay`
$= \lambda$ acoplado aplicado solo a las matrices de pesos (un `weight_decay` sobre todos los parámetros también decaería los
sesgos); no es el *weight decay* desacoplado de AdamW. Es la misma convención del MLP de la PD (Parte B). Una primera versión de
esta parte usaba $\lambda \sum \lVert W \rVert^2$ (equivalente a `weight_decay` $= 2\lambda$); la alineamos con la PD para que
el mismo rango de $\lambda$ en el espacio de búsqueda signifique la misma penalización en las dos partes, como pide la regla
de comparación justa. La parada temprana
mide la pérdida de validación tras cada época y, si no mejora en `paciencia` = 10 épocas seguidas, se detiene y restaura los
pesos de la mejor época.

Como contraste, la verosimilitud gaussiana con varianza constante lleva al error cuadrático medio,
$L = \frac{1}{n}\sum_i (\mu_i - y_i)^2$, que también apunta a la media condicional; pero supone varianza constante en un
objetivo acotado y, con salida sigmoide, su gradiente respecto al logit es $2(\mu - y)\,\mu(1 - \mu)$, que se apaga
cuando $\mu$ se acerca a 0 o a 1, que es hacia donde las LGD iguales a 1 (casi la mitad de los datos) empujan la predicción. El código la implementa como opción, pero no
la usamos en la búsqueda ni en los resultados.

**Búsqueda de hiperparámetros.** Optuna con el muestreador TPE [@akiba2019; @bergstra2011], **30 ensayos** (todos
completados) con semilla 0, sobre el espacio de la [[tab:lgd-busqueda]]. El criterio de selección es la **mínima pérdida de
validación del propio MLP** (la entropía cruzada media en validación, en la mejor época); el MAE de validación se registra al
lado pero no decide. Ganó el ensayo 29: cuatro capas ocultas de 32, 64, 64 y 32 neuronas con Leaky ReLU, *dropout* 0,392,
L2 0,00089, Adam con tasa de aprendizaje 0,0094 y lotes de 2 048, con pérdida de validación 0,28385 y MAE de validación
0,0760 (Fuente: `configs/lgd/mlp_final.yaml` y `results/data/lgd/hpo_ensayos.json`, bloque `mejor`).

**La superficie de búsqueda es casi plana.** De los 30 ensayos, **29 quedan a menos de 0,001 de la mejor pérdida de
validación** (entre 0,28385 y 0,28471), con configuraciones muy distintas: de una a cuatro capas, de 16 a 256 neuronas en la
primera capa, las tres activaciones y los dos optimizadores. El único que se separa es el ensayo 2 (SGD con tasa de
aprendizaje 0,00017), con 0,30321. Además, el ensayo con menor MAE de validación (el 9, con 0,0755) es el penúltimo por el
criterio de selección. Con diferencias de este tamaño, la configuración ganadora es una entre muchas casi equivalentes y no
conviene leer en ella una preferencia por una arquitectura. La tasa de aprendizaje elegida queda cerca del límite superior del
espacio (0,01): no sabemos si un rango más amplio habría elegido un valor mayor. Lo dejamos como limitación, porque ampliar el
rango rompería la regla de usar el mismo espacio de búsqueda que la PD (Parte B).

::: tabla lgd-busqueda
Título: Espacio de búsqueda del MLP de LGD y configuración elegida
Fuente: Elaboración propia a partir de results/data/lgd/hpo_ensayos.json (espacio, mejor) y configs/lgd/mlp_final.yaml. Escala logarítmica en L2 y en la tasa de aprendizaje.
| Hiperparámetro | Espacio | Elegido (ensayo 29) |
|---|---|---|
| Capas ocultas | 1 a 4 | 4 |
| Neuronas por capa | {16, 32, 64, 128, 256} | 32, 64, 64 y 32 |
| Activación oculta | ReLU, Leaky ReLU, tanh | Leaky ReLU |
| *Dropout* | [0; 0,5] | 0,392 |
| L2 ($\lambda$, en $\frac{\lambda}{2}\sum_l \lVert W_l \rVert^2$) | [10⁻⁶; 10⁻²] | 0,00089 |
| Optimizador | Adam, SGD (momento 0,9) | Adam |
| Tasa de aprendizaje | [10⁻⁴; 10⁻²] | 0,0094 |
| Tamaño del lote | {256, 512, 1024, 2048} | 2048 |
| Épocas máximas / paciencia | fijas: 200 / 10 | 200 / 10 |
| Salida y pérdida | fijas: sigmoide y entropía cruzada fraccional | ídem |
:::

**Treinta semillas.** Con la configuración elegida entrenamos el MLP con 30 semillas (0 a 29); cada una fija los pesos
iniciales, el *dropout* y el orden de los lotes. Las corridas se detuvieron entre las épocas 17 y 45, con la mejor época
entre la 7 y la 35 (Fuente: `results/data/lgd/semillas/`). La semilla 0 reproduce el ensayo 29 de la búsqueda: la misma
pérdida y el mismo MAE de validación. La regresión lineal es determinista y se ajusta una sola vez.
Para las figuras de predicciones usamos la **semilla mediana**: la semilla 22, la de MAE de **validación** mediano (al ordenar
las 30 semillas por MAE de validación y, en empate, por semilla, la de la posición $\lfloor (n-1)/2 \rfloor = 14$ contando
desde 0; Fuente:
`results/data/lgd/summary.json`, campos `mlp.semilla_mediana` y `mlp.criterio_semilla_mediana`). La elegimos con validación y
no con prueba, como en la PD, porque el conjunto de prueba solo mide y no puede intervenir en ninguna decisión del modelo.

### 8.4 Resultados

La [[tab:lgd-resultados]] reúne MAE y RMSE de validación y de prueba con las 30 semillas del MLP, junto con dos referencias
constantes que no usan ninguna variable: la media y la mediana de la LGD de entrenamiento. La [[tab:lgd-presupuesto]] resume
el costo de cada familia.

::: tabla lgd-resultados
Título: Error absoluto medio (MAE) y raíz del error cuadrático medio (RMSE) de la LGD en validación y en prueba
Fuente: Elaboración propia a partir de results/data/lgd/summary.json. Media, desviación estándar, mejor y peor valor entre corridas; la desviación se muestra con 5 decimales. Las constantes son un solo valor calculado con el entrenamiento (media = 0,9169; mediana = 0,9432).
| Modelo | Métrica | Media | Desv. | Mejor | Peor | n |
|---|---|---:|---:|---:|---:|---:|
| Regresión lineal | MAE validación | 0,0759 | 0 | 0,0759 | 0,0759 | 1 |
| | RMSE validación | 0,0928 | 0 | 0,0928 | 0,0928 | 1 |
| | MAE prueba | 0,0771 | 0 | 0,0771 | 0,0771 | 1 |
| | RMSE prueba | 0,0866 | 0 | 0,0866 | 0,0866 | 1 |
| MLP | MAE validación | 0,0761 | 0,00008 | 0,0760 | 0,0763 | 30 |
| | RMSE validación | 0,0924 | 0,00005 | 0,0923 | 0,0926 | 30 |
| | MAE prueba | 0,0769 | 0,00037 | 0,0761 | 0,0776 | 30 |
| | RMSE prueba | 0,0855 | 0,00040 | 0,0848 | 0,0864 | 30 |
| Constante: media de entrenamiento | MAE validación / prueba | 0,0764 / 0,0775 | — | — | — | — |
| | RMSE validación / prueba | 0,0925 / 0,0859 | — | — | — | — |
| Constante: mediana de entrenamiento | MAE validación / prueba | 0,0772 / 0,0699 | — | — | — | — |
| | RMSE validación / prueba | 0,0959 / 0,0811 | — | — | — | — |
:::

::: tabla lgd-presupuesto
Título: Parámetros entrenables, configuraciones evaluadas y tiempo por familia de modelos de LGD
Fuente: Elaboración propia a partir de results/data/lgd/summary.json (lineal.tiempo_total_s; mlp.tiempo_busqueda_s, mlp.tiempo_semillas_s y mlp.tiempo_total_s). El tiempo del MLP incluye la búsqueda (126,1 s) y las 30 semillas (39,2 s).
| Modelo | Parámetros entrenables | Configuraciones evaluadas | Tiempo total |
|---|---:|---:|---:|
| Regresión lineal | 49 | 1 | 0,03 s |
| MLP (32-64-64-32, Leaky ReLU) | 9 953 | 30 | 165,3 s |
| Constante: media o mediana | 1 | 1 | — |
:::

Lo que muestran las dos tablas:

- **El MLP y la regresión lineal quedan a milésimas, y el orden cambia de validación a prueba.** En MAE de validación la
  lineal (0,07587) es mejor que **las 30 semillas** del MLP (media 0,07614); en MAE de prueba se invierte: la media del MLP
  (0,07686) mejora a la lineal (0,07712) y **23 de las 30 semillas** la mejoran. La diferencia de prueba, 0,00026, es menor
  que la desviación entre semillas (0,00037), y el MAE de la lineal cae dentro del rango de las 30 semillas. En RMSE el MLP es
  mejor en las 30 semillas tanto en validación (0,0924 frente a 0,0928) como en prueba (0,0855 frente a 0,0866), pero por
  diferencias de 0,0003 y 0,0011 entre medias (Fuente: `results/data/lgd/semillas/`).
- **Ninguno de los dos supera con claridad a una constante.** La media de entrenamiento como predicción única obtiene MAE de
  prueba 0,0775 y RMSE 0,0859, a menos de una milésima de ambos modelos; en RMSE la lineal es incluso algo peor que esa
  constante en validación (0,0928 frente a 0,0925) y en prueba (0,0866 frente a 0,0859). La mediana de entrenamiento obtiene
  el **menor MAE y el menor RMSE de prueba** de toda la tabla (0,0699 y 0,0811), y ninguna de las 30 semillas la alcanza en
  ninguna de las dos métricas; en validación, en cambio, es la peor de las cuatro opciones (0,0772 y 0,0959). La sección 8.5
  explica por qué.
- **Variabilidad.** Entre semillas, el MAE de prueba del MLP va de 0,0761 a 0,0776 ([[fig:lgd-semillas]]). Esa dispersión,
  de 0,0015, es casi seis veces la diferencia entre la media del MLP y la lineal (0,00026).
- **Costo.** El MLP tiene 9 953 parámetros, unas 200 veces los 49 de la lineal, y evaluó 30 configuraciones frente a una; aun
  así, todo su ciclo (búsqueda más semillas) tomó 165,3 s, frente a 0,03 s de la lineal.

::: figura lgd-semillas
Título: MAE de prueba de cada una de las 30 semillas del MLP frente a la regresión lineal y a las dos constantes
Fuente: Elaboración propia a partir de results/data/lgd/semillas/ y results/data/lgd/summary.json. El círculo marca la semilla mediana por MAE de validación (22).
![Puntos del MAE de prueba por semilla entre 0,0761 y 0,0776, con líneas horizontales de la lineal, la media y la mediana de entrenamiento](figures/lgd-semillas.png)
:::

### 8.5 Distribución de las predicciones

El enunciado pide comparar la distribución de las predicciones con la observada, y es aquí donde se entiende la tabla anterior.
La [[fig:lgd-distribucion-predicha]] muestra que **ambos modelos predicen casi lo mismo para todos**: en prueba, la LGD
observada tiene desviación estándar 0,081 y va de 0 a 1, mientras que las predicciones de la lineal van de 0,863 a 0,981
(desviación 0,014) y las del MLP de la semilla mediana (22) en un rango aún más estrecho, **de 0,902 a 0,938** (desviación
0,006). Las dos se centran en la media de entrenamiento (0,917): sus medias en prueba son 0,917 y 0,918 (Fuente:
`results/data/lgd/pred_test.parquet`; desviaciones poblacionales). El MLP, en particular, se comporta casi como una constante:
todas sus predicciones caben en un intervalo de 0,036.

::: figura lgd-distribucion-predicha
Título: Distribución de la LGD predicha por la regresión lineal y por el MLP frente a la observada (prueba)
Fuente: Elaboración propia a partir de results/data/lgd/pred_test.parquet (MLP: semilla mediana) y results/data/lgd/summary.json (constantes). A la derecha, ampliación del intervalo [0,8; 1].
![Histogramas de la LGD observada y predicha en prueba: las predicciones se concentran entre 0,86 y 0,98 (las del MLP entre 0,90 y 0,94) mientras la observada tiene más de la mitad de la masa en 1](figures/lgd-distribucion-predicha.png)
:::

El gráfico cuantil-cuantil ([[fig:lgd-qq]]) lo confirma: entre los percentiles 1 y 99, los cuantiles observados recorren de
0,76 a 1, y los predichos quedan casi planos: entre 0,889 y 0,956 los de la lineal y entre 0,903 y 0,930 los del MLP. Ninguno de los dos modelos reproduce la masa en 1 ni la
cola izquierda; ambos predicen, en esencia, una media ligeramente ajustada por las variables.

::: figura lgd-qq
Título: Cuantiles de la LGD predicha frente a los de la observada en prueba (percentiles 1 a 99)
Fuente: Elaboración propia a partir de results/data/lgd/pred_test.parquet (MLP: semilla mediana).
![Gráfico Q-Q: las curvas de ambos modelos son casi horizontales cerca de 0,9 y cruzan la diagonal](figures/lgd-qq.png)
:::

La [[fig:lgd-error-por-banda]] descompone el error por banda de LGD observada (las bandas son de presentación, no del modelo).
Los dos modelos tienen errores casi idénticos en todas las bandas: grandes en las pocas LGD bajas (MAE de 0,758 y 0,759 en las 22 de
[0; 0,5] y de 0,195 en las 233 de (0,5; 0,8]), pequeños en el grueso intermedio (0,059 en 4 931 créditos) y de 0,082 en los
8 795 con LGD en (0,99; 1]. La constante mediana es peor en las tres primeras bandas, pero mejor en la última (0,057), que en
prueba concentra el 63 % de los créditos (Fuente: `results/data/lgd/pred_test.parquet`, función `tabla_error_por_banda` de
`src/viz/figs_lgd.py`).

::: figura lgd-error-por-banda
Título: MAE por banda de LGD observada en prueba para la regresión lineal, el MLP y la constante mediana
Fuente: Elaboración propia a partir de results/data/lgd/pred_test.parquet (MLP: semilla mediana) y results/data/lgd/summary.json. Bandas: [0; 0,5], (0,5; 0,8], (0,8; 0,99] y (0,99; 1].
![Barras de MAE por banda: altas en LGD baja, cerca de 0,06 en la banda (0,8; 0,99] y cerca de 0,08 en la banda (0,99; 1], donde la constante mediana es la más baja](figures/lgd-error-por-banda.png)
:::

**Por qué gana la constante mediana en prueba.** Hay dos razones que se suman y ninguna depende de una causa no probada:

1. **Qué estima cada uno.** La entropía cruzada lleva al MLP hacia la media condicional, y los mínimos cuadrados llevan a la
   lineal a su mejor aproximación lineal de esa media; el valor constante que minimiza el error absoluto, en cambio, es la mediana. Con una distribución tan
   asimétrica, media y mediana difieren (0,917 y 0,943 en entrenamiento).
2. **El desplazamiento de prueba.** La prueba está corrida hacia 1 (media 0,945, mediana 1,0, 62,7 % en LGD = 1; sección 8.2).
   La constante 0,943 queda más cerca de esa población que las predicciones de los modelos, ancladas en la media de
   entrenamiento (0,917). Por eso también gana en RMSE: la constante 0,943 está a 0,002 de la media de prueba. En validación,
   cuya mediana es 0,884, el mismo valor queda lejos y la constante mediana resulta la peor.

En otras palabras, la ventaja de la constante mediana en prueba no la da una mejor comprensión del riesgo, sino que el
conjunto de prueba se parece más a ese número que al de entrenamiento.

### 8.6 Qué aportó la red en la LGD

Con las 15 variables de la originación, el MLP no aportó de forma relevante frente a la regresión lineal: métricas a
milésimas de distancia (en MAE, dentro de la variabilidad entre semillas; en RMSE, mejor en las 30 semillas pero por 0,0003 en
validación y 0,0011 en prueba), unas 200 veces más parámetros, 30 configuraciones evaluadas frente a una y una interpretación
más difícil (la lineal se lee coeficiente por coeficiente; el MLP necesitaría herramientas adicionales). Las métricas de ambos modelos
quedan a menos de una milésima de las de predecir la media de entrenamiento, y una constante (la mediana) los supera en el conjunto de
prueba por el desplazamiento de su distribución. Lo que sí aportó el diseño de la red es una garantía de forma: la salida
sigmoide mantiene toda predicción en (0, 1) (en aritmética exacta) y la entropía cruzada fraccional apunta a la media condicional, que es lo que
necesita la pérdida esperada de la sección 9. La causa del poco poder predictivo (poca señal en las variables de la
originación, censura en las cosechas recientes o ambas) queda abierta; la discusión de conjunto está en la sección 10.

**Cuatro hallazgos y lo que permiten concluir.** Los resultados de esta sección encajan entre sí y con lo que la sección 8.2
documenta sobre la masa en LGD = 1 y la censura, aunque ninguno prueba por sí solo una causa:

1. **La constante mediana de entrenamiento gana en MAE y RMSE de prueba** (0,0699 y 0,0811) a la lineal y a las 30 semillas
   del MLP. Como se vio en la sección 8.5, una explicación posible, no contrastada, tiene dos partes: la mediana es el valor
   que minimiza el error absoluto, lo que explica la ventaja en MAE; y la prueba está desplazada hacia 1 (62,7 % con LGD = 1
   frente a 25,9 % en entrenamiento), de modo que la mediana de entrenamiento (0,943) queda más cerca de la media de prueba
   (0,945) que la media de entrenamiento (0,917), lo que explica también la ventaja en RMSE. Ese desplazamiento es
   consistente con censura, pero no se atribuye a ella de forma probada.
2. **La superficie de búsqueda es casi plana**: 29 de 30 ensayos a menos de 0,001 de la mejor pérdida de validación. Si las
   variables aportan poca señal sobre la LGD (sección 8.2), cabe esperar que arquitecturas muy distintas lleguen a casi la
   misma pérdida; no comprobamos que esa sea la razón.
3. **El MLP predice en un rango estrecho** (de 0,902 a 0,938 en prueba), más estrecho que el de la lineal y cercano a una
   constante. Es coherente con una pérdida que apunta a la media condicional cuando esta varía poco con las variables; no
   separamos cuánto se debe a la regularización elegida (*dropout* 0,392 y L2) y cuánto a los datos.
4. **El orden entre el MLP y la lineal cambia entre validación y prueba** en MAE: en validación la lineal es mejor que las 30
   semillas; en prueba 23 de 30 semillas la mejoran. Con diferencias del orden de 10⁻⁴ y una prueba cuya distribución difiere
   de la de validación (42,9 % frente a 62,7 % con LGD = 1), este cambio de orden no indica que un modelo generalice mejor que
   el otro; muestra que, en este caso, validación no anticipa el orden en prueba.
