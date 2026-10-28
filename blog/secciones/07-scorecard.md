<!-- Responsable: Juan José (C). Enunciado (Anexo del núcleo): derivación de la scorecard, variables de riesgo y cierre de hipótesis. Guía de sintaxis: blog/README.md -->
<!-- C7 (27-oct): cifras de results/data/scorecard/{scorecard,scorecard_mlp,shap_mlp,hipotesis}.json (python run.py scorecard
y python run.py scorecard --config configs/scorecard/mlp.yaml, que al final escribe hipotesis.json) y de
results/data/eda/post_preprocesamiento.json (python run.py eda --config configs/datos.yaml). Fórmulas: celdas markdown de
src/scorecard/{scorecard,puntos,shap_pd,hipotesis}.py. Figuras: python run.py figures (src/viz/figs_scorecard.py). -->
## 7. Scorecard y variables de riesgo

### 7.1 Qué es una scorecard y para qué sirve

Una scorecard resume el modelo de PD en un **puntaje**: una medida numérica de la solvencia crediticia (*creditworthiness*)
del solicitante en el momento de la solicitud, calculada solo con las 15 variables de originación. Un puntaje **mayor**
significa **menor** riesgo. El banco lo usa para ordenar a los solicitantes y decidir: aprobar, rechazar o revisar a mano
según el tramo de puntaje en el que cae cada uno. Los puntos de corte (desde qué puntaje se aprueba, se rechaza o se revisa
a mano) son una decisión de negocio que depende del apetito de riesgo del banco; este trabajo no los fija.

Construimos el puntaje como una función lineal del logaritmo de las odds, fijada con tres parámetros: un puntaje base, unas
odds de referencia y los puntos para duplicar las odds (PDO) [@siddiqi2017]. Lo hacemos para los dos modelos de PD de la
sección 6: la regresión logística y el MLP (semilla mediana). Los dos van a la aplicación web.

### 7.2 De la probabilidad al puntaje

Sea $p$ la PD ($y = 1$ es malo). Las **odds de buen pagador** (bueno:malo) son

$$
o = \frac{1 - p}{p}, \qquad \ln o = \ln\frac{1 - p}{p} = -\ln\frac{p}{1 - p} = -\operatorname{logit}(p).
$$

El puntaje es lineal en $\ln o$:

$$
\text{score}(o) = \text{offset} + \text{factor}\cdot\ln o .
$$

Se imponen dos condiciones: (1) $\text{score}(\text{odds}_{\text{base}}) = \text{base}$ y (2)
$\text{score}(2o) - \text{score}(o) = \text{PDO}$. De (2), $\text{factor}\,[\ln(2o) - \ln o] = \text{factor}\cdot\ln 2 =
\text{PDO}$; de (1), $\text{offset} + \text{factor}\cdot\ln(\text{odds}_{\text{base}}) = \text{base}$. Luego

$$
\text{factor} = \frac{\text{PDO}}{\ln 2}, \qquad \text{offset} = \text{base} - \text{factor}\cdot\ln(\text{odds}_{\text{base}}),
\qquad \text{score} = \text{offset} - \text{factor}\cdot\operatorname{logit}(p).
$$

Las dos condiciones valen para **cualquier** $o$: sustituyendo el offset,
$\text{score}(o) = \text{base} + \text{factor}\cdot\ln(o/\text{odds}_{\text{base}})$, que vale $\text{base}$ en
$o = \text{odds}_{\text{base}}$, y al duplicar $o$ suma $\text{factor}\cdot\ln 2 = \text{PDO}$ sin depender de $o$. El código
calcula el puntaje desde el logit del modelo (no desde $p$), en float64 y sin redondear.

Usamos base 600 con odds de referencia 50:1 y PDO 20. Entonces el factor es 28,85 y el offset 487,12. Un cliente con
$p = 0{,}5$ (odds 1:1, logit 0) recibe exactamente el offset, 487,12 puntos; uno con $p = 1/51$ (odds 50:1) recibe 600
([[fig:sc-transformacion]]). Los tres parámetros se cambian sin tocar el código, con `--set scorecard.*`. Con
`--set scorecard.pdo=40` el ancla sigue en 600 puntos con $p = 1/51$, el factor se duplica (57,71) y el offset baja a
$600 - 57{,}71\cdot\ln 50 = 374{,}25$: la pendiente se duplica, pero el **orden** de los clientes no cambia, porque el
puntaje es estrictamente decreciente en $p$ con cualquier PDO positivo.

::: figura sc-transformacion
Título: Puntaje de la scorecard en función de la PD con PDO 20 y PDO 40
Fuente: Elaboración propia a partir de results/data/scorecard/scorecard.json (base 600, odds 50:1, PDO 20) y de la misma transformación con PDO 40 (src/scorecard/scorecard.py).
![Dos curvas decrecientes del puntaje frente a la PD en escala logarítmica que se cruzan en el ancla de 600 puntos (PD de 0,0196); en PD = 0,5 la de PDO 20 vale 487,12 y la de PDO 40, 374,25](figures/sc-transformacion.png)
:::

Tres errores frecuentes cambian el resultado sin que el código falle. Usar odds malo:bueno, $p/(1 - p)$, cambia el **signo**:
el puntaje crecería con el riesgo. Mezclar bases, por ejemplo $\text{factor} = \text{PDO}/\ln 2$ con $\log_{10}$ en el
puntaje, hace que duplicar las odds sume $\text{PDO}\cdot\log_{10} 2/\ln 2 \approx 0{,}434\cdot\text{PDO}$ (8,69 puntos con
PDO 20) y no el PDO; si se usa $\log_{10}$ de forma consistente, también en el factor, el puntaje no cambia. Y el PDO **no** son "puntos por cada 1 % de PD": son los puntos que se suman al duplicar las odds, porque el puntaje es
lineal en $\ln o$, no en $p$.

### 7.3 Puntos por variable de la regresión logística

La logística es lineal en el logit sobre las columnas $x_j$ del preprocesador:
$\operatorname{logit}(p) = \beta_0 + \sum_j \beta_j x_j$. Sustituyendo en el puntaje,

$$
\text{score} = \underbrace{\text{offset} - \text{factor}\,\beta_0}_{\text{puntos base}}
+ \sum_j \underbrace{\left(-\text{factor}\,\beta_j\,x_j\right)}_{\text{puntos}_j} .
$$

La suma es exacta porque el puntaje es afín en el logit y el logit es lineal en $x$. En los tres conjuntos, la diferencia
máxima entre la suma de puntos y el puntaje es $1{,}1 \times 10^{-13}$, solo redondeo de float64. Como el factor es
positivo, un $\beta_j > 0$ (más log-odds de malo) da **menos** puntos.

Los puntos base valen 525,96. No son el puntaje medio de la población (538,45 en entrenamiento): son el puntaje de un
solicitante de referencia con cada numérica en su media de entrenamiento ($x_j = 0$ tras estandarizar, luego 0 puntos) y
cada categórica en el promedio no ponderado de los puntos de sus categorías.

**Numéricas.** El preprocesador imputa con la mediana, recorta a $[l, h]$ = percentiles 1 y 99 de entrenamiento, aplica
`log1p` si corresponde y estandariza con $\mu$ y $\sigma$ de entrenamiento. Con
$t(v) = \text{log1p}^{\,[\text{si aplica}]}\big(\operatorname{recorte}(\operatorname{imputación}(v))\big)$,

$$
\text{puntos}(v) = -\text{factor}\,\beta\;\frac{t(v) - \mu}{\sigma}.
$$

Es lineal en el valor estandarizado, pero no en el crudo: es plana fuera del recorte y logarítmica cuando hay `log1p`.

**Categóricas.** Una categórica con categorías $c_1, \dots, c_K$ ocupa $K$ columnas one-hot y en cada fila solo una vale 1,
así que $\sum_k -\text{factor}\,\beta_{c_k}\,x_{c_k} = -\text{factor}\,\beta_{c^\ast}$, con $c^\ast$ la categoría activa. Las
columnas one-hot de una variable suman 1 y son colineales con el intercepto: sumar $\delta$ a todos los $\beta_{c_k}$ y
restarlo a $\beta_0$ da el mismo logit. La penalización L2 de la logística no actúa sobre el intercepto y deja
$\sum_k \beta_{c_k} \approx 0$ en cada categórica, pero el nivel de cada categoría sigue dependiendo de esa convención: **solo
las diferencias** de puntos entre categorías de una misma variable se interpretan.

La [[tab:sc-puntos-logistica]] muestra los puntos en siete percentiles de entrenamiento de dos numéricas y en las dos
categorías del plazo. Con el ingreso anual los puntos de la variable van de −22,90 puntos en el percentil 1 a +22,96 en el 99; con el uso de
los cupos rotativos, de +20,06 a −16,36. Un plazo de 60 meses da 28,93 puntos menos que uno de 36. El peso de cada variable,
medido como su rango de puntos, está en la [[tab:discusion-scorecard]] de la sección 10.

::: tabla sc-puntos-logistica
Título: Puntos de la scorecard de la regresión logística en percentiles de entrenamiento (ingreso anual y uso de los cupos rotativos) y en las categorías del plazo
Fuente: Elaboración propia a partir de results/data/scorecard/scorecard.json (puntos_por_variable: tabla y categorias). Percentiles de los valores crudos no vacíos de entrenamiento; PDO 20.
| Variable | Percentil o categoría | Valor crudo | Puntos |
|---|---|---:|---:|
| `annual_inc` | p1 | 15 056,58 | −22,90 |
| `annual_inc` | p5 | 25 000 | −14,37 |
| `annual_inc` | p25 | 41 600 | −5,81 |
| `annual_inc` | p50 | 60 000 | 0,35 |
| `annual_inc` | p75 | 82 500 | 5,71 |
| `annual_inc` | p95 | 140 000 | 14,61 |
| `annual_inc` | p99 | 230 000 | 22,96 |
| `revol_util` | p1 | 0,0 | 20,06 |
| `revol_util` | p5 | 6,1 | 17,78 |
| `revol_util` | p25 | 33,8 | 7,44 |
| `revol_util` | p50 | 56,4 | −0,99 |
| `revol_util` | p75 | 75,6 | −8,15 |
| `revol_util` | p95 | 92,9 | −14,61 |
| `revol_util` | p99 | 97,6 | −16,36 |
| `term` | 36 meses (83,0 % de entrenamiento) | — | 14,46 |
| `term` | 60 meses (17,0 % de entrenamiento) | — | −14,46 |
:::

Los percentiles de la [[tab:sc-puntos-logistica]] se calculan sobre los valores crudos no vacíos de entrenamiento, mientras que
el recorte del preprocesador se calcula después de imputar la mediana. Por eso el p1 del ingreso (15 056,58) queda apenas por
debajo del límite inferior del recorte (15 058,54) y recibe los mismos puntos que ese límite.

La [[fig:sc-puntos-variable]] dibuja la función de puntos de la logística para el ingreso anual, el uso de los cupos y las
consultas de los últimos 6 meses, junto con los puntos medios del MLP por tramo (sección 7.4). En el ingreso la curva es
logarítmica y se aplana a partir de 230 000 USD (el recorte); en el uso de los cupos y en las consultas es una recta que se
aplana fuera del recorte (97,6 % y 5 consultas). Los puntos medios del MLP siguen la misma dirección en las tres variables.

::: figura sc-puntos-variable
Título: Puntos por variable de la regresión logística (curva) y del MLP (media por tramo) según el valor crudo
Fuente: Elaboración propia a partir de results/data/scorecard/scorecard.json y results/data/scorecard/scorecard_mlp.json (tramos con bordes en percentiles de entrenamiento; medias sobre los 1 000 créditos de prueba explicados con SHAP).
![Tres paneles: ingreso anual, uso de los cupos rotativos y consultas de crédito en los últimos 6 meses; la curva de la logística y los segmentos horizontales del MLP bajan o suben en la misma dirección](figures/sc-puntos-variable.png)
:::

### 7.4 Interpretabilidad del MLP con SHAP

El MLP no es lineal y sus pesos no se leen como puntos. Para darle una scorecard usamos los valores de Shapley (SHAP)
[@lundberg2017] del **logit** de la red, $f(x) = \operatorname{logit} p(x)$, con las $M = 15$ variables como jugadores. Para
una coalición $S \subseteq N = \{1, \dots, M\}$, el juego **intervencional** con un fondo $\{b^{(1)}, \dots, b^{(K)}\}$ de
$K = 100$ filas de **entrenamiento** es

$$
v_x(S) = \frac{1}{K}\sum_{k=1}^{K} f\big(x_S,\, b^{(k)}_{\bar S}\big):
$$

las variables de $S$ toman el valor del cliente y las demás el de cada fila del fondo. El valor de Shapley de la variable $j$
es su aporte marginal medio:

$$
\varphi_j(x) = \sum_{S \subseteq N\setminus\{j\}} \frac{|S|!\,(M - |S| - 1)!}{M!}
\big[v_x(S \cup \{j\}) - v_x(S)\big].
$$

Por la propiedad de **eficiencia**, con $\varphi_0 = v_x(\varnothing) = \mathbb{E}_{\text{fondo}}[f]$,

$$
f(x) = \varphi_0 + \sum_{j=1}^{M} \varphi_j(x).
$$

En nuestro fondo, $\varphi_0 = -1{,}4367$. Explicamos el logit y no la PD porque el puntaje es afín en el logit:
sustituyendo la eficiencia, la suma de puntos reproduce el puntaje exacto. Si explicáramos la probabilidad, $p$ sería la
suma de los aportes, pero $\text{offset} - \text{factor}\cdot\ln\frac{p}{1 - p}$ no es lineal en $p$ y los puntos no
sumarían el puntaje.

Los jugadores son las 15 variables, no las 48 columnas del preprocesador. Si cada columna one-hot fuera un jugador, ocultar
`term = 60 months` sin ocultar `term = 36 months` crearía clientes con dos plazos (o ninguno), que el modelo nunca vio; por
eso una categórica se oculta entera. La eficiencia sigue valiendo: solo cambia qué es un jugador.

**Cálculo.** Usamos el explainer de permutaciones de la biblioteca shap 0.54.1. Recorre cada permutación de las variables de
ida y de vuelta (muestreo antitético) y, en cada recorrido, los aportes marginales forman una suma telescópica que vale
exactamente $v_x(N) - v_x(\varnothing) = f(x) - \varphi_0$: la eficiencia se cumple **por construcción**, con cualquier número
de permutaciones. Con 15 jugadores, un presupuesto de 500 evaluaciones da $\lfloor 500 / 31 \rfloor = 16$ permutaciones por
fila. Explicamos 1 000 créditos de prueba. Descartamos los demás explainers de shap: DeepExplainer aproxima los valores con
reglas de DeepLIFT por nodo; GradientExplainer calcula gradientes esperados, que solo suman $f(x) - \mathbb{E}[f]$ en el límite
del muestreo; KernelExplainer estima los $\varphi$ con una regresión ponderada sobre coaliciones muestreadas y, por defecto,
selecciona como máximo 10 variables (con 15 jugadores dejaría al menos 5 en cero); y ExactExplainer enumera las $2^{15} = 32\,768$ coaliciones por
cliente, demasiado costoso para 1 000 clientes. Para que la telescopia fuera exacta tuvimos que hacer que shap comparara los
valores del cliente y del fondo por igualdad exacta (sección 12, hallazgo H-24).

**Puntos del MLP.** De la eficiencia,

$$
\text{score}(x) = \underbrace{\text{offset} - \text{factor}\,\varphi_0}_{\text{puntos base}}
+ \sum_j \underbrace{\left(-\text{factor}\,\varphi_j(x)\right)}_{\text{puntos}_j} .
$$

Los puntos base del MLP valen 528,58. El error máximo de aditividad (puntos base más puntos frente al puntaje) es
$2{,}3 \times 10^{-13}$ y el de eficiencia de SHAP, $6{,}2 \times 10^{-15}$. A diferencia de la logística, los puntos de una
variable **no** son una función de su valor: dependen del resto del cliente, porque la red tiene interacciones. Por eso la
[[fig:sc-puntos-variable]] muestra para el MLP la media de los puntos por tramo de valor, no una curva.

**Verificaciones.** Corrimos el mismo cálculo con la logística, cuya solución exacta se conoce: en un modelo aditivo,
$\varphi_j = \sum_{c \in j} \beta_c\,(x_c - \bar x_c)$ sobre las columnas de la variable, con $\bar x_c$ la media del fondo.
La diferencia máxima con la forma cerrada fue $1{,}6 \times 10^{-15}$ en 200 filas. Para medir la **estabilidad**, volvimos a
explicar esas 200 filas con otro fondo de entrenamiento y otras permutaciones (semilla base + 1): la correlación de Spearman
entre las dos importancias globales es 0,986 y las siete primeras variables salen en el mismo orden.

**Limitaciones.** SHAP intervencional trata las variables como independientes: al cambiar el monto por el de otra fila sin
cambiar el ingreso, evalúa el modelo en combinaciones poco realistas, y con variables correlacionadas el reparto depende de
esa extrapolación [@aas2021]. El contrapunto es que, para medir cuánto **usa** el modelo cada variable, la noción
intervencional es la adecuada [@janzing2020], y eso es lo que queremos para la scorecard. Además, con 16 permutaciones el
reparto entre variables es una estimación; la eficiencia, en cambio, es exacta.

**Contraste: importancia por permutación.** Para cada variable $j$ y repetición $r = 1, \dots, k$ permutamos sus valores
entre las filas de prueba (la categórica entera) y medimos

$$
\Delta\text{AUC}_{j,r} = \text{AUC}_0 - \text{AUC}\big(y, \hat p(Z^{\pi_r}_j)\big), \qquad
\Delta\text{LL}_{j,r} = \text{LL}\big(y, \hat p(Z^{\pi_r}_j)\big) - \text{LL}_0,
$$

con $k = 5$ y la media y la desviación muestral de las repeticiones, para los dos modelos y con las mismas permutaciones. La
idea viene de los bosques aleatorios [@breiman2001]; @fisher2019 la formalizan para cualquier modelo como *model reliance*,
que definen como un cociente, y mencionan también la versión en diferencia que usamos. Las dos medidas responden preguntas
distintas: la permutación es **global y sobre el desempeño** (cuánto empeora la discriminación si se rompe el vínculo de la
variable con $y$); SHAP es **local y sobre la salida** (cuánto mueve la variable el logit de cada cliente, sin mirar $y$).

### 7.5 Variables de riesgo

La [[tab:sc-variables-riesgo]] y la [[fig:sc-importancia]] ordenan las 15 variables por la media de $|\varphi_j|$ del MLP.
Las cinco primeras son el plazo, el uso de los cupos rotativos, el ingreso anual, las consultas de los últimos 6 meses y las
líneas de crédito abiertas. La importancia por permutación coincide en lo principal: plazo, ingreso anual y uso de los cupos
son las tres de mayor caída de AUC en la logística y en el MLP, y también están entre las cuatro de mayor rango de puntos de la
logística ([[tab:discusion-scorecard]]). Las tres medidas miden cosas distintas (efecto sobre el logit, sobre la AUC y sobre
los puntos), así que solo comparamos órdenes, no números. Los registros públicos negativos son la única variable con caída
de AUC **negativa** en los dos modelos: permutarlos mejora un poco la AUC de prueba (sección 7.6).

::: tabla sc-variables-riesgo
Título: Importancia y dirección de las variables de riesgo: SHAP del MLP e importancia por permutación de los dos modelos en prueba
Fuente: Elaboración propia a partir de results/data/scorecard/shap_mlp.json (importancia_global, direccion, permutacion) y results/data/scorecard/scorecard.json (etiquetas de la aplicación, abreviadas, y categorías). SHAP: 1 000 créditos de prueba; ρ de Spearman entre el valor crudo y φ (ρ > 0: a mayor valor, más log-odds de malo y menos puntos); en las categóricas, la categoría de φ medio mayor y menor con su número de créditos. Permutación: caída de AUC en prueba, media ± desviación muestral de 5 repeticiones.
| Puesto | Variable | Media \|φ\| | Dirección en SHAP | ΔAUC logística | ΔAUC MLP |
|---:|---|---:|---|---:|---:|
| 1 | `term` (Plazo) | 0,329 | Mayor: 60 meses (0,606; n = 281); menor: 36 meses (−0,221; n = 719) | 0,0579 ± 0,0005 | 0,0621 ± 0,0005 |
| 2 | `revol_util` (Uso de los cupos rotativos, %) | 0,219 | ρ = 0,995 | 0,0369 ± 0,0004 | 0,0367 ± 0,0004 |
| 3 | `annual_inc` (Ingreso anual, USD) | 0,194 | ρ = −0,997 | 0,0427 ± 0,0011 | 0,0386 ± 0,0011 |
| 4 | `inq_last_6mths` (Consultas de crédito en los últimos 6 meses) | 0,161 | ρ = 0,925 | 0,0101 ± 0,0002 | 0,0113 ± 0,0003 |
| 5 | `open_acc` (Líneas de crédito abiertas) | 0,124 | ρ = 0,996 | 0,0087 ± 0,0001 | 0,0116 ± 0,0001 |
| 6 | `loan_amnt` (Monto solicitado, USD) | 0,101 | ρ = 0,996 | 0,0095 ± 0,0008 | 0,0072 ± 0,0006 |
| 7 | `total_acc` (Total de líneas de crédito en el historial) | 0,089 | ρ = −0,995 | 0,0081 ± 0,0003 | 0,0094 ± 0,0003 |
| 8 | `purpose` (Destino del préstamo) | 0,081 | Mayor: Pequeña empresa (0,529; n = 10); menor: Tarjeta de crédito (−0,201; n = 197) | 0,0064 ± 0,0005 | 0,0070 ± 0,0006 |
| 9 | `revol_bal` (Saldo en tarjetas de crédito, USD) | 0,068 | ρ = −0,994 | 0,0034 ± 0,0002 | 0,0030 ± 0,0002 |
| 10 | `emp_length` (Antigüedad en el empleo) | 0,061 | Mayor: No lo reporto (0,241; n = 48); menor: 1 año (−0,110; n = 52) | 0,0003 ± 0,0001 | 0,0016 ± 0,0002 |
| 11 | `home_ownership` (Vivienda) | 0,058 | Mayor: Arriendo (0,052; n = 372); menor: Hipoteca (−0,069; n = 538) | 0,0040 ± 0,0001 | 0,0049 ± 0,0001 |
| 12 | `delinq_2yrs` (Moras de 30+ días en los últimos 2 años) | 0,048 | ρ = 0,714 | 0,0033 ± 0,0003 | 0,0040 ± 0,0004 |
| 13 | `pub_rec` (Registros públicos negativos) | 0,044 | ρ = 0,672 | −0,0011 ± 0,0004 | −0,0010 ± 0,0003 |
| 14 | `antiguedad_credito_anios` (Años de historial crediticio) | 0,036 | ρ = −0,990 | 0,0004 ± < 0,0001 | 0,0017 ± 0,0002 |
| 15 | `dti` (Deuda mensual sobre ingreso mensual, %) | 0,023 | ρ = 0,958 | 0,0049 ± 0,0003 | 0,0023 ± 0,0001 |
:::

::: figura sc-importancia
Título: Importancia de las variables: media de |φ| SHAP del MLP e importancia por permutación de los dos modelos en prueba
Fuente: Elaboración propia a partir de results/data/scorecard/shap_mlp.json (importancia_global y permutacion). Barras de error: desviación muestral de 5 repeticiones.
![Dos paneles con las variables en el orden de SHAP: barras de la media de |φ| del MLP, encabezadas por el plazo, y barras de la caída de AUC al permutar para la logística y el MLP, con el plazo, el ingreso y el uso de los cupos muy por encima del resto y los registros públicos por debajo de cero](figures/sc-importancia.png)
:::

La **dirección** se lee con el signo de $\rho$: con $\rho > 0$, a mayor valor de la variable, más log-odds de malo y menos
puntos. En el MLP, más uso de los cupos, más consultas, más líneas abiertas, más monto y más deuda sobre ingreso suben el
riesgo; más ingreso, más líneas en el historial, más saldo en tarjetas y un historial más largo lo bajan. Entre las
categóricas, el plazo de 60 meses, la pequeña empresa (con solo 10 créditos entre los explicados, un $\bar\varphi$ poco
preciso), no reportar la antigüedad en el empleo y arrendar son las categorías de mayor $\varphi$ medio. La
[[fig:sc-dependencia-shap]] muestra la forma de la relación en las cuatro numéricas más importantes. En el uso de los cupos y
en las líneas abiertas es casi lineal y se aplana en el recorte (97,6 % y 23 líneas); en el ingreso es decreciente y convexa,
y se aplana en los ingresos más altos; en las consultas sube por escalones, uno por cada valor entero. La relación negativa
del saldo en tarjetas es condicional a las demás variables: una explicación posible, que no verificamos, es que con el mismo
uso de los cupos un saldo mayor implica cupos más grandes.

::: figura sc-dependencia-shap
Título: Valor SHAP del MLP frente al valor crudo de las cuatro numéricas más importantes en prueba
Fuente: Elaboración propia a partir de results/data/scorecard/shap_mlp.json (φ de los 1 000 créditos explicados y ρ de Spearman) y de los valores crudos de data/processed/loans_pd.parquet. En el uso de los cupos, 999 créditos (uno sin valor).
![Cuatro paneles de dispersión: φ crece casi en línea recta con el uso de los cupos, baja de forma convexa con el ingreso, sube por escalones con las consultas y crece con las líneas abiertas hasta aplanarse](figures/sc-dependencia-shap.png)
:::

Los dos puntajes **ordenan** el riesgo ([[fig:sc-bandas]]). Con bandas fijadas en los deciles del puntaje de entrenamiento,
la tasa de malos de prueba baja de forma monótona de la banda 1 (menor puntaje) a la 10: de 44,8 % a 7,5 % con la logística y
de 45,5 % a 7,7 % con el MLP. En las diez bandas la PD media queda por debajo de la tasa observada: los dos modelos
subestiman el nivel del riesgo en prueba, como se vio en la calibración (sección 6.4), aunque lo ordenan bien.

::: figura sc-bandas
Título: Distribución del puntaje y tasa de malos y PD media por banda de puntaje en prueba, regresión logística y MLP
Fuente: Elaboración propia a partir de results/data/scorecard/scorecard.json y results/data/scorecard/scorecard_mlp.json (distribucion_scores.test). Bandas: deciles del puntaje de entrenamiento de cada modelo; banda 1 = menor puntaje (más riesgo).
![Dos paneles: histogramas del puntaje de prueba de los dos modelos, centrados entre 520 y 550 puntos; y tasa de malos y PD media por banda, las dos decrecientes, con la PD media siempre por debajo de la tasa observada](figures/sc-bandas.png)
:::

### 7.6 Cierre de hipótesis

La sección 3 formuló once hipótesis antes de entrenar y las contrastó con el conjunto de entrenamiento ([[tab:hipotesis]]).
Aquí las cerramos con cuatro fuentes y una regla fija:

1. **EDA** (entrenamiento, efecto marginal): la evidencia de la sección 3, tal cual.
2. **SHAP** (MLP, efecto condicional): en una numérica, a favor si $s\,\rho \ge 0{,}5$ y en contra si $s\,\rho \le -0{,}5$,
   con $s = \pm 1$ el signo que afirma la hipótesis; en una categórica, el signo de
   $\Delta\varphi = \bar\varphi(\text{más riesgosa}) - \bar\varphi(\text{referencia})$.
3. **Prueba fuera de tiempo** (efecto marginal): el mismo criterio del EDA, con las tasas de malos observadas en prueba y los
   grupos y cortes de **entrenamiento**.
4. **Compuerta de relevancia**: la permutación del MLP en prueba es relevante si
   $\overline{\Delta\text{AUC}} - 2\cdot\text{desv} > 0$.

Una hipótesis es **confirmada** si EDA, SHAP y prueba están a favor y la permutación del MLP es relevante; **refutada** si
al menos dos de {EDA, SHAP, prueba} están en contra y ninguna a favor; y **no concluyente** en cualquier otro caso. La
logística y el PSI de prueba se informan como contraste, sin voto. Una advertencia, tal como la registra el archivo de
resultados: "Las cifras del EDA, de SHAP y de la permutación ya se conocían (C4) cuando se fijó la regla; la regla solo es
ciega frente a las tasas de test, que no se habían calculado." Como análisis de sensibilidad repetimos el veredicto con
umbrales de $\rho$ de 0,3 y 0,7 y sin la compuerta: **ningún veredicto cambia**.

La [[tab:sc-hipotesis]] resume el resultado: **10 hipótesis confirmadas, una no concluyente (H9) y ninguna refutada.** Hay
dos matices. Primero, el EDA y la prueba miden la tasa de malos por grupo de una sola variable (efecto marginal), mientras
que SHAP mide cuánto mueve la variable el logit con las demás presentes (efecto condicional); pueden discrepar si la variable
está correlacionada con otras. Segundo, todo es asociación observada en los datos, no un efecto causal.

::: tabla sc-hipotesis
Título: Cierre de las hipótesis: señales del EDA, de SHAP y de la prueba fuera de tiempo, contrastes y veredicto
Fuente: Elaboración propia a partir de results/data/scorecard/hipotesis.json. SHAP: ρ de Spearman entre el valor crudo y φ (numéricas) o Δφ entre la categoría más riesgosa y la de referencia (categóricas). Prueba: tasa de malos del primer y del último grupo con los cortes de entrenamiento (o de las dos categorías) y ρ de Spearman entre el orden del grupo y su tasa. Permutación relevante: caída media de AUC del MLP − 2·desviación > 0. PSI: prueba frente a entrenamiento.
| Id | Hipótesis | Variable | EDA (entrenamiento) | SHAP (MLP) | Prueba | Permutación relevante | Logística coincide | PSI de prueba | Veredicto |
|---|---|---|---|---|---|---|---|---:|---|
| H1 | Los créditos a 60 meses son más riesgosos que los de 36 | `term` | A favor | A favor (Δφ = 0,827) | A favor (37,0 % vs. 22,1 %) | Sí | Sí | 0,064 | Confirmada |
| H2 | A mayor razón deuda/ingreso, mayor riesgo | `dti` | A favor | A favor (ρ = 0,958) | A favor (16,8 % → 37,2 %; ρ = 1,00) | Sí | Sí | 0,112 | Confirmada |
| H3 | A mayor ingreso anual, menor riesgo | `annual_inc` | A favor | A favor (ρ = −0,997) | A favor (38,5 % → 19,2 %; ρ = −1,00) | Sí | Sí | 0,038 | Confirmada |
| H4 | Más consultas recientes, más riesgo | `inq_last_6mths` | A favor | A favor (ρ = 0,925) | A favor (25,4 % → 29,9 %; ρ = 0,80) | Sí | Sí | 0,009 | Confirmada |
| H5 | A mayor uso de los cupos rotativos, mayor riesgo | `revol_util` | A favor | A favor (ρ = 0,995) | A favor (16,6 % → 33,6 %; ρ = 1,00) | Sí | Sí | 0,062 | Confirmada |
| H6 | Un historial crediticio más largo reduce el riesgo | `antiguedad_credito_anios` | A favor | A favor (ρ = −0,990) | A favor (33,9 % → 24,8 %; ρ = −0,98) | Sí | Sí | 0,113 | Confirmada |
| H7 | Quien arrienda es más riesgoso que quien paga hipoteca | `home_ownership` | A favor | A favor (Δφ = 0,121) | A favor (30,4 % vs. 22,7 %) | Sí | Sí | 0,039 | Confirmada |
| H8 | Pequeña empresa es más riesgosa que consolidación de deudas | `purpose` | A favor | A favor (Δφ = 0,553) | A favor (41,0 % vs. 26,8 %) | Sí | Sí | 0,270 | Confirmada |
| H9 | Los registros públicos negativos aumentan el riesgo | `pub_rec` | A favor | A favor (ρ = 0,672) | En contra (26,7 % → 24,4 %; ρ = −1,00) | No | Sí | 0,269 | No concluyente |
| H10 | A mayor monto, mayor riesgo | `loan_amnt` | A favor | A favor (ρ = 0,996) | A favor (19,7 % → 29,8 %; ρ = 0,95) | Sí | Sí | 0,089 | Confirmada |
| H11 | Las moras recientes aumentan el riesgo | `delinq_2yrs` | A favor | A favor (ρ = 0,714) | A favor (25,6 % → 32,2 %; ρ = 1,00) | Sí | Sí | 0,050 | Confirmada |
:::

En H4, H6 y H10 la relación se sostiene en dirección pero no es monótona en alguno de los dos conjuntos (ρ de grupos 0,80 en
la prueba de H4; 0,73 y 0,76 en el entrenamiento de H6 y H10). En H8, el $\bar\varphi$ de la pequeña empresa se apoya en solo
10 créditos explicados.

**H9 (registros públicos).** En entrenamiento, la tasa de malos es 16,0 % sin registros y 23,1 % con al menos uno; en prueba
la relación se invierte: 26,7 % frente a 24,4 %. La población también cambia: `pub_rec` está recortado en 1, así que la
proporción de valor 1 es la de solicitantes con al menos un registro, y pasa de 4,1 % en entrenamiento a 12,8 % en
validación y 19,5 % en prueba. El cambio ya empieza en validación. El PSI de prueba es 0,269 y la permutación del MLP no es
relevante (caída de AUC de −0,0010 ± 0,0003). SHAP sigue a favor ($\rho = 0{,}672$) porque describe lo que **aprendió** el
modelo con datos de entrenamiento, no si esa relación se mantiene después. Con dos fuentes a favor, una en contra y la
compuerta cerrada, la hipótesis queda **no concluyente**. El cambio de población de esta variable se discute en la sección 3
y en la sección 12 (hallazgo H-01).
<!-- Fuentes de H9: cifras de train y proporción de pub_rec de results/data/eda/post_preprocesamiento.json
(relacion_con_objetivo_train.pub_rec y distribucion_discretas.pub_rec). -->
