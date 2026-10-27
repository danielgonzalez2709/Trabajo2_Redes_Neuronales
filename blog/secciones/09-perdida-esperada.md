<!-- Responsable: Daniel (E). Enunciado (Anexo del núcleo): pérdida esperada E[L] = PD · LGD · EAD y definición de la EAD. Guía de sintaxis: blog/README.md -->
<!-- Cifras: results/data/el/summary.json y results/data/el/el_test.parquet (python run.py el --config configs/el/base.yaml);
results/data/pd/summary.json; results/data/lgd/summary.json y pred_pd_test.parquet; data/processed/montos.parquet. -->
## 9. Pérdida esperada

Las secciones 6 a 8 estiman por separado *si* un crédito incumple (PD) y *cuánto* se pierde cuando incumple (LGD). El
enunciado pide combinarlas en la **pérdida esperada** de cada crédito y definir y justificar la exposición que se usa. En esta
sección la calculamos para los 74 154 créditos del conjunto de prueba (originados en 2014), la agregamos por banda de
puntaje de la scorecard y la contrastamos con la pérdida que efectivamente se realizó, solo en los créditos cuyo desenlace
está cerrado.

### 9.1 La fórmula y de dónde sale cada factor

Siguiendo el enunciado del curso, la pérdida esperada del crédito $i$ es

$$
\mathrm{E}[L]_i = \mathrm{PD}_i \cdot \mathrm{LGD}_i \cdot \mathrm{EAD}_i ,
$$

donde cada factor viene de una parte distinta del trabajo:

- **$\mathrm{PD}_i$**: probabilidad de incumplimiento predicha por la **regresión logística** de la Parte B (sección 6), con AUC
  de prueba 0,670 (Fuente: `results/data/pd/summary.json`, `logistica.metricas.auc_test`). La pérdida esperada con el MLP
  de PD se obtiene con `python run.py el --config configs/el/base.yaml --set pd.modelo=mlp` y se compara con esta en la
  sección 10.4.
- **$\mathrm{LGD}_i$**: LGD predicha por el **MLP de la Parte D** (sección 8; semilla mediana 22, Fuente:
  `results/data/lgd/summary.json`, `mlp.semilla_mediana`) para **todos** los créditos de prueba, no solo para los
  castigados: la fórmula necesita la LGD *condicional a incumplir* de cada crédito, aunque el crédito haya terminado pagándose.
  Esas predicciones casi no varían entre créditos (desviación estándar 0,006; Fuente: `results/data/lgd/pred_pd_test.parquet`,
  columna `pred_mlp`), por lo que en la práctica la LGD actúa como una constante cercana a 0,917 (la LGD media de la
  [[tab:el-totales]]).
- **$\mathrm{EAD}_i$**: la exposición al incumplimiento, que definimos en la sección 9.2.

Los tres factores se unen por el identificador del crédito; el código rechaza PD o LGD fuera de $[0, 1]$, EAD negativas,
valores no finitos y créditos de prueba sin predicción (`src/el/perdida_esperada.py`).

### 9.2 La exposición al incumplimiento (EAD)

**Definición.** Usamos la **EAD de originación**,

$$
\mathrm{EAD}_i = F_i ,
$$

donde $F_i$ es `funded_amnt`, que el diccionario define como "The total amount committed to that loan at that point in
time." (Fuente: `data/processed/diccionario_loanstats.csv`; la sección 8.1 explica por qué lo preferimos a `loan_amnt`). Se
conoce en la originación, que es cuando se calcula la PD con las 15 variables de la sección 2.3; los montos posteriores
(`total_rec_prncp`, `recoveries`) solo se conocen después y no pueden entrar en una estimación *ex ante*.
<!-- TODO(cita): si se quiere anclar la definición de EAD o de E[L] en la regulación (p. ej. Basilea), falta una referencia
verificada en blog/references.bib; hoy la fórmula se atribuye solo al enunciado del curso. -->

**Por qué es conservadora.** Usar $F_i$ equivale a suponer que el incumplimiento ocurre antes de amortizar nada de
principal. En la realidad, quien incumple suele haber pagado una parte: entre los créditos castigados de prueba, la fracción
del monto desembolsado que seguía pendiente al castigo, $(F_i - P_i)/F_i$ con $P_i$ = `total_rec_prncp` ("Principal received
to date"), es en promedio **0,839** (Fuente: `results/data/el/summary.json`,
`fraccion_principal_pendiente_media_castigados`). Es decir, en promedio por crédito, alrededor del 16 % del principal ya se
había pagado al castigo (con el supuesto de la sección 8.1 de que lo cobrado después del castigo no se registra en
`total_rec_prncp`), y la EAD de originación lo cuenta como expuesto.

Hay además una diferencia de definición que conviene hacer explícita: la LGD de la Parte D es una fracción del **principal
pendiente** al castigo, $F_i - P_i$ (sección 8.1), y aquí se multiplica por el **monto desembolsado** $F_i$. El producto
$\mathrm{LGD}_i \cdot F_i$ es por lo tanto mayor o igual que la pérdida que la misma LGD implicaría sobre el principal
pendiente. Ese es exactamente el sesgo conservador de la EAD, y en la sección 9.5 lo medimos.

**La alternativa: exposición promedio según el plazo.** Un crédito de Lending Club se paga en cuotas mensuales
(`installment`: "The monthly payment owed by the borrower if the loan originates.") durante un plazo (`term`: "The number
of payments on the loan. Values are in months and can be either 36 or 60."), así que, si se paga según lo pactado, su
principal pendiente baja de $F_i$ hasta 0 a lo largo de 36 o 60 meses. Una EAD alternativa es el promedio del principal
pendiente durante la vida del crédito, que es menor que $F_i$. Tiene dos
costos: necesita la tabla de amortización de cada crédito y, sobre todo, un supuesto sobre **cuándo** ocurre el
incumplimiento (promediar sobre todo el plazo supone que es igual de probable en cualquier mes), que no verificamos. No la
cuantificamos: hacerlo, y estimar el momento del incumplimiento con los créditos de entrenamiento, queda como trabajo futuro.
Preferimos la EAD de originación porque no depende de ese supuesto y su sesgo tiene un signo conocido (sobreestima la
exposición).
<!-- revisar: autor (¿se quiere cuantificar la alternativa antes de la entrega?) -->

### 9.3 Resultados

Con la logística, el MLP de LGD y la EAD de originación, la **pérdida esperada de todo el conjunto de prueba es 197,03
millones de USD**, sobre 1 061,57 millones desembolsados, es decir, el 18,6 % del monto ([[tab:el-totales]]).

::: tabla el-totales
Título: Pérdida esperada del conjunto de prueba y de sus dos grupos según el desenlace
Fuente: Elaboración propia a partir de results/data/el/summary.json (bloques total, resueltos y en_mora; función tabla_totales de src/el/tablas.py). Montos en millones de USD. Resueltos = castigados y pagados; en mora = Late (31-120 days) y Default. La pérdida realizada solo está definida para los resueltos.
| Grupo | Créditos | EAD | E[L] | Pérdida realizada | E[L] / realizada | Tasa de malos | PD media | LGD media |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Todo el conjunto de prueba | 74 154 | 1 061,57 | 197,03 | — | — | 0,262 | 0,187 | 0,917 |
| Resueltos (contraste) | 68 694 | 977,74 | 178,16 | 168,20 | 1,059 | 0,204 | 0,183 | 0,917 |
| En mora (cobranza abierta) | 5 460 | 83,83 | 18,88 | — | — | 1,000 | 0,230 | 0,917 |
:::

Para agregarla por nivel de riesgo, partimos el conjunto de prueba en **deciles del puntaje de la scorecard** de la Parte C
(sección 7; base 600, odds 50:1 y 20 puntos para duplicar las odds, Fuente: `results/data/el/summary.json`, bloque
`scorecard`), numerados de la banda 1 (menor puntaje, más riesgo) a la 10. El puntaje se calcula a partir de la PD de la
logística como $\text{puntaje}_i = \text{offset} - \text{factor} \cdot \operatorname{logit}(\mathrm{PD}_i)$
(`src/el/experimento.py`; factor y offset en el mismo bloque `scorecard`), una transformación decreciente de la PD: las bandas
son, por lo tanto, deciles de la PD de la logística, y que la PD media baje de la banda 1 a la 10 es una consecuencia de cómo
se construyen.

La [[tab:el-bandas]] muestra que la pérdida esperada se concentra en las bandas de menor puntaje: la banda 1 suma 51,76
millones, 9,88 veces los 5,24 millones de la banda 10. La diferencia la produce sobre todo la PD: su media es 6,64 veces
mayor en la banda 1 (0,416 frente a 0,063), en parte por construcción de las bandas. La banda 1 tiene además 1,49 veces más
monto desembolsado (136,15 frente a 91,31 millones), mientras que la LGD media apenas se mueve entre bandas (de 0,914 a
0,919; 1,00 veces entre las bandas 1 y 10). Lo que no es por construcción es el contraste con lo observado: la tasa de malos
baja de forma monótona de la banda 1 a la 10 (de 0,476 a 0,087), así que el orden de la PD de la logística se sostiene en el
conjunto de prueba.

::: tabla el-bandas
Título: Pérdida esperada por banda de puntaje de la scorecard (todo el conjunto de prueba)
Fuente: Elaboración propia a partir de results/data/el/summary.json (por_banda, bloque total; función tabla_bandas de src/el/tablas.py). Montos en millones de USD. Bandas = deciles del puntaje; banda 1 = menor puntaje (más riesgo).
| Banda | Puntaje | Créditos | EAD | E[L] | PD media | Tasa de malos | LGD media |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 457,6–505,3 | 7 416 | 136,15 | 51,76 | 0,416 | 0,476 | 0,919 |
| 2 | 505,3–515,1 | 7 415 | 137,01 | 38,88 | 0,309 | 0,389 | 0,917 |
| 3 | 515,1–523,6 | 7 415 | 120,15 | 27,18 | 0,246 | 0,336 | 0,918 |
| 4 | 523,6–530,5 | 7 416 | 102,99 | 18,87 | 0,200 | 0,309 | 0,919 |
| 5 | 530,5–535,9 | 7 415 | 96,93 | 14,95 | 0,168 | 0,276 | 0,918 |
| 6 | 535,9–540,8 | 7 415 | 92,76 | 12,31 | 0,145 | 0,240 | 0,918 |
| 7 | 540,8–545,7 | 7 416 | 95,02 | 10,90 | 0,125 | 0,202 | 0,917 |
| 8 | 545,7–551,1 | 7 415 | 94,47 | 9,26 | 0,107 | 0,172 | 0,916 |
| 9 | 551,1–558,3 | 7 415 | 94,78 | 7,67 | 0,089 | 0,134 | 0,915 |
| 10 | 558,3–608,4 | 7 416 | 91,31 | 5,24 | 0,063 | 0,087 | 0,914 |
| **Total** | | **74 154** | **1 061,57** | **197,03** | 0,187 | 0,262 | 0,917 |
:::

### 9.4 Contraste con la pérdida realizada

**Qué se compara.** La pérdida esperada es una expectativa *ex ante* por crédito; la pérdida realizada de un crédito es 0 o
un monto positivo. Por eso el contraste se hace **por grupos** (todo el conjunto y cada banda), sumando ambas. La pérdida
realizada se mide en principal, neta de recuperaciones y sin intereses ni descuento del dinero en el tiempo, según el estado
del crédito (`src/el/perdida_esperada.py`):

$$
L_i =
\begin{cases}
\max\big\{0,\ (F_i - P_i) - (R_i - C_i)\big\} & \text{si el crédito fue castigado,} \\
0 & \text{si fue pagado,}
\end{cases}
$$

con $R_i$ = `recoveries` y $C_i$ = `collection_recovery_fee` ([[tab:lgd-campos]]). La razón que reportamos es
$\sum_{i \in \mathcal{R}} \mathrm{E}[L]_i \big/ \sum_{i \in \mathcal{R}} L_i$, donde $\mathcal{R}$ son los **créditos
resueltos**: los 13 981 castigados (*Charged Off*) y los 54 713 pagados (*Fully Paid*) de prueba (Fuente:
`data/processed/montos.parquet`, filas de prueba según `data/processed/loans_pd.parquet`). Para los castigados, $L_i$ es la
LGD observada de la sección 8 multiplicada por el principal pendiente.

**Por qué los créditos en mora no entran.** Los 5 460 créditos en mora (*Late (31-120 days)* y *Default*) cuentan como malos
en la PD, pero su cobranza no ha terminado y no hay pérdida realizada que medir. El código guarda aparte una pérdida
*provisional* con la misma cuenta, 61,33 millones de USD (Fuente: `results/data/el/summary.json`,
`en_mora.perdida_provisional`); como estos créditos no registran recuperaciones, esa cifra equivale a suponer LGD = 1 sobre su
principal pendiente. No es una pérdida observada ni una cota inferior de la pérdida final, y por eso **no la comparamos** con su
pérdida esperada (18,88 millones): ambas cifras miden cosas distintas.

**Resultado.** En los resueltos, la pérdida esperada suma **178,16 millones de USD** y la realizada **168,20 millones**: la
razón es **1,059** ([[tab:el-totales]]). Por banda, la razón va de 0,927 (banda 5) a 1,209 (banda 10) ([[tab:el-contraste]]
y [[fig:el-bandas]]): la pérdida esperada supera a la realizada en ocho de las diez bandas y queda por debajo en las bandas 5
y 6.

::: tabla el-contraste
Título: Pérdida esperada frente a pérdida realizada por banda de puntaje (créditos resueltos de prueba)
Fuente: Elaboración propia a partir de results/data/el/summary.json (por_banda, bloques resueltos y en_mora; función tabla_bandas de src/el/tablas.py). Montos en millones de USD. La última columna cuenta los créditos en mora de la banda, que no entran al contraste.
| Banda | Resueltos | E[L] | Pérdida realizada | E[L] / realizada | PD media | Tasa de malos | En mora (excluidos) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 464 | 45,03 | 39,72 | 1,134 | 0,415 | 0,399 | 952 |
| 2 | 6 613 | 34,62 | 32,37 | 1,069 | 0,309 | 0,315 | 802 |
| 3 | 6 712 | 24,61 | 22,73 | 1,083 | 0,246 | 0,266 | 703 |
| 4 | 6 739 | 17,13 | 16,48 | 1,040 | 0,200 | 0,240 | 677 |
| 5 | 6 846 | 13,76 | 14,84 | 0,927 | 0,168 | 0,216 | 569 |
| 6 | 6 925 | 11,52 | 12,27 | 0,939 | 0,145 | 0,187 | 490 |
| 7 | 6 987 | 10,24 | 10,18 | 1,006 | 0,125 | 0,153 | 429 |
| 8 | 7 043 | 8,78 | 8,76 | 1,003 | 0,107 | 0,128 | 372 |
| 9 | 7 134 | 7,37 | 6,64 | 1,110 | 0,089 | 0,100 | 281 |
| 10 | 7 231 | 5,10 | 4,22 | 1,209 | 0,062 | 0,064 | 185 |
| **Total** | **68 694** | **178,16** | **168,20** | **1,059** | 0,183 | 0,204 | **5 460** |
:::

::: figura el-bandas
Título: Pérdida esperada frente a pérdida realizada por banda de puntaje (créditos resueltos de prueba)
Fuente: Elaboración propia a partir de results/data/el/summary.json (python run.py figures). PD: regresión logística; LGD: MLP (semilla mediana); scorecard: base 600, odds 50:1, 20 puntos para duplicar las odds.
![Barras pareadas por banda de puntaje: la pérdida esperada y la realizada bajan de unos 40 a 45 millones de USD en la banda 1 a unos 4 a 5 millones en la banda 10, con diferencias pequeñas en cada banda](figures/el-bandas.png)
:::

**Una anomalía de los datos en los pagados.** En 1 284 créditos pagados de prueba el principal recibido es menor que el
monto desembolsado ($P_i < F_i$), por un total de 1 670 028,92 USD (Fuente: `results/data/el/summary.json`,
`anomalia_pagados`). Casi todos son diferencias de centavos: solo **197** superan 1 USD, y esos 197 suman 1 670 009,95 USD de
ese total; la mediana de las 1 284 diferencias es 0,01 USD. Los 197 y la mediana los calculamos uniendo
`data/processed/montos.parquet` con la columna `split` de `data/processed/loans_pd.parquet` (mismas filas y orden), tomando
los créditos de prueba con estado *Fully Paid* y contando cuántos tienen `funded_amnt − total_rec_prncp` mayor que 1. El
diccionario no explica por qué un crédito pagado tendría principal sin registrar, así que no interpretamos la diferencia:
estos créditos conservan pérdida realizada 0 y el monto se reporta aparte, sin sumarlo a la pérdida. Aun si se sumara entero,
sería menos del 1 % de la pérdida realizada de los resueltos.
<!-- revisar: autor (1 670 028,92 / 168 195 585,76 = 0,0099) -->

### 9.5 Lectura: por qué la razón queda cerca de 1

Que la razón total sea 1,059 no quiere decir que cada factor esté bien estimado. Los datos muestran tres efectos: dos a la
baja (la PD y la LGD se quedan cortas) y uno al alza (la EAD de originación sobreestima la exposición).

**La PD queda por debajo de la tasa de malos.** La [[fig:el-pd-vs-tasa]] muestra que, en todo el conjunto de prueba, la PD
media de la logística está por debajo de la tasa de malos observada **en las diez bandas** (0,187 frente a 0,262 en total).
Es lo esperable con la partición temporal: la logística se entrenó con créditos hasta 2012, cuya tasa de malos es 16,3 %, y la
tasa sube a 23,0 % en validación y 26,2 % en prueba ([[tab:particion]]; sección 2.4). En los resueltos la brecha es menor
(0,183 frente a 0,204, [[tab:el-totales]]) porque los 5 460 créditos en mora, todos malos, quedan fuera; aun así, la PD media
de los resueltos queda por debajo de su tasa de malos en las bandas 2 a 10 y por encima en la banda 1 ([[tab:el-contraste]]).

::: figura el-pd-vs-tasa
Título: PD media predicha frente a tasa de malos observada por banda de puntaje (todo el conjunto de prueba)
Fuente: Elaboración propia a partir de results/data/el/summary.json (por_banda, bloque total; python run.py figures). PD: regresión logística.
![Dos curvas decrecientes por banda de puntaje: la tasa de malos observada, de 0,48 a 0,09, siempre por encima de la PD media predicha, de 0,42 a 0,06](figures/el-pd-vs-tasa.png)
:::

**La EAD de originación sobreestima la exposición.** Va en sentido contrario, como se vio en la sección 9.2.

**Cuánto pesa cada efecto.** Para separarlos, escribimos la razón de los resueltos como un producto de tres factores. Sea
$\mathcal{C} \subset \mathcal{R}$ el conjunto de castigados, $\widehat{\mathrm{PD}}_i$ y $\widehat{\mathrm{LGD}}_i$ las
predicciones y $\mathrm{LGD}^{\text{obs}}_i = L_i / (F_i - P_i)$ la LGD observada de cada castigado (en prueba,
$F_i - P_i > 0$ en todos). Como $L_i = 0$ en los pagados,

$$
\frac{\sum_{\mathcal{R}} \widehat{\mathrm{PD}}_i\, \widehat{\mathrm{LGD}}_i\, F_i}{\sum_{\mathcal{R}} L_i}
=
\underbrace{\frac{\sum_{\mathcal{R}} \widehat{\mathrm{PD}}_i\, \widehat{\mathrm{LGD}}_i\, F_i}{\sum_{\mathcal{C}} \widehat{\mathrm{LGD}}_i\, F_i}}_{\text{factor PD}}
\cdot
\underbrace{\frac{\sum_{\mathcal{C}} \widehat{\mathrm{LGD}}_i\, F_i}{\sum_{\mathcal{C}} \mathrm{LGD}^{\text{obs}}_i\, F_i}}_{\text{factor LGD}}
\cdot
\underbrace{\frac{\sum_{\mathcal{C}} \mathrm{LGD}^{\text{obs}}_i\, F_i}{\sum_{\mathcal{C}} \mathrm{LGD}^{\text{obs}}_i\, (F_i - P_i)}}_{\text{factor EAD}} .
$$

Es una identidad (los términos intermedios se cancelan), y cada factor vale 1 si su pieza es perfecta: el factor PD, si la PD
fuera 1 en los castigados y 0 en los pagados; el factor LGD, si la LGD predicha coincidiera con la observada; y el factor EAD,
si la exposición fuera el principal pendiente al castigo. No es la única descomposición posible (el orden de los pasos cambia
los valores intermedios), pero sí la que separa las tres piezas de la fórmula. La lectura no depende del orden entre los pasos de
LGD y de EAD: si primero se cambia la EAD (con la LGD predicha) y después la LGD, los factores del total son 1,176 y 0,973, en
vez de 1,179 y 0,971. La [[tab:el-descomposicion]] aplica la descomposición a los resueltos de prueba.

::: tabla el-descomposicion
Título: Descomposición de la razón entre pérdida esperada y realizada en un factor de PD, uno de LGD y uno de EAD (créditos resueltos de prueba)
Fuente: Elaboración propia a partir de results/data/el/el_test.parquet (columnas pd, lgd, ead, ead_incumplimiento, estado, perdida_observada, observacion_completa y banda), con las sumas de la ecuación anterior. El producto de los tres factores reproduce la razón de results/data/el/summary.json.
| Banda | E[L] / realizada | Factor PD | Factor LGD | Factor EAD |
|---:|---:|---:|---:|---:|
| 1 | 1,134 | 1,051 | 0,978 | 1,103 |
| 2 | 1,069 | 0,983 | 0,970 | 1,121 |
| 3 | 1,083 | 0,959 | 0,973 | 1,159 |
| 4 | 1,040 | 0,874 | 0,969 | 1,228 |
| 5 | 0,927 | 0,770 | 0,968 | 1,244 |
| 6 | 0,939 | 0,775 | 0,966 | 1,254 |
| 7 | 1,006 | 0,829 | 0,967 | 1,255 |
| 8 | 1,003 | 0,818 | 0,962 | 1,274 |
| 9 | 1,110 | 0,903 | 0,967 | 1,272 |
| 10 | 1,209 | 0,981 | 0,967 | 1,275 |
| **Total** | **1,059** | **0,926** | **0,971** | **1,179** |
:::
<!-- revisar: autor (la tabla el-descomposicion se calculó fuera de src/: conviene llevar la función a src/el/tablas.py con su
prueba para que el notebook 05 la reproduzca; el cálculo está descrito en la Fuente) -->

En el total, **la PD se queda corta en un 7,4 % y la LGD en un 2,9 %, y la EAD de originación agrega un 17,9 %**: con
cuatro decimales, $0{,}9256 \times 0{,}9708 \times 1{,}1788 = 1{,}0592$. Por bandas se ve lo mismo con otros pesos:

- El **factor LGD** es casi constante (de 0,962 a 0,978). Es coherente con la sección 8.5: la LGD predicha queda anclada en
  la media de entrenamiento (0,917), mientras la LGD observada de los castigados de prueba está desplazada hacia 1
  (consistente con censura, sección 8.2).
- El **factor EAD** pasa de 1,103 en la banda 1 a 1,275 en la banda 10 (crece de una banda a la siguiente salvo de la 8 a
  la 9, donde baja de 1,274 a 1,272): los castigados de las bandas de mejor puntaje habían pagado, en promedio ponderado,
  una parte mayor de su principal antes del castigo, y la EAD de originación los sobreexpone más.
- El **factor PD** está por debajo de 1 en las bandas 2 a 10 y es mínimo en las bandas 5 y 6 (0,770 y 0,775). Justamente esas
  son las dos bandas donde la razón queda por debajo de 1: ahí la sobreexposición de la EAD (1,244 y 1,254) no alcanza a
  compensar la subestimación de la PD y de la LGD (factor LGD 0,968 y 0,966). En la banda 1 el factor PD supera a 1 (1,051)
  y en la 10 está cerca de 1 (0,981). La razón de la banda 10 (1,209, la más alta) combina un factor PD cercano a 1 con el
  factor EAD más alto (1,275): las bandas 8 y 9 tienen un factor EAD casi igual (1,274 y 1,272) pero un factor PD menor
  (0,818 y 0,903) y razones de 1,003 y 1,110, y la banda 2, con un factor PD parecido (0,983) pero un factor EAD de 1,121,
  queda en 1,069.

**Qué se puede concluir.** La razón cercana a 1 en los resueltos **no es evidencia de que la pérdida esperada esté bien
calibrada**: resulta de que una PD y una LGD que se quedan cortas en un conjunto de prueba más riesgoso que el de
entrenamiento se multiplican por una exposición que, a propósito, sobreestima. Los tres factores dependen de cosas distintas
(el cambio de la tasa de malos entre cosechas, el desplazamiento de la LGD de prueba hacia 1, consistente con censura según
la sección 8.2, y el ritmo de amortización antes del castigo), y nada garantiza que se sigan compensando en otra cosecha. Lo
que sí muestran los datos es que el orden de la PD de la logística, con el que se forman las bandas, se sostiene con lo
observado: la tasa de malos y la pérdida realizada por banda bajan de forma monótona de la banda 1 a la 10
([[tab:el-contraste]] y [[fig:el-bandas]]).

Tres limitaciones adicionales. Primera, el contraste deja fuera a los créditos en mora, cuyo desenlace todavía no se conoce.
Segunda, la pérdida esperada usa una sola PD (la logística es determinista) y una sola LGD (la semilla mediana del MLP); no
calculamos cómo varía la pérdida esperada entre las 30 semillas de la LGD ni entre las 30 del MLP de PD (cuya semilla
mediana se compara en la sección 10.4).
Tercera, tanto la pérdida esperada como la realizada se miden en principal, sin intereses ni descuento del dinero en el
tiempo (sección 8.1).
<!-- revisar: autor (variabilidad entre semillas de la E[L]: no calculada) -->
