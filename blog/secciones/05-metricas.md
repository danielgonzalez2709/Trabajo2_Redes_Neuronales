<!-- Responsable: Andrés (B). Enunciado (Anexo del núcleo): definición de AUC, Gini, KS, Brier y curva de calibración. Guía de sintaxis: blog/README.md -->
<!-- B8 (27-oct): definiciones tomadas de las celdas de texto de src/common/metricas.py (y la log-loss de
src/pd/logistica.py); lectura de la calibración según B-pd §3.6. Sin cifras de resultados: están en la sección 6. -->
## 5. Métricas de evaluación

Esta sección define cada métrica tal como la calcula `src/common/metricas.py`. Los valores obtenidos están en la sección 6.

### 5.1 Convención

$y_i \in \{0, 1\}$ es el desenlace real del crédito $i$ ($y_i = 1$ **malo**, $y_i = 0$ **bueno**) y $p_i \in [0, 1]$ su PD
predicha. Hay $n$ créditos, $P = \sum_i y_i$ malos y $N = n - P$ buenos. Una PD más alta significa más riesgo, así que
"señalar como malo con el umbral $t$" es $p_i \ge t$. Todas las métricas exigen que haya malos y buenos ($0 < P < n$): sin
las dos clases, la AUC y el KS no están definidos.

Las métricas se agrupan en dos familias. Las de **discriminación** (AUC, Gini, KS) miden si el modelo **ordena** bien: si
les da más PD a los malos que a los buenos. Las de **calibración** (Brier, log-loss, curva de calibración) miden si la PD es
una **probabilidad correcta**: si de cada 100 créditos con PD 0,2 incumplen unos 20.

### 5.2 Curva ROC y AUC

Para cada umbral $t$ (cada valor distinto de $p$, de mayor a menor) se cuentan los malos atrapados y los buenos señalados
por error:

$$
\mathrm{VP}(t) = \sum_{i} \mathbb{1}[p_i \ge t]\, y_i, \qquad
\mathrm{FP}(t) = \sum_{i} \mathbb{1}[p_i \ge t]\, (1 - y_i),
\qquad
\mathrm{TPR}(t) = \frac{\mathrm{VP}(t)}{P}, \qquad \mathrm{FPR}(t) = \frac{\mathrm{FP}(t)}{N}.
$$

En palabras: $\mathrm{TPR}(t)$ es la fracción de los malos que el umbral detecta y $\mathrm{FPR}(t)$, la fracción de los buenos
que señala por error. La **curva ROC** une los puntos $\big(\mathrm{FPR}(t), \mathrm{TPR}(t)\big)$, desde $(0, 0)$ (umbral
$+\infty$: nadie es señalado) hasta $(1, 1)$ (umbral mínimo: todos lo son). Los créditos con la misma PD entran juntos, así que
un empate forma un solo tramo diagonal.

La **AUC** es el área bajo esa curva, calculada con la regla del trapecio, que es exacta porque la ROC empírica es una
poligonal:

$$
\mathrm{AUC} = \int_0^1 \mathrm{TPR}\; d(\mathrm{FPR})
= \sum_{k=1}^{m} (x_k - x_{k-1})\, \frac{y_k + y_{k-1}}{2},
$$

con $(x_k, y_k)$ los vértices de la ROC (aquí $y_k$ es la altura del vértice, no un desenlace). Tiene una lectura directa
como probabilidad [@hanley1982]:

$$
\mathrm{AUC} = \Pr(p_{\text{malo}} > p_{\text{bueno}}) + \tfrac{1}{2}\Pr(p_{\text{malo}} = p_{\text{bueno}}),
$$

para un malo y un bueno tomados al azar. En palabras: la AUC es la probabilidad de que el modelo le dé más PD a un malo que a
un bueno (contando la mitad de los empates). @hanley1982 muestran que el área por trapecios coincide con el estadístico de
Wilcoxon, que cuenta 1, ½ o 0 por cada par malo-bueno. Un modelo que ordena al azar tiene AUC 0,5 y uno perfecto, 1. Solo
depende del **orden** de $p$: no cambia si se usa el logit en lugar de la PD, ni si todas las PD se multiplican por un
mismo factor positivo. Por eso la AUC no dice nada de la calibración.

### 5.3 Índice de Gini

$$
\mathrm{Gini} = 2\,\mathrm{AUC} - 1 .
$$

En palabras: reescala la AUC al intervalo $[-1, 1]$, de modo que 0 es azar, 1 es separación perfecta y −1 es el orden
invertido. No aporta información nueva respecto de la AUC, pero es la forma habitual de reportarla en riesgo de crédito.
Restar el 1 importa: $2\,\mathrm{AUC}$ sola daría 1 para un modelo al azar.

### 5.4 Estadístico KS

Las distribuciones acumuladas empíricas de la PD en los malos y en los buenos son

$$
F_1(t) = \frac{1}{P} \sum_{i:\, y_i = 1} \mathbb{1}[p_i \le t], \qquad
F_0(t) = \frac{1}{N} \sum_{i:\, y_i = 0} \mathbb{1}[p_i \le t],
$$

y el estadístico de Kolmogórov-Smirnov es su mayor distancia vertical:

$$
\mathrm{KS} = \max_t \bigl| F_1(t) - F_0(t) \bigr| .
$$

En palabras: $F_0(t)$ es la fracción de buenos con PD de a lo sumo $t$, y $F_1(t)$ lo mismo en los malos. Si el modelo
discrimina, los buenos se acumulan antes (tienen PD más bajas) y $F_0$ va por encima de $F_1$; el KS es la separación máxima
entre las dos curvas y el umbral donde ocurre es el que mejor separa las dos poblaciones. Va de 0 (no separa) a 1 (separa
perfectamente). Basta evaluar las curvas en cada valor distinto de $p$, porque entre dos valores consecutivos las dos son
constantes. Coincide con el estadístico de dos muestras de `scipy.stats.ks_2samp(p[y == 1], p[y == 0])`, con el que se
contrasta en las pruebas.

La relación con la ROC: $1 - F_1(t)$ es la fracción de malos con PD mayor que $t$ y $1 - F_0(t)$ la de buenos, así que
$F_0(t) - F_1(t) = \big(1 - F_1(t)\big) - \big(1 - F_0(t)\big)$ es la diferencia "TPR − FPR" del umbral siguiente. Por eso el
KS es también la mayor distancia vertical entre la curva ROC y la diagonal. A diferencia de la AUC, que promedia sobre todos
los umbrales, el KS mira solo el mejor.

### 5.5 Brier y log-loss

**Puntaje de Brier.**

$$
\mathrm{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2 .
$$

En palabras: es el error cuadrático medio de la probabilidad. Un crédito malo ($y = 1$) con PD 0,9 aporta $0{,}01$; con PD
0,1 aporta $0{,}81$. Va de 0 (perfecto) a 1 y menor es mejor. Se usa la versión binaria, la misma de
`sklearn.metrics.brier_score_loss`; la versión original de @brier1950 suma el error sobre las dos clases (malo y bueno) y en
el caso binario da el doble, $2(p - y)^2$.

El Brier premia la calibración. Si para un perfil la probabilidad verdadera de ser malo es $q$ y el modelo predice $p$, el
Brier esperado es

$$
\mathbb{E}\big[(p - y)^2\big] = q(1-p)^2 + (1-q)p^2 = (p - q)^2 + q(1 - q) .
$$

En palabras: el segundo término no depende del modelo (es el ruido propio del problema) y el primero es mínimo, cero,
cuando $p = q$. Un modelo solo puede bajar el Brier esperado acercando su PD a la probabilidad verdadera; por eso se dice que
es una **regla de puntuación propia**. Pero no mide solo calibración: si un modelo está calibrado, su Brier esperado es la
media de $p(1-p)$, que es menor cuanto más cerca de 0 o de 1 estén sus PD. Entre dos modelos calibrados gana, entonces, el
que separa mejor.

**Log-loss.** Es la entropía cruzada binaria de la sección 4.4, evaluada en validación o en prueba:

$$
\mathrm{LL} = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\Big].
$$

En palabras: el promedio de $-\log(\text{probabilidad que el modelo dio a lo que pasó})$. Castiga mucho más que el Brier los
errores hechos con confianza: un malo con PD 0,01 aporta $-\log 0{,}01 \approx 4{,}6$. También es una regla de puntuación
propia: con $w_+ = 1$, la cuenta de la sección 4.4 muestra que su valor esperado es mínimo en $p = q$. Se calcula con
`sklearn.metrics.log_loss(y, p, labels=[0, 1])`.

La log-loss de validación es el **criterio de selección** de los dos modelos: elige el $C$ de la logística y la configuración
del MLP (sección 4.8), y también la semilla mediana del MLP (sección 6). La elegimos porque es la misma función que se
minimiza al entrenar y porque, al ser propia, no premia a un modelo que ordena bien pero da probabilidades exageradas.

### 5.6 Curva de calibración

Se ordenan las PD y se parten en $B = 10$ grupos (*bins*) de tamaño parecido, con bordes en los cuantiles empíricos de $p$:

$$
b_k = Q_p\!\left(\tfrac{k}{B}\right), \qquad k = 0, 1, \dots, B,
$$

con interpolación lineal entre valores ordenados (el método por defecto de `np.percentile`), de modo que $b_0 = \min p$ y
$b_B = \max p$. El crédito $i$ va al bin

$$
k(i) = \#\{\, j \in \{1, \dots, B-1\} : b_j < p_i \,\},
$$

es decir, se compara solo con los bordes internos y un crédito cuya PD es igual a un borde va al bin inferior. Para cada bin
con $n_k > 0$ créditos:

$$
\bar p_k = \frac{1}{n_k} \sum_{i:\,k(i)=k} p_i, \qquad \bar y_k = \frac{1}{n_k} \sum_{i:\,k(i)=k} y_i .
$$

En palabras: $\bar p_k$ es la PD media que el modelo dio en el bin y $\bar y_k$ la fracción de malos que de verdad hubo. Los
bins vacíos (pueden aparecer si muchos créditos empatan en un borde) se omiten. Con bins por cuantiles, cada punto se apoya
en aproximadamente $n/10$ créditos: todos los puntos se apoyan en una cantidad parecida de datos (su precisión no es
idéntica, porque también depende de la tasa de cada bin).

**Cómo se lee.** Se grafica $\bar y_k$ (tasa observada, eje vertical) contra $\bar p_k$ (PD media, eje horizontal). Un modelo
bien calibrado queda sobre la diagonal $\bar y_k = \bar p_k$.

- Un punto **por encima** de la diagonal ($\bar y_k > \bar p_k$) significa que hubo más malos de los que el modelo predijo:
  el modelo **subestima** el riesgo en ese bin.
- Un punto **por debajo** ($\bar y_k < \bar p_k$) significa que el modelo **sobreestima** el riesgo.

Es fácil leerlo al revés (hallazgo H-22, sección 12): la regla es "arriba = la realidad es peor que la predicción". La
curva de calibración de prueba de los dos modelos está en la sección 6.4.

### 5.7 Por qué se reportan las dos familias

Discriminación y calibración miden cosas distintas, y un modelo puede ser bueno en una y malo en la otra. Si todas las PD se
multiplicaran por 2 (suponiendo que ninguna pasa de 1), el orden no cambiaría: la AUC, el Gini y el KS quedarían iguales, pero la PD dejaría de
ser una probabilidad correcta y la curva de calibración se iría por debajo de la diagonal. Al revés, un modelo que le diera
a todos la tasa media de malos estaría calibrado en promedio, pero no ordenaría nada (AUC 0,5).

En este trabajo importan las dos. La discriminación decide si la scorecard (sección 7) separa bien a los solicitantes. La
calibración decide si la pérdida esperada (sección 9) da montos correctos, porque ahí la PD se multiplica por la LGD y la
exposición, y una PD sistemáticamente baja da una pérdida esperada baja. La [[tab:metricas-resumen]] resume qué mide cada
métrica.

::: tabla metricas-resumen
Título: Métricas de evaluación de la PD: qué miden, rango y sentido
Fuente: Elaboración propia a partir de las definiciones de src/common/metricas.py y de la log-loss de src/pd/logistica.py.
| Métrica | Familia | Qué mide | Rango | Mejor |
|---|---|---|---|---|
| AUC | Discriminación | Probabilidad de que un malo tenga más PD que un bueno | [0, 1]; azar 0,5 | Mayor |
| Gini | Discriminación | La AUC reescalada, 2·AUC − 1 | [−1, 1]; azar 0 | Mayor |
| KS | Discriminación | Máxima separación entre las acumuladas de la PD en malos y buenos | [0, 1] | Mayor |
| Brier | Calibración y discriminación | Error cuadrático medio de la PD | [0, 1] | Menor |
| Log-loss | Calibración y discriminación | Log-verosimilitud negativa media (criterio de selección) | [0, ∞) | Menor |
| Curva de calibración | Calibración | PD media frente a tasa observada en 10 bins por cuantiles | Diagonal = calibrado | Cerca de la diagonal |
:::
