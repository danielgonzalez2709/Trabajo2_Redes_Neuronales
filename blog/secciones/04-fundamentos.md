<!-- Responsable: Andrés (B). Enunciado (Anexo del núcleo): neurona, MLP, activaciones, pérdida por máxima verosimilitud, retropropagación, optimizador, regularización. Guía de sintaxis: blog/README.md -->
<!-- B8 (27-oct): fórmulas tomadas de las celdas de texto de src/pd/mlp.py, src/pd/entrenar.py, src/pd/logistica.py y
src/pd/hpo.py. Cifras de configs/pd/mlp_final.yaml, configs/pd/hpo.yaml, models/preprocesador_pd.json (48 columnas),
results/data/pd/{summary,hpo_ensayos,busqueda_logistica}.json y data/processed/resumen_datos.json (n de train). -->
## 4. Fundamentos de los modelos de PD

Esta sección explica, paso a paso, las matemáticas de los dos modelos de probabilidad de incumplimiento (PD): la regresión
logística, que es la referencia, y el perceptrón multicapa (MLP). Los resultados están en la sección 6.

**Convención.** Para cada crédito $i$, $y_i = 1$ si fue **malo** (incumplió) y $y_i = 0$ si fue **bueno**. El vector
$\mathbf{x}_i$ tiene las variables de originación ya preprocesadas (sección 2.5). Los dos modelos producen primero un
número real $z$, el **logit**, y la PD es su sigmoide:

$$
z = \log\frac{p}{1-p}, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}} .
$$

En palabras: el logit es el logaritmo de la razón "probabilidad de ser malo / probabilidad de ser bueno"; la sigmoide lo
convierte de vuelta en una probabilidad entre 0 y 1. Un logit de 0 es una PD de 0,5; un logit negativo, una PD menor.

### 4.1 Regresión logística: el modelo de referencia

La logística es un modelo lineal en el logit:

$$
z = \mathbf{w}^\top \mathbf{x} + b, \qquad p = \sigma(\mathbf{w}^\top \mathbf{x} + b).
$$

En palabras: cada variable suma (o resta) riesgo en el logit con su peso $w_j$, y $b$ es el nivel base. Tiene un peso por
columna de entrada más el intercepto: con las 48 columnas del preprocesador son $48 + 1 = 49$ parámetros (Fuente:
`results/data/pd/summary.json`).

Con $n$ créditos de entrenamiento, `LogisticRegression(C, solver="lbfgs")` de scikit-learn minimiza

$$
f(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n} \Big[-y_i \log p_i - (1 - y_i)\log(1 - p_i)\Big]
+ \frac{1}{2Cn}\lVert \mathbf{w} \rVert_2^2, \qquad p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b).
$$

En palabras: el primer término es la entropía cruzada binaria (BCE) media, que la sección 4.4 deriva por máxima
verosimilitud; el segundo castiga los pesos grandes (penalización L2). El intercepto $b$ no se penaliza. $C$ es el
**inverso** de la fuerza de la penalización: $C$ grande deja los pesos casi libres y $C$ pequeño los encoge hacia 0. Con
$\lambda = 1/(Cn)$ el castigo es $(\lambda/2)\lVert \mathbf{w}\rVert^2$, la misma forma que la L2 del MLP (sección 4.7).

En el óptimo el gradiente es nulo:

$$
\nabla_{\mathbf{w}} f = \frac{1}{n} X^\top (\mathbf{p} - \mathbf{y}) + \frac{\mathbf{w}}{C n} = \mathbf{0},
\qquad \frac{\partial f}{\partial b} = \frac{1}{n} \sum_{i}(p_i - y_i) = 0 .
$$

En palabras: la segunda condición dice que, en entrenamiento, la PD media del modelo es exactamente la tasa de malos. El
término $p - y$ aparece aquí por la misma razón que en la retropropagación (sección 4.5).

$C$ se eligió en una rejilla de 13 valores, de $10^{-4}$ a $10^{2}$ cada medio orden de magnitud, con la log-loss de
validación como criterio (sección 4.8). Ganó $C = 0{,}0031623$ (Fuente: `results/data/pd/busqueda_logistica.json`); con los
90 347 créditos de entrenamiento (Fuente: `data/processed/resumen_datos.json`) equivale a $\lambda = 1/(Cn) \approx 0{,}0035$.
`lbfgs` es determinista: no usa azar, así que la logística se reporta con una sola corrida y su desviación entre semillas es 0.

### 4.2 Neurona artificial y perceptrón multicapa

Una **neurona** recibe un vector $\mathbf{x}$, calcula una suma ponderada más un sesgo y le aplica una función no lineal
$\varphi$, la **activación**: $\varphi(\mathbf{w}^\top \mathbf{x} + b)$. La logística es, de hecho, una sola neurona con
$\varphi = \sigma$.

Una **capa oculta** con $k$ neuronas hace eso $k$ veces en paralelo:

$$
\mathbf{h} = \varphi(W\mathbf{x} + \mathbf{b}), \qquad W \in \mathbb{R}^{k \times d},\; \mathbf{b} \in \mathbb{R}^{k},
$$

donde $\varphi$ se aplica a cada componente. En palabras: cada fila de $W$ es una neurona; la capa transforma las $d$
entradas en $k$ nuevas variables, $\mathbf{h}$, que la capa siguiente usa como entrada.

El **MLP** encadena $L$ capas ocultas de anchuras $k_1, \dots, k_L$ y termina en una neurona de salida sin activación:

$$
\mathbf{h}_0 = \mathbf{x}, \qquad
\mathbf{h}_\ell = \mathrm{Dropout}_p\big(\varphi(W_\ell \mathbf{h}_{\ell-1} + \mathbf{b}_\ell)\big) \quad (\ell = 1, \dots, L),
\qquad z = \mathbf{w}_{\text{sal}}^\top \mathbf{h}_L + b_{\text{sal}} .
$$

En palabras: cada capa oculta es `Linear → activación → Dropout` (el *dropout* se explica en la sección 4.7) y la salida
`Linear(k_L, 1)` devuelve el logit $z$, no la PD. La sigmoide se aplica después, dentro de la pérdida al entrenar
(sección 4.4) y al predecir. A diferencia de la logística, el logit ya no es lineal en $\mathbf{x}$: las capas ocultas
pueden representar interacciones y efectos no lineales entre variables.

**Cuántos parámetros tiene.** Cada `Linear(a, b)` tiene $a \cdot b$ pesos y $b$ sesgos. Con $k_0 = d$ entradas y
$k_{L+1} = 1$ salida,

$$
n_{\text{par}} = \sum_{\ell=1}^{L+1} \big(k_{\ell-1}\,k_\ell + k_\ell\big).
$$

En palabras: se suma, capa por capa, "entradas × neuronas" (los pesos) más "neuronas" (los sesgos). Se cuentan solo los
parámetros entrenables.

La entrada tiene $d = 48$ columnas (Fuente: `models/preprocesador_pd.json`, campo `columnas_salida`): 11 variables
numéricas, una columna cada una, y las columnas *one-hot* de las 4 categóricas, que son 3 de `term`, 13 de `emp_length`,
6 de `home_ownership` y 15 de `purpose` (37 en total, incluidas `DESCONOCIDO` y `OTRA`). La arquitectura elegida por la
búsqueda tiene tres capas ocultas de 32, 16 y 32 neuronas (Fuente: `configs/pd/mlp_final.yaml`). La [[tab:pd-parametros]]
muestra la cuenta: 2 673 parámetros, la cifra de `results/data/pd/summary.json`, frente a 49 de la logística.

::: tabla pd-parametros
Título: Parámetros entrenables del MLP de PD por capa (arquitectura [32, 16, 32] con 48 entradas)
Fuente: Elaboración propia a partir de models/preprocesador_pd.json (48 columnas), configs/pd/mlp_final.yaml (capas) y results/data/pd/summary.json (total).
| Capa | Entradas × salidas | Pesos | Sesgos | Total |
|---|---|---|---|---|
| Oculta 1 | 48 × 32 | 1 536 | 32 | 1 568 |
| Oculta 2 | 32 × 16 | 512 | 16 | 528 |
| Oculta 3 | 16 × 32 | 512 | 32 | 544 |
| Salida | 32 × 1 | 32 | 1 | 33 |
| **MLP** | | **2 592** | **81** | **2 673** |
:::

### 4.3 Activaciones

Sin activación no lineal, encadenar capas lineales daría otra vez un modelo lineal. El espacio de búsqueda (sección 4.8)
permitió tres activaciones ocultas, con $a$ la entrada de la neurona:

$$
\text{ReLU: } \varphi(a) = \max(0, a), \qquad
\text{Leaky ReLU: } \varphi(a) = \begin{cases} a & a \ge 0 \\ \alpha a & a < 0 \end{cases}, \qquad
\text{tanh: } \varphi(a) = \frac{e^{a} - e^{-a}}{e^{a} + e^{-a}} .
$$

En palabras:

- **ReLU** deja pasar los valores positivos y apaga los negativos. Su derivada es 1 si $a > 0$ y 0 si $a < 0$.
- **Leaky ReLU** es igual, pero en los negativos deja una pendiente pequeña, $\alpha = 0{,}01$ (el valor por defecto de
  PyTorch, que no cambiamos). Su derivada es 1 o $\alpha$, así que una neurona con entradas negativas sigue recibiendo
  gradiente.
- **tanh** aplasta la entrada en $(-1, 1)$ y es simétrica alrededor de 0. Su derivada es $1 - \tanh^2(a)$, que se acerca a 0
  cuando $|a|$ es grande.

La **sigmoide** solo se usa en la salida, para pasar del logit a la PD. Su derivada es

$$
\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)\,\big(1 - \sigma(z)\big) = p(1-p),
$$

que se usará en la sección 4.5. La búsqueda eligió **ReLU** (Fuente: `configs/pd/mlp_final.yaml`).

### 4.4 La pérdida, derivada por máxima verosimilitud

El desenlace de un crédito es 0 o 1. El modelo dice que, dado $\mathbf{x}_i$, el crédito es malo con probabilidad $p_i$;
es decir, $y_i$ sigue una distribución de **Bernoulli**:

$$
P(y_i \mid \mathbf{x}_i) = p_i^{\,y_i}\,(1 - p_i)^{\,1 - y_i} .
$$

En palabras: si $y_i = 1$ la fórmula vale $p_i$, y si $y_i = 0$ vale $1 - p_i$. Es la probabilidad que el modelo le dio a lo
que de verdad pasó.

Si los créditos se tratan como independientes, la **verosimilitud** de todos los datos es el producto,

$$
\mathcal{L}(\theta) = \prod_{i=1}^{n} p_i^{\,y_i}\,(1 - p_i)^{\,1 - y_i},
$$

donde $\theta$ son todos los parámetros (pesos y sesgos) de los que dependen los $p_i$. Máxima verosimilitud elige los
$\theta$ que hacen más probable lo observado. Como el logaritmo es creciente, maximizar $\mathcal{L}$ es lo mismo que
maximizar $\log \mathcal{L}$, que convierte el producto en suma. Cambiando el signo (para minimizar) y dividiendo entre $n$
(para promediar):

$$
\mathrm{BCE}(\theta) = -\frac{1}{n}\log \mathcal{L}(\theta)
= -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big] .
$$

En palabras: la **entropía cruzada binaria** es la log-verosimilitud negativa promedio. Cada crédito aporta
$-\log(\text{probabilidad que el modelo dio a su desenlace real})$: casi nada si el modelo acertó con confianza y mucho si
se equivocó con confianza. Es la misma función que minimiza la logística (sección 4.1) y la misma log-loss que se usa como
criterio de selección (sección 5.5).

**Por qué `BCEWithLogitsLoss`.** PyTorch la calcula a partir del logit, con la sigmoide por dentro (docstring de
`nn.BCEWithLogitsLoss`). Con $w_+$ = `mlp.pos_weight` y la media sobre el minilote:

$$
\ell = -\frac{1}{n}\sum_{i=1}^{n}\Big[w_+\,y_i \log \sigma(z_i) + (1-y_i)\log\big(1-\sigma(z_i)\big)\Big] .
$$

La razón es numérica. Si se calcula primero $p = \sigma(z)$ y luego $\log p$, para un logit muy negativo $p$ se redondea a 0
en coma flotante y $\log 0 = -\infty$. Trabajando con el logit, con $w_+ = 1$ la pérdida de un crédito se puede escribir de
forma equivalente como

$$
-\big[y \log \sigma(z) + (1-y)\log(1-\sigma(z))\big] = \max(z, 0) - z\,y + \log\big(1 + e^{-|z|}\big),
$$

que nunca exponencia un número positivo grande. En palabras: es la misma cuenta, reordenada para que no se desborde.
Usar `BCELoss` (que espera una probabilidad) sobre el logit sería un error.

**Por qué `pos_weight = 1`.** Sea $q = P(y = 1 \mid \mathbf{x})$ la probabilidad verdadera de ser malo para un perfil
$\mathbf{x}$. La pérdida esperada de predecir $p$ para ese perfil es $-\big[w_+ q \log p + (1-q)\log(1-p)\big]$. Derivando
respecto de $p$ e igualando a 0:

$$
-\frac{w_+ q}{p} + \frac{1-q}{1-p} = 0 \quad\Longrightarrow\quad p^\star = \frac{w_+ q}{w_+ q + 1 - q} .
$$

En palabras: con $w_+ = 1$ el mejor $p$ es exactamente $q$, la probabilidad verdadera. Con $w_+ > 1$ el óptimo queda por
encima de $q$: el modelo aprendería a exagerar la PD. Como la PD alimenta la scorecard (sección 7) y la pérdida esperada
(sección 9), necesitamos que sea una probabilidad bien calibrada; por eso `pos_weight` es 1 y no se busca.

### 4.5 Retropropagación

Entrenar es bajar la pérdida moviendo los parámetros en contra de su gradiente. La **retropropagación** calcula ese
gradiente con la regla de la cadena, de la salida hacia la entrada [@rumelhart1986]. En el código lo hace el diferenciador
automático de PyTorch; aquí se escribe a mano para un crédito.

**Paso 1: la salida.** Para un crédito con $w_+ = 1$, $\ell = -[y \log p + (1-y)\log(1-p)]$ con $p = \sigma(z)$. Por la
regla de la cadena y $\sigma'(z) = p(1-p)$ (sección 4.3):

$$
\frac{\partial \ell}{\partial z}
= \frac{\partial \ell}{\partial p}\,\frac{\partial p}{\partial z}
= \left(-\frac{y}{p} + \frac{1-y}{1-p}\right) p(1-p)
= -y(1-p) + (1-y)\,p
= p - y .
$$

En palabras: el error en el logit es simplemente "PD predicha menos desenlace real". Si el crédito fue malo ($y = 1$) y el
modelo dio $p = 0{,}2$, el gradiente es $-0{,}8$ y el paso del optimizador sube el logit. Es el mismo $p - y$ de las
condiciones de óptimo de la logística (sección 4.1). En el minilote la pérdida es una media, así que cada crédito aporta
$(p_i - y_i)/n$.

**Paso 2: los pesos de la salida.** Como $z = \mathbf{w}_{\text{sal}}^\top \mathbf{h}_L + b_{\text{sal}}$, llamando
$\delta_z = p - y$:

$$
\frac{\partial \ell}{\partial \mathbf{w}_{\text{sal}}} = \delta_z\,\mathbf{h}_L, \qquad
\frac{\partial \ell}{\partial b_{\text{sal}}} = \delta_z, \qquad
\frac{\partial \ell}{\partial \mathbf{h}_L} = \delta_z\,\mathbf{w}_{\text{sal}} .
$$

En palabras: un peso cambia en proporción al error y a lo que entró por él; y el error se pasa a la última capa oculta
repartido según los pesos de salida.

**Paso 3: cada capa oculta, de atrás hacia adelante.** Para la capa $\ell$, llamemos
$\mathbf{a}_\ell = W_\ell \mathbf{h}_{\ell-1} + \mathbf{b}_\ell$ a la entrada de las neuronas y
$\mathbf{h}_\ell = \mathbf{d}_\ell \odot \varphi(\mathbf{a}_\ell)$ a su salida, donde $\mathbf{d}_\ell$ es el vector de
factores del *dropout* (sección 4.7; al predecir es un vector de unos) y $\odot$ el producto componente a componente. Si ya
se conoce $\mathbf{g}_\ell = \partial \ell / \partial \mathbf{h}_\ell$:

$$
\boldsymbol{\delta}_\ell = \mathbf{g}_\ell \odot \mathbf{d}_\ell \odot \varphi'(\mathbf{a}_\ell), \qquad
\frac{\partial \ell}{\partial W_\ell} = \boldsymbol{\delta}_\ell\,\mathbf{h}_{\ell-1}^\top, \qquad
\frac{\partial \ell}{\partial \mathbf{b}_\ell} = \boldsymbol{\delta}_\ell, \qquad
\mathbf{g}_{\ell-1} = W_\ell^\top \boldsymbol{\delta}_\ell .
$$

En palabras: el error que llega a una capa se multiplica por la derivada de la activación (y por la máscara del
*dropout*: una neurona apagada no recibe gradiente); con él se obtienen los gradientes de los pesos y sesgos de esa capa, y
multiplicándolo por $W_\ell^\top$ se pasa a la capa anterior. Se repite desde $\ell = L$ hasta $\ell = 1$. Con ReLU,
$\varphi'(a)$ es 1 o 0 (sección 4.3), así que las neuronas inactivas cortan el paso del error.

### 4.6 Optimizadores

Con el gradiente $g_t$ del minilote $t$ (más la L2, sección 4.7), el optimizador decide cómo mover los parámetros
$\theta$. Las ecuaciones son las de PyTorch 2.14.1 (docstrings de `torch.optim.SGD` y `torch.optim.Adam`), con
$\gamma$ = `mlp.lr` la tasa de aprendizaje y

$$
g_t = \nabla_\theta\, \ell_t(\theta_{t-1}) + \lambda\,\theta_{t-1}
$$

($\lambda = 0$ en los sesgos). La búsqueda podía elegir entre dos optimizadores.

**SGD con momento** ($\mu$ = `mlp.momento` = 0,9; sin amortiguación ni Nesterov, los valores por defecto de PyTorch):

$$
\mathbf{b}_1 = g_1, \qquad \mathbf{b}_t = \mu\,\mathbf{b}_{t-1} + g_t \;\; (t > 1), \qquad
\theta_t = \theta_{t-1} - \gamma\,\mathbf{b}_t .
$$

En palabras: $\mathbf{b}_t$ es una "velocidad" que acumula los gradientes pasados con un olvido de $\mu$ por paso. Las
direcciones que se repiten se refuerzan y las que cambian de signo se cancelan, lo que suaviza el ruido de los minilotes.

**Adam** [@kingma2014], con $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\epsilon = 10^{-8}$ (valores por defecto de PyTorch,
que no cambiamos) y $m_0 = v_0 = 0$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2,
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad
\theta_t = \theta_{t-1} - \gamma\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} .
$$

En palabras: $m_t$ es un promedio móvil del gradiente (la dirección) y $v_t$ uno de su cuadrado (su tamaño típico), ambos
componente a componente. El paso es la dirección dividida por el tamaño típico, así que cada parámetro tiene su propia
escala de paso: los que reciben gradientes grandes avanzan con cautela y los que reciben gradientes pequeños, más rápido.

**La corrección de sesgo.** Desenrollando la recurrencia desde $m_0 = 0$:

$$
m_t = (1-\beta_1)\sum_{k=1}^{t} \beta_1^{\,t-k}\,g_k .
$$

Si los gradientes tienen una media estable $\mathbb{E}[g_k] = \bar g$, entonces

$$
\mathbb{E}[m_t] = (1-\beta_1)\,\bar g \sum_{k=1}^{t}\beta_1^{\,t-k} = (1-\beta_1)\,\bar g\,\frac{1-\beta_1^{t}}{1-\beta_1}
= (1-\beta_1^{t})\,\bar g .
$$

En palabras: como $m$ arranca en 0, en los primeros pasos subestima el gradiente medio por un factor $1-\beta_1^t$ (en el
primer paso, $1 - 0{,}9 = 0{,}1$). Dividir entre ese factor lo corrige; con $t$ grande el factor tiende a 1 y la corrección
desaparece. Lo mismo vale para $v_t$ con $\beta_2$. Es la justificación de @kingma2014.

`mlp.momento` solo lo usa SGD. La búsqueda eligió **Adam** con $\gamma = 0{,}004965$ y minilotes de 512 créditos (Fuente:
`configs/pd/mlp_final.yaml`).

### 4.7 Regularización

Un MLP con miles de parámetros puede memorizar el ruido de entrenamiento. Se usan tres regularizaciones, y las tres se
buscaron o se fijaron en la configuración.

**Penalización L2.** Sobre los pesos $W$ (todas las matrices de las capas `Linear`, **sin los sesgos**):

$$
\Omega(W) = \frac{\lambda}{2}\,\lVert W \rVert_2^2 = \frac{\lambda}{2}\sum_{j} w_j^2,
\qquad \nabla_W\,\Omega = \lambda\,W .
$$

En palabras: se suma a la pérdida un castigo proporcional al cuadrado de cada peso; su gradiente empuja cada peso hacia 0
con una fuerza proporcional a su tamaño. El $\tfrac12$ está para que el gradiente sea $\lambda W$ y no $2\lambda W$.

En el código no se escribe $\Omega$ a mano: el `weight_decay` $= \lambda$ de `torch.optim.Adam` y `torch.optim.SGD` suma
$\lambda\theta$ al gradiente antes del paso (la $g_t$ de la sección 4.6). Eso **es** sumar $(\lambda/2)\lVert W\rVert^2$ a la
pérdida, por lo que `weight_decay` = `mlp.l2` $= \lambda$. Los parámetros se reparten en dos grupos: los pesos (tensores de
dos dimensiones) con `weight_decay` $= \lambda$ y los sesgos con `weight_decay` $= 0$. Una versión anterior de la
especificación escribía $\lambda\lVert W\rVert^2$ sin el $\tfrac12$, cuyo gradiente es $2\lambda W$ y no equivale a
`weight_decay` $= \lambda$ (hallazgo H-08, sección 12).

**L2 no es AdamW.** @loshchilov2017 muestran que, con SGD estándar, la penalización L2 y el *weight decay* (encoger los
pesos directamente en cada paso) son equivalentes con un reescalado de $\lambda$, pero con Adam no lo son. Con Adam, el
$\lambda\theta$ entra en $g_t$ y por tanto también en $m_t$ y $v_t$: termina dividido entre $\sqrt{\hat v_t}$, así que los
pesos con gradientes grandes se regularizan menos. AdamW, que proponen @loshchilov2017, desacopla el decaimiento: lo aplica
directamente a los pesos, fuera de ese cociente. Nosotros usamos `Adam` con `weight_decay`, es decir, **L2 acoplada**, que
es lo que dice la fórmula de arriba; no usamos `AdamW`. La búsqueda eligió $\lambda = 4{,}451 \times 10^{-5}$ (Fuente:
`configs/pd/mlp_final.yaml`).

***Dropout* invertido** [@srivastava2014]. En esta subsección $p$ es la probabilidad de **apagar** una neurona
(`mlp.dropout`), no la PD. Al entrenar, cada salida $u_j$ de una capa oculta se reemplaza por

$$
\tilde u_j = \frac{m_j}{1-p}\,u_j, \qquad m_j \sim \mathrm{Bernoulli}(1-p),
$$

con una máscara $m_j$ nueva en cada minilote (el factor $m_j/(1-p)$ es la componente $d_j$ del vector $\mathbf{d}_\ell$ de la
sección 4.5). Su valor esperado es

$$
\mathbb{E}[\tilde u_j] = \frac{\mathbb{E}[m_j]}{1-p}\,u_j = \frac{1-p}{1-p}\,u_j = u_j .
$$

En palabras: en cada paso se apaga al azar una fracción $p$ de las neuronas y las que quedan se agrandan por $1/(1-p)$, de
modo que, en promedio, la señal que llega a la capa siguiente no cambia. La red no puede depender de una neurona concreta,
lo que reduce el sobreajuste. **Al predecir** (`eval()`) el *dropout* es la identidad: no se apaga nada ni se reescala. El
artículo original escala los pesos al predecir en lugar de escalar al entrenar, y usa $p$ para la probabilidad de
**conservar** [@srivastava2014]; las dos versiones dan la misma esperanza. La búsqueda eligió $p = 0{,}0942$ (Fuente:
`configs/pd/mlp_final.yaml`).

**Parada temprana.** Tras cada época $e$ (numeradas desde 1) se mide la pérdida de validación $L_e$, con la red en modo
evaluación. Sea $L^{*} = \min_{e' < e} L_{e'}$ la mejor hasta entonces ($L^{*} = +\infty$ al inicio):

- hay **mejora** si $L_e < L^{*}$, estrictamente (`min_delta` $= 0$: un empate no cuenta). Entonces se guarda una copia de
  los pesos y la cuenta de épocas sin mejora vuelve a 0;
- si no hay mejora, la cuenta sube en 1. Al llegar a `paciencia` (10) se detiene el entrenamiento.

Al terminar, por paciencia o por llegar a `max_epocas` (100), se **restauran los pesos de la mejor época** (Fuente de 10 y
100: `configs/pd/mlp_final.yaml`). En palabras: se entrena mientras el modelo mejore en datos que no ve, se espera un poco
por si la mejora vuelve y, al final, se devuelve la versión que mejor generalizó, no la última. La sección 6.2 muestra la
curva de aprendizaje de la semilla mediana.

### 4.8 Búsqueda de hiperparámetros

**Qué se busca.** El espacio del MLP tiene ocho hiperparámetros (Fuente: `configs/pd/hpo.yaml`): de 1 a 4 capas ocultas;
para cada capa, por separado, 16, 32, 64, 128 o 256 neuronas; activación ReLU, Leaky ReLU o tanh; *dropout* uniforme en
$[0;\ 0{,}5]$; L2 log-uniforme en $[10^{-6};\ 10^{-2}]$; optimizador Adam o SGD con momento; tasa de aprendizaje
log-uniforme en $[10^{-4};\ 10^{-2}]$; y lote de 256, 512, 1 024 o 2 048. No se buscan el momento, `max_epocas`,
`paciencia` ni `pos_weight`.

**Por qué TPE y no una rejilla.** Solo la parte discreta del espacio ya tiene
$(5 + 5^2 + 5^3 + 5^4) \times 3 \times 2 \times 4 = 780 \times 24 = 18\,720$ combinaciones, antes de discretizar las tres
variables continuas. Con un presupuesto de 40 entrenamientos, una rejilla solo podría cubrir unos pocos valores de cada
hiperparámetro. El *Tree-structured
Parzen Estimator* (TPE) es una búsqueda secuencial: usa los ensayos ya hechos para proponer el siguiente
[@bergstra2011]. Con $y$ la pérdida de un ensayo y $y^{*}$ un umbral tal que una fracción $\gamma$ de los ensayos queda por
debajo, modela

$$
p(\mathbf{x} \mid y) = \begin{cases} \ell(\mathbf{x}) & \text{si } y < y^{*} \\ g(\mathbf{x}) & \text{si } y \ge y^{*} \end{cases},
$$

donde $\mathbf{x}$ es aquí una configuración de hiperparámetros, $\ell$ es la densidad de las configuraciones "buenas" y $g$
la de las "malas" (estimadas con ventanas de Parzen). @bergstra2011 muestran que la mejora esperada es proporcional a
$\big(\gamma + \tfrac{g(\mathbf{x})}{\ell(\mathbf{x})}(1-\gamma)\big)^{-1}$, así que conviene proponer las configuraciones
con el mayor cociente $\ell(\mathbf{x})/g(\mathbf{x})$. En palabras: se prueba donde abundan las configuraciones que ya
dieron buena pérdida y escasean las que dieron mala.

**La implementación.** Usamos el `TPESampler` de Optuna [@akiba2019] con sus valores por defecto, semilla 0 y sin poda
(cada ensayo entrena hasta su parada temprana). Los primeros 10 ensayos son aleatorios y los 30 restantes los propone el
TPE (Fuente: `results/data/pd/hpo_ensayos.json`, campo `muestreador`). Un detalle importante: en Optuna 5.0.0, la versión
fijada, el TPE por defecto es **multivariado** en estudios de un objetivo, es decir, modela los hiperparámetros juntos y no
cada uno por separado como el TPE de @bergstra2011. Lo detectamos leyendo el código de Optuna (hallazgo H-16, sección 12) y
el JSON lo registra (`multivariate: true`). Para esta variante, la documentación de `TPESampler` remite a @falkner2018.

**Criterio, presupuesto y semilla.** Cada ensayo se puntúa con la **log-loss de validación** (sección 5.5), una regla de
puntuación propia que premia a la vez ordenar bien y estar calibrado; la AUC de validación se registra al lado pero no
decide, y prueba nunca se usa en la búsqueda. El mejor es el de menor log-loss (empate: menor número de ensayo). Todos los
ensayos usan la misma semilla (0) que el muestreador. El presupuesto es de 40 configuraciones, las 40 completadas, con
702,4 s de entrenamiento sumados (Fuente: `results/data/pd/hpo_ensayos.json`). La logística tuvo su propia búsqueda, una
rejilla de 13 valores de $C$, con el mismo criterio. La configuración ganadora (ensayo 11) es la de las secciones 4.2–4.7;
la sección 6.1 muestra la trayectoria de la búsqueda y la sección 6 compara los dos modelos con 30 semillas.
