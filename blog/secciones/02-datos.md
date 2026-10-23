<!-- Responsable: Daniel (A). Enunciado (Anexo del núcleo): Datos; Parte 1, "Variable objetivo" y tarea 5 (partición).
Cifras: data/processed/resumen_datos.json (python scripts/preparar_datos.py). Guía: blog/README.md -->
## 2. Datos

### 2.1 Fuente

Usamos el conjunto *Credit Risk Analysis* publicado en Kaggle (https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset),
con créditos de consumo otorgados por Lending Club, descargado el **23 de octubre de 2026**. El archivo de préstamos tiene
**887 379 registros y 74 variables**, con fechas de originación (`issue_d`) entre **junio de 2007 y diciembre de 2015**. El
conjunto incluye un diccionario de variables (`LCDataDictionary.xlsx`, hoja `LoanStats`); todas las definiciones que citamos
salen de él, textualmente, y están en `configs/features.yaml`. El SHA-256 de cada archivo está en `data/fuentes.md`.

Al contrastar el diccionario con el archivo encontramos diferencias que conviene conocer: el diccionario describe variables
de puntaje FICO (`fico_range_low`, `fico_range_high`) y una variable `is_inc_v` que **no están** en el CSV; la verificación
de ingresos aparece en el CSV como `verification_status`.

### 2.2 Variable objetivo

Construimos la variable binaria a partir de `loan_status` con la codificación del enunciado: 1 = malo (*Charged Off*,
*Default*, *Late (31-120 days)* y *Does not meet the credit policy. Status:Charged Off*), 0 = bueno (*Fully Paid* y *Does not
meet the credit policy. Status:Fully Paid*), y sin valor (NA) para los créditos cuyo desenlace todavía no se conoce. La
[[tab:loan-status]] muestra cuántos créditos hay en cada estado. Los diez textos del archivo coinciden exactamente con los de la
tabla del enunciado; el código rechaza cualquier estado que no esté en ella, en lugar de convertirlo en NA en silencio.

::: tabla loan-status
Título: Créditos por estado (`loan_status`) y su codificación
Fuente: Elaboración propia a partir de data/processed/resumen_datos.json.
| Estado | Créditos | Objetivo |
|---|---:|:---:|
| Current | 601 779 | NA |
| Fully Paid | 207 723 | 0 |
| Charged Off | 45 248 | 1 |
| Late (31-120 days) | 11 591 | 1 |
| Issued | 8 460 | NA |
| In Grace Period | 6 253 | NA |
| Late (16-30 days) | 2 357 | NA |
| Does not meet the credit policy. Status:Fully Paid | 1 988 | 0 |
| Default | 1 219 | 1 |
| Does not meet the credit policy. Status:Charged Off | 761 | 1 |
:::

De los 887 379 créditos, **618 849 (69,7 %) no tienen desenlace conocido** y no entran al entrenamiento.

### 2.3 Variables

Elegimos 15 variables que cumplen dos condiciones: se conocen en el momento de la solicitud (caso de uso, sección 1.1) y una
persona puede escribirlas en el formulario de la aplicación. La [[tab:variables]] las resume; catorce vienen del archivo y
una, los años de historial crediticio, se deriva como la diferencia entre `issue_d` y `earliest_cr_line` (la fecha de la
primera línea de crédito se conoce al solicitar).

::: tabla variables
Título: Variables del modelo de probabilidad de incumplimiento
Fuente: Elaboración propia; definiciones textuales de LCDataDictionary.xlsx (hoja LoanStats) en configs/features.yaml.
| Variable | Tipo | Qué mide |
|---|---|---|
| `loan_amnt` | numérica (log) | Monto solicitado |
| `term` | categórica | Plazo: 36 o 60 cuotas |
| `emp_length` | categórica | Antigüedad laboral |
| `home_ownership` | categórica | Situación de vivienda |
| `annual_inc` | numérica (log) | Ingreso anual declarado |
| `purpose` | categórica | Destino del crédito |
| `dti` | numérica | Cuotas de deuda (sin hipoteca ni el crédito pedido) / ingreso mensual |
| `delinq_2yrs` | numérica | Moras de 30+ días en los últimos 2 años |
| `antiguedad_credito_anios` | numérica (derivada) | Años desde la primera línea de crédito |
| `inq_last_6mths` | numérica | Consultas de crédito en los últimos 6 meses |
| `open_acc` | numérica | Líneas de crédito abiertas |
| `pub_rec` | numérica | Registros públicos negativos |
| `revol_bal` | numérica (log) | Saldo rotativo |
| `revol_util` | numérica | Uso del crédito rotativo |
| `total_acc` | numérica | Total de líneas de crédito en el historial |
:::

Dejamos por fuera, con su razón:

- **Posteriores al desembolso** (fuga de información): pagos y saldos (`total_pymnt`, `total_rec_prncp`, `out_prncp`,
  `last_pymnt_d`…), recuperaciones (`recoveries`, `collection_recovery_fee`, que sí usamos para construir la LGD) y
  `last_credit_pull_d`. Una prueba automática (`tests/test_fuga.py`) falla si alguna aparece en la lista de variables.
- **Evaluación de Lending Club:** `grade`, `sub_grade`, `int_rate` e `installment` (la cuota depende de la tasa).
- **Identificadores y texto libre:** `id`, `member_id`, `url`, `desc`, `title`, `emp_title`.
- **Sensibles:** `zip_code` y `addr_state`, por el riesgo de discriminar por ubicación.
- **No disponibles para el solicitante o casi vacías:** `verification_status` (la hace el banco, no la persona) y variables
  con más de la mitad de valores vacíos, como `mths_since_last_delinq`, donde un vacío significa "nunca ocurrió" y no un dato
  faltante.

### 2.4 Partición temporal

El enunciado pide al menos una partición temporal según la fecha de originación. La primera propuesta era entrenar con los
créditos hasta 2013, validar con 2014 y probar con 2015. Los datos la descartaron: la [[fig:datos-anio]] muestra que la
fracción de créditos sin desenlace crece con el año y llega al **93 % en 2015**. Los pocos créditos de 2015 que ya tienen
desenlace son, sobre todo, los que incumplieron pronto, y forman una muestra de prueba muy sesgada.

::: figura datos-anio
Título: Créditos sin desenlace y tasa de malos por año de originación, con la partición elegida
Fuente: Elaboración propia a partir de data/processed/resumen_datos.json (python run.py figures).
![Dos paneles de barras por año de 2007 a 2015: porcentaje de créditos sin desenlace y tasa de malos entre los resueltos](figures/datos-anio.png)
:::

Elegimos **entrenamiento = originados hasta 2012, validación = 2013 y prueba = 2014**, y dejamos fuera los 30 835 créditos
resueltos de 2015 ([[tab:particion]]). El mismo sesgo explica, al menos en parte, que la tasa de malos suba de un conjunto al
siguiente: entre los créditos recientes, los que ya terminaron están sobrerrepresentados por los incumplimientos tempranos.
Lo tendremos en cuenta al leer la calibración de los modelos en el conjunto de prueba.

::: tabla particion
Título: Tamaño y tasa de malos de cada conjunto
Fuente: Elaboración propia a partir de data/processed/resumen_datos.json.
| Conjunto | Años de originación | Créditos | Tasa de malos |
|---|---|---:|---:|
| Entrenamiento | 2007–2012 | 90 347 | 16,3 % |
| Validación | 2013 | 73 194 | 23,0 % |
| Prueba | 2014 | 74 154 | 26,2 % |
| **Total** | | **237 695** | 21,4 % |
:::

### 2.5 Preprocesamiento

Todos los modelos de una misma tarea usan el mismo preprocesamiento, ajustado **solo con el conjunto de entrenamiento**
(`src/common/preprocesamiento.py`). Para cada variable numérica, en este orden: imputación por la mediana, recorte a los
percentiles 1 y 99, logaritmo $\log(1+x)$ en las variables de monto (monto solicitado, ingreso y saldo rotativo) y
estandarización $z = (x - \bar{x})/s$. Las categóricas se codifican en una columna por categoría (*one-hot*); los vacíos van a
la categoría `DESCONOCIDO` y las categorías que no aparecieron en entrenamiento, a `OTRA`. Elegimos solo operaciones que la
aplicación web puede repetir exactamente en el navegador. La sección 3.3 describe qué efecto tienen sobre los datos.
