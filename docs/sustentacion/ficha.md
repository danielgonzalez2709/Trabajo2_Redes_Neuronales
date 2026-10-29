# Ficha de estudio para la sustentación (T5.4)

Todas las cifras salen de `results/`, `data/processed/resumen_datos.json` o del blog; si dudas de una, el archivo está al
lado. Formato esperado (como en el Trabajo 1): **(1)** una pregunta sobre hiperparámetros, resultados o comparaciones;
**(2)** ejecutar una corrida u optimización cambiando un hiperparámetro; **(3)** opinar sobre la decisión de un compañero
("¿por qué crees que X decidió esto?", "¿qué cambiarías de lo que hizo Y?"). Cualquiera puede responder sobre cualquier parte.

## 0. Antes de entrar (checklist)

- Cámara encendida, los 5 conectados **1 minuto antes**. Si alguien falta, queda descartado.
- `git pull origin main`, `.venv` activado y una corrida de prueba de cada comando de la sección 2 hecha antes (para que las
  bibliotecas ya estén en caché y nada pida internet).
- Tener abiertos: la terminal en la raíz del repo, `notebooks/00_sustentacion.ipynb`, el blog publicado y la app.
- Después de cualquier corrida en vivo: `git status` debe quedar limpio (las demos escriben en carpetas temporales o
  ignoradas). Si algo se pisó: `git checkout -- <archivo>`.

## 1. El trabajo en una página

| | |
|---|---|
| Datos | Lending Club (Kaggle, descarga 23-oct): 887 379 créditos × 74 variables, 2007-06 a 2015-12 |
| Caso de uso | Analista de originación: aprobar o no **antes del desembolso** → solo variables de la solicitud y del buró (15, `configs/features.yaml`) |
| Objetivo PD | Malo = *Charged Off*, *Default* o mora de 31–120 días; bueno = pagado. Los *Current* no tienen desenlace y no entran |
| Partición | Temporal: train ≤ 2012 (90 347), val 2013 (73 194), test 2014 (74 154). 2015 fuera: 93 % sin desenlace |
| Tasa de malos | train 16,3 % · val 22,9 % · test 26,2 % (sube por cosecha: explica que la PD media quede baja en test) |
| Excluidas | `grade` e `int_rate`: son la evaluación de riesgo de Lending Club, no información propia del banco (con ellas, +0,017 de AUC; B10) |

### PD (Parte B, Andrés)

| | Logística | MLP |
|---|---|---|
| Búsqueda | Rejilla de 13 valores de C (10⁻⁴ … 10²), criterio log-loss de val → C = 3,16·10⁻³ | Optuna TPE (multivariado) 40 ensayos, log-loss de val, 702 s; mejor #11 |
| Configuración final | 49 parámetros | [32, 16, 32] ReLU, dropout 0,094, L2 4,45·10⁻⁵, Adam lr 0,00497, batch 512, paciencia 10; 2 673 parámetros |
| AUC test | 0,6696 | 0,6716 ± 0,0031 (30 semillas) |
| KS test | 0,2465 | 0,2500 ± 0,0054 |
| Brier test | 0,1865 | 0,1832 ± 0,0011 (mejor en 30/30 semillas) |
| Log-loss test | 0,5600 | 0,5490 ± 0,0033 (mejor en 30/30) |
| Tiempo total | 60,8 s | 940,4 s (702,4 búsqueda + 238,0 semillas) |

- **Lectura:** el MLP discrimina igual (la diferencia de AUC es menor que la desviación entre semillas) y calibra un poco mejor.
  La mayor parte de esa mejora en Brier y log-loss es **nivel**: el MLP predice una PD media más alta (0,204 frente a 0,187;
  tasa 0,262). Al igualar el nivel, la brecha de log-loss baja de 0,0101 a 0,0018 (`python scripts/diagnostico_nivel_pd.py`).
- **Semilla representativa:** la de log-loss de val mediana, posición (n − 1)//2 → semilla 20 (la usan la app y la E[L]).
- **Mejor época muy temprana:** 1 en la semilla 20 (va de 1 a 20 entre semillas); val tiene más malos que train.

### Scorecard (Parte C, Juan José)

- **Fórmula:** puntaje = offset − factor · logit(PD), con factor = PDO / ln 2 = 20 / 0,693 = **28,85** y
  offset = base − factor · ln(odds) = 600 − 28,85 · ln 50 = **487,12**. 600 puntos ↔ 50 buenos por 1 malo; +20 puntos duplican las odds.
- **Puntos por variable (logística):** β·x estandarizado repartido en tramos de p1–p99 de train; puntos base 525,96; la suma
  reproduce el puntaje (error 1,1·10⁻¹³). Mayor rango: ingreso anual 45,9 puntos.
- **MLP:** SHAP intervencional **del logit** (15 jugadores, las categóricas enteras; fondo 100 de train, 1 000 de test).
  Las 5 más importantes: plazo, uso de cupos rotativos, ingreso, consultas de 6 meses, líneas abiertas; estabilidad entre
  semillas ρ = 0,986.
- **Hipótesis:** 10 de 11 confirmadas; H9 (`pub_rec`) no concluyente (permutación < 0, PSI 0,27).

### LGD (Parte D, Camila y José)

- **Población:** castigados (*Charged Off*); EAD al castigo = `funded_amnt − total_rec_prncp`;
  LGD = 1 − (`recoveries` − `collection_recovery_fee`) / EAD, recortada a [0, 1].
- **MLP:** salida **sigmoide** (la LGD está en [0, 1]) y **entropía cruzada fraccional** (cuasi-verosimilitud Bernoulli; apunta a la
  media condicional; Papke y Wooldridge, 1996). [32, 64, 64, 32] leaky ReLU, dropout 0,39, L2 8,9·10⁻⁴, Adam lr 0,0094, batch
  2 048; 9 953 parámetros. Optuna 30 ensayos (126 s).
- **Resultados test:** MLP MAE 0,07686 ± 0,00037, RMSE 0,08553; lineal MAE 0,07712, RMSE 0,08659; **constante mediana
  MAE 0,0699, RMSE 0,0811: gana a los dos.** Por qué: la mediana minimiza el MAE y test está desplazado hacia LGD = 1
  (62,7 % frente a 25,9 % en train), consistente con censura (los castigados recientes aún no recuperan).

### Pérdida esperada (Parte E, Daniel)

- E[L] = PD · LGD · EAD con **EAD = `funded_amnt`** (originación; conservadora porque supone incumplir antes de amortizar).
- Test: **197,03 M USD** (18,6 % del monto). Resueltos: E[L] / pérdida realizada = **1,059** = PD 0,926 · LGD 0,971 · EAD 1,179
  (`descomposicion_razon`): la PD y la LGD se quedan cortas y la EAD sobreestima; se compensan. Con el MLP de PD: 208,20 M y 1,123.

### Producto

- App: https://trabajo2-redes-neuronales.vercel.app (José). Next.js estático, la red corre en el navegador desde el JSON;
  paridad con Python en 1 000/1 000 casos por modelo (|ΔPD| < 10⁻⁶), comprobada en el CI.
- Blog: https://danielgonzalez2709.github.io/Trabajo2_Redes_Neuronales/ · Cacería: **29 hallazgos** (Código 17, Matemáticas 5,
  Datos 4, Conceptos 3). Clon limpio en otra máquina (runner Linux, sin red): todo en verde.

## 2. Ejecución en vivo (lo que más pesa: 10 %)

Todos desde la raíz con el `.venv` activado. Las demos escriben en carpetas ignoradas por git.

| Qué | Comando | Tiempo aquí |
|---|---|---|
| Demo PD | `python run.py pd --config configs/pd/demo.yaml` | ~13 s |
| … cambiando un hiperparámetro | `python run.py pd --config configs/pd/demo.yaml --set mlp.lr=0.05` (o `mlp.dropout=0.5`, `mlp.capas=[64,32]`, `mlp.max_epocas=2`, `logistica.rejilla_C=[1.0e-2]`) | ~13 s |
| Demo LGD | `python run.py lgd --config configs/lgd/demo.yaml --set mlp.l2=0.1` | ~5 s |
| Scorecard con otro PDO | `python run.py scorecard --config configs/scorecard/base.yaml --set scorecard.pdo=40 --set salida=<carpeta temporal>` | ~2 s |
| **Optimización (búsqueda) de la PD, corta** | `python run.py pd-hpo --config configs/pd/hpo.yaml --set hpo.n_ensayos=3 --set hpo.destino=<tmp>/mlp_final.yaml --set hpo.salida=<tmp>` | 3 ensayos: 96 s; depende de las configuraciones que salgan |
| … cambiando el espacio | añadir `--set "hpo.espacio.activacion=[tanh]"` (o `hpo.espacio.lr=[1.0e-3,1.0e-2]`) | 2 ensayos con solo tanh: 25 s |
| **Optimización de la LGD, corta** | `python run.py lgd-hpo --config configs/lgd/hpo.yaml --set hpo.n_ensayos=3 --set hpo.destino=<tmp>/mlp_final.yaml --set salida=<tmp>` | no medido con 3; con 5 ensayos: 138 s (se pasa de 2 min) |
| Notebook (si lo piden así) | `notebooks/00_sustentacion.ipynb`: `SET = {"demo": "pd"}`, `CAMBIO = {"mlp.lr": 0.05}`, `PREDICCION = "…"` | ~15 s |

**Nunca** correr `pd-hpo` o `lgd-hpo` sin `hpo.destino` temporal: reescriben `configs/*/mlp_final.yaml`.

**Antes de correr, decir la predicción.** La tabla "Predicciones de referencia" del notebook 00 tiene 12 cambios ya corridos,
con lo que pasó de verdad. Los más útiles:

- `mlp.lr` ×10 en la PD → el MLP empeora poco (AUC 0,6545 → 0,649): Adam oscila, pero la parada temprana guarda la mejor época.
- `mlp.max_epocas=2` en la PD → apenas cambia: la base ya paraba sola antes de la época 20 (mejores épocas 5, 7 y 3).
- `scorecard.pdo=40` → factor 28,85 → 57,71; los rangos de puntos se duplican; la PD y la tasa de malos por banda no cambian.
- `scorecard.odds_base=100` → todo el puntaje baja exactamente un PDO (20 puntos).
- `mlp.l2=0.1` en la LGD → el MAE sube a 0,0838 (peor que la constante media, 0,0779): con tanta L2 los pesos se van a 0.

## 3. Preguntas probables y respuesta corta

**Redes y entrenamiento**

- *¿Por qué ReLU en la PD y sigmoide a la salida?* La búsqueda eligió ReLU (también probó leaky ReLU y tanh); la salida es
  sigmoide porque la PD es una probabilidad en (0, 1) y con la BCE eso es máxima verosimilitud Bernoulli.
- *¿Qué hace el L2 y por qué (λ/2)‖W‖²?* Penaliza pesos grandes; con el factor ½ su gradiente es λW, igual a `weight_decay = λ`
  de Adam/SGD (sin aplicarlo a los sesgos). No usamos `AdamW`, que desacopla el decaimiento y ya no es L2 (H-08, V-07).
- *¿Dropout?* Apaga neuronas al azar en entrenamiento (no en evaluación); regulariza. En la PD es bajo (0,094) porque la red es
  pequeña.
- *¿Parada temprana?* Se detiene tras 10 épocas sin mejorar la pérdida de val y restaura la mejor época.
- *¿Por qué Optuna TPE y no rejilla?* Con 8 hiperparámetros una rejilla explota; TPE propone donde la log-loss ha sido buena.
  En Optuna 5 el TPE es multivariado por defecto (H-16; Falkner et al., 2018).
- *¿Comparación justa?* Mismos datos y partición, mismo criterio (log-loss de val), presupuesto declarado (13 configuraciones
  frente a 40, y su tiempo), 30 semillas del MLP y número de parámetros reportado.

**Métricas**

- *AUC, Gini, KS:* discriminación (ordenar malos arriba). Gini = 2·AUC − 1. KS = máxima distancia entre las acumuladas de
  buenos y malos.
- *Brier y log-loss:* calidad de la probabilidad (calibración + discriminación). Por eso el MLP gana ahí y no en AUC.
- *¿Por qué la PD media (0,187) queda por debajo de la tasa de test (0,262)?* Cambio entre cosechas: el modelo aprendió con
  ≤ 2012 (16,3 % de malos). Arreglarlo exigiría recalibrar con datos recientes, que en originación no se tienen.

**Decisiones de datos**

- *¿Por qué partición temporal?* Imita el uso real (se presta hoy con lo aprendido del pasado) y evita fuga entre cosechas.
- *¿Por qué no 2015?* El 93 % de esos créditos no tiene desenlace: quedarían solo los que fallaron rápido (sesgo).

**LGD y E[L]**

- *¿Por qué la constante gana a la red?* Las 15 variables de originación casi no explican la LGD y test está corrido hacia 1;
  la mediana de train (0,943) queda cerca de esa masa y minimiza el MAE.
- *¿Por qué EAD = monto desembolsado?* Se conoce en la originación; es conservadora (sobreestima la exposición: factor 1,179).
  Alternativa: el saldo promedio según la amortización, que requiere suponer cuándo se incumple.

## 4. "¿Por qué tu compañero decidió…?" / "¿Qué cambiarías?"

| Quién | Decisión que pueden preguntar | Por qué | Qué cambiaríamos (honesto) |
|---|---|---|---|
| Daniel (A, E) | Partición ≤ 2012 / 2013 / 2014 y 15 variables sin `grade`/`int_rate` | Sin fuga y como en el uso real | Medir la deriva por cosecha y recalibrar la PD |
| Daniel (E) | EAD = `funded_amnt` y E[L] por bandas de puntaje | Conocida en la originación y con sesgo de signo conocido | Probar la EAD por amortización |
| Andrés (B) | Criterio log-loss de val para elegir C y la arquitectura; semilla mediana por val | La log-loss mide la probabilidad, que es lo que usa la E[L]; test nunca decide | Revisar la parada en la época 1 (¿paciencia mayor o lr menor?) |
| Andrés (B) | Demo con 50 000 créditos y 3 semillas | Cabe en < 2 min sin tocar los resultados reales | — |
| Juan José (C) | Puntos desde el logit, base 600, 50:1, PDO 20; SHAP del logit con las categóricas enteras | El logit es aditivo; agrupar evita filas one-hot imposibles (H-25) | SHAP tarda ~90 s: no sirve para la demo en vivo |
| Camila y José (D) | Sigmoide + BCE fraccional; población solo castigados | La LGD es una fracción en [0, 1] y solo hay recuperación observada en castigados | Un modelo en dos partes (¿LGD = 1? y cuánto si no), por la masa en 1 |
| José (E) | Inferencia en el navegador con el JSON | PyTorch no cabe en una función de Vercel (715,5 MB; H-29) y el sitio es estático y gratis | — |

## 5. Cacería en 30 segundos

- Meta del enunciado (≥ 5 en ≥ 3 categorías) cumplida con 29 en 4 categorías; al menos 1 por parte.
- Patrón 1: 11 de los 17 errores de código están en las **pruebas** ("esta prueba cubre X" y no lo cubría); 9 se cazaron con mutantes.
- Patrón 2: se equivocan todos los papeles (sesión principal, implementador, redactor, incluso el cazador: H-15); casi siempre
  los encontró **otro** agente.
- Ejemplos para contar: H-24 (SHAP "exacto" que dependía de `np.isclose`), H-26 (una guarda que dejó sobrescribir
  `pred_test.parquet`), H-08 (el ½ del L2 en la especificación), H-22 (la diagonal de calibración leída al revés).
