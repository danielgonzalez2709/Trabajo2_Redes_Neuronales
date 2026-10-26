<!-- Responsable: Daniel (E). Enunciado (Anexo del núcleo), Entregable 1: enlaces al repositorio, a la aplicación web y
al video promocional. Las cifras de resultados se toman de lo que ya dicen y verificaron las secciones 9 y 10 (no se escribe
aquí una lectura nueva). Guía: blog/README.md -->
<!-- Cifras: results/data/pd/summary.json (auc_test de logistica y mlp; n_parametros), results/data/lgd/summary.json
(mae_test de lineal y mlp), results/data/el/summary.json (bloques total y resueltos). Lectura: secciones 9.3, 10.1, 10.3,
10.6 y 10.7. -->
## Resumen

Un banco quiere saber, antes de prestar, qué tan probable es que una persona no pague y, si no paga, cuánto va a perder.
Respondemos las dos preguntas con redes neuronales sobre **237 695 créditos de consumo de Lending Club** originados entre
2007 y 2014, usando solo las **15 variables que se conocen al momento de la solicitud**. Evaluamos con una partición
temporal: entrenamos con los créditos hasta 2012, ajustamos con los de 2013 y probamos con los de 2014.

Para la **probabilidad de incumplimiento** comparamos una regresión logística con un perceptrón multicapa bajo las mismas
condiciones: mismos datos y partición, presupuesto de búsqueda declarado, 30 semillas y número de parámetros. Después
representamos el modelo como una **scorecard** e identificamos las variables que más pesan en el riesgo. Para la **pérdida
dado incumplimiento** comparamos una regresión lineal con un perceptrón multicapa con salida sigmoide, y combinamos ambos
modelos en la **pérdida esperada** de cada crédito. Los dos modelos de probabilidad se publicarán en una aplicación web
(pendiente: hoy el repositorio solo tiene los modelos exportados para ella) donde cualquier persona podrá ver su puntaje, su
lugar frente a la población y cómo cambiaría si cambiara alguna de sus características.
<!-- revisar: autor (Daniel y Camila: pasar a presente cuando la app esté publicada, y confirmar que lleva los dos modelos
con el selector, como en la sección 10.5) -->

**Resultados.** En la probabilidad de incumplimiento, la red discrimina igual o apenas mejor que la logística (AUC de prueba
0,6716 ± 0,0031 entre 30 semillas frente a 0,6696, una diferencia menor que la variabilidad entre semillas) y mejora en las
30 semillas el Brier y la log-loss, según un diagnóstico con la semilla mediana sobre todo porque su probabilidad media
queda más cerca de la tasa de malos de 2014; a
cambio tiene unas 55 veces más parámetros y no se descompone en puntos por variable (secciones 10.1 y 10.6). En la pérdida
dado incumplimiento la red no aportó: empata con la regresión lineal en el error absoluto medio de prueba (0,07708 ± 0,00059
frente a 0,07712), y ninguno de los dos modelos supera con claridad a predecir una constante (sección 10.3). Con la
logística, la red de LGD y el monto desembolsado como exposición, la pérdida esperada de los 74 154 créditos de prueba es
196,71 millones de USD (el 18,5 % de lo desembolsado), y en los créditos ya resueltos es 1,057 veces la pérdida realizada,
una razón cercana a 1 que en parte sale de errores de la probabilidad y de la exposición que se compensan (secciones 9.3 y
10.4).

| Recurso | Enlace |
|---|---|
| Reporte publicado (este blog) | https://danielgonzalez2709.github.io/Trabajo2_Redes_Neuronales/ (se actualiza con cada cambio en `blog/` que llega a `main`) |
| Repositorio (código, datos procesados, modelos) | https://github.com/danielgonzalez2709/Trabajo2_Redes_Neuronales |
| Aplicación web | pendiente: se publica en Vercel (Parte E) |
| Video promocional | pendiente: se publica en YouTube (Parte E) |
| Notebooks (demostración y sustentación) | `notebooks/` en el repositorio |
