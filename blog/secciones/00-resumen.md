<!-- Responsable: Daniel (E). Enunciado (Anexo del núcleo), Entregable 1: enlaces al repositorio, a la aplicación web y
al video promocional. Las cifras de resultados se agregan cuando B, C y D tengan sus summary.json (no escribir ninguna antes).
Guía: blog/README.md -->
## Resumen

Un banco quiere saber, antes de prestar, qué tan probable es que una persona no pague y, si no paga, cuánto va a perder.
Respondemos las dos preguntas con redes neuronales sobre **237 695 créditos de consumo de Lending Club** originados entre
2007 y 2014, usando solo las **15 variables que se conocen al momento de la solicitud**. Evaluamos con una partición
temporal: entrenamos con los créditos hasta 2012, ajustamos con los de 2013 y probamos con los de 2014.

Para la **probabilidad de incumplimiento** comparamos una regresión logística con un perceptrón multicapa bajo las mismas
condiciones: mismos datos y partición, presupuesto de búsqueda declarado, 30 semillas y número de parámetros. Después
representamos el modelo como una **scorecard** e identificamos las variables que más pesan en el riesgo. Para la **pérdida
dado incumplimiento** comparamos una regresión lineal con un perceptrón multicapa con salida sigmoide, y combinamos ambos
modelos en la **pérdida esperada** de cada crédito. El modelo de probabilidad está publicado en una aplicación web donde
cualquier persona puede ver su puntaje, su lugar frente a la población y cómo cambiaría si cambiara alguna de sus
características.
<!-- TODO(resultados): 2–3 frases con lo que aportó la red frente a la logística y la lineal (AUC, Brier, MAE/RMSE con su
variabilidad entre semillas), desde results/data/pd/summary.json y results/data/lgd/summary.json. -->

| Recurso | Enlace |
|---|---|
| Repositorio (código, datos procesados, modelos) | https://github.com/danielgonzalez2709/Trabajo2_Redes_Neuronales |
| Aplicación web | pendiente: se publica en Vercel (Parte E) |
| Video promocional | pendiente: se publica en YouTube (Parte E) |
| Notebooks (demostración y sustentación) | `notebooks/` en el repositorio |
