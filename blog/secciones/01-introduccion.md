<!-- Responsable: Daniel (A). Enunciado (Anexo del núcleo): Parte 1, tarea 1 (caso de uso). Guía: blog/README.md -->
## 1. Introducción y caso de uso

Un banco que presta dinero necesita responder dos preguntas antes de aprobar un crédito: qué tan probable es que la persona
no pague (la **probabilidad de incumplimiento**, PD) y, si no paga, qué fracción de la deuda se pierde (la **pérdida dado
incumplimiento**, LGD). En este trabajo respondemos ambas con redes neuronales sobre créditos de consumo de la plataforma
Lending Club, las comparamos con modelos de baja complejidad bajo condiciones equivalentes y convertimos el modelo de PD en
una herramienta web que cualquier persona puede usar.

### 1.1 Caso de uso

- **Quién usa el modelo:** el analista de originación de un banco, o el motor de decisión que lo apoya.
- **Para qué decisión:** aprobar o rechazar una solicitud de crédito de consumo y fijar sus condiciones.
- **En qué momento:** al recibir la solicitud, **antes** del desembolso.
- **Con qué información:** solo lo que el solicitante declara en la solicitud y lo que muestra su historial de buró en ese
  momento.

La última condición gobierna todas las decisiones posteriores. Cualquier variable que se actualice después del desembolso
(pagos recibidos, saldo pendiente, recuperaciones, estado del crédito) **no existe** cuando se toma la decisión; usarla para
entrenar daría un modelo que parece excelente en los datos y es inútil en la práctica (fuga de información). Por la misma
razón excluimos el grado y la tasa de interés que asigna Lending Club: son el resultado de *su* evaluación de riesgo, y el
banco del caso de uso quiere la suya propia. La sección 2 detalla qué variables entran y por qué.
