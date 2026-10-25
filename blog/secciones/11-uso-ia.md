<!-- Responsable: Daniel + todos. Enunciado (Anexo del núcleo): "Sección de uso de IA: los principales prompts utilizados
y una discusión de su impacto en el resultado final". Cada parte agrega aquí sus prompts principales (los literales están en
alucinaciones/prompts/). Guía: blog/README.md -->
## 11. Uso de inteligencia artificial

### 11.1 Cómo la usamos

Trabajamos con Claude Code (modelo Claude Opus 5.5) siguiendo el mismo método del Trabajo 1: **desarrollo guiado por una
especificación y por subagentes**. Primero escribimos una especificación (`docs/spec/`): un núcleo con las reglas, los
contratos entre partes y el enunciado literal, y una especificación por parte. Después cada tarea la implementa un subagente
con un contexto mínimo (solo la tarea, su sección de la especificación y los contratos que usa) y la revisan otros
subagentes antes de que un integrante abra el *pull request*. La [[tab:subagentes]] resume los papeles.

::: tabla subagentes
Título: Subagentes del proyecto y su función
Fuente: Elaboración propia a partir de .claude/agents/ y del núcleo §10 de la especificación.
| Subagente | Qué hace | Qué no puede hacer |
|---|---|---|
| implementador / implementador-web | Escribe la prueba y luego el código de una tarea | Cambiar la especificación o los contratos |
| revisor-spec | Comprueba que la tarea cumpla la especificación, ni más ni menos | Corregir el código |
| revisor-calidad | Legibilidad y pruebas, pensando en la modificación en vivo | Corregir el código |
| verificador-matematico | Deriva y comprueba numéricamente cada fórmula | Declarar algo correcto sin evidencia |
| curador-datos | Carga y documenta los datos con el diccionario como única fuente | Inventar el significado de una variable |
| cazador-alucinaciones | Contrasta afirmaciones de la IA con fuentes primarias | Registrar un hallazgo sin prompt literal y evidencia |
| analista-experimentos, auditor-reproducibilidad, redactor-tecnico, tutor-sustentacion | Experimentos, clon limpio, redacción y preparación de la sustentación | Afirmar algo que los resultados no respaldan |
:::

Todos los prompts se guardan literalmente en `alucinaciones/prompts/` antes de enviarse. Esa bitácora es la base de esta
sección y de la cacería de la alucinación (sección 12).

### 11.2 Principales prompts e impacto

- **Planeación (Parte E).** Los prompts con los que el equipo le pidió a la IA leer el enunciado, proponer el reparto y
  armar la especificación (`prompts/asesoria_puente_claude.md`, prompts A a K). Impacto: la estructura del trabajo
  (contratos congelados para que nadie esperara a nadie, especificación por partes, regla de *pull requests*) salió de esa
  conversación. Pero varias decisiones de la IA cambiaron al contrastarlas con los datos. La partición temporal que propuso
  al inicio (prueba = 2015) habría dejado una muestra de prueba con el 93 % de los créditos sin desenlace (sección 2.4).
- **Datos y análisis descriptivo (Parte A).** "Empecemos con lo mio, primero lo mas importante, que es la parte A que de eso
  dependen los demas" (prompt I), con la especificación de la Parte A. Impacto: la IA escribió la carga, el objetivo, la
  partición, el preprocesador, el análisis descriptivo y el borrador de las secciones 1 a 3. La revisión independiente
  (`prompts/A-revision-spec.md`) la **rechazó en la primera ronda** con dos errores bloqueantes, y nueve recomendaciones más;
  uno de los bloqueantes es el hallazgo H-02 de la sección 12. La aprobó en la segunda, después de las correcciones.
<!-- Parte B: agregar aquí los prompts principales de los modelos de PD y su impacto. -->
<!-- Parte C: ídem para la scorecard y SHAP. -->
<!-- Parte D: ídem para la LGD. -->
<!-- Parte E: ídem para la app y el video. -->

### 11.3 Lo que aprendimos de trabajar con IA

- **La IA escribe rápido y con seguridad también lo incorrecto.** De los dos errores de la Parte A (sección 12), el de
  código pasó las pruebas que la misma IA había escrito y solo apareció al mirar un resultado con sospecha (un 0,000
  exacto); el otro era una frase del reporte que sonaba razonable y solo cayó al contrastarla con el diccionario.
- **Separar al que escribe del que revisa funciona.** El revisor no veía la conversación del implementador, solo la
  especificación y el código, y por eso encontró afirmaciones que el implementador daba por obvias.
- **La especificación hace verificable a la IA.** Una regla como "el significado de cada variable se cita del diccionario de
  Kaggle, nunca de una IA" convierte una impresión ("esto suena raro") en una comprobación concreta.
