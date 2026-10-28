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
- **PD: regresión logística y MLP (Parte B, B1 a B10).** Desde B2, Andrés pedía cada tarea en dos tiempos: primero el plan ("Solo
  quiero el PLAN; no escribas código todavía.", B2) y después las fases roja y verde del implementador. En la ronda 4 de B2
  el orquestador reconoció un error suyo ("Error mío, no tuyo"): en la ronda anterior había pedido borrar comentarios y
  Andrés exige "UN COMENTARIO EXPLICATIVO POR LÍNEA IMPLEMENTADA". Impacto: salieron las métricas (B1), el MLP y el bucle de
  entrenamiento (B2), la logística (B3), la búsqueda con Optuna de 40 ensayos (B4), las 30 semillas (B5), la demo, las
  figuras, los blogs 04 a 06, el notebook 02 y la sensibilidad con `grade` e `int_rate` (B10). Los revisores rechazaron
  con frecuencia. En B1, el revisor-calidad pidió cambios en la primera ronda (H-04). En B2 la rechazó dos veces y la
  aprobó en la ronda 4. En B4 rechazó la primera ronda (el valor que el objetivo devuelve a Optuna no estaba probado) y en
  B9 también (celdas sin texto previo). En B10, el revisor-spec rechazó la ronda 1 y el revisor-calidad las rondas 1 y 2: la
  guarda de rutas que protegía los resultados del modelo principal seguía con huecos (H-26). Los hallazgos de la sección 12
  salieron de ahí: H-03 y H-04 (B1), H-05 a H-08 (B2; H-08 era un error de la especificación que había escrito la IA), H-09
  (B3), H-16 y H-17 (B4), H-18 y H-19 (B5), H-22 y H-23 (B7) y H-26 (B10). Un caso contrario: en B10 Andrés sospechó que
  Daniel ya había hecho lo que la IA proponía y pidió "en caso de ser así, registralo como una alucinación". La revisión
  mostró que Daniel había entregado el parquet y la configuración, pero no la unión ni el preprocesador de 17 variables que
  la fila B10 le asigna a B ("No es alucinación"), y no se registró.
- **Scorecard, SHAP e hipótesis (Parte C, C1 a C8).** Juan José abría cada tarea con el protocolo del orquestador y una
  condición de control: "preséntame el plan (archivos, funciones con firmas, pruebas, supuestos) y ESPERA MI APROBACIÓN"
  (C1). Impacto: la transformación de probabilidad a puntaje con base 600, odds 50:1 y PDO 20 (C1), el exportador de los
  modelos a la app (C2, C5), la scorecard de la logística (C3), el SHAP del MLP con su permutación (C4), el cierre de las 11
  hipótesis (C6), el blog 07 y el notebook 03 (C7, C8). La mitad de los rechazos fueron por pruebas débiles. En C1, el
  verificador-matematico rechazó las pruebas de la fase roja por dos huecos (H-11) y el revisor-calidad pidió cambios porque
  texto y booleanos pasaban las transformaciones (H-12); se aprobó en la segunda ronda. En C3, el verificador cazó con
  mutantes una prueba cuyos parámetros coincidían (H-20). En C4, el revisor-calidad exigió cambios en la ronda 1 por el
  `np.isclose` del masker de SHAP (H-24) y el verificador-matematico rechazó las pruebas por débiles (H-25); el revisor-calidad
  aprobó en la ronda 2 y el verificador-matematico, en la siguiente. En C5, el revisor-calidad pidió que el export fuera "todo o nada", porque un fallo del segundo
  modelo dejaba mezclados los archivos del primero; no fue un hallazgo de la cacería. En C6, el revisor-calidad rechazó la
  ronda 1 por las pruebas de la regla del veredicto (H-27), y en C7 el verificador-matematico corrigió una nota sobre `log10`
  (H-28). El PR de Spec v0.25 (`prompts/C-spec-v0.25.md`) lo aprobó el revisor-spec sin bloqueantes. C8 no dejó hallazgos.
- **LGD (Parte D, D1 a D10).** Los prompts de Camila usan una plantilla de "Reglas duras": cero escrituras en git, prueba
  primero y, para los datos, "Nunca parafrasees el diccionario de memoria." (D1). Para la pérdida del MLP, la propia
  especificación advertía que "La BCE solo sirve con objetivos 0/1" es falso para la cuasi-verosimilitud fraccional y ordenaba
  "no lo afirmes en código ni docstrings" (D3). Impacto: el objetivo y la EAD citados del diccionario (D1), la población de
  43 236 créditos (D2), el modelo lineal y el MLP con sigmoide (D3), la búsqueda de 30 ensayos y las 30 semillas (D4), siete
  figuras (D6), el blog 08 y el notebook 04. En D3, el revisor-spec aprobó con siete recomendaciones, el revisor-calidad
  rechazó por el NaN en silencio y el verificador-matematico halló una consistencia atribuida al MLP (H-10, el único
  hallazgo de la Parte D); además, en la primera ronda el implementador hizo un commit pese a la prohibición y el
  orquestador lo deshizo. En D4, el revisor-spec y el revisor-calidad rechazaron la ronda 1: `mejor_config.json` pisaba los
  `--set mlp.*`, una falla que habría pasado en silencio en la sustentación; la ronda 2 fue aprobada. En D8, el revisor-spec
  rechazó la primera ronda y, según el prompt de la ronda 2, la causa fue "una omisión del encargo del orquestador". Después
  José (D9, D10, D4b y D7b) alineó la L2 con la de B, $(\lambda/2)\lVert W\rVert^2$, y rehizo la búsqueda y las 30 semillas:
  el MLP queda con un MAE de prueba de 0,07686 y la mediana constante sigue ganando con 0,06986 (`docs/MEMORIA.md`).
- **Pérdida esperada, app y blog (Parte E).** Daniel hizo E5, E7, E11 y T5.4; José, E1 a E3c. La primera versión de la
  pérdida esperada la escribió el orquestador directamente (24-oct); la del 26-oct pasó por subagentes con la regla
  "Prohibido inventar datos, constantes o referencias" (E5). El revisor-spec y el verificador-matematico rechazaron la ronda
  1: el contraste mezclaba créditos resueltos y en mora (H-13). Luego el
  revisor-calidad rechazó por un bloqueante (`CRITERIO_BANDAS` decía "deciles" con cualquier número de bandas) y la ronda 3
  lo corrigió. En el blog, el encargo de la sección 09 pedía explicar por qué la razón queda cerca de 1 "SIN afirmar más de
  lo que muestran los datos" (E7). Aun así, el verificador y el cazador rechazaron la ronda 1 por redacción (H-14), y en la
  ronda 2 el cazador rechazó una frase que venía de su propio informe (H-15). La sección 10 también se rechazó en la ronda
  1: el verificador halló 1 afirmación falsa y 2 imprecisas, y el cazador 5 incorrectas y 7 imprecisas, entre ellas la de
  H-21. La app de José (E1 a E3c) se pidió con "TDD: la prueba de paridad se escribe primero y debe fallar antes de
  implementar" (E1). La paridad entre Python y TypeScript da 1 000 de 1 000 casos por modelo, con |ΔPD| de 2,2e-16
  (`docs/MEMORIA.md`), y la app no aportó hallazgos a la cacería. Para el notebook 00, el tutor-sustentacion escribió 9
  predicciones de cambios en vivo antes de correrlas: 4 se cumplieron, 3 en parte y 2 no, y el prompt ordenaba dejar las
  fallidas ("es material para la sustentación, no se maquilla", T5.4). Esta misma sección se redactó con un prompt guardado
  en `prompts/E7.md`.

### 11.3 Lo que aprendimos de trabajar con IA

La sección 12 reúne 29 hallazgos confirmados. De ahí salen las lecciones.

- **La IA escribe rápido y con seguridad también lo incorrecto.** Los errores son 17 de código, 5 de matemáticas, 4 de datos
  del mundo real y 3 de conceptos; no hubo errores de bibliografía. En la Parte A, el de código pasó las pruebas que la
  misma IA había escrito y solo apareció al mirar un resultado con sospecha (un 0,000 exacto); el otro era una frase del
  reporte que sonaba razonable y solo cayó al contrastarla con el diccionario.
- **Las pruebas son el punto más débil.** Once de los 17 errores de código estaban en las pruebas o en la afirmación "esta
  prueba cubre X", y nueve de ellos se cazaron con mutantes. Por eso pedimos "prueba primero" y después comprobamos que cada
  prueba nueva falle con su mutante. Una prueba que no puede fallar (H-05) o cuyos parámetros coinciden (H-20) parece una
  cobertura que no existe.
- **Separar al que escribe del que revisa funciona.** El revisor no veía la conversación del implementador, solo la
  especificación y el código, y por eso encontró afirmaciones que el implementador daba por obvias. Solo 4 de los 29 errores
  (H-01, H-09, H-15 y H-22) los cazó el mismo agente que los cometió.
- **Todos los papeles se equivocan, también los que verifican.** Según el registro, 11 errores los cometió una sesión
  principal (la que dirige cada integrante o planeó el trabajo), 12 el implementador, 2 el redactor, 1 el cazador (H-15) y 3
  una sesión principal junto con otro subagente. La especificación que escribió la IA también tuvo errores (H-08 y H-16). Por
  eso ninguna cifra ni afirmación de un subagente se acepta sin otra revisión.
- **La especificación hace verificable a la IA.** Una regla como "el significado de cada variable se cita del diccionario de
  Kaggle, nunca de una IA" convierte una impresión ("esto suena raro") en una comprobación concreta, y lo mismo hace el
  "prohibido inventar" de los prompts. No todo estaba mal: al verificarlas, las 14 referencias que propuso la IA existen y
  sus DOI o direcciones oficiales llevan al trabajo esperado, con matices de año y de edición que vienen de los registros
  (sección 12).
- **El humano decide, y a veces la sospecha no se confirma.** Cada hallazgo lo confirma un integrante, y Andrés cazó H-05 al
  revisar el plan; pero su sospecha en B10 no se confirmó y no se registró. Un registro honesto incluye lo que no era un
  error.
- **Los revisores tienen costo.** Cada tarea pasa por varios subagentes y varias rondas, y eso consume tokens. Por eso Daniel
  fijó el 26-oct tres medidas (`docs/MEMORIA.md`): el blog con un solo verificador (el cazador, con el modelo sonnet),
  informes de subagente de 15 líneas como máximo y sesiones cortas. Los prompts de las partes B, C y E ya piden "Informe ≤ 15
  líneas".
