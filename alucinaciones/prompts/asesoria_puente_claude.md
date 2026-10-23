# Asesoría inicial del puente con Claude (planeación del Trabajo 2)

- **Herramienta y modelo:** Claude Code (app de escritorio), Claude Opus 5.5 · **Fecha:** 19-oct-2026 · **Quién:** Daniel
- Las respuestas completas están en la conversación de Claude Code; aquí se copian los prompts literales del usuario y se
  resumen las afirmaciones verificables de las respuestas (lista V-01…V-11 en `../registro.md`).

## Prompt A (literal)

> @"C:\Users\danie\dev\Trabajo 2 Redes Neuronales\Trabajo 2_ Redes neuronales para datos tabulares (¿me prestas.pdf"
> Necesito que me ayudes en mi segundo trabajo de mi materia de redes neuronales, somos 5 integrantes: andres, camila, yo, jose y juan jose. Adjunto te voy a dejar el enunciado, el cual contiene todo el enunciado y las instrucciones para la entrega:
>
> [se pegó el apartado "Propósito" del enunciado]
>
> No hagas nada todavía, solo lee, y proponme un plan para repartir las tareas, metodos que usar. Propongo para el despliegue de la aplicacion vercel, necesito que me confirmes eso, porque es una aplicacion web dinimica, con su base de datos, segun tengo entendido, la idea es desarrollar este trabajo gastando 0 pesos para el despliegue de la aplicacion; el video lo podemos hacer con remotion, con animaciones como si fuera una presentacion de un prototipo de apple, algo asi bien estetico e innovador

**Respuesta (resumen de lo verificable):** Vercel sirve gratis (Hobby); el enunciado no exige base de datos; no desplegar PyTorch en
funciones de Vercel por el límite de ~250 MB; inferencia del MLP en el navegador desde un JSON; prueba de paridad Python/JS;
Quarto para el blog (luego se cambió a MkDocs con el hook del Trabajo 1); Remotion para el video con clips reales de los 5;
métodos: exclusión de variables posteriores a la originación, partición temporal 2007–2013 / 2014 / 2015 con la advertencia de
sesgo por créditos aún no resueltos, Optuna TPE, `pos_weight` y calibración, scorecard con factor = PDO/ln 2, LGD con
`recoveries`, `collection_recovery_fee`, `funded_amnt` y `total_rec_prncp`, SHAP, referencias sugeridas (sin verificar).

## Prompt B (literal)

> La fecha de entrega es para el 27 de octubre a las 11:59pm, pero hagamos algo, varias cosas, primero yo voy a ser el unico que pueda aceptar PR, cada persona va a tener su propia rama de desarrollo y hacen PR, yo lo acepto y seguimos. Lo otro, necesito que me expliques bien lo de:
>
> [se pegó el apartado "b) No pongan PyTorch en un backend Python dentro de Vercel…" de la respuesta anterior]
>
> No lo entiendo muy bien y dime que implicaciones tiene en el desarrollo como equipo.
>
> Otra cosa, me parece mejor que cada persona se ocupe de una parte en especifico, digamos, andres, camila parte 1 del trabajo, es un ejemplo, no tiene que ser asi, pero es para que la persona que esta trabajando tenga siempre claro, sobre que esta trabajando y que debe tener en cuenta.
>
> La idea es que trabajemos con Spec Driven Development, junto con Subagent Driven Developmente, asi lo hicimos para el trabajo 1, hay que armar un muy buen spec teniendo en cuenta el tiempo y los requisitos que tiene el trabajo.
>
> Si tienes alguna duda de algo mas me puedes decir; entonces vuelve y reparte asi como te dije, por partes del trabajo y a mi me gustaría meterle mano a algun punto a parte pues de ser le puente, probar, desplegar la app, hacer el blog y el video

**Respuesta (resumen):** explicación de exportar pesos + preprocesamiento a JSON y hacer el forward en TypeScript; contrato de
exportación; preprocesamiento "exportable"; activaciones simples; 12–18 variables para el formulario; ruleset + CODEOWNERS
(gratis en repos públicos); reparto por partes A–E; cronograma 19–27 oct; estructura del Spec.

## Prompt C (literal)

> 1.La ruta del trabajo 1 es esta: C:\Users\danie\dev\Trabajo 1 Redes Neuronales
> 2.  danielgonzalez2709, ya el repo es publico, aca está el link
> 3. Es despues pero todo debe estar lisot para el 27
> 4. Pues yo diría que el PC de todos tiene GPU
> 5. Todavía no, estoy esperando a que digan que les gustaría hacer
>
> Otra cosa, que opcion hay que tenga mas compatiblidad para el despliegue de la app, que no cause tanta interferencia

**Respuesta:** este Spec v0.1, el PLAN, los agentes y las plantillas; comparación de opciones de despliegue.

## Prompt D (literal, 22-oct-2026)

> Los compañeros estan proponiendo que cada parte del trabajo depende una del otra, entonces basicamente tiene road blocks, y tambien proponen que cada parte tenga su propia spec, para que los agentes no lean informacion que no necesitan, que opinas al respecto, hay que repartir el trabajo entre los 5, porque estamos cogidos del tiempo

**Respuesta (resumen):** contratos congelados + datos sintéticos con el mismo esquema para quitar bloqueos; núcleo con los
contratos + una spec por parte; reparto y cronograma comprimidos al 27-oct.

## Prompt E (literal)

> Dame el spec yo lo reviso

**Respuesta:** Spec v0.2 (núcleo + `docs/spec/partes/`), PLAN como tablero.

## Prompt F (literal)

> La idea es que la parte A la haga yo, junto con E que la hacemos Camila y yo, ella me va a ayudar con el frontend de la app;  la parte B y C lo van a desarrollar Andres y Juan Jose, y la parte D la van a desarrollar entre Camila y Jose. Necestito que en el spec este explicatemente el enunciado del trabajo, para que siempre tenga el contexto necesario y ten en cuenta que cualquier persona puede hacer cambios en el spec bajo su criterio; muestrame el spec que construiste y te doy el visto bueno

**Respuesta:** Spec v0.3 (equipo asignado, enunciado en el Anexo, regla §0.1 de cambios).

## Prompt G (literal)

> Dos cosas, primero el profe en el primer trabajo nos dio un consejo al momento de correr el codigo con algunos requerimientos nosotros debiamos correr un comando para que cambiara y el dijo que le parecería mas facil si hubieran sido celdas de un jupyter notebook, se podrían hacer a parte por fuera de la logica del trabajo o exportarlo, no se lo que sientas que sea mas eficiente; la otra es que no veo que en el spec menciones  un analisis descriptivo despues del preprocesamiento de los datos, si lo hay dime en que parte si no se deberia de agregar;

**Respuesta:** Spec v0.4 (notebooks como capa delgada, §5.8; descriptivo después del preprocesamiento, A §3.8 y D §3.2b;
afirmación por verificar: fórmula y umbrales del PSI).

## Prompt H (literal)

> Visto bueno. Haz el primer commit, súbelo a main, crea las ramas dev/ de cada uno y dame los pasos del ruleset.

## Prompt I (literal, 23-oct-2026)

> Empecemos con lo mio, primero lo mas importante, que es la parte A que de eso dependen los demas

**Respuesta:** A1–A5 (carga, objetivo, variables, partición, preprocesador, parquet) en `dev/daniel`.

## Prompt J (literal)

> Eso te iba a decir, nunca hagas push directamente al repo, primero siempre PR, yo lo autorizo y mergeo con tu ayuda, esto es para todas las partes, debe estar en el spec general; en este momento no tengo como seleccionar en que cuenta Se debe hacer el push, entonces, pues porque no estoy en la casa, estoy lejos del computador. Entonces, si quieres, puedes ir adelantando, deja los commits y ya después, cuando llegue, hacemos el PR necesario. O sea, desarrolla todo, haz los commits localmente, y ya cuando llegue, pasamos el PR.

## Prompt K (literal)

> Pues eso a lo que me refiero con el permiso para el push, obviamente es para la rama main. Para mi rama, si hagámoslo todo directamente, pues pushamos todo de una. Y para mergear lo de mi rama a la principal, sí debe ser siempre por PR. Eso para todas las ramas, pues de todos los compañeros. Pero para cada rama sí se puede hacer push tranquilamente sin hacer PR. Pero lo que sí quiero es que vayas trabajando con el commit, con todos los commits que haya que hacer, y ya cuando llegue yo hacemos push a mi rama. Y ya después verificamos y hacemos el PR hacia la principal para que los compañeros puedan seguir.

**Respuesta:** regla de push en el núcleo §7.2 (Spec v0.6); A6 y A6b (EDA, hipótesis, descriptivo posterior al
preprocesamiento) con commits locales; hallazgo H-01 (PSI).
