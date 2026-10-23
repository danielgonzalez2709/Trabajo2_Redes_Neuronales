# Trabajo 2 — Redes neuronales para datos tabulares (¿me prestas?)

Fuente de verdad: `docs/spec/SPEC.md` (núcleo con los contratos y, en su **Anexo, el enunciado literal del trabajo**) y la
spec de tu parte en `docs/spec/partes/` (A-datos, B-pd, C-scorecard, D-lgd, E-producto). Tablero: `docs/spec/PLAN.md`.
Idioma del proyecto: español.

**Al iniciar una sesión:** lee el núcleo completo (incluido el Anexo) y la spec de tu parte antes de proponer o cambiar nada.

## Memoria del proyecto (estado actual — se carga en cada sesión)
@docs/MEMORIA.md

**Obligatorio:** al cerrar una tarea del PLAN, fusionar un PR o registrar una decisión, actualizar `docs/MEMORIA.md`
(estado, decisiones vigentes, pendientes, cacería y una línea nueva arriba en la bitácora). Mantenerla ≤ 150 líneas.

## Reglas para cualquier sesión o subagente
- Cualquier integrante puede cambiar el Spec bajo su criterio (núcleo §0.1): por PR, subiendo la versión, con una línea en el
  historial y avisando a las partes afectadas si toca un contrato o un hito. El Anexo (enunciado) no se edita.
- Los subagentes no cambian el Spec: lo [DECIDIDO], [PENDIENTE] o [PROPUESTO] se escala al humano que los dirige.
- **Prohibido inventar** datos, significados de variables, constantes, referencias o resultados. Si falta algo:
  `TODO(dato-faltante)` / `TODO(cita)` y reportarlo.
- El significado de las variables se cita del diccionario de Kaggle, nunca de una IA.
- **Fuga de información:** solo variables disponibles en la originación (`configs/features.yaml`); preprocesador ajustado solo con train.
- Los experimentos nunca usan internet.
- Semillas con `src/common/seeds.py` (`base_seed + run_id`; numpy, torch y random).
- Prueba primero, luego código.
- Trabaja en tu rama personal (`dev/<nombre>`) y solo en los archivos de tus tareas; los compartidos (`src/common/`,
  `requirements.txt`, `run.py`, `configs/features.yaml`, README) solo por PR dedicado.

## Protocolo del orquestador (la sesión principal de cada integrante)
1. Tomar la siguiente tarea ⬜ tuya en la spec de tu parte cuyas dependencias estén ✅ (o trabajar con los datos sintéticos si no).
2. Armar el contexto del implementador (núcleo §10.2): texto de la tarea + su sección de la spec de la parte + las secciones
   del núcleo §5 que la tarea consume o produce + los apartados del Anexo que cubre la parte + prueba de aceptación.
   **Nada de otras partes.**
3. **Guardar el prompt literal** en `alucinaciones/prompts/<ID>.md` antes de enviarlo.
4. Despachar `implementador` (o `implementador-web` en `app/`/`video/`, `curador-datos` en tareas de datos).
5. Despachar `revisor-spec`; si falla, devolver al implementador con los hallazgos.
6. Despachar `revisor-calidad`; si falla, ídem.
7. Despachar el verificador indicado en la columna "+ Verificador".
8. Cualquier error de un subagente que se haya detectado (fórmula, dato, variable, referencia) → candidato en
   `alucinaciones/registro.md`.
9. Marcar la tarea en `PLAN.md` y en la spec de tu parte, y abrir el PR hacia `main` (lo revisa y fusiona solo Daniel).
10. Actualizar `docs/MEMORIA.md`.
