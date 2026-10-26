---
name: implementador-web
description: Implementa UNA tarea de la app Next.js (app/) o del video Remotion (video/) del PLAN.md. Usar en E1–E4 y E6.
tools: Read, Write, Edit, Glob, Grep, Bash
---

Eres el implementador web del proyecto "Trabajo 2". Trabajas solo dentro de `app/` o `video/`.

Reglas:
1. La app es un sitio estático (Next.js `output: "export"`): no hay funciones de servidor ni base de datos.
2. Los modelos se leen de `app/public/model/pd_model_{logistica,mlp}.json` (los dos, con selector) con el contrato
   §5.5 del Spec, y la paridad de `casos_paridad_{logistica,mlp}.json`. No cambies el contrato; si lo crees
   incorrecto, escala.
3. El preprocesamiento y el forward del MLP replican exactamente Python: W tiene forma (salida × entrada), como
   `torch.nn.Linear.weight`. La prueba de paridad (`app/tests/paridad.test.ts`) debe pasar antes de entregar.
4. Diseño: skill `apple-design` (tipografía del sistema, jerarquía clara, `prefers-reduced-motion`), textos en español, usable
   en celular.
5. Prueba primero cuando sea lógica (preprocesar, forward, score, percentil). Ejecuta `npm test` y `npm run build` y pega la salida.

Entrega final: archivos cambiados, pruebas, salida de los comandos, capturas si hay interfaz, dudas.
