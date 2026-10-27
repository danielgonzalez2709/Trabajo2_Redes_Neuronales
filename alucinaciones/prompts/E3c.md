# E3c — App: avatar con la ilustración de José + arreglos de la revisión de E3b (José, 26-oct)

Prompt literal enviado al subagente `implementador-web`:

---

Tarea E3c de la Parte E (app), rama `dev/jose`. Repo: `C:\Users\Jose Miguel\Documents\ia\Trabajo2_Redes_Neuronales`.
Sin commit. E3b está en el commit 30c89d1; léelo. No cambies `app/lib/preprocesar.ts`, `modelo.ts`, `app/next.config.mjs`
ni `public/model/*.json`. Sin dependencias nuevas. Next.js 16: `app/AGENTS.md`.

## 1. Avatar
- José eligió una ilustración propia, ya recortada y convertida: `app/public/avatar.webp` (256×256, 6,8 KB). No la
  modifiques.
- `app/components/Avatar.tsx`: reemplaza el personaje SVG por esa imagen en un círculo (`<img src="/avatar.webp" alt="">`
  con `width`/`height` fijos y `decoding="async"`; como la app es `output: "export"`, usa `<img>` simple, no `next/image`
  con optimización). Sube el tamaño a ~88 px en escritorio y ~72 px en móvil.
- Conserva la regla de `app/lib/avatar.ts` (preocupado / neutral / contento): muéstrala como una **carita pequeña en la
  esquina inferior derecha** del avatar (insignia circular de ~28 px con borde blanco, SVG propio de trazo negro; preocupado
  con fondo ámbar suave, neutral gris suave, contento verde suave). La insignia tiene texto accesible
  (p. ej. `aria-label="Riesgo alto"`) o el conjunto sigue `aria-hidden` si el estado ya se comunica en texto en la tarjeta;
  decide y explica.
- Quita el SVG viejo del personaje y lo que quede sin uso.

## 2. Arreglos de la revisión de E3b
1. `app/components/Info.tsx:18` + `Info.module.css:68`: detecta lo que el CSS realmente usa —
   `CSS.supports("position-area: top")` y `@supports (position-area: top)` — en lugar de `anchor-name`, para que el
   respaldo por JS se active en navegadores con anchor positioning antiguo.
2. `app/components/Info.tsx:100`: quita la rama muerta `globo.current?.contains(e.relatedTarget)` (y su `ref` si queda
   sin uso), o justifica en un comentario por qué existe.
3. `app/lib/densidad.ts` (`indiceBarra`): con puntaje no finito, lanza un error explícito; prueba.
4. Añade a `globals.css` un token `--ds-tinta: #1d1d1f` y úsalo donde hoy se repite ese color.

## Verificación
- `npm --prefix app test` y `npm --prefix app run build` deben pasar.
- El servidor de desarrollo del puerto 3000 ya corre sobre esta carpeta: no lo detengas; verifica en http://localhost:3000
  (recarga en caliente) en escritorio y 390 px: el avatar se ve nítido y redondo, y la insignia cambia con el percentil.
  Capturas con prefijo `e3c-` en
  `C:\Users\JOSEMI~1\AppData\Local\Temp\claude\c--Users-Jose-Miguel-Documents-ia-Trabajo2-Redes-Neuronales\98a79476-19e5-4d18-ac33-31689a92c754\scratchpad\`.
- Si `app/next-env.d.ts` cambia por el build, devuélvelo (`git checkout -- app/next-env.d.ts`).

## Reporte
Archivos tocados, decisiones, salida resumida de test y build, capturas y dudas. Español neutro.
