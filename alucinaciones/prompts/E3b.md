# E3b — App: mejoras visuales pedidas por José (tooltips, tarjeta del resultado, gráfica, avatar) (José, 26-oct)

Prompt literal enviado al subagente `implementador-web`:

---

Tarea E3b de la Parte E (app), rama `dev/jose`. Repo: `C:\Users\Jose Miguel\Documents\ia\Trabajo2_Redes_Neuronales`
(Windows; Node 24). Sin commit (lo hace el orquestador). Prueba primero para toda lógica nueva. E1–E3 están en los commits
654ae06, c5a865f y c8b57ef: léelos antes de tocar nada. Mantén la arquitectura: lógica pura en `app/lib/` con Vitest,
hooks como adaptadores, componentes presentacionales con CSS Modules, `page.tsx` único contenedor, sin dependencias nuevas.
**No cambies `app/lib/preprocesar.ts` ni `modelo.ts`** (la paridad debe seguir pasando). **No toques `app/next.config.mjs`**
(tiene un cambio local que no es de esta tarea). Next.js 16: `app/AGENTS.md` y `app/node_modules/next/dist/docs/`.

Imágenes de referencia que mandó José (míralas con la herramienta de lectura de imágenes):
- Tarjeta: `C:\Users\JOSEMI~1\AppData\Local\Temp\claude\c--Users-Jose-Miguel-Documents-ia-Trabajo2-Redes-Neuronales\98a79476-19e5-4d18-ac33-31689a92c754\images\1.png`
- Gráfica: `...\images\2.png` (misma carpeta)
- Avatar: `...\images\3.png` (misma carpeta)

## 1. Tooltips de información en cada campo
- Junto a la etiqueta de cada control, un botón pequeño "i" (≥ 24×24 px de área táctil) que muestra la `ayuda_es` del
  JSON. Debe funcionar con hover, con foco de teclado y con toque en el celular (toggletip), y cerrarse con Esc o al tocar
  fuera. Sigue la guía `C:\Users\JOSEMI~1\AppData\Local\Temp\claude\c--Users-Jose-Miguel-Documents-ia-Trabajo2-Redes-Neuronales\98a79476-19e5-4d18-ac33-31689a92c754\scratchpad\guia-tooltips.md`
  (Popover API + anchor positioning) **con su fallback** para navegadores sin anchor positioning (`@supports`), sin polyfills.
- Quita el texto de ayuda siempre visible bajo cada etiqueta (para despejar el formulario), pero conserva el enlace accesible:
  el control sigue con `aria-describedby` hacia el texto de ayuda (oculto visualmente con `.oculta`).
- No inventes textos: solo `ayuda_es`.

## 2. Tarjeta del resultado como la referencia 1
- Encabezado "Tu resultado" con una píldora oscura con el nombre del modelo elegido. El selector segmentado MLP/logística se
  mantiene (encima de la tarjeta o en ella; decide tú lo más claro).
- **Puntaje grande en semibold (font-weight 600)** — no fino como la referencia. Debajo, la PD en %.
- **Anillo del percentil** a la derecha (SVG, `stroke-dasharray`), con el número y "percentil" dentro; accesible
  (`role="img"` + `aria-label`).
- Frase: "Tu riesgo estimado es menor que el de aproximadamente el **X %** de los créditos del conjunto de prueba." (X = el
  percentil ya calculado; más puntaje = menos riesgo).
- Dos bloques breves de explicación:
  - "¿Qué es el puntaje?": más alto = menos riesgo; **los números salen del bloque `scorecard` del JSON** (`base`,
    `odds_base`, `pdo`), no escritos a mano: "{base} puntos equivalen a {odds_base} personas que pagan por cada una que no
    paga, y cada {pdo} puntos más duplican esa proporción."
  - "¿Qué es la probabilidad de incumplimiento?": "La probabilidad estimada de que el crédito no se pague: que termine
    declarado incobrable, en impago o con atrasos de 31 a 120 días." (es la definición de "malo" del enunciado,
    `docs/spec/SPEC.md`, Anexo, "Variable objetivo"; no la cambies).
- La fila del otro modelo ("Con Regresión logística: 541 puntos · 13,4 % de incumplimiento") en una píldora clara.
- Línea final tenue con el desempeño del modelo elegido desde `metricas` del JSON: "Desempeño de este modelo en prueba:
  AUC 0,67 · KS 24,9 % · Brier 0,18" (formateado desde los valores, no escritos a mano).
- Fondo de la tarjeta: un tinte suave lavanda derivado de `#4a3aa7` (tono de la paleta del repo `src/viz/paleta.py`, que no
  es el color de ningún modelo), bordes redondeados grandes; el color de cada modelo se conserva en los puntos y marcas.
  Puedes añadir una ilustración decorativa sutil (SVG propio, `aria-hidden`) como la alcancía de la referencia. Contraste AA.
- Se conservan: el delta base → escenario con ▲/▼ y texto, y los botones "Guardar como mi perfil" / "Volver a mi perfil".

## 3. Gráfica como la referencia 2
- Tarjeta propia "Dónde te ubicas" con "Créditos de prueba" a la derecha.
- Barras verticales redondeadas de **ancho de puntaje igual** (~25 barras entre p0 y p100). La masa de cada barra se calcula
  en `app/lib/densidad.ts` interpolando linealmente la función de distribución a partir de los 101 `percentiles_score`
  (masa = F(b) − F(a); suma = 1), con pruebas. Barras en tinte claro del color del modelo; **la barra que contiene el puntaje
  de la persona, en color sólido del modelo**, con una etiqueta en píldora oscura encima "Tú · 541".
- Eje: mínimo y máximo, "← Más riesgo", "Puntaje", "Menos riesgo →". Pie: "Cada barra muestra qué parte de los créditos de
  prueba tiene un puntaje en ese rango (aproximada a partir de sus percentiles). La barra de color es la tuya."
- Accesible (`role="img"` + descripción). Respeta `prefers-reduced-motion` si animas la barra.

## 4. Avatar con personalidad
- Un personaje ilustrado propio en SVG inline (estilo línea negra y relleno blanco como la referencia 3, **no copies la
  imagen**), con más personalidad: su expresión cambia según el percentil del escenario (por ejemplo < 33 preocupado,
  33–66 neutral, ≥ 66 contento) y algún detalle propio (accesorio, mejillas, gesto). La regla de la expresión va en `lib/`
  con prueba. Colócalo donde se vea junto al resultado o al encabezado de "Tu perfil" (decide tú). `aria-hidden`.

## 5. Encabezado
- Quita el título "¿Me prestas?" (suena extraño). El `h1` pasa a ser "Conoce tu puntaje y tu probabilidad de
  incumplimiento". Ajusta el `<title>`/`metadata` de `layout.tsx` de forma coherente (sin "¿Me prestas?").

## 6. Arreglos pendientes de la revisión de E3
1. `YSi.tsx`: el texto de lista vacía debe decir que ningún cambio de una sola variable sube el puntaje **al menos 1 punto**.
2. Al aplicar una tarjeta del "¿Y si…?", no pierdas el foco: devuélvelo al encabezado de la sección (`tabIndex={-1}`).
3. `tests/escenarios.test.ts:150-159`: quita la aserción de tiempo frágil (`ms < 50`); deja solo el reporte.

## Verificación
- `npm --prefix app test` (con paridad) y `npm --prefix app run build` deben pasar.
- Levanta el servidor de desarrollo **en otro puerto (`npx next dev -p 3100` dentro de `app/`)** porque el 3000 está en
  uso por otro proceso: no lo detengas. Verifica en el navegador (escritorio y 390 px): tooltips con ratón, teclado y toque;
  tarjeta, anillo, gráfica, avatar con sus tres expresiones (cambia entradas para recorrerlas); sin errores de consola.
  Detén tu servidor al terminar. Capturas con prefijo `e3b-` en
  `C:\Users\JOSEMI~1\AppData\Local\Temp\claude\c--Users-Jose-Miguel-Documents-ia-Trabajo2-Redes-Neuronales\98a79476-19e5-4d18-ac33-31689a92c754\scratchpad\`.
- Si `app/next-env.d.ts` cambia solo por correr dev/build, devuélvelo (`git checkout -- app/next-env.d.ts`).

## Reglas
- Solo tocas `app/` (menos `next.config.mjs` y `public/model/*.json`). Textos en español neutro con tuteo; nunca voseo.
- Tus herramientas pueden convertir `\uXXXX` en el carácter real; verifica los bytes si escribes escapes.

## Reporte
Archivos creados/modificados, decisiones de diseño, salida resumida de test y build, qué verificaste en el navegador (rutas
de las capturas) y dudas.
