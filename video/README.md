# Video promocional (Remotion)

Guion: [`guion.md`](guion.md). Composición principal: `Promocional` (2:30, 1920×1080, 30 fps), con cada escena registrada
también por separado en la carpeta *Escenas* del Studio. Remotion es gratis para individuos y organizaciones sin ánimo de
lucro (LICENSE.md oficial, verificado el 24-oct-2026; registro de la cacería, V-05).

## Ver y editar

```bash
npm --prefix video install
npm --prefix video run dev
```
Abre el Studio en http://localhost:3000. Los cambios en `src/` se ven al guardar.

## Agregar los clips grabados (sábado 24)

1. Copiar cada clip a `video/public/clips/` (p. ej. `andres.mp4`, `equipo.mp4`, `app.mp4`).
2. En `src/Promocional.tsx`, poner el nombre del archivo en el `archivo=""` de su escena. Mientras esté vacío, la escena
   muestra un aviso con quién habla y qué dice.
3. Faltan también: la curva de calibración real (Parte B) en `src/escenas/Honesta.tsx` y la URL de la app en el cierre.

## Cifras

Las cifras salen del repositorio, no se escriben a mano: `src/datos.ts` lee `data/processed/resumen_datos.json`, y
`src/features.json` es la lista de `configs/features.yaml`. Si cambian las variables, se regenera desde la raíz con:

```bash
.venv\Scripts\python -c "import yaml, json; f=yaml.safe_load(open('configs/features.yaml',encoding='utf-8'))['features']; json.dump([{'nombre':x['nombre'],'etiqueta_es':x['etiqueta_es']} for x in f], open('video/src/features.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)"
```

## Exportar el MP4 (al final)

```bash
cd video
npx remotion render Promocional out/promocional.mp4
```
