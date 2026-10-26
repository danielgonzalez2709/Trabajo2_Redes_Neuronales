// Formato de lo que se muestra (no cambia ningún cálculo). Dominio puro.
import type { Entrada } from "./contrato";

const UNIDAD_FINAL = /\s*\(([^()]+)\)\s*$/; // "Monto solicitado (USD)" -> "USD"

/** PD en porcentaje con 1 decimal y coma decimal: 0,18349 -> "18,3 %". */
export function formatearPD(pd: number): string {
  return `${(pd * 100).toFixed(1).replace(".", ",")} %`;
}

/** Puntaje redondeado a entero solo para mostrar. */
export function formatearPuntaje(s: number): string {
  return String(Math.round(s));
}

/** Percentil entero hacia abajo, para que "supera al X %" nunca exagere. */
export function formatearPercentil(p: number): string {
  return String(Math.floor(p));
}

/** Decimales de un paso: 25 -> 0, 0,1 -> 1, 0,05 -> 2. */
function decimales(paso: number): number {
  const t = String(paso);
  const i = t.indexOf(".");
  return i < 0 ? 0 : t.length - i - 1;
}

/** Número para leer (no para editar): punto de miles y coma decimal, con los decimales del paso. */
export function formatearNumero(x: number, paso: number): string {
  const d = decimales(paso);
  const [entera, frac] = Math.abs(x).toFixed(d).split(".");
  const miles = entera.replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  return `${x < 0 ? "-" : ""}${miles}${frac ? `,${frac}` : ""}`;
}

/** Unidad solo si la `etiqueta_es` la declara entre paréntesis al final; si no, null (no se inventa). */
export function unidad(e: Entrada): string | null {
  return UNIDAD_FINAL.exec(e.etiqueta_es)?.[1] ?? null;
}

export function etiquetaSinUnidad(e: Entrada): string {
  return e.etiqueta_es.replace(UNIDAD_FINAL, "");
}

const REFERENCIAS: Record<string, string> = { test: "prueba", val: "validación", train: "entrenamiento" };

/** Nombre en español de `poblacion.referencia` (partición; src/viz/paleta.py NOMBRE_SPLIT). */
export function nombreReferencia(r: string): string {
  return REFERENCIAS[r] ?? r;
}
