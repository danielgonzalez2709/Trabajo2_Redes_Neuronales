// Formato de lo que se muestra (no cambia ningún cálculo). Dominio puro.
import { decimales } from "./campo";
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

/** Valor de una entrada para leer: número con su unidad (si la etiqueta la declara) o etiqueta de la categoría. */
export function formatearValor(e: Entrada, v: number | string): string {
  if (e.tipo === "cat") return e.etiquetas_categorias[String(v)] ?? String(v);
  const u = unidad(e);
  return `${formatearNumero(Number(v), e.paso)}${u ? ` ${u}` : ""}`;
}

const MENOS = "−"; // signo menos tipográfico

/** Diferencia entera de puntos con signo explícito: +13, −13, 0. */
export function formatearDelta(d: number): string {
  return d > 0 ? `+${d}` : d < 0 ? `${MENOS}${-d}` : "0";
}

/** Diferencia de los puntajes tal como se muestran (enteros): "532 → 545" y "+13" siempre cuadran. */
export function deltaPuntaje(antes: number, despues: number): number {
  return Math.round(despues) - Math.round(antes);
}

/** "+13 puntos", "−1 punto", "0 puntos". */
export function formatearPuntos(d: number): string {
  return `${formatearDelta(d)} ${Math.abs(d) === 1 ? "punto" : "puntos"}`;
}

export function etiquetaSinUnidad(e: Entrada): string {
  return e.etiqueta_es.replace(UNIDAD_FINAL, "");
}

const REFERENCIAS: Record<string, string> = { test: "prueba", val: "validación", train: "entrenamiento" };

/** Nombre en español de `poblacion.referencia` (partición; src/viz/paleta.py NOMBRE_SPLIT). */
export function nombreReferencia(r: string): string {
  return REFERENCIAS[r] ?? r;
}
