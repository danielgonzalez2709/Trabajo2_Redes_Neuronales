// Regla del campo numérico (rango y paso del JSON). Dominio puro: la usan el control y los escenarios.
import type { EntradaNum } from "./contrato";

/** Decimales de un paso: 25 -> 0, 0,1 -> 1, 0,05 -> 2. */
export function decimales(paso: number): number {
  const t = String(paso);
  const i = t.indexOf(".");
  return i < 0 ? 0 : t.length - i - 1;
}

/** Mientras se escribe solo se comunica un número finito dentro de [min, max] (aunque no caiga en el paso). */
export function esValido(e: EntradaNum, x: number): boolean {
  return Number.isFinite(x) && x >= e.min && x <= e.max;
}

/** Redondea a la rejilla min + k·paso (la del deslizador), sin ruido de coma flotante. No acota al rango. */
export function alPaso(e: EntradaNum, x: number): number {
  const k = Math.round((x - e.min) / e.paso);
  const d = Math.max(decimales(e.paso), decimales(e.min));
  return Number((e.min + k * e.paso).toFixed(d));
}

/** Al salir del campo: ajusta al paso y acota a [min, max]; null si lo escrito no es un número. */
export function acotar(e: EntradaNum, x: number): number | null {
  if (!Number.isFinite(x)) return null;
  return Math.min(Math.max(alPaso(e, x), e.min), e.max);
}
