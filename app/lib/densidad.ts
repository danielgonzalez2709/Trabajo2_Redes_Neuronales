// Distribución de la población por barras de puntaje a partir de sus percentiles (`poblacion.percentiles_score`).
// Dominio puro.
import { percentil } from "./modelo";

export interface Barra {
  desde: number;
  hasta: number;
  /** Proporción de la población con puntaje en [desde, hasta): F(hasta) − F(desde). */
  masa: number;
}

/** Barras de ancho de puntaje igual entre p0 y p100. */
export const BARRAS = 25;

/**
 * `n` barras de ancho igual entre p0 y p100. La función de distribución F se interpola linealmente entre los 101
 * percentiles (F(p_k) = k/100 con p0..p100): es la misma regla del percentil que se muestra (`percentil` de lib/modelo), así que
 * la barra de la persona y su percentil cuadran. masa = F(hasta) − F(desde); las masas suman 1 (F(p0) = 0, F(p100) = 1).
 * Sin rango (todos los percentiles iguales) o con menos de 2 valores no hay barras.
 */
export function barrasDesdePercentiles(p: number[], n = BARRAS): Barra[] {
  if (!Number.isInteger(n) || n < 1) throw new Error(`número de barras inválido: ${n}`);
  for (let k = 0; k < p.length; k++) {
    if (!Number.isFinite(p[k])) throw new Error(`percentiles_score[${k}] no es finito`);
    if (k > 0 && p[k] < p[k - 1]) throw new Error(`percentiles_score decrece en el índice ${k}`);
  }
  if (p.length < 2) return [];
  const min = p[0];
  const max = p[p.length - 1];
  if (max === min) return [];
  const ancho = (max - min) / n;
  // `percentil` da el índice fraccionario entre percentiles (con 101 valores, el %) y acota a 100 desde p100.
  const F = (x: number) => (x >= max ? 1 : percentil(x, p) / (p.length - 1));
  const borde = (j: number) => (j === n ? max : min + j * ancho); // el último borde es p100 exacto
  return Array.from({ length: n }, (_, j) => {
    const desde = borde(j);
    const hasta = borde(j + 1);
    return { desde, hasta, masa: F(hasta) - F(desde) };
  });
}

/** Índice de la barra que contiene el puntaje (el borde va con la barra de la derecha); fuera, se acota; sin barras, -1. */
export function indiceBarra(barras: Barra[], puntaje: number): number {
  if (!Number.isFinite(puntaje)) throw new Error(`puntaje no finito: ${puntaje}`);
  if (barras.length === 0) return -1;
  const min = barras[0].desde;
  const ancho = (barras[barras.length - 1].hasta - min) / barras.length;
  const j = Math.floor((puntaje - min) / ancho);
  return Math.min(Math.max(j, 0), barras.length - 1);
}
