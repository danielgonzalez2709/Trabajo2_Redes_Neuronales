// Densidad de la población a partir de sus percentiles del puntaje (`poblacion.percentiles_score`). Dominio puro.

export interface Tramo {
  desde: number;
  hasta: number;
  /** Probabilidad del tramo (con la de los tramos de ancho 0 fusionados). */
  masa: number;
  /** masa / (hasta - desde): el área del tramo es su probabilidad. */
  densidad: number;
}

/**
 * Entre dos percentiles consecutivos cae 1/(n-1) de la población (0,01 con los 101 de p0..p100), así que la densidad
 * del tramo k es 0,01 / (p[k+1] - p[k]). Un tramo de ancho 0 (percentiles repetidos) no tiene densidad finita: su masa
 * se suma al siguiente tramo con ancho (o al anterior si está al final). El área total es 1.
 */
export function densidadDesdePercentiles(p: number[]): Tramo[] {
  if (p.length < 2) return [];
  for (let k = 0; k < p.length; k++) {
    if (!Number.isFinite(p[k])) throw new Error(`percentiles_score[${k}] no es finito`);
    if (k > 0 && p[k] < p[k - 1]) throw new Error(`percentiles_score decrece en el índice ${k}`);
  }
  const masaTramo = 1 / (p.length - 1);
  const tramos: Tramo[] = [];
  let pendiente = 0; // masa de los tramos de ancho 0 aún sin asignar
  for (let k = 0; k < p.length - 1; k++) {
    if (p[k + 1] === p[k]) {
      pendiente += masaTramo;
      continue;
    }
    tramos.push({ desde: p[k], hasta: p[k + 1], masa: masaTramo + pendiente, densidad: 0 });
    pendiente = 0;
  }
  if (tramos.length === 0) return [];
  tramos[tramos.length - 1].masa += pendiente;
  for (const t of tramos) t.densidad = t.masa / (t.hasta - t.desde);
  return tramos;
}
