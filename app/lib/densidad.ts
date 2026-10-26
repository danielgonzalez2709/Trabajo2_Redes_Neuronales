// Densidad de la población a partir de sus percentiles del puntaje (`poblacion.percentiles_score`). Dominio puro.

export interface Tramo {
  desde: number;
  hasta: number;
  /** Probabilidad del tramo (con la de los tramos de ancho 0 fusionados). */
  masa: number;
  /** masa / (hasta - desde): el área del tramo es su probabilidad. */
  densidad: number;
}

/** Percentiles por tramo de la densidad: 20 tramos de 5 percentiles con los 101 de p0..p100 (menos ruido que 100). */
export const PERCENTILES_POR_TRAMO = 5;

/**
 * Los tramos van de p[c·j] a p[c·(j+1)] con c = porTramo; en cada uno cae c/(n-1) de la población (0,05 con los 101
 * de p0..p100 y c = 5), así que su densidad es 0,05 / (p[5(j+1)] - p[5j]). Un tramo de ancho 0 (percentiles
 * repetidos) no tiene densidad finita: su masa se suma al siguiente tramo con ancho (o al anterior si está al final).
 * El área total es 1.
 */
export function densidadDesdePercentiles(p: number[], porTramo = PERCENTILES_POR_TRAMO): Tramo[] {
  if (p.length < 2) return [];
  for (let k = 0; k < p.length; k++) {
    if (!Number.isFinite(p[k])) throw new Error(`percentiles_score[${k}] no es finito`);
    if (k > 0 && p[k] < p[k - 1]) throw new Error(`percentiles_score decrece en el índice ${k}`);
  }
  if (!Number.isInteger(porTramo) || porTramo < 1 || (p.length - 1) % porTramo !== 0) {
    throw new Error(`porTramo = ${porTramo} no divide los ${p.length - 1} tramos entre percentiles`);
  }
  const q = p.filter((_, k) => k % porTramo === 0); // p0, p5, …, p100
  const masaTramo = 1 / (q.length - 1);
  const tramos: Tramo[] = [];
  let pendiente = 0; // masa de los tramos de ancho 0 aún sin asignar
  for (let k = 0; k < q.length - 1; k++) {
    if (q[k + 1] === q[k]) {
      pendiente += masaTramo;
      continue;
    }
    tramos.push({ desde: q[k], hasta: q[k + 1], masa: masaTramo + pendiente, densidad: 0 });
    pendiente = 0;
  }
  if (tramos.length === 0) return [];
  tramos[tramos.length - 1].masa += pendiente;
  for (const t of tramos) t.densidad = t.masa / (t.hasta - t.desde);
  return tramos;
}
