// "¿Y si…?": candidatos por variable y los cambios de UNA variable que más suben el puntaje. Dominio puro.
import { alPaso } from "./campo";
import type { Entrada, ModeloPD } from "./contrato";
import type { Valores } from "./entradas";
import { inferir, type Resultado } from "./modelo";

/** Tope de valores que se prueban por variable numérica (la rejilla completa del monto tiene 1 347). */
export const MAX_CANDIDATOS = 60;

/**
 * Valores que se prueban para una entrada. Numérica: la rejilla min, min + paso, …, max; si pasa de MAX_CANDIDATOS,
 * MAX_CANDIDATOS puntos uniformes (índices redondeados, así que todos caen en el paso), siempre con min y max.
 * Categórica: todas sus `categorias` (las que ofrece el control).
 */
export function candidatos(e: Entrada): (number | string)[] {
  if (e.tipo === "cat") return [...e.categorias];
  const n = Math.round((e.max - e.min) / e.paso); // pasos de min a max
  const puntos = Math.min(n + 1, MAX_CANDIDATOS);
  const valores: number[] = [];
  for (let i = 0; i < puntos; i++) {
    const k = puntos === 1 ? 0 : Math.round((i * n) / (puntos - 1));
    const x = Math.min(alPaso(e, e.min + k * e.paso), e.max);
    if (valores[valores.length - 1] !== x) valores.push(x);
  }
  if (valores[valores.length - 1] !== e.max) valores.push(e.max); // max fuera de la rejilla: igual se prueba
  return valores;
}

export interface Cambio {
  nombre: string;
  desde: number | string;
  hasta: number | string;
  /** Puntaje nuevo − puntaje actual, sin redondear (> 0). */
  delta: number;
  antes: Resultado;
  despues: Resultado;
}

export interface OpcionesCambios {
  /** Solo cuentan los cambios con delta > minimo (por defecto 0). */
  minimo?: number;
}

/** Distancia al valor actual para desempatar dentro de una variable (categóricas: el orden de `categorias`). */
const distancia = (a: number | string, b: number | string) =>
  typeof a === "number" && typeof b === "number" ? Math.abs(a - b) : 0;

/**
 * Para cada variable, con las demás fijas, el candidato (distinto del actual) que más sube el puntaje del modelo;
 * en empate dentro de la variable, el más cercano al valor actual. Devuelve los k mejores con delta > minimo,
 * por delta descendente y, en empate, en el orden de `modelo.entradas`. Una variable aparece a lo sumo una vez.
 */
export function mejoresCambios(modelo: ModeloPD, valores: Valores, k = 3, opciones: OpcionesCambios = {}): Cambio[] {
  const minimo = opciones.minimo ?? 0;
  const antes = inferir(modelo, valores);
  const mejores: Cambio[] = [];
  for (const e of modelo.entradas) {
    const desde = valores[e.nombre];
    let mejor: Cambio | null = null;
    for (const hasta of candidatos(e)) {
      if (hasta === desde) continue;
      const despues = inferir(modelo, { ...valores, [e.nombre]: hasta });
      const delta = despues.score - antes.score;
      if (delta <= minimo) continue;
      if (!mejor || delta > mejor.delta || (delta === mejor.delta && distancia(hasta, desde) < distancia(mejor.hasta, desde))) {
        mejor = { nombre: e.nombre, desde, hasta, delta, antes, despues };
      }
    }
    if (mejor) mejores.push(mejor);
  }
  // sort es estable: en empate se conserva el orden de `entradas`.
  return mejores.sort((a, b) => b.delta - a.delta).slice(0, k);
}

/** ¿Mismo valor en todas las entradas del contrato? (perfil base frente al escenario). */
export function mismosValores(entradas: Entrada[], a: Valores, b: Valores): boolean {
  return entradas.every((e) => a[e.nombre] === b[e.nombre]);
}
