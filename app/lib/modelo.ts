// Forward de la red del JSON, PD, puntaje y percentil. Función pura (src/export/exportar_app.py, receta de la app).
import type { Activacion, Entradas, ModeloPD, Red, Scorecard } from "./contrato";
import { preprocesar } from "./preprocesar";

const PENDIENTE_LEAKY = 0.01; // negative_slope por defecto de torch.nn.LeakyReLU

const ACTIVACIONES: Record<Activacion, (a: number) => number> = {
  relu: (a) => Math.max(0, a),
  leaky_relu: (a) => (a >= 0 ? a : PENDIENTE_LEAKY * a),
  tanh: Math.tanh,
  identidad: (a) => a,
};

/** Logit: h <- phi(W h + b) por capa, con W de forma salida x entrada. */
export function logit(red: Red, x: number[]): number {
  let h = x;
  red.capas.forEach((capa, i) => {
    const f = ACTIVACIONES[capa.activacion];
    if (!f) throw new Error(`Capa ${i}: activación desconocida ${capa.activacion}`);
    if (capa.b.length !== capa.W.length) throw new Error(`Capa ${i}: b no encaja con W`);
    const entrada = h;
    h = capa.W.map((fila, j) => {
      if (fila.length !== entrada.length) {
        throw new Error(`Capa ${i}: W espera ${fila.length} entradas y llegan ${entrada.length}`);
      }
      let s = capa.b[j];
      for (let k = 0; k < fila.length; k++) s += fila[k] * entrada[k];
      return f(s);
    });
  });
  if (h.length !== 1) throw new Error(`La última capa debe tener una salida; tiene ${h.length}`);
  return h[0];
}

/** Sigmoide estable: nunca evalúa exp de un número positivo grande. */
export function sigmoide(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** Puntaje desde el logit (no desde la PD), sin redondear: offset - factor * z. */
export function puntaje(z: number, sc: Scorecard): number {
  return sc.offset - sc.factor * z;
}

/**
 * Percentil (0 a 100) del puntaje frente a la población de referencia (`percentiles_score`, p0..p100, no decrecientes).
 * Regla: interpolación lineal entre los 101 puntos; fuera del rango se acota a 0 o 100. Si hay valores repetidos
 * (tramo plano) se toma el último índice con p[i] <= s (la proporción de la población con puntaje <= s).
 * Más puntaje = mejor pagador: un percentil alto significa "mejor que ese porcentaje de la población".
 */
export function percentil(s: number, p: number[]): number {
  const n = p.length - 1;
  if (n < 1) throw new Error("percentiles_score debe tener al menos 2 valores");
  if (s <= p[0]) return 0;
  if (s >= p[n]) return 100;
  let i = 0;
  while (i < n - 1 && p[i + 1] <= s) i++; // último i con p[i] <= s
  return i + (s - p[i]) / (p[i + 1] - p[i]); // p[i+1] > s >= p[i]: sin división por cero
}

export interface Resultado {
  logit: number;
  pd: number;
  score: number;
  percentil: number;
}

/** Entradas crudas -> logit, PD, puntaje y percentil con un `pd_model_*.json`. */
export function inferir(modelo: ModeloPD, entradas: Entradas): Resultado {
  if (modelo.red.salida !== "sigmoid") throw new Error(`Salida de la red desconocida: ${modelo.red.salida}`);
  const z = logit(modelo.red, preprocesar(modelo.preprocesamiento, entradas));
  const score = puntaje(z, modelo.scorecard);
  return { logit: z, pd: sigmoide(z), score, percentil: percentil(score, modelo.poblacion.percentiles_score) };
}
