// Expresión del avatar según el percentil del escenario (más percentil = menos riesgo). Dominio puro.

export type Expresion = "preocupado" | "neutral" | "contento";

/** Cortes en tercios del percentil: < 33 preocupado, de 33 a menos de 66 neutral, ≥ 66 contento. */
export const CORTES_EXPRESION = [33, 66] as const;

export function expresion(percentil: number): Expresion {
  if (!Number.isFinite(percentil) || percentil < 0 || percentil > 100) {
    throw new Error(`percentil fuera de 0–100: ${percentil}`);
  }
  if (percentil < CORTES_EXPRESION[0]) return "preocupado";
  if (percentil < CORTES_EXPRESION[1]) return "neutral";
  return "contento";
}
