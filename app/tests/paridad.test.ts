import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { CasosParidad, ModeloPD } from "../lib/contrato";
import { inferir } from "../lib/modelo";

const DIR = join(__dirname, "..", "public", "model");
const leer = <T>(nombre: string): T => JSON.parse(readFileSync(join(DIR, nombre), "utf8")) as T;

describe.each(["logistica", "mlp"] as const)("paridad TS vs Python: %s", (nombre) => {
  const modelo = leer<ModeloPD>(`pd_model_${nombre}.json`);
  const casos = leer<CasosParidad>(`casos_paridad_${nombre}.json`);

  it("el contrato declara el modelo y las tolerancias esperadas", () => {
    expect(modelo.modelo).toBe(nombre);
    expect(casos.modelo).toBe(nombre);
    expect(casos.casos).toHaveLength(1000);
    expect(casos.tolerancia).toEqual({ pd: 1e-6, score: 1e-3 });
  });

  it("1000/1000 casos dentro de la tolerancia (|dPD| < 1e-6, |dscore| < 1e-3)", () => {
    let maxPD = 0;
    let maxScore = 0;
    const fallos: { id: unknown; dPD: number; dScore: number }[] = [];
    for (const c of casos.casos) {
      const r = inferir(modelo, c.entradas);
      const dPD = Math.abs(r.pd - c.pd);
      const dS = Math.abs(r.score - c.score);
      maxPD = Math.max(maxPD, dPD);
      maxScore = Math.max(maxScore, dS);
      if (!(dPD < casos.tolerancia.pd) || !(dS < casos.tolerancia.score)) {
        fallos.push({ id: c.id, dPD, dScore: dS });
      }
    }
    console.log(`[${nombre}] max|dPD| = ${maxPD.toExponential(3)}  max|dscore| = ${maxScore.toExponential(3)}`);
    // Un array vacío en lugar de un contador: Vitest muestra los casos que fallaron.
    expect(fallos).toEqual([]);
  });
});
