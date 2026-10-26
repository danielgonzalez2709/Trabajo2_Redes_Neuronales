import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { ModeloPD } from "../lib/contrato";
import { densidadDesdePercentiles } from "../lib/densidad";

const area = (t: { desde: number; hasta: number; densidad: number }[]) =>
  t.reduce((s, x) => s + (x.hasta - x.desde) * x.densidad, 0);

describe("densidadDesdePercentiles", () => {
  it("cada tramo tiene masa 1/(n-1) y densidad masa/ancho", () => {
    const t = densidadDesdePercentiles([0, 1, 3, 4]);
    expect(t).toHaveLength(3);
    expect(t[0]).toMatchObject({ desde: 0, hasta: 1 });
    expect(t[0].densidad).toBeCloseTo(1 / 3, 12);
    expect(t[1].densidad).toBeCloseTo(1 / 3 / 2, 12);
    expect(area(t)).toBeCloseTo(1, 12);
  });

  it("101 percentiles: densidad del tramo k = 0,01 / (p[k+1] - p[k])", () => {
    const p = Array.from({ length: 101 }, (_, k) => 400 + k * k * 0.02);
    const t = densidadDesdePercentiles(p);
    expect(t).toHaveLength(100);
    expect(t[10].densidad).toBeCloseTo(0.01 / (p[11] - p[10]), 12);
    expect(area(t)).toBeCloseTo(1, 12);
  });

  it("un tramo de ancho 0 se fusiona con el siguiente (su masa no se pierde)", () => {
    const t = densidadDesdePercentiles([0, 1, 1, 2, 4]);
    expect(t.map((x) => [x.desde, x.hasta])).toEqual([[0, 1], [1, 2], [2, 4]]);
    expect(t[1].masa).toBeCloseTo(0.5, 12); // 0,25 propio + 0,25 del tramo [1, 1]
    expect(t[1].densidad).toBeCloseTo(0.5, 12);
    expect(area(t)).toBeCloseTo(1, 12);
  });

  it("varios tramos de ancho 0 seguidos y al final se fusionan con el vecino", () => {
    const t = densidadDesdePercentiles([0, 2, 2, 2, 3, 3]);
    expect(t.map((x) => [x.desde, x.hasta])).toEqual([[0, 2], [2, 3]]);
    expect(t[1].masa).toBeCloseTo(0.8, 12); // 2 ceros + [2, 3] + el cero final
    expect(area(t)).toBeCloseTo(1, 12);
  });

  it("todos los tramos de ancho 0 o menos de 2 valores: sin densidad", () => {
    expect(densidadDesdePercentiles([5, 5, 5])).toEqual([]);
    expect(densidadDesdePercentiles([5])).toEqual([]);
  });

  it("rechaza percentiles decrecientes o no finitos", () => {
    expect(() => densidadDesdePercentiles([0, 2, 1])).toThrow();
    expect(() => densidadDesdePercentiles([0, NaN, 1])).toThrow();
  });

  it.each(["logistica", "mlp"])("JSON real (%s): área total = 1", (nombre) => {
    const m = JSON.parse(
      readFileSync(join(__dirname, "..", "public", "model", `pd_model_${nombre}.json`), "utf8"),
    ) as ModeloPD;
    const t = densidadDesdePercentiles(m.poblacion.percentiles_score);
    expect(t.length).toBeGreaterThan(0);
    expect(area(t)).toBeCloseTo(1, 10);
    expect(t.every((x) => x.hasta > x.desde && Number.isFinite(x.densidad))).toBe(true);
  });
});
