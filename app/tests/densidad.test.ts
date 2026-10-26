import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { ModeloPD } from "../lib/contrato";
import { BARRAS, barrasDesdePercentiles, indiceBarra } from "../lib/densidad";
import { percentil } from "../lib/modelo";

const suma = (b: { masa: number }[]) => b.reduce((s, x) => s + x.masa, 0);

describe("barrasDesdePercentiles", () => {
  it("por defecto, 25 barras de ancho igual entre p0 y p100", () => {
    expect(BARRAS).toBe(25);
    const p = Array.from({ length: 101 }, (_, k) => 500 + k); // uniforme en [500, 600]
    const b = barrasDesdePercentiles(p);
    expect(b).toHaveLength(25);
    expect(b[0].desde).toBe(500);
    expect(b[24].hasta).toBe(600);
    for (const x of b) {
      expect(x.hasta - x.desde).toBeCloseTo(4, 12);
      expect(x.masa).toBeCloseTo(0.04, 12); // uniforme: cada barra tiene 1/25
    }
    for (let j = 1; j < b.length; j++) expect(b[j].desde).toBe(b[j - 1].hasta);
  });

  it("la masa es F(hasta) − F(desde) con F lineal entre percentiles (a mano)", () => {
    // F(0) = 0, F(1) = 0,5, F(4) = 1; barras [0, 2] y [2, 4]; F(2) = 0,5 + 0,5 · 1/3.
    const b = barrasDesdePercentiles([0, 1, 4], 2);
    expect(b.map((x) => [x.desde, x.hasta])).toEqual([[0, 2], [2, 4]]);
    expect(b[0].masa).toBeCloseTo(2 / 3, 12);
    expect(b[1].masa).toBeCloseTo(1 / 3, 12);
  });

  it("la F es la misma del percentil que se muestra (lib/modelo)", () => {
    const p = Array.from({ length: 101 }, (_, k) => 400 + k * k * 0.02);
    const b = barrasDesdePercentiles(p);
    for (const x of b) expect(x.masa).toBeCloseTo((percentil(x.hasta, p) - percentil(x.desde, p)) / 100, 12);
    expect(suma(b)).toBeCloseTo(1, 12);
  });

  it("percentiles repetidos (masa puntual): la masa no se pierde y la suma es 1", () => {
    const b = barrasDesdePercentiles([0, 0, 0, 4], 2); // dos tercios de la población en 0
    expect(b[0].masa).toBeCloseTo(5 / 6, 12);
    expect(b[1].masa).toBeCloseTo(1 / 6, 12);
    expect(suma(b)).toBeCloseTo(1, 12);
  });

  it("sin rango (todos iguales) o con menos de 2 valores: sin barras", () => {
    expect(barrasDesdePercentiles([5, 5, 5])).toEqual([]);
    expect(barrasDesdePercentiles([5])).toEqual([]);
  });

  it("rechaza percentiles decrecientes o no finitos y un número de barras inválido", () => {
    expect(() => barrasDesdePercentiles([0, 2, 1])).toThrow();
    expect(() => barrasDesdePercentiles([0, NaN, 1])).toThrow();
    expect(() => barrasDesdePercentiles([0, 1, 2], 0)).toThrow();
    expect(() => barrasDesdePercentiles([0, 1, 2], 2.5)).toThrow();
  });

  it.each(["logistica", "mlp"])("JSON real (%s): 25 barras, masas ≥ 0 y suma 1", (nombre) => {
    const m = JSON.parse(
      readFileSync(join(__dirname, "..", "public", "model", `pd_model_${nombre}.json`), "utf8"),
    ) as ModeloPD;
    const b = barrasDesdePercentiles(m.poblacion.percentiles_score);
    expect(b).toHaveLength(25);
    expect(b.every((x) => x.masa >= 0 && x.hasta > x.desde)).toBe(true);
    expect(suma(b)).toBeCloseTo(1, 12);
  });
});

describe("indiceBarra", () => {
  const b = barrasDesdePercentiles(Array.from({ length: 101 }, (_, k) => 500 + k)); // ancho 4

  it("la barra que contiene el puntaje; en el borde, la de la derecha", () => {
    expect(indiceBarra(b, 500)).toBe(0);
    expect(indiceBarra(b, 503.9)).toBe(0);
    expect(indiceBarra(b, 504)).toBe(1);
    expect(indiceBarra(b, 541)).toBe(10);
  });

  it("p100 cae en la última; fuera del rango se acota a la primera o la última", () => {
    expect(indiceBarra(b, 600)).toBe(24);
    expect(indiceBarra(b, 450)).toBe(0);
    expect(indiceBarra(b, 700)).toBe(24);
  });

  it("sin barras: -1", () => {
    expect(indiceBarra([], 541)).toBe(-1);
  });
});
