import { describe, expect, it } from "vitest";
import { expresion } from "../lib/avatar";

describe("expresion del avatar según el percentil del escenario", () => {
  it("< 33 preocupado, 33–66 neutral, ≥ 66 contento", () => {
    expect(expresion(0)).toBe("preocupado");
    expect(expresion(32.99)).toBe("preocupado");
    expect(expresion(33)).toBe("neutral");
    expect(expresion(50)).toBe("neutral");
    expect(expresion(65.99)).toBe("neutral");
    expect(expresion(66)).toBe("contento");
    expect(expresion(100)).toBe("contento");
  });

  it("rechaza un percentil no finito o fuera de 0–100", () => {
    expect(() => expresion(NaN)).toThrow();
    expect(() => expresion(-1)).toThrow();
    expect(() => expresion(101)).toThrow();
  });
});
