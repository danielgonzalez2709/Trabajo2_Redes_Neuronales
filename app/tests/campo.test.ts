import { describe, expect, it } from "vitest";
import type { EntradaNum } from "../lib/contrato";
import { acotar, alPaso, decimales, esValido } from "../lib/campo";

const MONTO: EntradaNum = {
  nombre: "loan_amnt", tipo: "num", etiqueta_es: "Monto (USD)", ayuda_es: "",
  min: 1000, max: 5000, paso: 25, valor_defecto: 2000,
};
const DTI: EntradaNum = { ...MONTO, nombre: "dti", min: 0.6, max: 32.5, paso: 0.1, valor_defecto: 15 };

describe("decimales del paso", () => {
  it("25 -> 0, 0,1 -> 1, 0,05 -> 2", () => {
    expect(decimales(25)).toBe(0);
    expect(decimales(0.1)).toBe(1);
    expect(decimales(0.05)).toBe(2);
  });
});

describe("esValido (lo que se comunica mientras se escribe)", () => {
  it("dentro de [min, max] y finito", () => {
    expect(esValido(MONTO, 1000)).toBe(true);
    expect(esValido(MONTO, 5000)).toBe(true);
    expect(esValido(MONTO, 1013)).toBe(true); // fuera del paso pero en rango: se ajusta al salir
  });

  it.each([999, 5001, NaN, Infinity, -Infinity])("rechaza %s", (x) => {
    expect(esValido(MONTO, x)).toBe(false);
  });
});

describe("alPaso", () => {
  it("redondea a la rejilla min + k * paso", () => {
    expect(alPaso(MONTO, 1013)).toBe(1025); // 0,52 pasos -> 1
    expect(alPaso(MONTO, 1012)).toBe(1000);
    expect(alPaso(MONTO, 4999)).toBe(5000);
  });

  it("sin ruido de coma flotante", () => {
    expect(alPaso(DTI, 0.1 + 0.2)).toBe(0.3); // no acota: solo redondea (0,30000000000000004 -> 0,3)
    expect(alPaso(DTI, 12.34)).toBe(12.3);
    expect(alPaso(DTI, 0.1 + 0.2 + 12)).toBe(12.3);
  });
});

describe("acotar (al salir del campo)", () => {
  it("acota al rango y ajusta al paso", () => {
    expect(acotar(MONTO, 99999)).toBe(5000);
    expect(acotar(MONTO, 0)).toBe(1000);
    expect(acotar(MONTO, 1013)).toBe(1025);
    expect(acotar(DTI, 40)).toBe(32.5);
    expect(acotar(DTI, 15.06)).toBe(15.1);
  });

  it("un valor que no es número no se comunica (null)", () => {
    expect(acotar(MONTO, NaN)).toBeNull();
    expect(acotar(MONTO, Infinity)).toBeNull();
  });

  it("nunca sale del rango aunque max no caiga en la rejilla", () => {
    const raro: EntradaNum = { ...MONTO, min: 0, max: 10, paso: 3 };
    expect(acotar(raro, 9.9)).toBe(9);
    expect(acotar(raro, 11)).toBe(10);
  });
});
