import { describe, expect, it } from "vitest";
import type { Preprocesamiento, Red } from "../lib/contrato";
import { logit, percentil, puntaje, sigmoide } from "../lib/modelo";
import { preprocesar, stripPy } from "../lib/preprocesar";

const PRE: Preprocesamiento = {
  version: 1,
  percentiles_recorte: [1, 99],
  categoricas: { limpieza: "strip", vacio: "DESCONOCIDO", no_vista: "OTRA" },
  orden_numericas: ["imputar_mediana", "recortar", "log1p_si_aplica", "estandarizar"],
  columnas_salida: ["x", "c=a", "c=OTRA", "l"],
  variables: [
    { nombre: "x", tipo: "num", mediana: 5, recorte: [0, 10], log1p: false, media: 5, desv: 2 },
    { nombre: "c", tipo: "cat", categorias: ["a", "OTRA"] },
    { nombre: "l", tipo: "num", mediana: 1, recorte: [0, 3], log1p: true, media: 0.5, desv: 0.5 },
  ],
};

describe("preprocesar", () => {
  it("recorta, estandariza, one-hot y log1p", () => {
    const v = preprocesar(PRE, { x: 99, c: "a", l: 1 });
    expect(v[0]).toBeCloseTo((10 - 5) / 2, 12); // recortado a 10
    expect(v.slice(1, 3)).toEqual([1, 0]);
    expect(v[3]).toBeCloseTo((Math.log1p(1) - 0.5) / 0.5, 12);
  });

  it("valor faltante: null, undefined, vacío y solo espacios van a la mediana (no a 0)", () => {
    for (const vacio of [null, undefined, "", "   ", NaN]) {
      expect(preprocesar(PRE, { x: vacio, c: "a", l: 1 })[0]).toBe(0); // mediana 5 -> (5-5)/2
    }
  });

  it("numérica en texto con espacios se lee como número; texto no numérico falla", () => {
    expect(preprocesar(PRE, { x: " 7 ", c: "a", l: 1 })[0]).toBeCloseTo(1, 12);
    expect(() => preprocesar(PRE, { x: "abc", c: "a", l: 1 })).toThrow(/no es un número/);
  });

  it("categoría no vista -> OTRA; ausente -> DESCONOCIDO -> OTRA si no se vio; strip", () => {
    expect(preprocesar(PRE, { x: 5, c: "zzz", l: 1 }).slice(1, 3)).toEqual([0, 1]);
    expect(preprocesar(PRE, { x: 5, c: null, l: 1 }).slice(1, 3)).toEqual([0, 1]);
    expect(preprocesar(PRE, { x: 5, c: "  a ", l: 1 }).slice(1, 3)).toEqual([1, 0]);
    expect(preprocesar(PRE, { x: 5, c: "nan", l: 1 }).slice(1, 3)).toEqual([0, 1]); // "nan" es un valor real
  });

  it("falta una entrada: error", () => {
    expect(() => preprocesar(PRE, { x: 5, c: "a" })).toThrow(/Falta/);
  });

  it("stripPy imita str.strip (quita U+00A0 y \\x1c, no U+FEFF)", () => {
    // U+00A0 (espacio duro) y U+3000 (espacio ideográfico) sí se quitan
    expect(stripPy("\u00a0a\u3000")).toBe("a");
    // \x1c-\x1f (separadores de archivo/registro/unidad) son whitespace en Python
    expect(stripPy("\x1ca\x1f")).toBe("a");
    // U+FEFF (BOM) no es whitespace en Python: se conserva
    expect(stripPy("\uFEFFa")).toBe("\uFEFFa");
  });
});

describe("modelo", () => {
  it("una capa a mano con W salida x entrada (no transpuesta)", () => {
    const red: Red = {
      salida: "sigmoid",
      capas: [
        { W: [[1, 2, 3], [0, -1, 0]], b: [0.5, 0], activacion: "relu" }, // 2 salidas x 3 entradas
        { W: [[2, 10]], b: [1], activacion: "identidad" },
      ],
    };
    // capa 1: [1*1+2*2+3*3+0.5, max(0,-2)] = [14.5, 0]; capa 2: 2*14.5 + 10*0 + 1 = 30
    expect(logit(red, [1, 2, 3])).toBeCloseTo(30, 12);
  });

  it("dimensiones incompatibles fallan", () => {
    const red: Red = { salida: "sigmoid", capas: [{ W: [[1, 2]], b: [0], activacion: "identidad" }] };
    expect(() => logit(red, [1, 2, 3])).toThrow(/W espera 2 entradas y llegan 3/);
  });

  it("leaky_relu usa pendiente 0.01 y tanh funciona", () => {
    const red = (a: "leaky_relu" | "tanh"): Red => ({
      salida: "sigmoid",
      capas: [{ W: [[1]], b: [0], activacion: a }],
    });
    expect(logit(red("leaky_relu"), [-2])).toBeCloseTo(-0.02, 12);
    expect(logit(red("tanh"), [0.5])).toBeCloseTo(Math.tanh(0.5), 12);
  });

  it("sigmoide estable en los extremos", () => {
    expect(sigmoide(0)).toBe(0.5);
    expect(sigmoide(-800)).toBe(0);
    expect(sigmoide(800)).toBe(1);
    expect(sigmoide(-745)).toBeGreaterThan(0);
  });

  it("puntaje = offset - factor * logit, sin redondear", () => {
    const sc = { base: 600, odds_base: 50, pdo: 20, factor: 28.85390081777927, offset: 487.1228762045055 };
    expect(puntaje(0, sc)).toBe(sc.offset);
    expect(puntaje(1, sc)).toBeCloseTo(sc.offset - sc.factor, 12);
  });

  it("percentil: interpola, acota y maneja tramos planos", () => {
    const p = Array.from({ length: 101 }, (_, i) => 400 + i); // p_i = 400 + i
    expect(percentil(450, p)).toBeCloseTo(50, 12);
    expect(percentil(450.5, p)).toBeCloseTo(50.5, 12);
    expect(percentil(100, p)).toBe(0);
    expect(percentil(999, p)).toBe(100);
    const plano = [0, 1, 1, 1, 2];
    expect(percentil(1, plano)).toBe(3); // último índice con p[i] <= 1
  });
});
