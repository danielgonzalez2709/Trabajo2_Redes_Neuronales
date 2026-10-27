import { describe, expect, it } from "vitest";
import { transicion, type EstadoToggletip, type EventoToggletip } from "../lib/toggletip";

const correr = (eventos: EventoToggletip[], inicio: EstadoToggletip = "cerrado") =>
  eventos.reduce<EstadoToggletip>((s, e) => transicion(s, e), inicio);

describe("toggletip: hover, foco y toque", () => {
  it("hover abre de forma pasajera y salir lo cierra", () => {
    expect(correr(["entrar"])).toBe("pasajero");
    expect(correr(["entrar", "salir"])).toBe("cerrado");
  });

  it("foco de teclado abre; perder el foco cierra", () => {
    expect(correr(["foco"])).toBe("pasajero");
    expect(correr(["foco", "desenfoque"])).toBe("cerrado");
  });

  it("clic (o toque) con el globo cerrado lo fija; otro clic lo cierra", () => {
    expect(correr(["clic"])).toBe("fijo");
    expect(correr(["clic", "clic"])).toBe("cerrado");
  });

  it("clic con el ratón encima (abierto por hover y foco) lo fija en vez de cerrarlo", () => {
    expect(correr(["entrar", "foco", "clic"])).toBe("fijo");
    expect(correr(["entrar", "foco", "clic", "salir"])).toBe("fijo"); // fijo: salir con el ratón no lo cierra
  });

  it("toque en el celular (foco + clic) lo deja abierto", () => {
    expect(correr(["foco", "clic"])).toBe("fijo");
  });

  it("Esc o tocar fuera lo cierran siempre", () => {
    for (const inicio of ["pasajero", "fijo"] as const) {
      expect(transicion(inicio, "escape")).toBe("cerrado");
      expect(transicion(inicio, "fuera")).toBe("cerrado");
    }
  });

  it("perder el foco cierra también el fijo (pasar con Tab al siguiente campo)", () => {
    expect(correr(["clic", "desenfoque"])).toBe("cerrado");
  });

  it("cerrado: salir, desenfoque, Esc y fuera no hacen nada", () => {
    for (const e of ["salir", "desenfoque", "escape", "fuera"] as const) expect(transicion("cerrado", e)).toBe("cerrado");
  });
});
