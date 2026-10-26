"use client";
// Botón "i" junto a la etiqueta de un campo con su `ayuda_es` en un globo (toggletip): se abre al pasar el ratón, con el
// foco del teclado o con un toque, y se cierra con Esc, al tocar fuera o al salir. Popover API (top layer) con anchor
// positioning; donde no hay anchor positioning, el globo se coloca con getBoundingClientRect (Info.module.css,
// @supports). Sin polyfills. El campo conserva su `aria-describedby` hacia el texto de ayuda (oculto visualmente).
import type { CSSProperties, FocusEvent, PointerEvent as EventoPuntero } from "react";
import { useEffect, useId, useReducer, useRef } from "react";
import { transicion } from "../lib/toggletip";
import estilos from "./Info.module.css";

const ESPERA_SALIR_MS = 150; // margen para pasar el ratón del botón al globo sin que se cierre
const SEPARACION_PX = 6;
const BORDE_PX = 8;

/** Un solo globo abierto a la vez: abrir uno cierra el anterior. */
let cerrarAbierto: (() => void) | null = null;

const hayAnclas = () => typeof CSS !== "undefined" && CSS.supports("anchor-name: --a");

/** Respaldo sin anchor positioning: encima del botón (debajo si no cabe), centrado y dentro de la ventana. */
function colocar(boton: HTMLElement, globo: HTMLElement) {
  const b = boton.getBoundingClientRect();
  const alto = globo.offsetHeight;
  const ancho = globo.offsetWidth;
  const arriba = b.top - alto - SEPARACION_PX;
  const top = arriba >= BORDE_PX ? arriba : b.bottom + SEPARACION_PX;
  const left = Math.min(Math.max(b.left + b.width / 2 - ancho / 2, BORDE_PX), window.innerWidth - ancho - BORDE_PX);
  globo.style.top = `${Math.round(top)}px`;
  globo.style.left = `${Math.round(Math.max(left, BORDE_PX))}px`;
}

interface Props {
  /** Etiqueta del campo, para el nombre accesible del botón. */
  etiqueta: string;
  texto: string;
}

export function Info({ etiqueta, texto }: Props) {
  const id = useId();
  const globoId = `${id}-globo`;
  const ancla = `--info-${id.replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const [estado, despachar] = useReducer(transicion, "cerrado");
  const boton = useRef<HTMLButtonElement>(null);
  const globo = useRef<HTMLDivElement>(null);
  const salida = useRef<number | undefined>(undefined);
  const abierto = estado !== "cerrado";

  useEffect(() => {
    const b = boton.current;
    const g = globo.current;
    if (!b || !g) return;
    const conPopover = typeof g.showPopover === "function";
    if (!abierto) {
      if (conPopover && g.matches(":popover-open")) g.hidePopover();
      return;
    }
    const cerrar = () => despachar("fuera");
    cerrarAbierto?.();
    cerrarAbierto = cerrar;
    if (conPopover && !g.matches(":popover-open")) g.showPopover();
    const respaldo = !hayAnclas();
    if (respaldo) colocar(b, g);

    const alTecla = (e: KeyboardEvent) => {
      if (e.key === "Escape") despachar("escape");
    };
    const alPuntero = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!b.contains(t) && !g.contains(t)) despachar("fuera");
    };
    const alMover = () => colocar(b, g);
    document.addEventListener("keydown", alTecla);
    document.addEventListener("pointerdown", alPuntero);
    if (respaldo) {
      window.addEventListener("scroll", alMover, true);
      window.addEventListener("resize", alMover);
    }
    return () => {
      document.removeEventListener("keydown", alTecla);
      document.removeEventListener("pointerdown", alPuntero);
      window.removeEventListener("scroll", alMover, true);
      window.removeEventListener("resize", alMover);
      if (cerrarAbierto === cerrar) cerrarAbierto = null;
    };
  }, [abierto]);

  useEffect(() => () => window.clearTimeout(salida.current), []);

  const entrar = (e: EventoPuntero) => {
    if (e.pointerType !== "mouse") return; // el toque se maneja con el clic
    window.clearTimeout(salida.current);
    despachar("entrar");
  };
  const salir = (e: EventoPuntero) => {
    if (e.pointerType !== "mouse") return;
    window.clearTimeout(salida.current);
    salida.current = window.setTimeout(() => despachar("salir"), ESPERA_SALIR_MS);
  };
  const desenfocar = (e: FocusEvent) => {
    if (globo.current?.contains(e.relatedTarget as Node | null)) return;
    despachar("desenfoque");
  };

  return (
    <>
      <button
        ref={boton}
        type="button"
        className={estilos.boton}
        style={{ "--ancla": ancla } as CSSProperties}
        aria-label={`Qué significa: ${etiqueta}`}
        aria-expanded={abierto}
        aria-controls={globoId}
        onClick={() => despachar("clic")}
        onFocus={() => despachar("foco")}
        onBlur={desenfocar}
        onPointerEnter={entrar}
        onPointerLeave={salir}
      >
        <span aria-hidden="true">i</span>
      </button>
      <div
        ref={globo}
        id={globoId}
        popover="manual"
        className={estilos.globo}
        data-abierto={abierto || undefined}
        style={{ "--ancla": ancla } as CSSProperties}
        onPointerEnter={entrar}
        onPointerLeave={salir}
        onPointerDown={(e) => e.preventDefault()} // seleccionar o tocar el texto no le quita el foco al botón
      >
        {texto}
      </div>
    </>
  );
}
