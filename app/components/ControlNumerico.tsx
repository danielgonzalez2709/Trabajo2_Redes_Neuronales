"use client";
// Deslizador + campo numérico con el mismo valor. El campo guarda un borrador local para poder escribir libremente;
// solo comunica valores válidos (lib/campo) y, al salir del campo, ajusta lo escrito al paso y al rango.
import type { CSSProperties } from "react";
import { useId, useState } from "react";
import { acotar, esValido } from "../lib/campo";
import type { EntradaNum } from "../lib/contrato";
import { etiquetaSinUnidad, formatearNumero, unidad } from "../lib/formato";
import estilos from "./Control.module.css";
import { Info } from "./Info";

interface Props {
  entrada: EntradaNum;
  valor: number;
  /** Valor del perfil base: si difiere del actual, se marca tenue sobre el deslizador. */
  valorBase?: number;
  onCambio: (v: number) => void;
}

export function ControlNumerico({ entrada: e, valor, valorBase, onCambio }: Props) {
  const id = useId();
  const [borrador, setBorrador] = useState<string | null>(null);
  const u = unidad(e);
  const conUnidad = (x: number) => `${formatearNumero(x, e.paso)}${u ? ` ${u}` : ""}`;
  const rango = `Entre ${formatearNumero(e.min, e.paso)} y ${conUnidad(e.max)}.`;
  const base = valorBase !== undefined && valorBase !== valor ? valorBase : null; // solo se marca si difiere

  const alEscribir = (texto: string, x: number) => {
    setBorrador(texto);
    if (esValido(e, x)) onCambio(x);
  };

  const alSalir = (x: number) => {
    setBorrador(null);
    const ajustado = acotar(e, x);
    if (ajustado !== null && ajustado !== valor) onCambio(ajustado);
  };

  return (
    <div className={estilos.control}>
      <div className={estilos.cabeza}>
        <span className={estilos.etiquetaFila}>
          <label className={estilos.etiqueta} id={`${id}-etiqueta`} htmlFor={`${id}-campo`}>
            {etiquetaSinUnidad(e)}
          </label>
          <Info etiqueta={etiquetaSinUnidad(e)} texto={e.ayuda_es} />
        </span>
        <span className={estilos.campoCaja}>
          <input
            id={`${id}-campo`}
            className={estilos.campo}
            type="number"
            inputMode="decimal"
            min={e.min}
            max={e.max}
            step={e.paso}
            value={borrador ?? String(valor)}
            aria-describedby={`${id}-ayuda ${id}-rango`}
            onChange={(ev) => alEscribir(ev.target.value, ev.target.valueAsNumber)}
            onBlur={(ev) => alSalir(ev.target.valueAsNumber)}
            onInvalid={(ev) => ev.preventDefault()} // sin globo nativo: el valor se ajusta al salir
          />
          <span className={estilos.unidad} aria-hidden={u ? undefined : true}>
            {u}
          </span>
        </span>
      </div>
      <p className="oculta" id={`${id}-ayuda`}>
        {e.ayuda_es}
      </p>
      <div className={estilos.pista}>
        {base !== null && (
          <span
            className={estilos.marcaBase}
            style={{ "--fraccion": (base - e.min) / (e.max - e.min) } as CSSProperties}
            title={`Tu perfil: ${conUnidad(base)}`}
            aria-hidden="true"
          />
        )}
        <input
          className={estilos.deslizador}
          type="range"
          min={e.min}
          max={e.max}
          step={e.paso}
          value={valor}
          aria-labelledby={`${id}-etiqueta`}
          aria-describedby={`${id}-ayuda${base !== null ? ` ${id}-base` : ""}`}
          aria-valuetext={conUnidad(valor)}
          onChange={(ev) => {
            setBorrador(null);
            onCambio(Number(ev.target.value));
          }}
        />
      </div>
      {base !== null && (
        <span className="oculta" id={`${id}-base`}>
          En tu perfil: {conUnidad(base)}.
        </span>
      )}
      <div className={estilos.extremos} id={`${id}-rango`}>
        <span aria-hidden="true">{formatearNumero(e.min, e.paso)}</span>
        <span className="oculta">{rango}</span>
        <span aria-hidden="true">{formatearNumero(e.max, e.paso)}</span>
      </div>
    </div>
  );
}
