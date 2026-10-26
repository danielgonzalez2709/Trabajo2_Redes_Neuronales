"use client";
// Deslizador + campo numérico con el mismo valor. El campo guarda un borrador local para poder escribir libremente;
// solo comunica valores dentro de [min, max] y, al salir del campo, acota lo escrito al rango.
import { useId, useState } from "react";
import type { EntradaNum } from "../lib/contrato";
import { etiquetaSinUnidad, formatearNumero, unidad } from "../lib/formato";
import estilos from "./Control.module.css";

interface Props {
  entrada: EntradaNum;
  valor: number;
  onCambio: (v: number) => void;
}

export function ControlNumerico({ entrada: e, valor, onCambio }: Props) {
  const id = useId();
  const [borrador, setBorrador] = useState<string | null>(null);
  const u = unidad(e);
  const rango = `Entre ${formatearNumero(e.min, e.paso)} y ${formatearNumero(e.max, e.paso)}${u ? ` ${u}` : ""}.`;

  const alEscribir = (texto: string, x: number) => {
    setBorrador(texto);
    if (Number.isFinite(x) && x >= e.min && x <= e.max) onCambio(x);
  };

  const alSalir = (x: number) => {
    setBorrador(null);
    if (Number.isFinite(x) && (x < e.min || x > e.max)) onCambio(Math.min(Math.max(x, e.min), e.max));
  };

  return (
    <div className={estilos.control}>
      <div className={estilos.cabeza}>
        <label className={estilos.etiqueta} id={`${id}-etiqueta`} htmlFor={`${id}-campo`}>
          {etiquetaSinUnidad(e)}
        </label>
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
          />
          <span className={estilos.unidad} aria-hidden={u ? undefined : true}>
            {u}
          </span>
        </span>
      </div>
      <p className={estilos.ayuda} id={`${id}-ayuda`}>
        {e.ayuda_es}
      </p>
      <input
        className={estilos.deslizador}
        type="range"
        min={e.min}
        max={e.max}
        step={e.paso}
        value={valor}
        aria-labelledby={`${id}-etiqueta`}
        aria-describedby={`${id}-ayuda`}
        aria-valuetext={`${formatearNumero(valor, e.paso)}${u ? ` ${u}` : ""}`}
        onChange={(ev) => {
          setBorrador(null);
          onCambio(Number(ev.target.value));
        }}
      />
      <div className={estilos.extremos} id={`${id}-rango`}>
        <span aria-hidden="true">{formatearNumero(e.min, e.paso)}</span>
        <span className={estilos.oculta}>{rango}</span>
        <span aria-hidden="true">{formatearNumero(e.max, e.paso)}</span>
      </div>
    </div>
  );
}
