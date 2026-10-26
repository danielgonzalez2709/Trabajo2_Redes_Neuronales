"use client";
// Categórica: control segmentado con 3 opciones o menos; lista desplegable con más.
import { useId } from "react";
import type { EntradaCat } from "../lib/contrato";
import estilos from "./Control.module.css";
import { Segmentado } from "./Segmentado";

const MAX_SEGMENTADO = 3;

interface Props {
  entrada: EntradaCat;
  valor: string;
  onCambio: (v: string) => void;
}

export function ControlCategorico({ entrada: e, valor, onCambio }: Props) {
  const id = useId();
  const opciones = e.categorias.map((c) => ({ valor: c, etiqueta: e.etiquetas_categorias[c] ?? c }));
  const ayuda = (
    <p className={estilos.ayuda} id={`${id}-ayuda`}>
      {e.ayuda_es}
    </p>
  );

  if (opciones.length <= MAX_SEGMENTADO) {
    return (
      <div className={estilos.control}>
        <Segmentado
          nombre={`${id}-${e.nombre}`}
          leyenda={e.etiqueta_es}
          opciones={opciones}
          valor={valor}
          onCambio={onCambio}
          ayudaId={`${id}-ayuda`}
        />
        {ayuda}
      </div>
    );
  }

  return (
    <div className={estilos.control}>
      <label className={estilos.etiqueta} htmlFor={`${id}-lista`}>
        {e.etiqueta_es}
      </label>
      {ayuda}
      <select
        id={`${id}-lista`}
        className={estilos.lista}
        value={valor}
        aria-describedby={`${id}-ayuda`}
        onChange={(ev) => onCambio(ev.target.value)}
      >
        {opciones.map((o) => (
          <option key={o.valor} value={o.valor}>
            {o.etiqueta}
          </option>
        ))}
      </select>
    </div>
  );
}
