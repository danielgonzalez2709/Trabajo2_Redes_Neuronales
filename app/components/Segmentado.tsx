// Control segmentado accesible: un grupo de radios nativos (teclado y lector de pantalla sin trabajo extra).
import type { ReactNode } from "react";
import { useId } from "react";
import estilos from "./Segmentado.module.css";

export interface OpcionSegmentada {
  valor: string;
  etiqueta: string;
  /** Punto de color de la entidad (p. ej. el modelo); el texto sigue en tinta. */
  color?: string;
}

interface Props {
  nombre: string;
  leyenda: string;
  leyendaVisible?: boolean;
  opciones: OpcionSegmentada[];
  valor: string;
  onCambio: (valor: string) => void;
  ayudaId?: string;
  /** Junto a la leyenda (p. ej. el botón de ayuda); el nombre del grupo sigue siendo solo la leyenda. */
  extra?: ReactNode;
}

export function Segmentado(p: Props) {
  const { nombre, leyenda, leyendaVisible = true, opciones, valor, onCambio, ayudaId, extra } = p;
  const leyendaId = `${useId()}-leyenda`;
  return (
    <fieldset className={estilos.grupo} aria-describedby={ayudaId} aria-labelledby={extra ? leyendaId : undefined}>
      <legend className={leyendaVisible ? estilos.leyenda : "oculta"}>
        <span id={leyendaId}>{leyenda}</span>
        {extra}
      </legend>
      <div className={estilos.pista} style={{ gridTemplateColumns: `repeat(${opciones.length}, minmax(0, 1fr))` }}>
        {opciones.map((o) => (
          <label key={o.valor} className={estilos.opcion}>
            <input
              type="radio"
              className={estilos.radio}
              name={nombre}
              value={o.valor}
              checked={o.valor === valor}
              onChange={() => onCambio(o.valor)}
            />
            <span className={estilos.cara}>
              {o.color && <span className={estilos.punto} style={{ background: o.color }} aria-hidden="true" />}
              {o.etiqueta}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
