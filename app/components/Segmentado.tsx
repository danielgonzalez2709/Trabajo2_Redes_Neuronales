// Control segmentado accesible: un grupo de radios nativos (teclado y lector de pantalla sin trabajo extra).
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
}

export function Segmentado({ nombre, leyenda, leyendaVisible = true, opciones, valor, onCambio, ayudaId }: Props) {
  return (
    <fieldset className={estilos.grupo} aria-describedby={ayudaId}>
      <legend className={leyendaVisible ? estilos.leyenda : "oculta"}>{leyenda}</legend>
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
