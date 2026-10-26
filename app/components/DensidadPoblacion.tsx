// Densidad de los puntajes de la población (tramos entre percentiles, área = probabilidad) con la marca de la persona.
import type { Tramo } from "../lib/densidad";
import { formatearPuntaje } from "../lib/formato";
import estilos from "./DensidadPoblacion.module.css";

const ANCHO = 320;
const ALTO = 112;
const MARGEN = { izq: 6, der: 6, arriba: 20, abajo: 20 };

interface Props {
  tramos: Tramo[];
  puntaje: number;
  color: string;
  descripcion: string;
}

export function DensidadPoblacion({ tramos, puntaje, color, descripcion }: Props) {
  if (tramos.length === 0) return null;
  const min = tramos[0].desde;
  const max = tramos[tramos.length - 1].hasta;
  const dMax = Math.max(...tramos.map((t) => t.densidad));
  const base = ALTO - MARGEN.abajo;
  const x = (s: number) => MARGEN.izq + ((s - min) / (max - min)) * (ANCHO - MARGEN.izq - MARGEN.der);
  const y = (d: number) => base - (d / dMax) * (base - MARGEN.arriba);

  let trazo = `M${x(min).toFixed(2)},${base}`;
  for (const t of tramos) trazo += `L${x(t.desde).toFixed(2)},${y(t.densidad).toFixed(2)}H${x(t.hasta).toFixed(2)}`;
  trazo += `V${base}Z`;

  const xm = x(Math.min(Math.max(puntaje, min), max));
  const anclaTu = xm < 24 ? "start" : xm > ANCHO - 24 ? "end" : "middle";

  return (
    <svg className={estilos.figura} viewBox={`0 0 ${ANCHO} ${ALTO}`} role="img" aria-label={descripcion}>
      <path d={trazo} fill={color} fillOpacity={0.16} stroke={color} strokeWidth={1.25} strokeLinejoin="round" />
      <line x1={MARGEN.izq} x2={ANCHO - MARGEN.der} y1={base} y2={base} stroke="var(--ds-eje)" strokeWidth={1} />
      <g className={estilos.marca} style={{ transform: `translateX(${xm.toFixed(2)}px)` }}>
        <line x1={0} x2={0} y1={MARGEN.arriba - 4} y2={base} stroke={color} strokeWidth={2.5} strokeLinecap="round" />
        <text x={0} y={MARGEN.arriba - 8} textAnchor={anclaTu} className={estilos.tu}>
          Tú
        </text>
      </g>
      <text x={MARGEN.izq} y={ALTO - 4} className={estilos.eje}>
        {formatearPuntaje(min)}
      </text>
      <text x={ANCHO - MARGEN.der} y={ALTO - 4} textAnchor="end" className={estilos.eje}>
        {formatearPuntaje(max)}
      </text>
    </svg>
  );
}
