// Anillo del percentil (SVG con stroke-dasharray) con el número y "percentil" dentro.
import estilos from "./Anillo.module.css";

const R = 44;
const GROSOR = 10;
const LADO = 2 * (R + GROSOR / 2);
const CIRCUNFERENCIA = 2 * Math.PI * R;

interface Props {
  /** Percentil 0–100 sin redondear (el arco). */
  percentil: number;
  /** El número que se muestra (ya redondeado hacia abajo). */
  texto: string;
  descripcion: string;
}

export function Anillo({ percentil, texto, descripcion }: Props) {
  const arco = (Math.min(Math.max(percentil, 0), 100) / 100) * CIRCUNFERENCIA;
  return (
    <div className={estilos.anillo} role="img" aria-label={descripcion}>
      <svg viewBox={`0 0 ${LADO} ${LADO}`} aria-hidden="true">
        <circle className={estilos.pista} cx={LADO / 2} cy={LADO / 2} r={R} fill="none" strokeWidth={GROSOR} />
        <circle
          className={estilos.arco}
          cx={LADO / 2}
          cy={LADO / 2}
          r={R}
          fill="none"
          strokeWidth={GROSOR}
          strokeLinecap={arco > 0 ? "round" : "butt"}
          strokeDasharray={`${arco.toFixed(2)} ${CIRCUNFERENCIA.toFixed(2)}`}
          transform={`rotate(-90 ${LADO / 2} ${LADO / 2})`}
        />
      </svg>
      <span className={estilos.centro} aria-hidden="true">
        <span className={estilos.numero}>{texto}</span>
        <span className={estilos.rotulo}>percentil</span>
      </span>
    </div>
  );
}
