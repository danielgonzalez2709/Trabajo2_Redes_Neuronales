// Ilustración de José en un círculo; su gesto (regla en lib/avatar) va como una carita en una insignia en la esquina.
// Decorativo: aria-hidden, porque el resultado (puntaje, PD y percentil) ya se dice en texto en la tarjeta.
import type { Expresion } from "../lib/avatar";
import estilos from "./Avatar.module.css";

const TRAZO = {
  stroke: "var(--ds-tinta)",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  fill: "none",
} as const;

function Carita({ expresion }: { expresion: Expresion }) {
  switch (expresion) {
    case "preocupado":
      return (
        <>
          <path d="M6.2 8 L9 7.2 M17.8 8 L15 7.2" {...TRAZO} strokeWidth={1.5} />
          <circle cx={9} cy={10.5} r={1.1} fill="var(--ds-tinta)" />
          <circle cx={15} cy={10.5} r={1.1} fill="var(--ds-tinta)" />
          <path d="M8.5 17 Q12 13.8 15.5 17" {...TRAZO} />
        </>
      );
    case "neutral":
      return (
        <>
          <circle cx={9} cy={10} r={1.1} fill="var(--ds-tinta)" />
          <circle cx={15} cy={10} r={1.1} fill="var(--ds-tinta)" />
          <path d="M9 15.5 H15" {...TRAZO} />
        </>
      );
    case "contento":
      return (
        <>
          <circle cx={9} cy={10} r={1.1} fill="var(--ds-tinta)" />
          <circle cx={15} cy={10} r={1.1} fill="var(--ds-tinta)" />
          <path d="M8 14 Q12 19 16 14" {...TRAZO} />
        </>
      );
  }
}

export function Avatar({ expresion }: { expresion: Expresion }) {
  return (
    <span className={estilos.avatar} aria-hidden="true" data-expresion={expresion}>
      <img className={estilos.foto} src="/avatar.webp" alt="" width={256} height={256} decoding="async" />
      <svg className={`${estilos.insignia} ${estilos[expresion]}`} viewBox="0 0 24 24">
        <g key={expresion} className={estilos.cara}>
          <Carita expresion={expresion} />
        </g>
      </svg>
    </span>
  );
}
