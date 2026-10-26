// Personaje propio en línea negra y relleno blanco; su gesto sigue al percentil del escenario (regla en lib/avatar).
// Decorativo: aria-hidden (el resultado ya se dice en texto).
import type { Expresion } from "../lib/avatar";
import estilos from "./Avatar.module.css";

const TINTA = "#1d1d1f";
const TRAZO = { stroke: TINTA, strokeWidth: 2.4, strokeLinecap: "round", strokeLinejoin: "round" } as const;

function Cara({ expresion }: { expresion: Expresion }) {
  switch (expresion) {
    case "preocupado":
      return (
        <g>
          {/* cejas con la punta interior levantada, ojos y boca ondulada; una gota de sudor */}
          <path d="M44 45.5 L53 43" {...TRAZO} fill="none" />
          <path d="M76 45.5 L67 43" {...TRAZO} fill="none" />
          <circle cx={50} cy={52} r={2.6} fill={TINTA} />
          <circle cx={70} cy={52} r={2.6} fill={TINTA} />
          <path d="M52 66 Q55.5 62.5 60 65 T68 64.5" {...TRAZO} fill="none" />
          <path d="M88 38 Q91 43.5 91 45.5 A3 3 0 0 1 85 45.5 Q85 43.5 88 38 Z" fill="#cfe6ff" {...TRAZO} strokeWidth={1.6} />
        </g>
      );
    case "neutral":
      return (
        <g>
          <circle cx={50} cy={53} r={2.6} fill={TINTA} />
          <circle cx={70} cy={53} r={2.6} fill={TINTA} />
          <path d="M55 65 H65" {...TRAZO} fill="none" />
        </g>
      );
    case "contento":
      return (
        <g>
          {/* ojos cerrados de gusto, mejillas y sonrisa; un destello */}
          <path d="M46 54 Q50 49 54 54" {...TRAZO} fill="none" />
          <path d="M66 54 Q70 49 74 54" {...TRAZO} fill="none" />
          <ellipse cx={45} cy={61} rx={4.5} ry={2.8} fill="#f6a5a5" opacity={0.85} />
          <ellipse cx={75} cy={61} rx={4.5} ry={2.8} fill="#f6a5a5" opacity={0.85} />
          <path d="M52 62.5 Q60 72 68 62.5 Z" fill={TINTA} {...TRAZO} strokeWidth={2} />
          <path d="M92 30 v8 M88 34 h8" {...TRAZO} strokeWidth={1.8} fill="none" />
        </g>
      );
  }
}

export function Avatar({ expresion }: { expresion: Expresion }) {
  return (
    <svg className={estilos.avatar} viewBox="0 0 120 120" aria-hidden="true" data-expresion={expresion}>
      {/* torso con camisa de cuello */}
      <path d="M20 121 C22 99 38 88 60 88 C82 88 98 99 100 121 Z" fill="#ffffff" {...TRAZO} />
      <path d="M60 99 V121" {...TRAZO} fill="none" />
      <path d="M33 104 V121 M87 104 V121" {...TRAZO} fill="none" />
      {/* cuello */}
      <path d="M53 74 V90 H67 V74" fill="#ffffff" {...TRAZO} />
      <path d="M49 88 L60 98 L71 88 L66 85 L60 92 L54 85 Z" fill="#ffffff" {...TRAZO} />
      {/* orejas y cabeza */}
      <ellipse cx={36.5} cy={56} rx={4.5} ry={6} fill="#ffffff" {...TRAZO} />
      <ellipse cx={83.5} cy={56} rx={4.5} ry={6} fill="#ffffff" {...TRAZO} />
      <ellipse cx={60} cy={53} rx={23.5} ry={25} fill="#ffffff" {...TRAZO} />
      {/* pelo con flequillo, moño y una pinza de estrella */}
      <path
        d="M35.5 50 C33 30 45 21.5 60 21.5 C76 21.5 88 31 84.5 51 C82 43 77 37.5 70 35 C66 40 57 42.5 49 40 C45 42 39 45 35.5 50 Z"
        fill={TINTA}
        {...TRAZO}
      />
      <circle cx={60} cy={15.5} r={8} fill={TINTA} />
      <path
        d="M77 24.5 l1.9 3.9 4.3 0.6 -3.1 3 0.7 4.3 -3.8 -2 -3.8 2 0.7 -4.3 -3.1 -3 4.3 -0.6 Z"
        fill="#f2c94c"
        stroke={TINTA}
        strokeWidth={1.4}
        strokeLinejoin="round"
      />
      <g className={estilos.cara}>
        <Cara expresion={expresion} />
      </g>
    </svg>
  );
}
