import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { tema } from "../tema";

type Props = { readonly pregunta1: string; readonly pregunta2: string };

// Escena 3 (0:25–0:40): las dos preguntas del banco, una tras otra.
export const DosPreguntas: React.FC<Props> = ({ pregunta1, pregunta2 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const entra = (inicio: number) => ({
    opacity: interpolate(frame, [inicio, inicio + 0.8 * fps], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.bezier(0.16, 1, 0.3, 1),
    }),
    translate: interpolate(frame, [inicio, inicio + 0.8 * fps], ["0px 30px", "0px 0px"], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.bezier(0.16, 1, 0.3, 1),
    }),
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: tema.fondo,
        justifyContent: "center",
        padding: "0 200px",
        gap: 80,
        fontFamily: tema.fuente,
        fontWeight: 700,
        letterSpacing: "-0.02em",
        lineHeight: 1.1,
      }}
    >
      <div style={{ fontSize: 110, color: tema.texto, ...entra(0.5 * fps) }}>{pregunta1}</div>
      <div style={{ fontSize: 110, color: tema.acento2, ...entra(5 * fps) }}>{pregunta2}</div>
    </AbsoluteFill>
  );
};
