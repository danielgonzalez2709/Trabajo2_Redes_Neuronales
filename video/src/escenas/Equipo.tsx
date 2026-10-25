import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Clip } from "../Clip";
import { tema } from "../tema";

type Props = { readonly archivo: string; readonly nombres: string[] };

// Escena 8 (2:10–2:25): clip grupal de los cinco y sus nombres en la parte inferior.
export const Equipo: React.FC<Props> = ({ archivo, nombres }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: tema.fondo }}>
      <Clip archivo={archivo} quien="los cinco" linea="Somos estudiantes de la Universidad Nacional. Pruébala." />
      <div
        style={{
          position: "absolute",
          left: 100,
          right: 100,
          bottom: 100,
          display: "flex",
          justifyContent: "space-between",
          fontFamily: tema.fuente,
          fontSize: 40,
          fontWeight: 600,
          color: tema.texto,
          opacity: interpolate(frame, [1 * fps, 2 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        {nombres.map((n) => (
          <span key={n}>{n}</span>
        ))}
      </div>
    </AbsoluteFill>
  );
};
