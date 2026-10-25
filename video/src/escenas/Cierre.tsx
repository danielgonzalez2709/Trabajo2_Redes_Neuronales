import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { tema } from "../tema";

type Props = { readonly llamado: string; readonly url: string; readonly pie: string };

// Escena 9 (2:25–2:30): llamado a la acción con la URL de la app.
export const Cierre: React.FC<Props> = ({ llamado, url, pie }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const aparece = interpolate(frame, [0, 0.8 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill
      style={{ backgroundColor: tema.fondo, justifyContent: "center", alignItems: "center", fontFamily: tema.fuente, gap: 36 }}
    >
      <div style={{ fontSize: 150, fontWeight: 700, letterSpacing: "-0.03em", color: tema.texto, opacity: aparece }}>{llamado}</div>
      <div style={{ fontSize: 64, color: tema.acento, opacity: aparece }}>{url}</div>
      <div style={{ position: "absolute", bottom: 100, fontSize: 36, color: tema.textoTenue }}>{pie}</div>
    </AbsoluteFill>
  );
};
