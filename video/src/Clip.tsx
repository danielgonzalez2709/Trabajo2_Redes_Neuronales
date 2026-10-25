import { Video } from "@remotion/media";
import { AbsoluteFill, staticFile, useVideoConfig } from "remotion";
import { tema } from "./tema";

type Props = {
  // Archivo dentro de public/clips/ (p. ej. "andres.mp4"). Vacío = todavía no se ha grabado: se muestra un aviso.
  readonly archivo: string;
  readonly quien: string;
  readonly linea: string;
};

// Clip grabado por un integrante (sábado 24). Mientras no exista, un marcador muestra quién habla y qué dice.
export const Clip: React.FC<Props> = ({ archivo, quien, linea }) => {
  const { fps } = useVideoConfig();
  if (archivo) {
    return (
      <Video
        name={`Clip de ${quien}`}
        src={staticFile(`clips/${archivo}`)}
        premountFor={fps}
        objectFit="cover"
        style={{ position: "absolute", width: "100%", height: "100%" }}
      />
    );
  }
  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#111114",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: tema.fuente,
        color: tema.textoTenue,
        textAlign: "center",
        padding: 160,
        gap: 24,
      }}
    >
      <div style={{ fontSize: 44, letterSpacing: "0.08em", textTransform: "uppercase" }}>Clip pendiente · {quien}</div>
      <div style={{ fontSize: 56, color: tema.texto, lineHeight: 1.3 }}>“{linea}”</div>
    </AbsoluteFill>
  );
};
