import { Composition, Folder } from "remotion";
import { Cierre } from "./escenas/Cierre";
import { Cifra } from "./escenas/Cifra";
import { Demo } from "./escenas/Demo";
import { DosPreguntas } from "./escenas/DosPreguntas";
import { Equipo } from "./escenas/Equipo";
import { Honesta } from "./escenas/Honesta";
import { RedNeuronal } from "./escenas/RedNeuronal";
import { Titulo } from "./escenas/Titulo";
import { Promocional } from "./Promocional";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition id="Promocional" component={Promocional} durationInFrames={4500} fps={30} width={1920} height={1080} />
      <Folder name="Escenas">
        <Composition
          id="Titulo"
          component={Titulo}
          durationInFrames={300}
          fps={30}
          width={1920}
          height={1080}
          defaultProps={{ texto: "¿Me prestas?" }}
        />
        <Composition
          id="DosPreguntas"
          component={DosPreguntas}
          durationInFrames={450}
          fps={30}
          width={1920}
          height={1080}
          defaultProps={{ pregunta1: "¿Qué tan probable es que no me paguen?", pregunta2: "Y si no me pagan, ¿cuánto pierdo?" }}
        />
        <Composition id="Cifra" component={Cifra} durationInFrames={450} fps={30} width={1920} height={1080} />
        <Composition
          id="Demo"
          component={Demo}
          durationInFrames={1200}
          fps={30}
          width={1920}
          height={1080}
          defaultProps={{ archivo: "" }}
        />
        <Composition
          id="RedNeuronal"
          component={RedNeuronal}
          durationInFrames={600}
          fps={30}
          width={1920}
          height={1080}
          defaultProps={{ capas: [5, 6, 4, 1], texto: "Una red neuronal, comparada de forma justa." }}
        />
        <Composition
          id="Honesta"
          component={Honesta}
          durationInFrames={450}
          fps={30}
          width={1920}
          height={1080}
          defaultProps={{
            titulo: "Y es honesta.",
            subtitulo: "Te muestra qué pesa en tu puntaje y qué tan calibradas están sus probabilidades.",
          }}
        />
        <Composition
          id="Equipo"
          component={Equipo}
          durationInFrames={450}
          fps={30}
          width={1920}
          height={1080}
          defaultProps={{ archivo: "", nombres: ["Andrés", "Camila", "Daniel", "José", "Juan José"] }}
        />
        <Composition
          id="Cierre"
          component={Cierre}
          durationInFrames={150}
          fps={30}
          width={1920}
          height={1080}
          defaultProps={{
            llamado: "Pruébala.",
            url: "URL de la app (pendiente del despliegue)",
            pie: "UNAL · Redes neuronales y algoritmos bioinspirados · 2026-02",
          }}
        />
      </Folder>
    </>
  );
};
