// Cifras del video: se leen de los resultados del repositorio, nunca se escriben a mano (núcleo §0.7).
import features from "./features.json";
import resumen from "../../data/processed/resumen_datos.json";

export const nModelado: number = resumen.n_modelado;
export const nVariables: number = features.length;
export const periodo = {
  desde: Number(resumen.periodo.min.slice(0, 4)),
  hasta: Number(resumen.particion.test_hasta.slice(0, 4)),
};
