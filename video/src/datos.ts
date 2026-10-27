// Cifras del video: se leen de los resultados del repositorio, nunca se escriben a mano (núcleo §0.7).
import features from "./features.json";
import resumen from "../../data/processed/resumen_datos.json";

export const nModelado: number = resumen.n_modelado;
export const nVariables: number = features.length;
export const periodo = {
  desde: Number(resumen.periodo.min.slice(0, 4)),
  hasta: Number(resumen.particion.test_hasta.slice(0, 4)),
};

// Curva de calibración del MLP de PD (semilla mediana, partición de prueba): la genera `python run.py figures` (B7).
import calibracion from "../../blog/docs/figures/pd-calibracion.json";

// PD predicha media de cada bin (eje x de la curva).
export const calibracionPd: readonly number[] = calibracion.modelos.mlp.pd_media;
// Tasa observada de malos de cada bin (eje y de la curva).
export const calibracionTasa: readonly number[] = calibracion.modelos.mlp.tasa_observada;
