// Estética de keynote (skill apple-design): fondo oscuro, tipografía del sistema, un color de acento.
// Colores tomados de la paleta del repositorio (src/viz/paleta.py) para que el video y las figuras del blog combinen.
export const tema = {
  fondo: "#000000",
  texto: "#f5f5f7",
  textoTenue: "#86868b",
  acento: "#2a78d6", // azul de la paleta (entrenamiento / modelo principal)
  acento2: "#eb6834", // naranja de la paleta (prueba / MLP)
  verde: "#1baf7a",
  fuente: '"SF Pro Display", -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
} as const;

export const FPS = 30;
export const ANCHO = 1920;
export const ALTO = 1080;
