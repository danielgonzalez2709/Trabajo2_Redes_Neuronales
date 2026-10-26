import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "¿Me prestas? Puntaje y probabilidad de impago",
  description: "Herramienta académica: puntaje y probabilidad de impago con regresión logística y red neuronal.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
