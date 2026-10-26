import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "¿Me prestas? Puntaje y probabilidad de incumplimiento",
  description: "Herramienta académica: puntaje y probabilidad de incumplimiento con regresión logística y red neuronal.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#f5f5f3" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
