"use client";
// Adaptador: el escenario (las entradas que se ven, sincronizadas con la URL ?modelo=…&loan_amnt=…), el modelo elegido
// y el perfil base con el que se compara el escenario (solo en memoria, nunca en la URL).
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { Entrada, NombreModelo } from "../lib/contrato";
import { escribirParams, leerModelo, leerValores, type Valores } from "../lib/entradas";

export const ESPERA_URL_MS = 200; // al arrastrar un deslizador, la URL se escribe tras una pausa breve

export interface EstadoEntradas {
  modelo: NombreModelo;
  setModelo: (m: NombreModelo) => void;
  /** Escenario; null mientras no se conocen las entradas (modelos sin cargar). */
  valores: Valores | null;
  setValor: (nombre: string, v: number | string) => void;
  /** Perfil base: las entradas con que arrancó la página (URL o defectos) hasta que se guarda otro. */
  base: Valores | null;
  /** base <- escenario. */
  guardarBase: () => void;
  /** escenario <- base. */
  volverABase: () => void;
}

export function useEntradas(entradas: Entrada[] | null): EstadoEntradas {
  const params = useSearchParams();
  const router = useRouter();
  const ruta = usePathname();
  // Copia de la URL al montar: el perfil base no se mueve cuando la URL empieza a reflejar el escenario.
  const [paramsIniciales] = useState(() => new URLSearchParams(params.toString()));
  const [modelo, setModeloEstado] = useState<NombreModelo>(() => leerModelo(params));
  // Mientras nadie toca nada, los valores salen de la URL; después, del estado (la URL solo lo refleja).
  const [editados, setEditados] = useState<Valores | null>(null);
  const [guardada, setGuardada] = useState<Valores | null>(null);
  const [tocado, setTocado] = useState(false);

  // Memorizados: misma identidad entre renders mientras no cambian (useMemo y useDeferredValue dependen de ello).
  const valores = useMemo(() => editados ?? (entradas ? leerValores(entradas, params) : null), [editados, entradas, params]);
  const base = useMemo(
    () => guardada ?? (entradas ? leerValores(entradas, paramsIniciales) : null),
    [guardada, entradas, paramsIniciales],
  );

  const setModelo = useCallback((m: NombreModelo) => {
    setModeloEstado(m);
    setTocado(true);
  }, []);

  const setValor = useCallback(
    (nombre: string, v: number | string) => {
      if (!entradas) return;
      setEditados((prev) => ({ ...(prev ?? leerValores(entradas, params)), [nombre]: v }));
      setTocado(true);
    },
    [entradas, params],
  );

  const guardarBase = useCallback(() => {
    if (valores) setGuardada({ ...valores });
  }, [valores]);

  const volverABase = useCallback(() => {
    if (!base) return;
    setEditados({ ...base });
    setTocado(true);
  }, [base]);

  useEffect(() => {
    if (!tocado || !entradas || !valores) return;
    const qs = escribirParams(entradas, modelo, valores).toString();
    if (qs === params.toString()) return;
    const t = setTimeout(() => router.replace(qs ? `${ruta}?${qs}` : ruta, { scroll: false }), ESPERA_URL_MS);
    return () => clearTimeout(t);
  }, [tocado, entradas, valores, modelo, params, router, ruta]);

  return { modelo, setModelo, valores, setValor, base, guardarBase, volverABase };
}
