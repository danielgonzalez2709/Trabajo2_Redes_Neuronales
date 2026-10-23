# 🎯 Registro de la Cacería de la alucinación

> Meta: ≥ 5 hallazgos **genuinos** en ≥ 3 categorías, al menos 1 por parte. **Nunca inventar.** Un candidato sin prompt
> literal o sin evidencia no cuenta. Evidencias en `alucinaciones/evidencias/`. Prompts literales en `alucinaciones/prompts/`.

## Cobertura actual
| Categoría | Candidatos | Confirmados |
|---|---|---|
| Matemáticas | 0 | 0 |
| Código | 0 | 0 |
| Datos del mundo real | 0 | 0 |
| Bibliografía | 0 | 0 |
| Conceptos | 0 | 0 |

## Afirmaciones pendientes de verificar (de la asesoría inicial, `prompts/asesoria_puente_claude.md`)
Afirmaciones que hizo Claude al planear el trabajo (19-oct-2026). **No son hallazgos**: hay que verificarlas, y solo si resultan
falsas se registran como candidatos.

| # | Afirmación | Cómo verificarla | Resultado |
|---|---|---|---|
| V-01 | Las funciones serverless de Vercel tienen un límite de ~250 MB y PyTorch no cabe | Documentación oficial de límites de Vercel | ⬜ |
| V-02 | El plan Hobby de Vercel es gratis y para uso personal/no comercial | Términos y precios oficiales de Vercel | ⬜ |
| V-03 | Neon Postgres tiene un plan gratis en el Marketplace de Vercel | Documentación de Vercel/Neon | ⬜ |
| V-04 | Render (plan gratis) duerme los servicios tras un periodo sin uso | Documentación de Render | ⬜ |
| V-05 | Remotion es gratis para individuos y equipos pequeños | Licencia oficial de Remotion | ⬜ |
| V-06 | Los créditos recientes ya resueltos sesgan la tasa de malos hacia arriba | Experimento A4 (tasa de malos y fracción de NA por año) | ⬜ |
| V-07 | `AdamW` aplica weight decay desacoplado, que no equivale a L2 con Adam | Loshchilov & Hutter + documentación de PyTorch | ⬜ |
| V-08 | El conjunto cubre créditos de 2007 a 2015 (afirmación implícita en la partición propuesta) | `resumen_datos.json` (A1) | ⬜ |
| V-09 | La EAD al castigo = `funded_amnt` − `total_rec_prncp`, y LGD = 1 − (`recoveries` − `collection_recovery_fee`)/EAD | Diccionario de Kaggle + casos reales (D1) | ⬜ |
| V-10 | Material for MkDocs está en mantenimiento con correcciones hasta el 5-nov-2026 (dato del Trabajo 1) | Anuncio oficial del proyecto | ⬜ |
| V-11 | Referencias propuestas: Bergstra et al. (2011), Akiba et al. (2019), Lundberg & Lee (2017), Papke & Wooldridge (1996), Loshchilov & Hutter (2019), Grinsztajn et al. (2022), Siddiqi (2017) | Crossref / DOI de cada una | ⬜ |

---

## Plantilla
```markdown
### H-XX — <título corto>
- **Estado:** candidato / confirmado / descartado
- **Categoría:** Matemáticas | Código | Datos | Bibliografía | Conceptos
- **Parte:** A | B | C | D | E
- **Herramienta y modelo:** · **Fecha:** · **Quién lo cazó:**
1. **Prompt (literal):**
2. **Respuesta relevante de la IA (cita textual o captura):**
3. **Cómo sospechamos:**
4. **Evidencia:** (derivación / experimento / fuente primaria con URL y fecha)
5. **Corrección:**
6. **Lección aprendida — ¿qué verificación lo habría evitado?:**
```

---

## Candidatos

_(ninguno todavía)_
