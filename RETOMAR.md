# RETOMAR.md — SPH-IncipientMotion
> Ultima actualizacion: 2026-02-24 (sesion 11)

## Estado Actual
**Codebase reorganizada. Tesis en standby esperando Dr. Moris.**

## TAREA INMEDIATA AL RETOMAR
1. **Esperar respuesta Dr. Moris** (email enviado 2026-02-20) con rangos parametricos
2. **Cuando responda**: editar `config/param_ranges.json` y lanzar produccion
3. **Render Blender pendiente**: agua PLY no se importo en WS, verificar en consola de Blender
4. **Revisar DEFENSA_TECNICA.md** antes de reunion con profesor

## Estructura del Proyecto (reorganizada sesion 11)

```
Tesis/
├── CLAUDE.md                     # Instrucciones del proyecto
├── RETOMAR.md                    # Punto de entrada (LEER PRIMERO)
├── DEFENSA_TECNICA.md            # Documento de defensa (650+ lineas)
├── PLAN.md                       # Plan de implementacion
├── PLAN_RENDER.md                # Plan render Blender
├── EDUCACION_TECNICA.md          # Guia tecnica SPH
├── AUDITORIA_FISICA.md           # Auditoria de parametros fisicos
│
├── src/                          # Pipeline principal (4 modulos)
│   ├── geometry_builder.py
│   ├── batch_runner.py
│   ├── data_cleaner.py
│   ├── ml_surrogate.py
│   └── main_orchestrator.py
│
├── scripts/                      # Scripts standalone activos
│   ├── run_convergence.py        # Correr estudio de convergencia
│   ├── convergencia_formal.py    # Analisis formal de convergencia
│   ├── figuras_convergencia_7dp_es.py  # Figuras convergencia ES (7dp)
│   ├── run_for_render.py         # Script render para WS
│   ├── run_production.py         # Script produccion para WS
│   ├── blender_render.py         # Script Blender render
│   ├── convert_vtk_to_ply.py     # Convertir VTK a PLY
│   └── test_ml.py                # Test GP surrogate
│
├── config/                       # Configuracion
│   ├── template_base.xml         # Template XML DualSPHysics
│   ├── Floating_Materials.xml    # Materiales Chrono
│   ├── experiment_matrix.csv     # Matriz LHS
│   ├── param_ranges.json         # Rangos parametricos (editar con Dr. Moris)
│   └── dsph_config.json          # Config paths DualSPHysics
│
├── models/                       # Geometrias STL
│   ├── BLIR3.stl                 # Boulder irregular
│   └── Canal_Playa_1esa20_750cm.stl  # Canal + playa
│
├── data/                         # Resultados
│   ├── processed/                # CSVs procesados (lhs_001-005, test_diego)
│   ├── figuras_7dp/              # Figuras convergencia EN (VIGENTES)
│   ├── figuras_7dp_es/           # Figuras convergencia ES (VIGENTES)
│   ├── figuras_tesis/            # Figuras para capitulos de tesis
│   ├── figuras_ml/               # Figuras ML surrogate
│   ├── figuras_ml_test/          # Figuras ML test
│   ├── reporte_convergencia_7dp.csv  # Reporte convergencia final
│   └── production_status.json    # Estado produccion
│
├── tesis/                        # Capitulos de tesis
│   ├── cap3_metodologia.md
│   ├── cap4_resultados_convergencia.md
│   └── cap5_pipeline.md
│
├── cases/                        # Casos de simulacion (con outputs)
│   ├── lhs_001/ ... lhs_005/    # 5 casos LHS
│   └── test_diego_reference/     # Caso referencia de Diego
│
├── ENTREGA_KEVIN/                # Archivos originales de Diego (referencia)
│
├── archive/                      # Material archivado (ya no se usa)
│   ├── scripts/                  # Scripts obsoletos
│   │   ├── app.py                # Dashboard Dash (no activo)
│   │   ├── monitor_dp003.py      # Monitor GitHub (cumplio funcion)
│   │   ├── figuras_convergencia_v2.py      # Supersedido por 7dp
│   │   ├── figuras_convergencia_v2_es.py   # Supersedido por 7dp
│   │   ├── generar_todas_figuras.py        # Generador viejo
│   │   ├── verificar_setup.py              # Check one-time
│   │   ├── preview_paraview.py             # Utilidad ParaView
│   │   ├── auditor_stl.py                  # Auditor STL
│   │   ├── generar_figuras_tesis.py        # Generador figuras viejo
│   │   └── analisis_convergencia_paper.py  # Analisis paper viejo
│   ├── docs/                     # Contextos y docs viejos
│   │   ├── contexto_previo_gemini_3.1_pro.md
│   │   ├── contexto.md
│   │   ├── contexto-1227.md
│   │   ├── RESEARCH_PROJECT_CONTEXT.md
│   │   ├── INFORME_SESION.md
│   │   ├── english_technical_guide.md
│   │   ├── upwork_profile.md
│   │   ├── Oferta capstone...pdf
│   │   └── contexto-lead-architect/   # Contexto Diego (7 archivos)
│   └── data/                     # Figuras y datos supersedidos
│       ├── figuras_convergencia_v2/       # Figuras v2 EN (sin 7dp)
│       ├── figuras_convergencia_v2_es/    # Figuras v2 ES (sin 7dp)
│       ├── figuras_6dp/                   # Figuras 6dp (pre-7dp)
│       ├── figuras_paper/                 # Figuras paper viejas
│       ├── reporte_convergencia.csv       # Reporte viejo
│       └── reporte_convergencia_6dp.csv   # Reporte 6dp viejo
│
└── .agente/                      # Contexto del agente
    ├── conversacion/
    └── codigo/
```

## Lo que se hizo en sesion 11 (2026-02-24)

### Reorganizacion de codebase
- Creado `scripts/` para 8 scripts activos (antes sueltos en root)
- Archivados 10 scripts obsoletos en `archive/scripts/`
- Archivados 10+ docs/contextos viejos en `archive/docs/`
- Archivadas 4 carpetas de figuras supersedidas + 2 CSVs viejos en `archive/data/`
- Eliminado path anidado roto: `cases/test_diego_reference/cases/...`
- Consolidado `archive/` que antes tenia scripts sueltos en raiz

### Sesiones previas (resumen)
- Sesion 10: DEFENSA_TECNICA.md + repo Git + revision tecnica
- Sesion 8-9: Convergencia 7dp + render pipeline
- Sesion 1-7: Pipeline completo, convergencia, figuras, tesis

## Resultados de Convergencia (sin cambios)
| dp | Desplaz | Delta% | Rot | F_SPH | F_cont | Tiempo |
|---|---|---|---|---|---|---|
| 0.020 | 3.495m | — | 95.8 | 166.4N | 2254N | 13 min |
| 0.015 | 3.433m | 1.8% | 97.2 | 77.0N | 4915N | 12 min |
| 0.010 | 3.069m | 10.6% | 60.3 | 45.3N | 131N | 24 min |
| 0.008 | 2.408m | 21.5% | 87.2 | 34.9N | 3229N | 30 min |
| 0.005 | 1.725m | 28.4% | 86.8 | 23.0N | 3083N | 118 min |
| 0.004 | 1.615m | 6.4% | 84.8 | 22.8N | 359N | 260 min |
| 0.003 | 1.553m | 3.9% | 90.2 | 22.2N | 450N | 812 min |

## Decisiones Acumuladas
- Chrono (RigidAlgorithm=3) obligatorio
- massbody > rhopbody, FtPause=0.5
- dp produccion: 0.004 (convergido)
- Contact force: NO usar como criterio (CV=82%)
- Render: dp=0.004, TimeOut=0.5, 21 frames
- WS sin git: deploy via ZIP (Invoke-WebRequest)
- Blender: pestaña Scripting para ejecutar scripts (--python no funciona bien)
- DEFENSA_TECNICA.md sin menciones a Diego (lenguaje neutro)
- Repo Git local creado + GitHub remote

## Infraestructura
- **Laptop**: git + gh CLI instalados, repo en C:\Seba\Tesis\
- **GitHub**: https://github.com/kcortes765/Tesis (publico)
- **WS UCN**: `C:\Users\ProyBloq_Cortes\Desktop\SPH-Convergencia` (sin git)
- **Blender WS**: `C:\Program Files\Blender Foundation\Blender 4.2\blender.exe`

## Cuando Dr. Moris Responda
1. Editar `config/param_ranges.json` con rangos reales
2. `python scripts/run_production.py --generate 50`
3. Deploy en WS via ZIP
4. `python scripts/run_production.py --prod`

## Email enviado al Dr. Moris (2026-02-20)
Preguntas pendientes:
1. Geometria: ¿solo BLIR3.stl o otras formas?
2. Altura de flujo: rango realista para canal UCN
3. Masa del bloque: rango presupuestado para testeo fisico
4. Criterio de movimiento incipiente: umbrales de desplazamiento/rotacion
