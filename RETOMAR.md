# RETOMAR.md — SPH-IncipientMotion
> Ultima actualizacion: 2026-02-24 (sesion 16)

## Estado Actual
**50 casos LHS corriendo en WS (RTX 5090). ETA: 6 marzo. Viz 3D local completada. Bug fixes aplicados.**

## TAREA INMEDIATA AL RETOMAR

### ANTES de que terminen las 50 sims:
1. **Borrar results.sqlite** (tiene datos dev dp=0.02, causara problemas con datos reales)
2. **Escribir Cap 1 (Intro) y Cap 2 (Marco Teorico)** — no dependen de datos
3. **Render Blender en WS** — ver `INSTRUCCIONES_RENDER_WS.md` (CPU, en paralelo con sims)

### Si las simulaciones terminaron (o hay >= 30 casos):
1. **Descargar results.sqlite desde WS** (o copiar CSVs procesados)
2. **Correr analisis completo:**
   ```bash
   python scripts/analisis_completo.py --validar
   ```
   Esto genera **31 figuras** automaticamente:
   - 3 figuras GP (data/figuras_ml/)
   - 6 figuras UQ (data/figuras_uq/)
   - 22 figuras piloto (data/figuras_piloto/)
   - Validacion contra lhs_001_old
3. **Revisar Sobol:** si rot_z contribuye <5%, se puede fijar para estudio multi-forma
4. **Escribir Cap. 3 de tesis:** Screening de sensibilidad

### Si las simulaciones aun corren:
- Monitorear via ntfy (topic: `sph-kevin-tesis-2026`)
- Puedes correr analisis parcial cuando haya >= 10 casos:
  ```bash
  python scripts/analisis_completo.py
  ```
- Analizar a los 30 casos si Sobol ya convergio → posiblemente parar ahi

### Analisis de caso individual (NUEVO sesion 15):
```bash
# Generar 12 figuras para cualquier caso:
python scripts/analisis_caso_individual.py --case lhs_001
python scripts/analisis_caso_individual.py --case test_diego_reference
```
Ya generados: `data/figuras_lhs_001/` y `data/figuras_test_diego_reference/`

### Pendiente siempre:
- **Email a Moris** pidiendo: 6 STLs adicionales + validacion de rangos
- **Render Blender** en WS (agua PLY pendiente)

## CONTEXTO CRITICO

### Reencuadre del estudio (sesion 14)
- Las 50 sims son un **estudio preliminar de screening**, NO el estudio definitivo
- Objetivo: determinar que parametros importan (Sobol) y donde esta la transicion
- Esto informa el diseno del estudio principal con 7 formas de boulder
- Los 50 casos usan un LHS DEDICADO de 50 puntos (no subset de 100)

### Alineacion con Fondecyt Moris (sesion 15)
- **Fondecyt Iniciacion 2025** (Dr. Joaquin Moris, FaCIC-UCN, abril 2025)
- Objetivo Fondecyt: mejorar ecuaciones predictivas de transporte incipiente incorporando FORMA
- Ecuaciones actuales (Nott 2003, Nandasena 2011) tratan bloques como prismas rectangulares
- Metodologia: experimentos laboratorio + simulaciones numericas
- Sitio: Bahia Cisnes (Atacama), bloques ~40 ton transportados por tsunami ~600 años
- **Nuestra tesis es el engranaje numerico** de la cadena terremoto→tsunami→transporte→deposito
- **Direccion recomendada post-screening**: curvas de movimiento incipiente
  - Barrer h_dam bajas para encontrar h_critico donde boulder apenas se mueve
  - No depende de STLs extra (se puede hacer solo con BLIR3)
  - Es literalmente el titulo de la tesis: "Incipient Motion"

### Caso de validacion independiente
- `data/validacion_lhs001_old.csv` — caso de la corrida anterior (LHS 100)
- h=0.293, M=1.452, rot=53.2 → disp=2.839m, MOVIMIENTO
- Usar para validar GP: si predice ~2.8m, el modelo funciona

### Plan del Fondecyt (7 formas)
```
Screening (corriendo)  →  Incipient Motion  →  Estudio Multi-forma  →  Paper + Tesis
  50 LHS, 1 forma          h_critico curves      7 formas, ~35/forma     GP + UQ + Sobol
  ~9.4 dias GPU             solo BLIR3             ~245 sims               comparar formas
```

### Decision sobre parametros
- 3 params: dam_height [0.10,0.50], boulder_mass [0.80,1.60], rot_z [0,90]
- Rangos NO validados por Moris — son provisionales
- Si Sobol dice rot_z no importa → fijar en estudio principal (2 params, menos sims)
- Para 7 formas: densidad fija (2650 kg/m3) probablemente mejor que masa variable

## Scripts de Analisis

| Script | Que hace | Figuras |
|--------|----------|---------|
| `scripts/analisis_completo.py` | **Maestro** — corre todo en secuencia | 31 total |
| `scripts/analisis_caso_individual.py` | 12 figuras para 1 caso (displacement, vel, forces, gauges, energy) | 12 |
| `src/ml_surrogate.py` | Entrena GP, valida LOO | 3 |
| `scripts/run_uq_analysis.py` | Monte Carlo + Sobol | 6 |
| `scripts/figuras_piloto.py` | 22 figuras thesis-quality | 22 |

### Comando rapido
```bash
# Todo de una (31 figuras):
python scripts/analisis_completo.py --validar

# Solo un caso (12 figuras):
python scripts/analisis_caso_individual.py --case lhs_001

# Solo figuras (si GP ya entrenado):
python scripts/analisis_completo.py --solo-figuras

# Testing sin datos reales:
python scripts/analisis_completo.py --synthetic
```

## Lo que se hizo en sesion 16 (2026-02-24)

### Visualizacion 3D completa
- Simulacion dp=0.01 en laptop (RTX 4060): 1.77M particulas, 101 frames, 5s, ~50 min
- Pipeline: GenCase -> DualSPHysics GPU -> PartVTK -> pyvista player
- `player_sph.py`: player interactivo con 1.34M particulas de agua + boulder
- `player_3d.py`: player trayectoria con slider y teclas
- Screenshots: `data/render_3d_showcase.png`, `data/render_3d_side.png`, `data/render_3d_impact.png`
- Caso en `cases/viz_dp01/` (101 VTKs fluid + 101 VTKs boulder, 5.3 GB .bi4)

### Bug fixes criticos
- `data_cleaner.py`: INSERT con nombres de columna explicitos (evita scramble por ALTER TABLE)
- `analisis_caso_individual.py`: filtro predictor `astype(str) == 'False'` (antes comparaba string con bool)

### Auditoria tesis (agente background)
- 3 de ~7 caps escritos. Cap 4 (convergencia) es el mas maduro
- Caps 1 (Intro) y 2 (Marco Teorico) se pueden escribir YA
- Duplicacion entre caps 3 y 4 identificada

### Instrucciones render Blender WS
- `INSTRUCCIONES_RENDER_WS.md`: pipeline paso a paso, CPU rendering, en paralelo con sims

## Lo que se hizo en sesion 15 (2026-02-24)

### Analisis caso individual
- Script `scripts/analisis_caso_individual.py` — 12 figuras thesis-quality por caso
- lhs_001: disp=6.10m, v_peak=2.30 m/s, rot=91°, F_SPH=218N, F_contact=184N
- test_diego_reference: disp=4.48m, v_peak=1.40 m/s, rot=132°, F_contact=494N
- Comparacion consistente, diferencias por distintas condiciones iniciales (dam height)
- Hallazgo: ChronoBody_forces fx/fy/fz INCLUYEN gravedad (fz baseline = -mg)

### Auditoria codebase
- Solo 3 issues cosmeticos encontrados (corregidos):
  - PLAN.md: RigidAlgorithm=1→3, FtPause=0.0→0.5
  - run_production.py: comment "100 casos"→"50 casos"
  - main_orchestrator.py: comment dp "0.005"→"0.004"
- Codigo y configs 100% correctos

### Alineacion Fondecyt
- Articulo UCN julio 2025 sobre Fondecyt de Moris
- Nuestra tesis alimenta directamente la investigacion
- Direccion post-screening: curvas de incipient motion

## Lo que se hizo en sesion 14 (2026-02-24)

### Correcciones criticas
- **Bug timeout_s=None** en batch_runner.py (TypeError en log formatting)
- **Logging mejorado**: progress banner con barra ASCII, ETA, tasa exito, resumen por caso
- **Notificaciones ntfy mejoradas**: metricas detalladas por caso

### Figuras: 14 → 22
8 figuras nuevas: heatmap parametrico, boxplots, correlacion, violines, energia adimensional, dependencia parcial GP, dashboard multioutput, altura critica

### Rediseno experimental
- 100 → 50 casos LHS DEDICADO (no subset)
- Reencuadre como "screening de sensibilidad" (no estudio definitivo)
- Confirmado: LHS(50, seed=42) != primeros 50 de LHS(100, seed=42)

### Script analisis_completo.py
- Maestro que corre GP + UQ + 22 figuras + validacion en secuencia
- 31 figuras totales en un solo comando
- Incluye validacion contra caso lhs_001_old

## Decisiones Acumuladas
- Chrono (RigidAlgorithm=3) obligatorio
- massbody > rhopbody, FtPause=0.5
- dp produccion: 0.004 (convergido con 7 dp)
- Contact force: NO usar como criterio (CV=82%)
- 3 parametros: dam_height [0.10,0.50], boulder_mass [0.80,1.60], boulder_rot_z [0,90]
- Deploy WS via ZIP desde GitHub
- Figuras: 300dpi PNG + PDF, espanol, serif fonts
- UQ: Monte Carlo 10k + Sobol (Saltelli manual)
- Timeout: None (sin limite, GPU dedicada)
- Notificaciones: ntfy.sh topic sph-kevin-tesis-2026
- 50 casos LHS dedicado para screening (no 100)
- Screening informa diseno multi-forma (7 STLs)
- Curvas de incipient motion como siguiente estudio (alineado con Fondecyt Moris)

## WS Status
- **Corrida activa**: 50 casos LHS, dp=0.004, iniciada 2026-02-24 14:49
- **Caso 1 viejo**: guardado en `cases/_validacion/lhs_001_old/`
- **ETA fin**: ~2026-03-06 (9.4 dias)
