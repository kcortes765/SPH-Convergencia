# RETOMAR.md — SPH-IncipientMotion
> Ultima actualizacion: 2026-02-24 (sesion 14)

## Estado Actual
**50 casos LHS corriendo en WS (RTX 5090). Screening de sensibilidad con 1 forma (BLIR3). ~9.4 dias.**

## TAREA INMEDIATA AL RETOMAR

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

### Pendiente siempre:
- **Email a Moris** pidiendo: 6 STLs adicionales + validacion de rangos
- **Render Blender** en WS (agua PLY pendiente)

## CONTEXTO CRITICO

### Reencuadre del estudio (sesion 14)
- Las 50 sims son un **estudio preliminar de screening**, NO el estudio definitivo
- Objetivo: determinar que parametros importan (Sobol) y donde esta la transicion
- Esto informa el diseno del estudio principal con 7 formas de boulder
- Los 50 casos usan un LHS DEDICADO de 50 puntos (no subset de 100)

### Caso de validacion independiente
- `data/validacion_lhs001_old.csv` — caso de la corrida anterior (LHS 100)
- h=0.293, M=1.452, rot=53.2 → disp=2.839m, MOVIMIENTO
- Usar para validar GP: si predice ~2.8m, el modelo funciona

### Plan del Fondecyt (7 formas)
```
Screening (corriendo)  →  Estudio Principal (7 formas)  →  Paper + Tesis
  50 LHS, 1 forma          ~35/forma, informed by Sobol     GP + UQ + Sobol
  ~9.4 dias GPU             ~7x35 = 245 sims                comparar formas
```

### Decision sobre parametros
- 3 params: dam_height [0.10,0.50], boulder_mass [0.80,1.60], rot_z [0,90]
- Rangos NO validados por Moris — son provisionales
- Si Sobol dice rot_z no importa → fijar en estudio principal (2 params, menos sims)
- Para 7 formas: densidad fija (2650 kg/m3) probablemente mejor que masa variable

## Scripts de Analisis (post-simulacion)

| Script | Que hace | Figuras |
|--------|----------|---------|
| `scripts/analisis_completo.py` | **Maestro** — corre todo en secuencia | 31 total |
| `src/ml_surrogate.py` | Entrena GP, valida LOO | 3 |
| `scripts/run_uq_analysis.py` | Monte Carlo + Sobol | 6 |
| `scripts/figuras_piloto.py` | 22 figuras thesis-quality | 22 |

### Comando rapido
```bash
# Todo de una:
python scripts/analisis_completo.py --validar

# Solo figuras (si GP ya entrenado):
python scripts/analisis_completo.py --solo-figuras

# Testing sin datos reales:
python scripts/analisis_completo.py --synthetic
```

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

### Discusion de parametros
- 3 params actuales son provisionales (no validados por Moris)
- Para 7 formas: considerar densidad fija vs masa variable
- rot_z podria eliminarse si Sobol < 5%

### Git
```
2bc6b28 Redesign pilot as 50-case screening study (dedicated LHS)
5501e4e Add 8 new thesis figures + detailed progress logging per case
db7a04f Fix TypeError when timeout_s is None in log formatting
```

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

## WS Status
- **Corrida activa**: 50 casos LHS, dp=0.004, iniciada 2026-02-24 14:49
- **Caso 1 viejo**: guardado en `cases/_validacion/lhs_001_old/`
- **ETA fin**: ~2026-03-06 (9.4 dias)
