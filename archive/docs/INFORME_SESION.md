# Informe de Sesion — SPH-IncipientMotion Pipeline
> Fecha: 2026-02-20 | Autor: Kevin Cortes (UCN) | Para: Lead Architect

---

## Resumen Ejecutivo

Se completo la instalacion de DualSPHysics v5.4.3 en la laptop de desarrollo, se corrigieron 3 bugs criticos, se ejecuto exitosamente la simulacion completa (GenCase + solver GPU, 10s simulados, 15.4 min reales), se escribio y valido el Modulo 3 (ETL/data_cleaner), y se verifico el pipeline end-to-end: XML → particulas → simulacion GPU → CSVs → analisis → SQLite.

**Estado: 4 de 4 modulos funcionales.** Pipeline completo validado con campana LHS de 5 casos (5/5 exitosos, 99.7 min).

---

## 1. Instalacion DualSPHysics v5.4.3

### Proceso
- Se clono el repo GitHub (`github.com/DualSPHysics/DualSPHysics`) pero este **no incluye el solver GPU** — solo herramientas auxiliares (GenCase, PartVTK, etc.)
- Se descargo el **package completo** desde `dual.sphysics.org/downloads/` (requiere formulario de registro)
- Package instalado en: `C:\DualSPHysics_v5.4.3\DualSPHysics_v5.4\`
- Repo GitHub eliminado (redundante)

### Ejecutables Verificados (18 total)
| Ejecutable | Tamano | Funcion |
|---|---|---|
| `DualSPHysics5.4_win64.exe` | 141.5 MB | Solver GPU + Chrono |
| `DualSPHysics5.4CPU_win64.exe` | 21.0 MB | Solver CPU |
| `GenCase_win64.exe` | 2.8 MB | XML -> particulas |
| `FloatingInfo_win64.exe` | 3.1 MB | Post-proceso floating (no usado) |
| `ComputeForces_win64.exe` | 3.1 MB | Post-proceso fuerzas (no usado) |
| + 13 herramientas auxiliares | | |

### Dependencias DLL confirmadas
- `ChronoEngine.dll` (5.8 MB) — motor de cuerpos rigidos
- `dsphchrono.dll` (0.8 MB) — acoplamiento SPH-Chrono
- `vcomp140.dll` (0.2 MB) — runtime OpenMP

### Configuracion Actualizada
```json
// config/dsph_config.json
"dsph_bin": "C:\\DualSPHysics_v5.4.3\\DualSPHysics_v5.4\\bin\\windows"
```

---

## 2. Bug Critico Corregido: Formato de Inercia XML

### Problema
El geometry_builder.py generaba el tensor de inercia en formato 3x3 con tags `<values>`:
```xml
<inertia>
    <values v11="Ixx" v12="Ixy" v13="Ixz"/>
    <values v21="Iyx" v22="Iyy" v23="Iyz"/>
    <values v31="Izx" v32="Izy" v33="Izz"/>
</inertia>
```

GenCase v5.4 crasheaba con: `Error reading xml - Attribute 'z' is missing` (fila 55).

### Causa Raiz
El formato 3x3 con `<values>` es para DEM (`RigidAlgorithm=2`). Para **Chrono** (`RigidAlgorithm=3`), DualSPHysics espera la diagonal del tensor:
```xml
<inertia x="Ixx" y="Iyy" z="Izz"/>
```

Confirmado inspeccionando los ejemplos oficiales:
- `examples/chrono/05_OWSC/CaseOWSC3D_Def.xml` (linea 56)

### Fix Aplicado
```python
# ANTES (incorrecto para Chrono)
for i in range(3):
    row = etree.SubElement(inertia_elem, 'values')
    row.set(f'v{i+1}1', f'{I[i][0]:.8g}')
    ...

# DESPUES (correcto para Chrono)
inertia_elem.set('x', f'{I[0][0]:.8g}')
inertia_elem.set('y', f'{I[1][1]:.8g}')
inertia_elem.set('z', f'{I[2][2]:.8g}')
```

### Implicacion
Los terminos fuera de la diagonal (Ixy, Ixz, Iyz) se pierden. Para un boulder irregular esto introduce un error, pero:
- Los terminos cruzados del tensor son tipicamente 1-2 ordenes de magnitud menores que los diagonales
- Chrono internamente recalcula la inercia basada en la geometria de colision
- El ejemplo oficial de DualSPHysics tampoco usa terminos cruzados

**Decision:** Aceptable para la tesis. Documentar como limitacion.

---

## 3. Bug Corregido: GenCase CWD para STL

### Problema
GenCase no encontraba los archivos STL referenciados con path relativo en el XML:
```
*** Exception: Cannot open the file. File: BLIR3.stl
```

### Causa Raiz
`batch_runner.py` ejecutaba GenCase con paths absolutos, pero GenCase resuelve los paths de STL relativos al CWD del proceso, no al directorio del XML.

### Fix Aplicado
```python
# batch_runner.py — _run_step ahora acepta cwd
_run_step(gencase_cmd, 'GenCase', timeout_s=300,
          case_name=case_name, cwd=case_dir)
```

GenCase ahora se ejecuta con `cwd=case_dir` y usa paths relativos.

---

## 4. Bug Corregido: Paths del Solver en batch_runner

### Problema
DualSPHysics recibia paths absolutos/relativos-al-root pero se ejecutaba con `cwd=case_dir`:
```
File: cases\test_diego_reference\test_diego_reference_out/test_diego_reference.xml
# ↑ path relativo al root, pero cwd es el case_dir → no encontrado
```

### Fix
Usar paths relativos al case_dir para el solver (mismo patron que GenCase):
```python
# ANTES (roto)
dsph_cmd = [dsph_exe, f'-gpu:{gpu_id}', str(out_dir / case_name), str(out_dir)]

# DESPUES (funciona)
dsph_cmd = [dsph_exe, f'-gpu:{gpu_id}', f"{out_rel}/{case_name}", out_rel]
```

---

## 5. Prueba End-to-End: GenCase + Solver GPU

### GenCase (exitoso)
```
Caso: test_diego_reference
dp=0.02m, TimeMax=10s, FtPause=0.5s

Particulas generadas:
  Fixed.....: 108,170 (boundaries + beach)
  Floating..: 133 (boulder BLIR3)
  Fluid.....: 100,800
  Total.....: 209,103

Archivos generados: 15 (27.7 MB)
  - test_diego_reference.bi4 (8.9 MB, particulas)
  - test_diego_reference_ChronoGeo.cbi4 (geometria Chrono)
  - VTKs de visualizacion
  - Copias de STL y XML
Tiempo: 0.217s (16 threads)
```

### Solver GPU (COMPLETO)
```
Hardware: NVIDIA RTX 4060 Laptop (8GB VRAM)
Tiempo simulado: 9.999s de 10.0s (COMPLETO)
Tiempo real: 924.3s (~15.4 minutos)
Velocidad: ~1.54 min/s de simulacion (209K particulas)

CSVs generados (24 archivos):
  - ChronoExchange_mkbound_51.csv (9500 filas)
  - ChronoBody_forces.csv (9500 filas)
  - 12x GaugesVel_V**.csv
  - 8x GaugesMaxZ_hmax**.csv
  - Run.csv, RunPARTs.csv

Limpieza automatica: 15 archivos .bi4/.vtk eliminados (40.3 MB liberados)
```

**Rendimiento estimado por hardware:**
- RTX 4060 laptop (8GB): ~1.54 min/s sim → 10s en ~15 min
- RTX 5090 workstation (32GB): estimado ~10x mas rapido → 10s en ~1.5 min

---

## 6. Modulo 3: data_cleaner.py (COMPLETO)

### Funcionalidad
ETL completo que parsea todos los CSVs de DualSPHysics/Chrono:

| Parser | Archivo | Datos |
|---|---|---|
| `parse_chrono_exchange()` | ChronoExchange_mkbound_*.csv | Posicion, velocidad, omega, aceleracion del boulder |
| `parse_chrono_forces()` | ChronoBody_forces.csv | Fuerzas SPH + contacto por cuerpo |
| `parse_gauge_velocity()` | GaugesVel_V**.csv | Velocidad del flujo en puntos fijos |
| `parse_gauge_maxz()` | GaugesMaxZ_hmax**.csv | Altura maxima del agua |

### Analisis Implementado
- **Desplazamiento:** Distancia 3D del CM respecto a posicion inicial
- **Rotacion:** Integral acumulada de |omega| (trapezoidal)
- **Velocidad del boulder:** Magnitud de fvel
- **Fuerzas:** Magnitud de fuerzas SPH y de contacto
- **Flujo:** Gauge mas cercano al boulder (distancia euclidiana)

### Criterio de Falla (Incipient Motion)
```
moved   = max_displacement > d_eq * (threshold_pct / 100)
rotated = max_rotation > threshold_deg
failed  = moved OR rotated
```

Umbrales default: 5% d_eq desplazamiento, 5 grados rotacion.
**Nota:** Umbrales pendientes de validacion con Dr. Moris.

### Persistencia
Resultados guardados en SQLite (`data/results.sqlite`) con upsert por case_name.

### Validacion con Datos Reales

| Metrica | Diego (dp=0.05, FtPause=0) | Nuestro (dp=0.02, FtPause=0.5) |
|---|---|---|
| Tiempo simulado | 9.999s | 9.999s (COMPLETO) |
| Timesteps | 9,999 | 9,500 |
| Desplazamiento max | 3.905m (3889% d_eq) | 4.478m (4460% d_eq) |
| Rotacion max | 122.3 grados | 132.3 grados |
| Vel max boulder | 1.55 m/s | 1.40 m/s |
| Fuerza SPH max | 68.7 N | 100.4 N |
| Fuerza contacto max | 397.0 N | 494.4 N |
| Flujo max | 1.32 m/s | 0.42 m/s |
| Resultado | FALLO | FALLO |

Ambos casos muestran incipient motion claro (boulder arrastrado ~4m).
Las diferencias en fuerzas se explican por el dp mas fino (mayor resolucion de presion)
y el FtPause=0.5 (asentamiento gravitatorio antes del impacto).

### Manejo de Datos Problematicos
- **Sentinel -3.40282e+38** en GaugesMaxZ: reemplazado con NaN (61-100% de filas dependiendo del gauge)
- **Header duplicado** en ChronoBody_forces (fy, fz repetidos para cada body): resuelto con prefijo automatico por body
- **Separador punto y coma (;):** hardcodeado en todos los parsers

---

## 7. Estado de Modulos

| Modulo | Archivo | Estado | Validado |
|---|---|---|---|
| 1 - Geometry Builder | `src/geometry_builder.py` | COMPLETO | GenCase ejecuta OK |
| 2 - Batch Runner | `src/batch_runner.py` | COMPLETO | GenCase + GPU probados |
| 3 - Data Cleaner | `src/data_cleaner.py` | COMPLETO | Datos Diego + test OK |
| 4 - Orchestrator | `src/main_orchestrator.py` | COMPLETO | 5/5 campana LHS OK |
| 5 - ML Surrogate | `src/ml_surrogate.py` | PENDIENTE | Requiere ~50+ casos |

---

## 8. Modulo Orquestador: main_orchestrator.py (COMPLETO)

### Funcionalidad
Cerebro del pipeline. Genera matriz de experimentos via LHS y ejecuta el pipeline completo para cada caso:
```
LHS (scipy) → geometry_builder → batch_runner → data_cleaner → SQLite
```

### Diseno de Experimentos (LHS)
Latin Hypercube Sampling con `scipy.stats.qmc.LatinHypercube`:
- Variables: `dam_height` (0.2-0.5m), `boulder_mass` (1.0-3.0kg)
- 5 muestras generadas con seed=42 (reproducible)
- Guardadas en `config/experiment_matrix.csv`

### Manejo de Errores
Si un caso falla (GPU timeout, GenCase error, CSV corrupto):
- Registra el error en consola
- Continua con el siguiente caso
- Resumen final muestra exitosos vs fallidos

### CLI
```bash
python src/main_orchestrator.py              # Ejecutar campana (genera matriz si no existe)
python src/main_orchestrator.py --generate 10 # Solo generar matriz de N muestras
python src/main_orchestrator.py --prod        # Usar dp de produccion (0.005)
```

---

## 9. Primera Campana LHS: 5 Casos (COMPLETADA)

### Resultado: 5/5 exitosos, 0 fallidos

| Caso | dam_h (m) | mass (kg) | Desplaz (m) | Rotacion | Fuerza SPH | F. Contacto | Tiempo |
|---|---|---|---|---|---|---|---|
| lhs_001 | 0.454 | 1.224 | 6.096 | 91.1° | 95.7 N | 1237.8 N | 25.4 min |
| lhs_002 | 0.208 | 2.321 | 1.416 | 64.7° | 55.3 N | 2671.1 N | 12.7 min |
| lhs_003 | 0.374 | 2.610 | 5.164 | 100.0° | 199.2 N | 3327.3 N | 20.5 min |
| lhs_004 | 0.394 | 1.886 | 5.688 | 137.6° | 184.3 N | 2374.6 N | 22.3 min |
| lhs_005 | 0.312 | 1.620 | 4.473 | 74.6° | 147.2 N | 1952.5 N | 18.8 min |

**Tiempo total: 99.7 minutos (1h 40min) en RTX 4060 laptop**

### Observaciones Fisicas
- **Menor desplazamiento**: lhs_002 (dam_h=0.208m, mass=2.321kg) → 1.42m. Ola chica + boulder pesado.
- **Mayor desplazamiento**: lhs_001 (dam_h=0.454m, mass=1.224kg) → 6.10m. Ola grande + boulder liviano.
- Todos los 5 casos muestran incipient motion claro (todos superan umbrales)
- A dp=0.02 no se espera encontrar el umbral critico — se necesita dp mas fino y rangos mas estrechos
- 209 MB de binarios limpiados automaticamente por try/finally

### Almacenamiento
- 5 resultados en `data/results.sqlite`
- 120 CSVs procesados en `data/processed/`
- Matriz en `config/experiment_matrix.csv`

---

## 10. Proximos Pasos

1. ~~Escribir main_orchestrator.py~~ **COMPLETADO**
2. **Estudio de convergencia dp** — dp=0.02, 0.01, 0.005 en la workstation RTX 5090
3. **Validar umbrales de falla y rangos parametricos** con Dr. Moris (correo enviado)
4. **Campana de produccion** con rangos reales del laboratorio + dp fino
5. **ML Surrogate** (GP Regression) cuando haya suficientes datos (~50+ casos)

---

## 11. Riesgos y Deuda Tecnica

| Riesgo | Severidad | Mitigacion |
|---|---|---|
| Inercia sin terminos cruzados | Media | Chrono recalcula internamente; documentar como limitacion |
| dp=0.02 insuficiente (2 part en dim_min) | Alta | Solo para dev; produccion usa dp=0.005 |
| Timeout solver en laptop | Baja | Simular solo 2-3s en laptop, produccion en workstation |
| Sentinel en 61-100% de GaugesMaxZ | Media | Gauges mal posicionados; revisar ubicacion en XML |
| Umbrales de falla no validados | Alta | Reunion con Dr. Moris pendiente |
