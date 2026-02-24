# PLAN MAESTRO: Render 3D Fotorrealista de Simulacion SPH Dam-Break + Boulder

**Proyecto:** SPH-IncipientMotion (Tesis UCN 2026, Kevin Cortes)
**Fecha:** 2026-02-21
**Objetivo:** Producir un render fotorrealista basado 100% en datos reales de convergencia DualSPHysics v5.4.3

---

## PARTE 1: SELECCION DEL ESCENARIO

### 1.1 Caso Recomendado: dp=0.004 (26.1M particulas)

De los 7 niveles de resolucion del estudio de convergencia, dp=0.004 es el sweet spot para visualizacion:

| Criterio | dp=0.005 | dp=0.004 | dp=0.003 |
|----------|----------|----------|----------|
| Particulas | 13.4M | 26.1M | 61.9M |
| Desplazamiento | 1.725m | 1.615m | Pendiente |
| Rotacion | ~85 deg | ~85 deg | Pendiente |
| Convergencia delta | 28.4% | 6.4% | -- |
| Tiempo computo | 118 min | 260 min | ~9h |
| Calidad visual | Buena | Excelente | Mejor pero puede no estar listo |

**Razones para dp=0.004:**
- Convergencia solida: solo 6.4% de cambio respecto a dp=0.005
- 26.1M particulas: densidad suficiente para superficie de agua suave
- 1.615m de desplazamiento + ~85 grados de rotacion: visualmente dramatico
- Los datos ya existen en la workstation
- dp=0.003 puede aun estar corriendo y no seria significativamente mejor visualmente

### 1.2 Momentos Clave a Capturar

La simulacion cubre t=0 a t=10 segundos:

| Momento | Tiempo | Descripcion | Uso |
|---------|--------|-------------|-----|
| Estado inicial | t=0.0s | Columna de agua a la izquierda, boulder quieto en playa inclinada | Referencia "antes" |
| Propagacion | t=1.0-1.5s | Frente de onda avanzando por el canal hacia la playa | Contexto de flujo |
| **Primer impacto** | **t=2.0-2.5s** | **Ola colisiona con boulder, splash espectacular** | **HERO SHOT** |
| Transporte activo | t=3.0-4.0s | Boulder en movimiento, rotando, agua envolviendolo | Momento de accion |
| Maximo caos | t=4.0-5.0s | Boulder desplazado ~0.8m, turbulencia maxima | Drama |
| Equilibrio | t=8.0-10.0s | Boulder desplazado 1.6m, rotado 85 grados, agua calmandose | Estado final |

**Para still image (primer deliverable)**: t=2.0-2.5s (impacto con splash)
**Para animacion completa**: t=0.0 a t=10.0s, con posible slow-motion en t=2.0-3.0s

### 1.3 Problema Critico: Los .bi4 fueron eliminados

El `batch_runner.py` elimina .bi4/.vtk/.bt4 despues de extraer CSV. Los .bi4 son INDISPENSABLES para generar VTK/isosurfaces.

**Solucion**: Re-ejecutar dp=0.004 en la workstation, ejecutando post-procesado visual ANTES de la limpieza.

---

## PARTE 2: TOOLCHAIN COMPLETO

### Pipeline de Datos

```
DualSPHysics Solver (.bi4 binarios)
        |
        |--- PartVTK_win64.exe ----> fluid_XXXX.vtk  (particulas fluido)
        |--- PartVTK_win64.exe ----> boulder_XXXX.vtk (particulas boulder)
        |--- BoundaryVTK_win64.exe -> channel.vtk      (canal/playa)
        |--- IsoSurface_win64.exe --> water_XXXX.vtk   (superficie agua)
        |
        v
   [splashsurf] (alternativa superior para calidad visual)
        |
        v
   VTK / OBJ meshes
        |
        |--- SciBlend / import directo ---> Blender
        |
        v
   Blender (Cycles renderer, GPU)
        |--- Materiales (agua Glass BSDF, roca PBR, hormigon)
        |--- Iluminacion (HDRI + 3-point lighting)
        |--- Camara (angulos cinematicos)
        |--- Render (GPU accelerated, OptiX denoiser)
        |
        v
   Deliverables: PNG stills 4K + MP4 animacion 1080p
```

### Software Necesario

| Software | Proposito | Costo |
|----------|-----------|-------|
| DualSPHysics v5.4.3 | Post-procesamiento .bi4 a VTK | Gratis (ya instalado) |
| Blender 4.5+ LTS | Motor de render 3D fotorrealista | Gratis |
| SciBlend Advanced Core | Importar series VTK a Blender | Gratis (extension Blender) |
| splashsurf | Reconstruccion de superficie de particulas (mejor que IsoSurface) | Gratis |
| ParaView 5.12+ | Preview rapida y debugging visual | Gratis |
| ffmpeg | Compilar PNG a MP4 | Gratis |

---

## PARTE 3: ESTILO VISUAL

### Opcion Recomendada: Semi-Fotorrealista

- **Agua**: Semi-transparente azul con Glass BSDF (IOR=1.33), absorcion volumetrica azul-verdosa
- **Boulder**: Material PBR de roca caliza con textura procedural
- **Canal/Playa**: Hormigon gris claro, ligeramente rugoso
- **Iluminacion**: HDRI cielo despejado + luces de relleno suaves
- **Fondo**: Gradiente limpio o HDRI desaturado

### Estrategia Dual

1. **Render hero semi-fotorrealista** (Blender) -- para portada, presentacion, defensa
2. **Figuras tecnicas con colormap** (ParaView) -- para el cuerpo de la tesis

---

## PARTE 4: WORKFLOW DETALLADO

### FASE 1: Re-simulacion con Output Visual (~5h)

**Ubicacion**: Workstation UCN (RTX 5090)

#### TimeOut para visualizacion

| TimeOut | Frames | Disco estimado | Uso |
|---------|--------|----------------|-----|
| 0.5s | 20 | ~17 GB | Suficiente para still + animacion basica |
| 0.1s | 100 | ~83 GB | Animacion suave |
| 0.02s | 500 | ~418 GB | Imposible |

**Recomendacion**: `TimeOut=0.5` para primer render (20 frames, 17 GB).

#### Crear `run_for_render.py`

Script que:
1. Ejecuta simulacion dp=0.004 con `TimeOut=0.5`
2. Ejecuta PartVTK, IsoSurface, BoundaryVTK inmediatamente
3. LUEGO limpia .bi4

### FASE 2: Post-procesado DualSPHysics (~30-60 min)

**Prerequisito**: .bi4 deben existir (antes de cleanup)

```
DSPH_BIN = C:\Users\ProyBloq_Cortes\Desktop\CarpetaTesis\ejecutablesWindows
CASE_OUT = cases\conv_dp0004\conv_dp0004_out
RENDER   = data\render\dp004
```

#### Particulas de fluido
```bat
%DSPH_BIN%\PartVTK_win64.exe ^
    -dirdata %CASE_OUT% -filexml %CASE_OUT%\conv_dp0004.xml ^
    -savevtk %RENDER%\fluid\PartFluid ^
    -onlytype:-all,+fluid -vars:+idp,+vel,+rhop,+press
```

#### Particulas del boulder
```bat
%DSPH_BIN%\PartVTK_win64.exe ^
    -dirdata %CASE_OUT% -filexml %CASE_OUT%\conv_dp0004.xml ^
    -savevtk %RENDER%\boulder\PartBoulder ^
    -onlymk:51 -vars:+idp,+vel
```

#### Isosurface del agua (CRITICO para calidad visual)
```bat
%DSPH_BIN%\IsoSurface_win64.exe ^
    -dirdata %CASE_OUT% -filexml %CASE_OUT%\conv_dp0004.xml ^
    -saveiso %RENDER%\surface\WaterSurface ^
    -onlytype:-all,+fluid ^
    -distnode_dp:1.5 -distinter_2h:1.0 -vars:+vel,+press
```

#### Geometria del canal
```bat
%DSPH_BIN%\BoundaryVTK_win64.exe ^
    -loadvtk %CASE_OUT%\conv_dp0004__Actual.vtk ^
    -filexml %CASE_OUT%\conv_dp0004.xml ^
    -savevtk %RENDER%\channel\Channel -onlymk:0
```

#### Alternativa superior: splashsurf
```bash
splashsurf reconstruct %RENDER%\fluid\PartFluid_0005.vtk ^
    --particle-radius=0.004 --smoothing-length=2.0 ^
    --cube-size=0.75 --surface-threshold=0.6 ^
    --mesh-smoothing-iters=20 --mesh-smoothing-weights=on ^
    --mesh-cleanup=on --normals=on ^
    --output-file=%RENDER%\surface_hq\WaterSurface_0005.obj
```

### FASE 3: Preview en ParaView (~30 min)

1. Verificar isosurface (sin huecos, splash visible)
2. Verificar boulder (desplazamiento correcto)
3. Identificar frame optimo para hero shot (t=2.0-2.5s, frame 4-5)
4. Exportar figuras tecnicas con colormap para tesis

### FASE 4: Setup en Blender (~3-5h)

#### Importar geometria
1. Canal: `models/Canal_Playa_1esa20_750cm.stl`
2. Boulder: `models/BLIR3.stl` (Scale=0.04, Location=(8.5, 0.5, 0.1))
3. Agua: isosurface VTK/OBJ via SciBlend o import directo

#### Animar boulder con datos Chrono
Script Python en Blender que lee `ChronoExchange_mkbound_51.csv` y aplica keyframes de posicion/rotacion.

### FASE 5: Materiales (~1-2h)

#### Agua "Realistic Water"
```
Principled BSDF:
  Base Color:          (0.05, 0.15, 0.25)
  Roughness:           0.02
  IOR:                 1.333
  Transmission Weight: 1.0

Volume Absorption:
  Color:    (0.4, 0.75, 0.9)
  Density:  2.0

Bump (micro-ondas):
  Noise Texture Scale: 500
  Bump Strength: 0.015

Light Paths (CRITICO):
  Transmission Bounces: 12+ (si no, agua se ve negra)
```

#### Boulder "Limestone Rock"
```
Noise Texture -> Color Ramp (marron a gris)
Roughness: 0.85
Voronoi Texture -> Bump (Strength: 0.3)
```

#### Canal "Concrete"
```
Base Color: (0.50, 0.50, 0.48)
Roughness: 0.88
Musgrave Texture -> Bump (Strength: 0.08)
```

### FASE 6: Iluminacion y Camara (~1h)

#### HDRI + 3-Point Lighting
- **HDRI**: "kloofendal_48d_partly_cloudy" (polyhaven.com), Strength=1.5
- **Key Light**: Area 3x3m, 800W, posicion (7, -1.5, 3)
- **Fill Light**: Area 4x4m, 200W, azul 7000K, posicion (10, 2, 1.5)
- **Rim Light**: Spot 400W, detras del boulder

#### Camaras
| Camara | Posicion | Focal | Uso |
|--------|----------|-------|-----|
| Hero Shot (3/4) | (6, -2, 1.5) | 35mm | Toma principal |
| Vista lateral | (8.5, -3, 0.8) | 50mm | Para tesis |
| Vista aerea | (8.5, 0.5, 4) | 28mm | Patron de flujo |
| Close-up dramatico | (7.5, 0, 0.5) | 85mm | Maximo drama |

### FASE 7: Render Settings

#### Still Image 4K
```
Engine: Cycles, GPU Compute
Samples: 1024, OptiX Denoiser
Resolution: 3840x2160
Transmission Bounces: 12
View Transform: Filmic, Medium Contrast
```

| Hardware | Tiempo estimado |
|----------|----------------|
| RTX 4060 (laptop) | 15-25 min |
| RTX 5090 (workstation) | 3-5 min |

#### Animacion 1080p
```
Samples: 256 + OptiX Denoiser
Resolution: 1920x1080
300 frames (10s a 30fps)
```

| Hardware | Tiempo total |
|----------|-------------|
| RTX 4060 | ~10-15 horas |
| RTX 5090 | ~2-2.5 horas |

### FASE 8: Post-Procesamiento (~1-2h)

#### Still Image
- Glare, Color Balance, Lens Distortion, Vignette en Blender Compositor
- Anotaciones: barra escala, timestamp, etiquetas, caption

#### Animacion
```bash
ffmpeg -framerate 30 ^
    -i "frame_%%04d.png" ^
    -c:v libx264 -crf 18 -pix_fmt yuv420p ^
    "SPH_DamBreak_dp004.mp4"
```

---

## PARTE 5: PRIORIDADES

1. **PRIMERO**: Still image (1 frame, feedback rapido, inmediatamente usable)
2. **DESPUES**: Animacion (repetir render 300 veces una vez que el setup esta perfecto)

---

## PARTE 6: TIMELINE REALISTA

| Dia | Actividad | Horas humanas | Horas maquina |
|-----|-----------|---------------|---------------|
| 1 manana | Crear run_for_render.py, deploy a workstation | 2h | -- |
| 1 tarde | Lanzar simulacion, instalar Blender | 1h | 5h (bg) |
| 2 manana | Post-procesado + transferir datos + ParaView preview | 2h | 0.5h |
| 2 tarde | Import Blender, materiales, luces, camara | 3h | -- |
| 2 tarde | Test renders y ajustes | 1.5h | -- |
| 2 noche | Render final still 4K | 0.5h | 0.5h |
| 2 noche | Post-procesado y anotaciones | 1h | -- |
| **TOTAL still** | | **~11h humanas** | **~6h maquina** |
| 3 | Animacion setup + render + compilar | 4h | 2.5h |
| **TOTAL con video** | | **~15h humanas** | **~8.5h maquina** |

---

## PARTE 7: CHECKLIST

**Ya disponibles:**
- [x] `models/BLIR3.stl` -- mesh del boulder
- [x] `models/Canal_Playa_1esa20_750cm.stl` -- mesh del canal
- [x] `config/template_base.xml` -- template del caso
- [x] `src/batch_runner.py` -- runner existente

**A generar:**
- [ ] `run_for_render.py` -- script dedicado
- [ ] `data/render/dp004/surface/WaterSurface_XXXX.vtk` -- isosurfaces
- [ ] `data/render/dp004/fluid/PartFluid_XXXX.vtk` -- particulas (ParaView)
- [ ] ChronoExchange CSV (puede ya existir)

**A instalar:**
- [ ] Blender 4.5+ LTS
- [ ] SciBlend extension
- [ ] splashsurf
- [ ] ffmpeg
