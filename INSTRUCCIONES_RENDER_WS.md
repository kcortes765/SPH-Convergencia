# Instrucciones: Render Blender en Workstation UCN

**Fecha:** 2026-02-24
**Prerequisito:** 50 sims LHS corriendo en GPU (ETA ~6 marzo)

---

## Puedo correr el render en paralelo con las 50 sims?

**SI, pero solo con render CPU.** Explicacion:

| Recurso | 50 sims LHS | Blender GPU | Blender CPU |
|---------|-------------|-------------|-------------|
| RTX 5090 GPU | **OCUPADA** (DualSPHysics) | **CONFLICTO** (CUDA OOM) | No la usa |
| i9-14900KF CPU | Chrono usa 1 thread | No la usa | **24 cores disponibles** |
| RAM 64GB | ~2-4 GB | ~4 GB | ~4 GB |

**Regla:** Blender debe usar **CPU rendering**, NO GPU, mientras las sims corren.

En `blender_render.py` cambiar la linea 59:
```python
# CAMBIAR ESTO:
scene.cycles.device = 'GPU'
# POR ESTO:
scene.cycles.device = 'CPU'
```

O mejor: al abrir Blender manualmente, ir a Edit > Preferences > System > Cycles Render Devices > seleccionar **None** (CPU only).

Render CPU en el i9-14900KF con 24 cores: ~15-20 min por frame 1080p (256 samples + denoiser). Perfectamente viable.

---

## Pipeline paso a paso en la WS

### Paso 0: Verificar que los datos de render existen

La simulacion de render (dp=0.004, 21 frames) ya se completo el 2026-02-22. Verificar:

```powershell
# En la WS, abrir PowerShell
cd C:\Users\ProyBloq_Cortes\Desktop\SPH-Convergencia

# Verificar que existen los VTKs o .bi4
dir data\render\dp004\surface\   # Deberian existir WaterSurface_*.vtk
dir data\render\dp004\fluid\     # Deberian existir PartFluid_*.vtk
dir data\render\dp004\boulder\   # Deberian existir PartBoulder_*.vtk
```

Si NO existen los VTKs pero SI los .bi4:
```powershell
python run_for_render.py --skip-sim   # Solo post-procesado
```

Si NO existe nada (los .bi4 se borraron), hay que re-simular:
```powershell
python run_for_render.py --keep-bi4
# ATENCION: esto usa la GPU. Solo correr si las 50 sims NO estan activas,
# o si ya terminaron. Tarda ~4.5h con dp=0.004.
```

### Paso 1: Convertir VTK a PLY (para Blender)

Blender no lee VTK nativamente. Necesitamos convertir las isosurfaces a PLY.

```powershell
python -c "
import pyvista as pv
from pathlib import Path
import os

render_dir = Path('data/render/dp004')
ply_dir = render_dir / 'ply'
ply_dir.mkdir(exist_ok=True)

vtk_files = sorted((render_dir / 'surface').glob('WaterSurface_*.vtk'))
print(f'Convirtiendo {len(vtk_files)} VTKs a PLY...')

for i, vtk_path in enumerate(vtk_files):
    mesh = pv.read(str(vtk_path))
    out = ply_dir / f'water_{i:04d}.ply'
    mesh.save(str(out))
    print(f'  [{i+1}/{len(vtk_files)}] {out.name} ({mesh.n_points:,} puntos)')

print('Conversion completada.')
"
```

### Paso 2: Abrir Blender y configurar CPU

1. Abrir **Blender 4.2** (ya instalado en WS)
2. Ir a **Edit > Preferences > System**
3. En "Cycles Render Devices": seleccionar **None** (fuerza CPU)
4. Cerrar Preferences

### Paso 3: Ejecutar blender_render.py

**Opcion A — Con GUI (recomendado para primera vez):**
1. En Blender, ir a la pestana **Scripting**
2. Click **Open** > navegar a `scripts\blender_render.py`
3. Antes de correr, editar linea 59: cambiar `'GPU'` por `'CPU'`
4. Click **Run Script**
5. Esperar ~15-20 min (1 frame 1080p con CPU)

**Opcion B — Headless (si ya funciono con GUI):**
```powershell
& "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --background --python scripts\blender_render.py
```

### Paso 4: Verificar output

```powershell
dir data\render\dp004\blender_output\
# Deberia existir: render_0005.png (hero shot del impacto a t=2.5s)
```

### Paso 5: (Opcional) Render de animacion completa

Si el hero shot se ve bien y quieres la animacion completa (20 frames):

```python
# Modificar blender_render.py:
# En vez de renderizar solo HERO_FRAME=5, hacer un loop:
for frame in range(20):
    # importar agua frame, renderizar, limpiar
```

**Tiempo estimado (CPU, 20 frames):** ~5-7 horas. Puede correr overnight.

---

## Resumen de tiempos

| Paso | Tiempo | Usa GPU? |
|------|--------|----------|
| Verificar datos | 1 min | No |
| Convertir VTK a PLY | 5 min | No |
| Configurar Blender | 5 min | No |
| Render 1 frame (CPU) | 15-20 min | **No** |
| Render 20 frames (CPU) | 5-7 horas | **No** |

**Total para hero shot:** ~30 min de trabajo humano.
**Conflicto con sims:** NINGUNO (todo CPU).

---

## Troubleshooting

**"Water PLY not found"**: Paso 1 no se completo. Convertir VTK a PLY primero.

**Blender crashea al importar**: El PLY puede ser muy grande. Reducir con:
```python
mesh = pv.read('WaterSurface_0005.vtk')
decimated = mesh.decimate(0.5)  # reducir 50% triangulos
decimated.save('water_0005.ply')
```

**Render se ve negro**: Aumentar Transmission Bounces en Cycles a 12+.
Blender > Render Properties > Light Paths > Max Bounces > Transmission: 12

**Out of memory CPU**: Reducir RENDER_SAMPLES a 128 (con denoiser sigue viendose bien).
