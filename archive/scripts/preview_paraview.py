"""
preview_paraview.py — Abre ParaView con los VTKs del render cargados y configurados.

Ejecutar en la WS con:
  & "C:\Program Files\ParaView 6.0.1\bin\pvpython.exe" preview_paraview.py

Autor: Kevin Cortes (UCN 2026)
"""

from paraview.simple import *
import glob
import os

# === CONFIGURACION ===
RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "render", "dp004")
SURFACE_DIR = os.path.join(RENDER_DIR, "surface")
BOULDER_DIR = os.path.join(RENDER_DIR, "boulder")
FLUID_DIR = os.path.join(RENDER_DIR, "fluid")

# STLs para geometria estatica
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CANAL_STL = os.path.join(PROJECT_ROOT, "models", "Canal_Playa_1esa20_750cm.stl")

print("=" * 60)
print("  PARAVIEW PREVIEW — SPH Render dp=0.004")
print("=" * 60)

# === 1. CARGAR ISOSURFACE DEL AGUA ===
surface_files = sorted(glob.glob(os.path.join(SURFACE_DIR, "WaterSurface_*.vtk")))
if surface_files:
    print(f"\n[1/3] Cargando {len(surface_files)} isosurfaces del agua...")
    water = LegacyVTKReader(FileNames=surface_files)
    water_display = Show(water)
    water_display.Representation = 'Surface'
    # Colorear por velocidad
    ColorBy(water_display, ('POINTS', 'Vel', 'Magnitude'))
    vel_lut = GetColorTransferFunction('Vel')
    vel_lut.RescaleTransferFunction(0.0, 2.0)
    vel_lut.ApplyPreset('Cool to Warm', True)
    water_display.Opacity = 0.85
    print(f"  OK: {len(surface_files)} frames de agua cargados")
else:
    print("  AVISO: No se encontraron isosurfaces")

# === 2. CARGAR BOULDER ===
boulder_files = sorted(glob.glob(os.path.join(BOULDER_DIR, "PartBoulder_*.vtk")))
if boulder_files:
    print(f"\n[2/3] Cargando {len(boulder_files)} frames del boulder...")
    boulder = LegacyVTKReader(FileNames=boulder_files)
    boulder_display = Show(boulder)
    boulder_display.Representation = 'Surface'
    boulder_display.DiffuseColor = [0.6, 0.4, 0.2]  # Marron roca
    boulder_display.Opacity = 1.0
    print(f"  OK: {len(boulder_files)} frames del boulder cargados")
else:
    print("  AVISO: No se encontraron VTKs del boulder")

# === 3. CARGAR CANAL (STL estatico) ===
if os.path.exists(CANAL_STL):
    print(f"\n[3/3] Cargando geometria del canal...")
    canal = STLReader(FileNames=[CANAL_STL])
    canal_display = Show(canal)
    canal_display.Representation = 'Surface'
    canal_display.DiffuseColor = [0.7, 0.7, 0.7]  # Gris
    canal_display.Opacity = 0.5
    print(f"  OK: Canal cargado")
else:
    print(f"  AVISO: Canal STL no encontrado en {CANAL_STL}")

# === 4. CONFIGURAR CAMARA ===
print("\nConfigurando camara...")
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1920, 1080]
view.Background = [0.1, 0.1, 0.15]  # Fondo oscuro

# Camara lateral (vista del impacto)
view.CameraPosition = [8.5, -3.0, 1.5]   # Posicion camara
view.CameraFocalPoint = [8.5, 0.5, 0.2]  # Mirar al boulder
view.CameraViewUp = [0.0, 0.0, 1.0]      # Z arriba

# Barra de color
if surface_files:
    vel_bar = GetScalarBar(vel_lut, view)
    vel_bar.Title = 'Velocidad (m/s)'
    vel_bar.Visibility = 1

# === 5. RENDER ===
Render()

# Ir al frame del impacto (~frame 5, t=2.5s)
animationScene = GetAnimationScene()
animationScene.GoToFirst()
if len(surface_files) > 5:
    for _ in range(5):
        animationScene.GoToNext()

Render()

print("\n" + "=" * 60)
print("  LISTO! ParaView deberia estar mostrando la simulacion.")
print("  - Play para animar")
print("  - Ctrl+Rueda para zoom")
print("  - Click+drag para rotar")
print("  - Frame actual: ~t=2.5s (impacto)")
print("=" * 60)

# Mantener abierto en modo interactivo
Interact()
