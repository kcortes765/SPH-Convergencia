"""
convert_vtk_to_ply.py — Convierte isosurfaces VTK a PLY para importar en Blender.

Requiere: pip install pyvista
Ejecutar: python convert_vtk_to_ply.py

Autor: Kevin Cortes (UCN 2026)
"""

import os
import glob
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(PROJECT_ROOT, "data", "render", "dp004")
SURFACE_DIR = os.path.join(RENDER_DIR, "surface")
PLY_DIR = os.path.join(RENDER_DIR, "ply")


def convert_all():
    try:
        import pyvista as pv
    except ImportError:
        print("Instalando pyvista...")
        os.system(f"{sys.executable} -m pip install pyvista")
        import pyvista as pv

    os.makedirs(PLY_DIR, exist_ok=True)

    vtk_files = sorted(glob.glob(os.path.join(SURFACE_DIR, "WaterSurface_*.vtk")))
    if not vtk_files:
        print(f"No se encontraron VTKs en {SURFACE_DIR}")
        return

    print(f"Convirtiendo {len(vtk_files)} VTKs a PLY...")

    for i, vtk_path in enumerate(vtk_files):
        ply_path = os.path.join(PLY_DIR, f"water_{i:04d}.ply")
        try:
            mesh = pv.read(vtk_path)
            # Extraer solo la superficie si es volumen
            if hasattr(mesh, 'extract_surface'):
                surface = mesh.extract_surface()
            else:
                surface = mesh
            surface.save(ply_path)
            size_mb = os.path.getsize(ply_path) / (1024 * 1024)
            print(f"  [{i+1}/{len(vtk_files)}] {os.path.basename(vtk_path)} -> {os.path.basename(ply_path)} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ERROR en {vtk_path}: {e}")

    print(f"\nListo! {len(vtk_files)} PLYs en {PLY_DIR}")
    print("Ahora ejecuta Blender:")
    print(f'  blender --python blender_render.py')


if __name__ == "__main__":
    convert_all()
