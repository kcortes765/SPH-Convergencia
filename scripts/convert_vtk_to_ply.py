"""
convert_vtk_to_ply.py — Convierte isosurfaces VTK a PLY para importar en Blender.

Requiere: pip install pyvista
Ejecutar: python scripts/convert_vtk_to_ply.py
          python scripts/convert_vtk_to_ply.py --decimate 0.5   (reducir 50% triangulos)

Autor: Kevin Cortes (UCN 2026)
"""

import os
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_DIR = os.path.join(PROJECT_ROOT, "data", "render", "dp004")
SURFACE_DIR = os.path.join(RENDER_DIR, "surface")
PLY_DIR = os.path.join(RENDER_DIR, "ply")


def convert_all(decimate_ratio=None):
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
        print(f"Ejecutar primero: python scripts/run_for_render.py")
        return

    print(f"Convirtiendo {len(vtk_files)} VTKs a PLY...")
    if decimate_ratio:
        print(f"  Decimation: {decimate_ratio} (reduccion de {(1-decimate_ratio)*100:.0f}%)")

    for i, vtk_path in enumerate(vtk_files):
        ply_path = os.path.join(PLY_DIR, f"water_{i:04d}.ply")
        try:
            mesh = pv.read(vtk_path)
            if hasattr(mesh, 'extract_surface'):
                surface = mesh.extract_surface()
            else:
                surface = mesh

            if decimate_ratio and hasattr(surface, 'decimate'):
                surface = surface.decimate(decimate_ratio)

            surface.save(ply_path)
            size_mb = os.path.getsize(ply_path) / (1024 * 1024)
            print(f"  [{i+1}/{len(vtk_files)}] {os.path.basename(vtk_path)} -> water_{i:04d}.ply ({size_mb:.1f} MB, {surface.n_points:,} pts)")
        except Exception as e:
            print(f"  ERROR en {vtk_path}: {e}")

    print(f"\nListo! {len(vtk_files)} PLYs en {PLY_DIR}")
    print("Siguiente paso:")
    print("  Blender GUI:     Scripting > Open > scripts/blender_render.py > Run Script")
    print("  Blender headless: blender --background --python scripts/blender_render.py")


if __name__ == "__main__":
    decimate = None
    if '--decimate' in sys.argv:
        idx = sys.argv.index('--decimate')
        if idx + 1 < len(sys.argv):
            decimate = float(sys.argv[idx + 1])

    convert_all(decimate_ratio=decimate)
