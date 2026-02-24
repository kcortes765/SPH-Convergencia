#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-procesado: convierte .bi4 a .vtk usando PartVTK de DualSPHysics.
Genera VTKs de fluido (con velocidad) y boundary (boulder).

Uso:
    python scripts/postprocess_vtk.py
    python scripts/postprocess_vtk.py --case viz_dp01
"""

import argparse
import subprocess
import sys
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"
PARTVTK = Path(r"C:\DualSPHysics_v5.4.3\DualSPHysics_v5.4\bin\windows\PartVTK_win64.exe")


def run_partvtk(case_id):
    case_dir = CASES_DIR / case_id
    out_dir = case_dir / f"{case_id}_out"
    data_dir = out_dir / "data"

    vtk_fluid_dir = out_dir / "vtk_fluid"
    vtk_bound_dir = out_dir / "vtk_bound"
    vtk_fluid_dir.mkdir(exist_ok=True)
    vtk_bound_dir.mkdir(exist_ok=True)

    # Count available parts
    parts = sorted(data_dir.glob("Part_*.bi4"))
    n_parts = len(parts)
    print(f"Found {n_parts} .bi4 files in {data_dir}")

    if n_parts == 0:
        print("ERROR: No .bi4 files found. Run simulation first.")
        sys.exit(1)

    # PartVTK for FLUID particles (with velocity scalar)
    print(f"\n--- Exporting fluid VTKs ({n_parts} frames) ---")
    cmd_fluid = [
        str(PARTVTK),
        "-dirin", str(data_dir),
        "-filexml", str(out_dir / f"{case_id}.xml"),
        "-savevtk", str(vtk_fluid_dir / "fluid"),
        "-onlytype:-all,+fluid",
        "-vars:+idp,+vel,+rhop",
    ]
    print(f"  CMD: {' '.join(cmd_fluid)}")
    result = subprocess.run(cmd_fluid, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:] if result.stderr else 'unknown'}")
    else:
        n_vtk = len(list(vtk_fluid_dir.glob("*.vtk")))
        print(f"  OK: {n_vtk} fluid VTK files generated")

    # PartVTK for BOUNDARY particles (boulder only, mkbound=51)
    print(f"\n--- Exporting boulder VTKs ({n_parts} frames) ---")
    cmd_bound = [
        str(PARTVTK),
        "-dirin", str(data_dir),
        "-filexml", str(out_dir / f"{case_id}.xml"),
        "-savevtk", str(vtk_bound_dir / "boulder"),
        "-onlytype:-all,+floating",
        "-vars:+idp,+vel",
    ]
    print(f"  CMD: {' '.join(cmd_bound)}")
    result = subprocess.run(cmd_bound, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:] if result.stderr else 'unknown'}")
    else:
        n_vtk = len(list(vtk_bound_dir.glob("*.vtk")))
        print(f"  OK: {n_vtk} boulder VTK files generated")

    print(f"\nVTK export complete.")
    print(f"  Fluid: {vtk_fluid_dir}")
    print(f"  Boulder: {vtk_bound_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="viz_dp01")
    args = p.parse_args()
    run_partvtk(args.case)
