#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Player SPH con particulas de agua — Visualizacion completa.
Carga VTKs generados por PartVTK y reproduce la simulacion frame a frame.

Pipeline:
    1. Simulacion DualSPHysics -> .bi4
    2. PartVTK -> .vtk (fluid + boulder)
    3. Este script -> ventana interactiva

Controles:
    Slider          Scrub temporal
    LEFT/RIGHT      Frame a frame
    , / .           Saltar 5%
    1               Camara Follow (sigue al boulder)
    2               Camara Lateral
    3               Camara Top-down
    4               Camara Close-up impacto
    R               Reset
    Q               Salir

Uso:
    python scripts/player_sph.py
    python scripts/player_sph.py --case viz_dp01
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pyvista as pv

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


def discover_vtks(case_id):
    """Find all VTK files for fluid and boulder."""
    case_dir = CASES_DIR / case_id
    out_dir = case_dir / f"{case_id}_out"
    vtk_fluid_dir = out_dir / "vtk_fluid"
    vtk_bound_dir = out_dir / "vtk_bound"

    fluid_files = sorted(vtk_fluid_dir.glob("fluid_*.vtk"))
    boulder_files = sorted(vtk_bound_dir.glob("boulder_*.vtk"))

    if not fluid_files:
        print(f"ERROR: No fluid VTKs in {vtk_fluid_dir}")
        print("Run: python scripts/postprocess_vtk.py --case", case_id)
        sys.exit(1)

    # Also load canal STL for static geometry
    canal_stl = case_dir / "Canal_Playa_1esa20_750cm.stl"

    return {
        'case_id': case_id,
        'fluid_files': fluid_files,
        'boulder_files': boulder_files,
        'canal_stl': canal_stl if canal_stl.exists() else None,
        'n_frames': len(fluid_files),
        'dt': 0.05,  # TimeOut from XML
    }


def run_player(info):
    n = info['n_frames']
    dt = info['dt']

    print(f"Loading frame 0 for initial setup...")

    # Load first frame to get structure
    fluid0 = pv.read(str(info['fluid_files'][0]))
    has_boulder = len(info['boulder_files']) > 0
    boulder0 = pv.read(str(info['boulder_files'][0])) if has_boulder else None

    # Compute velocity magnitude for fluid
    if 'Vel' in fluid0.array_names:
        vel = fluid0['Vel']
        vmag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)
        fluid0['vel_mag'] = vmag
    else:
        fluid0['vel_mag'] = np.zeros(fluid0.n_points)

    # ── Plotter ──
    pl = pv.Plotter(window_size=[1920, 1080],
                     title=f"SPH Simulation - {info['case_id']}")
    pl.set_background('#0a1628', top='#162a45')

    # ── Static: Canal ──
    if info['canal_stl']:
        canal = pv.read(str(info['canal_stl']))
        pl.add_mesh(canal, color='#d4b896', opacity=0.6,
                    smooth_shading=True, specular=0.15)

    # ── Dynamic: Fluid particles ──
    fluid_actor = pl.add_mesh(
        fluid0, scalars='vel_mag', cmap='coolwarm',
        point_size=2, render_points_as_spheres=True,
        clim=[0, 3.0],  # velocity range
        scalar_bar_args={
            'title': 'Vel [m/s]', 'title_font_size': 13,
            'label_font_size': 11, 'shadow': True, 'n_labels': 5,
            'fmt': '%.1f', 'position_x': 0.78, 'position_y': 0.05,
            'width': 0.18, 'height': 0.05,
        },
        opacity=0.8,
    )

    # ── Dynamic: Boulder particles ──
    boulder_actor = None
    if boulder0 is not None and boulder0.n_points > 0:
        boulder_actor = pl.add_mesh(
            boulder0, color='#d84315',
            point_size=4, render_points_as_spheres=True,
        )

    # ── HUD ──
    pl.add_text(
        f"{info['case_id']} | t=0.00s | frame 0/{n-1}",
        position='upper_edge', font_size=12, color='white',
        shadow=True, name='hud'
    )

    pl.add_text(
        "Slider=tiempo  LEFT/RIGHT=frame  1=Follow 2=Lateral 3=Top 4=Close  Q=salir",
        position='lower_left', font_size=9, color='#6688aa', shadow=True
    )

    # ── State ──
    state = {'frame': 0, 'cam': 'lateral'}

    # Cache for loaded frames
    cache = {}

    def load_frame(idx):
        """Load and cache a frame's VTK data."""
        if idx in cache:
            return cache[idx]

        fluid = pv.read(str(info['fluid_files'][idx]))
        if 'Vel' in fluid.array_names:
            vel = fluid['Vel']
            fluid['vel_mag'] = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)
        else:
            fluid['vel_mag'] = np.zeros(fluid.n_points)

        boulder = None
        if has_boulder and idx < len(info['boulder_files']):
            boulder = pv.read(str(info['boulder_files'][idx]))

        result = {'fluid': fluid, 'boulder': boulder}

        # Cache last 10 frames for smooth scrubbing
        cache[idx] = result
        if len(cache) > 10:
            oldest = min(cache.keys())
            del cache[oldest]

        return result

    def update_scene(idx):
        idx = max(0, min(n - 1, int(idx)))
        state['frame'] = idx

        frame_data = load_frame(idx)
        fluid = frame_data['fluid']

        # Update fluid
        fluid_actor.mapper.dataset.copy_from(fluid)

        # Update boulder
        if boulder_actor is not None and frame_data['boulder'] is not None:
            boulder_actor.mapper.dataset.copy_from(frame_data['boulder'])

        # Compute boulder center for camera
        b_center = None
        if frame_data['boulder'] is not None and frame_data['boulder'].n_points > 0:
            b_center = np.mean(frame_data['boulder'].points, axis=0)

        # HUD
        t_sim = idx * dt
        pl.remove_actor('hud')
        pl.add_text(
            f"{info['case_id']} | t={t_sim:.2f}s | frame {idx}/{n-1}",
            position='upper_edge', font_size=12, color='white',
            shadow=True, name='hud'
        )

        # Camera
        if state['cam'] == 'follow' and b_center is not None:
            focal = b_center + np.array([0.3, 0, 0.05])
            cam = focal + np.array([-0.6, -1.2, 0.9])
            pl.camera_position = [cam.tolist(), focal.tolist(), [0, 0, 1]]
        elif state['cam'] == 'top' and b_center is not None:
            pl.camera_position = [
                [b_center[0], b_center[1], 3.0],
                [b_center[0], b_center[1], 0.0],
                [1, 0, 0]]
        elif state['cam'] == 'closeup' and b_center is not None:
            cam = b_center + np.array([-0.3, -0.5, 0.4])
            pl.camera_position = [cam.tolist(), b_center.tolist(), [0, 0, 1]]

        pl.render()

    # ── Slider ──
    slider = pl.add_slider_widget(
        lambda v: update_scene(int(v)),
        rng=[0, n - 1], value=0,
        title="Frame",
        pointa=(0.15, 0.92), pointb=(0.85, 0.92),
        style='modern', color='white',
    )

    # ── Key events ──
    def next_f():
        f = min(state['frame'] + 1, n - 1)
        slider.GetRepresentation().SetValue(f)
        update_scene(f)

    def prev_f():
        f = max(state['frame'] - 1, 0)
        slider.GetRepresentation().SetValue(f)
        update_scene(f)

    def skip_fwd():
        f = min(state['frame'] + n // 20, n - 1)
        slider.GetRepresentation().SetValue(f)
        update_scene(f)

    def skip_bwd():
        f = max(state['frame'] - n // 20, 0)
        slider.GetRepresentation().SetValue(f)
        update_scene(f)

    def cam_follow():
        state['cam'] = 'follow'
        update_scene(state['frame'])

    def cam_lateral():
        state['cam'] = 'lateral'
        pl.camera_position = [
            (7.0, -3.5, 3.0),
            (9.0, 0.5, 0.15),
            (0, 0, 1)]
        pl.render()

    def cam_top():
        state['cam'] = 'top'
        update_scene(state['frame'])

    def cam_closeup():
        state['cam'] = 'closeup'
        update_scene(state['frame'])

    def reset():
        state['frame'] = 0
        state['cam'] = 'lateral'
        slider.GetRepresentation().SetValue(0)
        update_scene(0)
        cam_lateral()

    pl.add_key_event('Right', next_f)
    pl.add_key_event('Left', prev_f)
    pl.add_key_event('period', skip_fwd)
    pl.add_key_event('comma', skip_bwd)
    pl.add_key_event('1', cam_follow)
    pl.add_key_event('2', cam_lateral)
    pl.add_key_event('3', cam_top)
    pl.add_key_event('4', cam_closeup)
    pl.add_key_event('r', reset)

    # Initial camera: lateral view of boulder zone
    cam_lateral()

    print(f"\n{'='*55}")
    print(f"  SPH Particle Player - {info['case_id']}")
    print(f"  {n} frames | dt={dt}s | T=[0, {n*dt:.1f}]s")
    print(f"  Fluid: {fluid0.n_points:,} particles")
    if boulder0:
        print(f"  Boulder: {boulder0.n_points:,} particles")
    print(f"{'='*55}")
    print(f"  Slider     = Scrub temporal")
    print(f"  LEFT/RIGHT = Frame a frame")
    print(f"  < / >      = Saltar 5%")
    print(f"  1 = Follow  2 = Lateral  3 = Top  4 = Close-up")
    print(f"  R = Reset   Q = Salir")
    print(f"{'='*55}\n")

    pl.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SPH Particle Player")
    p.add_argument("--case", default="viz_dp01")
    args = p.parse_args()

    print(f"Buscando VTKs para {args.case}...")
    info = discover_vtks(args.case)
    print(f"  {info['n_frames']} frames encontrados")
    run_player(info)
