#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Player 3D — SPH Boulder Transport
Slider scrubber + teclas para avanzar/retroceder.

Controles:
    Slider          Arrastrar para ir a cualquier momento
    Flecha derecha  Avanzar 1 frame
    Flecha izquierda Retroceder 1 frame
    1               Camara Follow
    2               Camara Lateral fija
    3               Camara Top-down
    R               Reset
    Q               Salir

Uso:
    python scripts/player_3d.py
    python scripts/player_3d.py --case test_diego_reference
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"
STL_SCALE = 0.04


def load_data(case_id):
    case_dir = CASES_DIR / case_id
    out_dir = case_dir / f"{case_id}_out"

    kin = pd.read_csv(out_dir / 'ChronoExchange_mkbound_51.csv', sep=';')
    n_total = len(kin)
    step = max(1, n_total // 400)
    kin = kin.iloc[::step].reset_index(drop=True)

    t = kin['time [s]'].values
    cx = kin['fcenter.x [m]'].values
    cy = kin['fcenter.y [m]'].values
    cz = kin['fcenter.z [m]'].values
    vx = kin['fvel.x [m/s]'].values
    vy = kin['fvel.y [m/s]'].values
    vz = kin['fvel.z [m/s]'].values
    vel = np.sqrt(vx**2 + vy**2 + vz**2)

    canal = pv.read(str(case_dir / 'Canal_Playa_1esa20_750cm.stl'))
    boulder = pv.read(str(case_dir / 'BLIR3.stl'))
    boulder.points *= STL_SCALE
    bc = np.mean(boulder.points, axis=0)

    return {
        'case_id': case_id,
        'canal': canal,
        'boulder': boulder,
        'bc': bc,
        't': t, 'cx': cx, 'cy': cy, 'cz': cz,
        'vel': vel,
        'n': len(t),
    }


def run(data):
    n = data['n']
    cx, cy, cz = data['cx'], data['cy'], data['cz']
    vel = data['vel']
    t = data['t']
    bc = data['bc']

    disp_total = np.sqrt((cx[-1]-cx[0])**2 + (cy[-1]-cy[0])**2 + (cz[-1]-cz[0])**2)

    pl = pv.Plotter(window_size=[1920, 1080],
                     title=f"SPH Player - {data['case_id']}")
    pl.set_background('#0a1628', top='#162a45')

    # ── Static: Canal ──
    pl.add_mesh(data['canal'], color='#d4b896', opacity=0.75,
                smooth_shading=True, specular=0.15)

    # ── Static: Dam ghost ──
    dam = pv.Box(bounds=[0, 3.0, 0, 1.0, 0, 0.4536])
    pl.add_mesh(dam, color='#2196F3', opacity=0.15, smooth_shading=True)

    # ── Static: Full trajectory (faint) ──
    all_pts = np.column_stack([cx, cy, cz])
    spline = pv.Spline(all_pts, n_points=min(800, n * 2))
    spline['vel'] = np.interp(
        np.linspace(0, 1, spline.n_points),
        np.linspace(0, 1, n), vel
    )
    tube = spline.tube(radius=0.008, n_sides=8)
    pl.add_mesh(tube, scalars='vel', cmap='coolwarm', opacity=0.2,
                smooth_shading=True, scalar_bar_args={
                    'title': 'Vel [m/s]', 'title_font_size': 12,
                    'label_font_size': 10, 'n_labels': 4, 'fmt': '%.1f',
                    'position_x': 0.75, 'position_y': 0.05,
                    'width': 0.2, 'height': 0.04})

    # ── Static: Start/End markers ──
    pl.add_mesh(pv.Sphere(0.03, center=[cx[0], cy[0], cz[0]]),
                color='#00e676', smooth_shading=True)
    pl.add_mesh(pv.Sphere(0.03, center=[cx[-1], cy[-1], cz[-1]]),
                color='#ff1744', smooth_shading=True)

    # ── Dynamic: Boulder ──
    b0 = data['boulder'].copy()
    b0.translate(np.array([cx[0], cy[0], cz[0]]) - bc, inplace=True)
    boulder_actor = pl.add_mesh(b0, color='#d84315', smooth_shading=True,
                                specular=0.7)

    # ── Dynamic: Trail points ──
    trail = pv.PolyData(np.array([[cx[0], cy[0], cz[0]]]))
    trail_actor = pl.add_mesh(trail, color='#ff7043', point_size=5,
                              render_points_as_spheres=True, opacity=0.8)

    # ── Dynamic: Position sphere ──
    pos_sphere = pv.Sphere(0.025, center=[cx[0], cy[0], cz[0]])
    pos_actor = pl.add_mesh(pos_sphere, color='#ffeb3b', smooth_shading=True)

    # ── HUD ──
    hud_text = pl.add_text(
        f"{data['case_id']} | t={t[0]:.2f}s | v=0.00 m/s | frame 0/{n-1}",
        position='upper_edge', font_size=12, color='white', shadow=True,
        name='hud'
    )

    pl.add_text(
        "Slider=tiempo  LEFT/RIGHT=frame  1=Follow 2=Lateral 3=Top  Q=salir",
        position='lower_left', font_size=9, color='#5577aa', shadow=True
    )

    # State
    state = {'frame': 0, 'cam': 'follow'}

    def update(idx):
        idx = max(0, min(n - 1, int(idx)))
        state['frame'] = idx
        pos = np.array([cx[idx], cy[idx], cz[idx]])

        # Boulder
        new_b = data['boulder'].copy()
        new_b.translate(pos - bc, inplace=True)
        boulder_actor.mapper.dataset.copy_from(new_b)

        # Trail
        step_t = max(1, idx // 150)
        t_idx = np.arange(0, idx + 1, step_t)
        if len(t_idx) < 1:
            t_idx = np.array([0])
        pts = np.column_stack([cx[t_idx], cy[t_idx], cz[t_idx]])
        trail_actor.mapper.dataset.copy_from(pv.PolyData(pts))

        # Position marker
        pos_actor.mapper.dataset.copy_from(
            pv.Sphere(0.025, center=pos.tolist()))

        # HUD
        disp = np.sqrt((cx[idx]-cx[0])**2 + (cy[idx]-cy[0])**2 + (cz[idx]-cz[0])**2)
        pl.remove_actor('hud')
        pl.add_text(
            f"{data['case_id']} | t={t[idx]:.2f}s | v={vel[idx]:.2f} m/s | "
            f"disp={disp:.2f}m | frame {idx}/{n-1}",
            position='upper_edge', font_size=12, color='white',
            shadow=True, name='hud')

        # Camera
        if state['cam'] == 'follow':
            ahead = min(idx + 15, n - 1)
            focal = np.array([(pos[0] + cx[ahead]) / 2, pos[1], pos[2] + 0.05])
            cam = focal + np.array([-0.5, -1.0, 0.8])
            pl.camera_position = [cam.tolist(), focal.tolist(), [0, 0, 1]]
        elif state['cam'] == 'top':
            pl.camera_position = [
                [pos[0], pos[1], 3.0],
                [pos[0], pos[1], 0.0],
                [1, 0, 0]]

    # Slider
    slider_w = pl.add_slider_widget(
        lambda v: update(int(v)),
        rng=[0, n - 1], value=0,
        title="Tiempo",
        pointa=(0.15, 0.92), pointb=(0.85, 0.92),
        style='modern', color='white',
    )

    # Key events
    def next_f():
        f = min(state['frame'] + 1, n - 1)
        slider_w.GetRepresentation().SetValue(f)
        update(f)

    def prev_f():
        f = max(state['frame'] - 1, 0)
        slider_w.GetRepresentation().SetValue(f)
        update(f)

    def skip_fwd():
        f = min(state['frame'] + n // 20, n - 1)
        slider_w.GetRepresentation().SetValue(f)
        update(f)

    def skip_bwd():
        f = max(state['frame'] - n // 20, 0)
        slider_w.GetRepresentation().SetValue(f)
        update(f)

    def cam_follow():
        state['cam'] = 'follow'
        update(state['frame'])

    def cam_lateral():
        state['cam'] = 'fixed'
        mx = (cx[0] + cx[-1]) / 2
        pl.camera_position = [
            (mx, -3.0, 2.5),
            (mx, 0.5, 0.25),
            (0, 0, 1)]
        pl.render()

    def cam_top():
        state['cam'] = 'top'
        update(state['frame'])

    def reset():
        state['frame'] = 0
        state['cam'] = 'follow'
        slider_w.GetRepresentation().SetValue(0)
        update(0)

    pl.add_key_event('Right', next_f)
    pl.add_key_event('Left', prev_f)
    pl.add_key_event('period', skip_fwd)
    pl.add_key_event('comma', skip_bwd)
    pl.add_key_event('1', cam_follow)
    pl.add_key_event('2', cam_lateral)
    pl.add_key_event('3', cam_top)
    pl.add_key_event('r', reset)

    # Initial camera
    update(0)

    print(f"\n{'='*55}")
    print(f"  SPH Player - {data['case_id']}")
    print(f"  {n} frames | t=[{t[0]:.2f}, {t[-1]:.2f}]s")
    print(f"  Desp: {disp_total:.2f}m | V_peak: {vel.max():.2f} m/s")
    print(f"{'='*55}")
    print(f"  Slider     Arrastrar = ir a cualquier momento")
    print(f"  LEFT/RIGHT Avanzar/retroceder 1 frame")
    print(f"  < / >      Saltar 5%")
    print(f"  1          Camara Follow (sigue al boulder)")
    print(f"  2          Camara Lateral (panoramica)")
    print(f"  3          Camara Top-down")
    print(f"  R          Reset al inicio")
    print(f"  Q          Salir")
    print(f"{'='*55}\n")

    pl.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="lhs_001")
    args = p.parse_args()

    print(f"Cargando {args.case}...")
    data = load_data(args.case)
    print(f"  {data['n']} frames listos.")
    run(data)
