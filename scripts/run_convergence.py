"""
run_convergence.py — Estudio de Convergencia de Malla Oficial

Mantiene la fisica constante (caso base de Diego, corregido) e itera
sobre la resolucion de particulas (dp) para encontrar el dp optimo
donde las metricas convergen.

dp probados: 0.020, 0.015, 0.010, 0.008, 0.005

Uso:
    python run_convergence.py              # Corre todos los dp
    python run_convergence.py --desde 0.010  # Retoma desde dp=0.010

Autor: Kevin Cortes (UCN 2026)
"""

import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# --- Setup de imports desde src/ ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from geometry_builder import CaseParams, build_case, compute_boulder_properties
from batch_runner import load_config, run_case
from data_cleaner import process_case, save_to_sqlite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuracion del estudio (NO MODIFICAR entre corridas)
# ---------------------------------------------------------------------------

DP_LIST = [0.020, 0.015, 0.010, 0.008, 0.005]

# Caso base: parametros de Diego corregidos (material roca, no PVC)
BASE_PARAMS = {
    'dam_height': 0.3,
    'boulder_mass': 1.06053,
    'boulder_scale': 0.04,
    'boulder_pos': (8.487, 0.523, 0.124),
    'boulder_rot': (0.0, 0.0, 0.0),
    'material': 'lime-stone',
    'time_max': 10.0,
}


# ---------------------------------------------------------------------------
# Status file (monitoreo remoto)
# ---------------------------------------------------------------------------

STATUS_FILE = PROJECT_ROOT / 'data' / 'convergencia_status.json'


def update_status(dp_actual: float, indice: int, total: int,
                  estado: str, resultados: list, inicio: float):
    """Escribe un JSON con el estado actual para monitoreo remoto."""
    elapsed_min = (time.time() - inicio) / 60

    ok = [r for r in resultados if r.get('status') == 'OK']
    fails = [r for r in resultados if r.get('status') != 'OK']

    status = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dp_actual': dp_actual,
        'estado': estado,
        'progreso': f"{indice}/{total}",
        'completados_ok': len(ok),
        'fallidos': len(fails),
        'tiempo_transcurrido_min': round(elapsed_min, 1),
        'dp_pendientes': [dp for dp in DP_LIST[indice:]],
        'ultimo_resultado': ok[-1] if ok else None,
    }

    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Pipeline para un dp
# ---------------------------------------------------------------------------

def run_single_dp(dp: float, config: dict, d_eq: float) -> dict:
    """
    Ejecuta el pipeline completo para un valor de dp.

    Returns:
        dict con metricas del caso o info de error.
    """
    case_name = f"conv_dp{dp:.3f}".replace('.', '')
    start = time.time()

    entry = {'dp': dp, 'case_name': case_name, 'status': 'ERROR'}

    # Rutas
    template_xml = PROJECT_ROOT / config['paths']['template_xml']
    boulder_stl = PROJECT_ROOT / config['paths']['boulder_stl']
    beach_stl = PROJECT_ROOT / config['paths']['beach_stl']
    materials_xml = PROJECT_ROOT / config['paths']['materials_xml']
    cases_dir = PROJECT_ROOT / config['paths']['cases_dir']
    processed_dir = PROJECT_ROOT / config['paths']['processed_dir'] / case_name

    # --- 1. Geometry ---
    params = CaseParams(
        case_name=case_name,
        dp=dp,
        dam_height=BASE_PARAMS['dam_height'],
        boulder_mass=BASE_PARAMS['boulder_mass'],
        boulder_scale=BASE_PARAMS['boulder_scale'],
        boulder_pos=BASE_PARAMS['boulder_pos'],
        boulder_rot=BASE_PARAMS['boulder_rot'],
        material=BASE_PARAMS['material'],
        time_max=BASE_PARAMS['time_max'],
        time_out=BASE_PARAMS['time_max'],   # Solo output al final
        ft_pause=config['defaults']['ft_pause'],
    )

    xml_path = build_case(
        template_xml=template_xml,
        boulder_stl=boulder_stl,
        beach_stl=beach_stl,
        materials_xml=materials_xml,
        output_dir=cases_dir,
        params=params,
    )
    case_dir = xml_path.parent
    logger.info("  [1/3] Geometry: OK")

    # --- 2. Simulacion GPU ---
    run_result = run_case(case_dir, config, processed_dir)

    if not run_result['success']:
        entry['status'] = 'FALLO_SIM'
        entry['error'] = run_result['error_message']
        entry['tiempo_computo_min'] = (time.time() - start) / 60
        logger.error(f"  [2/3] Simulacion: FALLO — {run_result['error_message']}")
        return entry

    logger.info(f"  [2/3] Simulacion: OK ({run_result['duration_s']:.0f}s)")

    # --- 3. Analisis ---
    case_result = process_case(processed_dir, d_eq=d_eq)

    elapsed_min = (time.time() - start) / 60
    logger.info(f"  [3/3] Analisis: OK — total {elapsed_min:.1f} min")

    entry.update({
        'status': 'OK',
        'max_displacement_m': case_result.max_displacement,
        'max_displacement_pct': case_result.max_displacement_rel,
        'max_rotation_deg': case_result.max_rotation,
        'max_velocity_ms': case_result.max_velocity,
        'max_sph_force_N': case_result.max_sph_force,
        'max_contact_force_N': case_result.max_contact_force,
        'max_flow_velocity_ms': case_result.max_flow_velocity,
        'max_water_height_m': case_result.max_water_height,
        'sim_time_reached_s': case_result.sim_time_reached,
        'n_timesteps': case_result.n_timesteps,
        'tiempo_computo_min': elapsed_min,
        '_case_result': case_result,  # Para SQLite (no va al CSV)
    })
    return entry


# ---------------------------------------------------------------------------
# Estudio completo
# ---------------------------------------------------------------------------

def run_convergence_study(desde_dp: float = None):
    """Ejecuta el estudio de convergencia iterando sobre dp."""

    # Filtrar dp si se pide retomar
    dp_to_run = DP_LIST
    if desde_dp is not None:
        dp_to_run = [dp for dp in DP_LIST if dp <= desde_dp]
        logger.info(f"Retomando desde dp={desde_dp}: {dp_to_run}")

    logger.info("=" * 60)
    logger.info("ESTUDIO DE CONVERGENCIA DE MALLA")
    logger.info(f"dp a probar: {dp_to_run}")
    logger.info(f"Fisica congelada: dam_h={BASE_PARAMS['dam_height']}m, "
                f"mass={BASE_PARAMS['boulder_mass']}kg, "
                f"material={BASE_PARAMS['material']}")
    logger.info("=" * 60)

    # Cargar config
    config = load_config(PROJECT_ROOT / 'config' / 'dsph_config.json')

    # d_eq es constante (misma geometria en todos los dp)
    boulder_props = compute_boulder_properties(
        stl_path=PROJECT_ROOT / config['paths']['boulder_stl'],
        scale=BASE_PARAMS['boulder_scale'],
        rotation_deg=BASE_PARAMS['boulder_rot'],
        mass_kg=BASE_PARAMS['boulder_mass'],
    )
    d_eq = boulder_props['d_eq']
    logger.info(f"d_eq = {d_eq:.6f} m (constante para todos los dp)\n")

    # --- Loop principal ---
    resultados = []
    total_start = time.time()

    for i, dp in enumerate(dp_to_run, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(dp_to_run)}] dp = {dp} m")
        logger.info(f"{'='*60}")

        update_status(dp, i - 1, len(dp_to_run),
                      f"CORRIENDO dp={dp}", resultados, total_start)

        try:
            entry = run_single_dp(dp, config, d_eq)
            resultados.append(entry)
        except Exception as e:
            logger.error(f"  Excepcion no controlada: {e}", exc_info=True)
            resultados.append({
                'dp': dp,
                'case_name': f"conv_dp{dp:.3f}".replace('.', ''),
                'status': 'ERROR',
                'error': str(e),
                'tiempo_computo_min': 0.0,
            })

        update_status(dp, i, len(dp_to_run),
                      f"TERMINADO dp={dp}", resultados, total_start)

    total_min = (time.time() - total_start) / 60

    # --- Guardar CSV ---
    if not resultados:
        logger.error("Ningun caso completado.")
        return

    # Separar _case_result antes de crear el DataFrame
    case_results = [r.pop('_case_result') for r in resultados
                    if '_case_result' in r]

    df = pd.DataFrame(resultados)
    report_path = PROJECT_ROOT / 'data' / 'reporte_convergencia.csv'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_path, index=False, sep=';')
    logger.info(f"\nCSV guardado: {report_path}")

    # --- Guardar en SQLite (tabla separada) ---
    if case_results:
        db_path = PROJECT_ROOT / 'data' / 'results.sqlite'
        save_to_sqlite(case_results, db_path, table='convergence')
        logger.info(f"SQLite: {len(case_results)} resultados en tabla 'convergence'")

    # --- Tabla resumen con delta % ---
    ok = [r for r in resultados if r['status'] == 'OK']

    logger.info(f"\n{'='*60}")
    logger.info("REPORTE DE CONVERGENCIA")
    logger.info(f"{'='*60}")

    if ok:
        header = (f"{'dp':>8}  {'Desplaz(m)':>12}  {'delta%':>7}  "
                  f"{'Rot(deg)':>9}  {'F_sph(N)':>10}  "
                  f"{'F_cont(N)':>10}  {'Tiempo(min)':>11}")
        logger.info(header)
        logger.info("-" * len(header))

        prev_disp = None
        for r in ok:
            disp = r['max_displacement_m']
            delta = ''
            if prev_disp is not None and prev_disp > 0:
                pct = abs(disp - prev_disp) / prev_disp * 100
                delta = f"{pct:.1f}%"

            logger.info(
                f"{r['dp']:>8.3f}  {disp:>12.6f}  {delta:>7}  "
                f"{r['max_rotation_deg']:>9.2f}  "
                f"{r['max_sph_force_N']:>10.4f}  "
                f"{r['max_contact_force_N']:>10.4f}  "
                f"{r['tiempo_computo_min']:>11.1f}"
            )
            prev_disp = disp

    failed = [r for r in resultados if r['status'] != 'OK']
    if failed:
        logger.warning(f"\nCasos fallidos:")
        for r in failed:
            logger.warning(f"  {r['case_name']}: {r.get('error', 'desconocido')}")

    logger.info(f"\nResumen: {len(ok)}/{len(resultados)} exitosos, "
                f"tiempo total: {total_min:.1f} min")
    logger.info(f"CSV: {report_path}")

    # Status final
    update_status(0, len(dp_to_run), len(dp_to_run),
                  "COMPLETADO", resultados, total_start)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Log a consola + archivo
    log_file = PROJECT_ROOT / 'data' / 'convergencia.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8'),
        ],
    )

    # --desde dp para retomar si crashea a mitad de camino
    desde = None
    if len(sys.argv) > 1 and sys.argv[1] == '--desde':
        desde = float(sys.argv[2])

    run_convergence_study(desde_dp=desde)
