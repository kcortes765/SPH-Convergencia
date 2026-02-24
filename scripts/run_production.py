"""
run_production.py — Script "Boton Rojo" para campana de produccion masiva

Lee config/param_ranges.json, genera matriz LHS, ejecuta el pipeline
completo en la GPU, y sincroniza status via archivo JSON.

Ejecutar:
    python run_production.py --generate 50       # Solo generar matriz
    python run_production.py --dry-run            # Simular sin ejecutar GPU
    python run_production.py                      # Correr campaña (dev dp)
    python run_production.py --prod               # Correr campaña (prod dp)
    python run_production.py --prod --desde 15    # Recovery: continuar desde caso 15

Autor: Kevin Cortes (UCN 2026)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Agregar src/ al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main_orchestrator import load_param_ranges, generate_experiment_matrix, run_pipeline_case
from batch_runner import load_config
from data_cleaner import save_to_sqlite
from ml_surrogate import run_surrogate

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
STATUS_FILE = PROJECT_ROOT / "data" / "production_status.json"


def update_status(status: dict):
    """Escribe status JSON atomicamente para monitoreo remoto."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    status['updated'] = datetime.now().isoformat()
    tmp = STATUS_FILE.with_suffix('.tmp')
    try:
        with open(tmp, 'w') as f:
            json.dump(status, f, indent=2)
        tmp.replace(STATUS_FILE)
    except Exception as e:
        logger.warning(f"Error escribiendo status: {e}")
        if tmp.exists():
            tmp.unlink()


MAX_FAIL_RATE = 0.30  # Abort si >30% de los casos fallan


def preflight_check(config: dict) -> bool:
    """Verificaciones pre-produccion. Retorna True si todo OK."""
    checks_passed = True

    # 1. Ejecutables existen
    for exe_key in ['gencase', 'dualsphysics_gpu']:
        exe_name = config['executables'][exe_key]
        exe_path = Path(config['dsph_bin']) / exe_name
        if exe_path.exists():
            logger.info(f"  OK: {exe_key} -> {exe_path}")
        else:
            logger.error(f"  FALTA: {exe_key} -> {exe_path}")
            checks_passed = False

    # 2. Template XML existe
    template = PROJECT_ROOT / config['paths']['template_xml']
    if template.exists():
        logger.info(f"  OK: template XML -> {template}")
    else:
        logger.error(f"  FALTA: template XML -> {template}")
        checks_passed = False

    # 3. Boulder STL existe
    stl = PROJECT_ROOT / config['paths']['boulder_stl']
    if stl.exists():
        logger.info(f"  OK: boulder STL -> {stl}")
    else:
        logger.error(f"  FALTA: boulder STL -> {stl}")
        checks_passed = False

    # 4. Espacio en disco (estimar 20 GB por caso temporal)
    import shutil as _shutil
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    usage = _shutil.disk_usage(str(data_dir))
    free_gb = usage.free / (1024**3)
    logger.info(f"  Disco libre: {free_gb:.1f} GB")
    if free_gb < 50:
        logger.warning(f"  ADVERTENCIA: Poco espacio libre ({free_gb:.1f} GB < 50 GB)")

    return checks_passed


def run_production(args):
    """Pipeline de produccion completo."""
    config_path = PROJECT_ROOT / 'config' / 'dsph_config.json'
    config = load_config(config_path)
    matrix_csv = PROJECT_ROOT / 'config' / 'experiment_matrix.csv'

    # Cargar rangos
    param_ranges = load_param_ranges()

    # Determinar dp
    if args.prod:
        dp = config['defaults']['dp_prod']
        logger.info(f"MODO PRODUCCION: dp={dp}")
    else:
        dp = config['defaults']['dp_dev']
        logger.info(f"MODO DESARROLLO: dp={dp}")

    # Determinar n_samples
    ranges_json = PROJECT_ROOT / 'config' / 'param_ranges.json'
    with open(ranges_json) as f:
        ranges_cfg = json.load(f)
    n_samples = ranges_cfg['sampling']['n_samples_prod' if args.prod else 'n_samples_dev']

    # Solo generar?
    if args.generate:
        n = args.generate
        generate_experiment_matrix(n, output_csv=matrix_csv, param_ranges=param_ranges)
        print(f"Matriz generada: {matrix_csv} ({n} muestras)")
        return

    # Generar si no existe
    if not matrix_csv.exists():
        logger.info(f"Generando matriz LHS ({n_samples} muestras)...")
        generate_experiment_matrix(n_samples, output_csv=matrix_csv, param_ranges=param_ranges)

    # Leer matriz
    matrix = pd.read_csv(matrix_csv)
    n_total = len(matrix)

    # Recovery: saltar casos ya completados
    desde = args.desde if args.desde else 0
    if desde > 0:
        logger.info(f"RECOVERY: saltando casos 1-{desde}, empezando desde {desde+1}")
        matrix = matrix.iloc[desde:]

    n_pending = len(matrix)

    # Status inicial
    status = {
        'phase': 'production',
        'dp': dp,
        'mode': 'prod' if args.prod else 'dev',
        'total_cases': n_total,
        'desde': desde,
        'pending': n_pending,
        'completed': 0,
        'failed': 0,
        'current_case': '',
        'start_time': datetime.now().isoformat(),
        'dry_run': args.dry_run,
    }
    update_status(status)

    # Pre-flight check
    logger.info("Pre-flight check...")
    if not preflight_check(config):
        logger.error("PRE-FLIGHT FALLIDO. Corregir errores antes de continuar.")
        sys.exit(1)
    logger.info("Pre-flight OK.\n")

    logger.info(f"\n{'#'*60}")
    logger.info(f"# PRODUCCION: {n_pending} casos pendientes (de {n_total} total)")
    logger.info(f"# dp={dp}, GPU={config['defaults']['gpu_id']}")
    logger.info(f"# {'DRY RUN' if args.dry_run else 'EJECUCION REAL'}")
    logger.info(f"{'#'*60}\n")

    all_results = []
    successful_results = []

    for i, (_, row) in enumerate(matrix.iterrows(), 1):
        case_id = row['case_id']
        logger.info(f"\n--- Caso {desde + i}/{n_total} [{case_id}] ---")

        status['current_case'] = case_id
        status['progress'] = f"{i}/{n_pending}"
        update_status(status)

        if args.dry_run:
            logger.info(f"  [DRY RUN] Saltando simulacion GPU")
            continue

        try:
            result = run_pipeline_case(row, PROJECT_ROOT, config, dp)
            all_results.append(result)

            if result['success'] and result['result'] is not None:
                successful_results.append(result['result'])
                status['completed'] += 1
                logger.info(f"  OK ({result['duration_s']:.1f}s)")
            else:
                status['failed'] += 1
                logger.error(f"  FALLO: {result['error']}")

        except Exception as e:
            status['failed'] += 1
            logger.error(f"  EXCEPCION: {e}", exc_info=True)

        update_status(status)

        # Abort si tasa de fallos excede umbral
        total_run = status['completed'] + status['failed']
        if total_run >= 3 and status['failed'] / total_run > MAX_FAIL_RATE:
            fail_pct = status['failed'] / total_run * 100
            logger.critical(
                f"\nABORT: Tasa de fallos {fail_pct:.0f}% > {MAX_FAIL_RATE*100:.0f}% "
                f"({status['failed']}/{total_run} casos fallidos). "
                f"Revisar configuracion antes de continuar."
            )
            status['phase'] = 'aborted'
            status['abort_reason'] = f"Fail rate {fail_pct:.0f}% after {total_run} cases"
            update_status(status)
            break

        # Guardar a SQLite despues de cada caso exitoso (crash safety)
        if successful_results:
            db_path = PROJECT_ROOT / 'data' / 'results.sqlite'
            save_to_sqlite(successful_results, db_path)
            successful_results = []  # Reset para no duplicar

    # Status final
    status['phase'] = 'completed'
    status['end_time'] = datetime.now().isoformat()
    update_status(status)

    if args.dry_run:
        logger.info(f"\nDRY RUN completado. {n_pending} casos simulados.")
        return

    # Resumen
    ok = status['completed']
    fail = status['failed']
    logger.info(f"\n{'#'*60}")
    logger.info(f"# PRODUCCION COMPLETADA")
    logger.info(f"# Exitosos: {ok}/{n_pending}")
    logger.info(f"# Fallidos: {fail}/{n_pending}")
    logger.info(f"{'#'*60}")

    # Re-entrenar surrogate si hay suficientes datos
    if ok >= 10:
        logger.info("\nRe-entrenando GP surrogate con datos frescos...")
        run_surrogate()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                PROJECT_ROOT / 'data' / f'production_{datetime.now():%Y%m%d_%H%M}.log',
                encoding='utf-8',
            ),
        ],
    )

    parser = argparse.ArgumentParser(description='Produccion masiva SPH-IncipientMotion')
    parser.add_argument('--prod', action='store_true',
                        help='Usar dp de produccion (convergido)')
    parser.add_argument('--generate', type=int, default=0,
                        help='Solo generar N muestras LHS y salir')
    parser.add_argument('--desde', type=int, default=0,
                        help='Recovery: continuar desde caso N (0-indexed)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simular campaña sin ejecutar GPU')
    args = parser.parse_args()

    run_production(args)
