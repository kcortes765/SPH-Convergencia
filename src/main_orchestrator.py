"""
main_orchestrator.py — Cerebro del Pipeline SPH-IncipientMotion

Genera una matriz de experimentos via Latin Hypercube Sampling (LHS),
luego ejecuta el pipeline completo para cada caso:

  LHS → geometry_builder → batch_runner → data_cleaner → SQLite

Si un caso falla (GPU timeout, GenCase error, CSV corrupto), el orquestador
registra el error y continua con el siguiente. La fabrica no se detiene.

Autor: Kevin Cortes (UCN 2026)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from geometry_builder import CaseParams, build_case, compute_boulder_properties
from batch_runner import load_config, run_case
from data_cleaner import process_case, save_to_sqlite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diseno de Experimentos (LHS)
# ---------------------------------------------------------------------------

# Rangos por defecto (se sobreescriben con config/param_ranges.json)
DEFAULT_PARAM_RANGES = {
    'dam_height': (0.2, 0.5),
    'boulder_mass': (1.0, 3.0),
}


def load_param_ranges(json_path: Path = None) -> dict:
    """
    Carga rangos parametricos desde config/param_ranges.json.

    Returns:
        dict: {param_name: (min, max)} para cada parametro activo.
    """
    if json_path is None:
        json_path = Path(__file__).resolve().parent.parent / 'config' / 'param_ranges.json'

    if not json_path.exists():
        logger.warning(f"param_ranges.json no encontrado en {json_path}, usando defaults")
        return DEFAULT_PARAM_RANGES

    with open(json_path) as f:
        cfg = json.load(f)

    ranges = {}
    for name, spec in cfg.get('parameters', {}).items():
        ranges[name] = (spec['min'], spec['max'])

    logger.info(f"Rangos cargados de {json_path.name}: {list(ranges.keys())}")
    if cfg.get('_status', '').startswith('PLACEHOLDER'):
        logger.warning("  ATENCION: rangos son PLACEHOLDER, esperando validacion Dr. Moris")

    return ranges


def generate_experiment_matrix(n_samples: int, seed: int = 42,
                               output_csv: Path = None,
                               param_ranges: dict = None) -> pd.DataFrame:
    """
    Genera una matriz de experimentos usando Latin Hypercube Sampling.

    LHS garantiza cobertura uniforme del espacio parametrico con menos
    muestras que un grid completo. Ideal para simulaciones costosas.

    Args:
        n_samples: Numero de combinaciones a generar.
        seed: Semilla para reproducibilidad.
        output_csv: Si se proporciona, guarda la matriz en CSV.
        param_ranges: dict {name: (min, max)}. Si None, carga de JSON.

    Returns:
        DataFrame con columnas: case_id + una columna por parametro.
    """
    if param_ranges is None:
        param_ranges = load_param_ranges()

    n_params = len(param_ranges)
    sampler = LatinHypercube(d=n_params, seed=seed)
    samples = sampler.random(n=n_samples)  # Valores en [0, 1]

    # Escalar a los rangos reales
    param_names = list(param_ranges.keys())
    for i, name in enumerate(param_names):
        low, high = param_ranges[name]
        samples[:, i] = low + samples[:, i] * (high - low)

    # Construir DataFrame
    df = pd.DataFrame(samples, columns=param_names)
    df.insert(0, 'case_id', [f"lhs_{i+1:03d}" for i in range(n_samples)])

    # Redondear para legibilidad
    for col in param_names:
        df[col] = df[col].round(4)

    logger.info(f"LHS: {n_samples} muestras generadas ({n_params} parametros)")
    for _, row in df.iterrows():
        parts = [f"{col}={row[col]:.3f}" for col in param_names]
        logger.info(f"  {row['case_id']}: {', '.join(parts)}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"  Guardado en: {output_csv}")

    return df


# ---------------------------------------------------------------------------
# Pipeline completo para un caso
# ---------------------------------------------------------------------------

def run_pipeline_case(row: pd.Series, project_root: Path,
                      config: dict, dp: float) -> dict:
    """
    Ejecuta el pipeline completo para UN caso de la matriz.

    Pasos:
    1. geometry_builder.build_case() → carpeta con XML + STLs
    2. batch_runner.run_case() → simulacion GPU + limpieza .bi4
    3. data_cleaner.process_case() → analisis → CaseResult

    Args:
        row: Fila del DataFrame de la matriz (case_id, dam_height, boulder_mass).
        project_root: Raiz del proyecto.
        config: Dict de configuracion (dsph_config.json).
        dp: Distancia entre particulas (m).

    Returns:
        dict con: case_id, success, result (CaseResult o None), error, duration_s
    """
    case_id = row['case_id']
    start = datetime.now()

    rot_z = float(row.get('boulder_rot_z', 0.0))

    pipeline_result = {
        'case_id': case_id,
        'dam_height': row['dam_height'],
        'boulder_mass': row['boulder_mass'],
        'boulder_rot_z': rot_z,
        'dp': dp,
        'success': False,
        'result': None,
        'error': None,
        'duration_s': 0.0,
    }

    # Rutas
    template_xml = project_root / config['paths']['template_xml']
    boulder_stl = project_root / config['paths']['boulder_stl']
    beach_stl = project_root / config['paths']['beach_stl']
    materials_xml = project_root / config['paths']['materials_xml']
    cases_dir = project_root / config['paths']['cases_dir']
    processed_dir = project_root / config['paths']['processed_dir'] / case_id

    try:
        # --- PASO 1: Geometry Builder ---
        logger.info(f"\n{'='*60}")
        logger.info(f"CASO {case_id}: dam_h={row['dam_height']:.3f}m, "
                     f"mass={row['boulder_mass']:.3f}kg, rot_z={rot_z:.1f}deg, dp={dp}")
        logger.info(f"{'='*60}")

        params = CaseParams(
            case_name=case_id,
            dp=dp,
            dam_height=row['dam_height'],
            boulder_mass=row['boulder_mass'],
            # Defaults del template para el resto
            boulder_scale=0.04,
            boulder_pos=(8.5, 0.5, 0.1),
            boulder_rot=(0.0, 0.0, rot_z),
            material="pvc",
            time_max=config['defaults']['time_max'],
            time_out=config['defaults']['time_out'],
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
        logger.info(f"  [1/3] Geometry: OK ({xml_path.name})")

        # --- PASO 2: Batch Runner ---
        run_result = run_case(case_dir, config, processed_dir, dp=dp)

        if not run_result['success']:
            pipeline_result['error'] = f"Simulacion fallida: {run_result['error_message']}"
            logger.error(f"  [2/3] Simulacion: FALLO — {run_result['error_message']}")
            return pipeline_result

        logger.info(f"  [2/3] Simulacion: OK ({run_result['duration_s']:.1f}s, "
                     f"{len(run_result['csvs_collected'])} CSVs)")

        # --- PASO 3: Data Cleaner ---
        # Calcular d_eq para este caso (necesita las propiedades del boulder)
        boulder_props = compute_boulder_properties(
            stl_path=boulder_stl,
            scale=params.boulder_scale,
            rotation_deg=params.boulder_rot,
            mass_kg=params.boulder_mass,
        )
        d_eq = boulder_props['d_eq']

        case_result = process_case(processed_dir, d_eq=d_eq)
        # Inyectar parametros de entrada para ML surrogate
        case_result.dam_height = row['dam_height']
        case_result.boulder_mass = row['boulder_mass']
        case_result.boulder_rot_z = rot_z
        case_result.dp = dp
        case_result.stl_file = config['paths'].get('boulder_stl', '').split('/')[-1]
        pipeline_result['result'] = case_result
        pipeline_result['success'] = True

        status = "FALLO" if case_result.failed else "ESTABLE"
        logger.info(f"  [3/3] Analisis: OK → {status} "
                     f"(disp={case_result.max_displacement:.4f}m, "
                     f"rot={case_result.max_rotation:.1f}deg)")

    except Exception as e:
        pipeline_result['error'] = f"{type(e).__name__}: {e}"
        logger.error(f"  [{case_id}] ERROR: {e}", exc_info=True)

    finally:
        pipeline_result['duration_s'] = (datetime.now() - start).total_seconds()

    return pipeline_result


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def run_campaign(matrix_csv: Path, project_root: Path,
                 config: dict, dp: float) -> list:
    """
    Ejecuta una campana completa de simulaciones.

    Lee la matriz de experimentos y ejecuta el pipeline para cada caso.
    Si un caso falla, lo registra y continua con el siguiente.

    Args:
        matrix_csv: Ruta al CSV con la matriz de experimentos.
        project_root: Raiz del proyecto.
        config: Dict de configuracion.
        dp: Distancia entre particulas (m).

    Returns:
        Lista de dicts con resultados por caso.
    """
    matrix = pd.read_csv(matrix_csv)
    n_cases = len(matrix)

    logger.info(f"\n{'#'*60}")
    logger.info(f"# CAMPANA DE SIMULACION: {n_cases} casos")
    logger.info(f"# Matriz: {matrix_csv}")
    logger.info(f"# dp={dp}m, GPU={config['defaults']['gpu_id']}")
    logger.info(f"{'#'*60}\n")

    all_results = []
    successful_results = []

    for i, (_, row) in enumerate(matrix.iterrows(), 1):
        logger.info(f"\n--- Caso {i}/{n_cases} ---")

        pipeline_result = run_pipeline_case(row, project_root, config, dp)
        all_results.append(pipeline_result)

        if pipeline_result['success'] and pipeline_result['result'] is not None:
            successful_results.append(pipeline_result['result'])

        status = "OK" if pipeline_result['success'] else "FALLO"
        logger.info(f"  [{pipeline_result['case_id']}] {status} "
                     f"({pipeline_result['duration_s']:.1f}s)")

    # Guardar resultados exitosos en SQLite
    if successful_results:
        db_path = project_root / 'data' / 'results.sqlite'
        save_to_sqlite(successful_results, db_path)
        logger.info(f"\nSQLite: {len(successful_results)} resultados guardados en {db_path}")

    # Resumen final
    ok = sum(1 for r in all_results if r['success'])
    fail = n_cases - ok
    total_time = sum(r['duration_s'] for r in all_results)

    logger.info(f"\n{'#'*60}")
    logger.info(f"# CAMPANA COMPLETADA")
    logger.info(f"# Exitosos: {ok}/{n_cases}")
    logger.info(f"# Fallidos: {fail}/{n_cases}")
    logger.info(f"# Tiempo total: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"{'#'*60}")

    if fail > 0:
        logger.warning(f"\nCasos fallidos:")
        for r in all_results:
            if not r['success']:
                logger.warning(f"  {r['case_id']}: {r['error']}")

    # Tabla resumen
    logger.info(f"\nResumen:")
    logger.info(f"{'Case':<12} {'dam_h':>6} {'mass':>6} {'Status':>8} "
                f"{'Disp(m)':>9} {'Rot(deg)':>9} {'Time(s)':>8}")
    logger.info("-" * 70)
    for r in all_results:
        if r['success'] and r['result']:
            cr = r['result']
            status = "FALLO" if cr.failed else "ESTABLE"
            logger.info(f"{r['case_id']:<12} {r['dam_height']:>6.3f} {r['boulder_mass']:>6.3f} "
                         f"{status:>8} {cr.max_displacement:>9.4f} "
                         f"{cr.max_rotation:>9.1f} {r['duration_s']:>8.1f}")
        else:
            logger.info(f"{r['case_id']:<12} {r['dam_height']:>6.3f} {r['boulder_mass']:>6.3f} "
                         f"{'ERROR':>8} {'---':>9} {'---':>9} {r['duration_s']:>8.1f}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / 'config' / 'dsph_config.json'
    config = load_config(config_path)

    matrix_csv = project_root / 'config' / 'experiment_matrix.csv'

    # Argumentos CLI
    n_samples = 5
    dp = config['defaults']['dp_dev']  # 0.02 para desarrollo

    if len(sys.argv) > 1:
        if sys.argv[1] == '--generate':
            # Solo generar la matriz, no simular
            n = int(sys.argv[2]) if len(sys.argv) > 2 else n_samples
            generate_experiment_matrix(n, output_csv=matrix_csv)
            print(f"Matriz generada: {matrix_csv}")
            sys.exit(0)
        elif sys.argv[1] == '--prod':
            dp = config['defaults']['dp_prod']  # 0.004 para produccion
            logger.info(f"MODO PRODUCCION: dp={dp}")

    # Generar matriz si no existe
    if not matrix_csv.exists():
        logger.info("Matriz no encontrada, generando con LHS...")
        generate_experiment_matrix(n_samples, output_csv=matrix_csv)

    # Ejecutar campana
    results = run_campaign(matrix_csv, project_root, config, dp)

    # Exit code: 0 si todos exitosos, 1 si algun fallo
    sys.exit(0 if all(r['success'] for r in results) else 1)
