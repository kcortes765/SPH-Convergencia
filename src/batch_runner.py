"""
batch_runner.py — Modulo 2 del Pipeline SPH-IncipientMotion

Ejecuta la cadena GenCase -> DualSPHysics GPU para casos generados por geometry_builder.
Implementa limpieza obligatoria de binarios (.bi4) en bloque try/finally.

Cadena de ejecucion (simplificada vs Diego):
  GenCase  CaseName_Def  outdir/CaseName  -save:all
  DualSPHysics5.4_win64.exe  -gpu  outdir/CaseName  outdir

Los CSVs de Chrono y Gauges se generan AUTOMATICAMENTE durante la simulacion,
no requieren post-proceso separado (FloatingInfo/ComputeForces NO se usan).

Archivos que se CONSERVAN despues de la simulacion:
  - *.csv (ChronoExchange, ChronoBody_forces, GaugesVel, GaugesMaxZ, Run, RunPARTs)
  - *.xml (caso de definicion y generado)
  - *.stl (geometrias)
  - boulder_properties.txt

Archivos que se ELIMINAN (try/finally blindado):
  - *.bi4 (binarios de particulas — 32 bytes/particula/snapshot)
  - *.vtk (visualizacion — no los generamos, pero por seguridad)
  - *.bt4 (boundaries)
  - *.out (logs de DualSPHysics temporales)

Autor: Kevin Cortes (UCN 2026)
"""

import json
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Extensiones de archivos pesados a limpiar
CLEANUP_EXTENSIONS = {'.bi4', '.vtk', '.bt4'}

# Extensiones de archivos a CONSERVAR siempre
KEEP_EXTENSIONS = {'.csv', '.xml', '.stl', '.txt', '.json'}

# Sin timeout — dejar que cada caso corra hasta que termine.
# La RTX 5090 no se comparte, no hay razon para abortar por tiempo.
# Si algo se cuelga de verdad, el usuario mata el proceso manualmente.
TIMEOUT_BY_DP = None  # No se usa


def get_timeout_for_dp(dp: float, config: dict) -> int:
    """Retorna timeout para subprocess. None = sin limite."""
    override = config.get('defaults', {}).get('timeout_seconds_override')
    if override:
        return int(override)
    # Sin timeout — dejar correr indefinidamente
    return None


def load_config(config_path: Path) -> dict:
    """Carga la configuracion de dsph_config.json."""
    with open(config_path) as f:
        return json.load(f)


def _get_exe(config: dict, name: str) -> Path:
    """Resuelve la ruta completa a un ejecutable de DualSPHysics."""
    dsph_bin = Path(config['dsph_bin'])
    exe_name = config['executables'][name]
    exe_path = dsph_bin / exe_name
    return exe_path


def _run_step(cmd: list, step_name: str, timeout_s: int,
              case_name: str, cwd: Path = None) -> subprocess.CompletedProcess:
    """
    Ejecuta un paso de la cadena con subprocess.run().
    Captura stdout/stderr y verifica returncode.

    Raises:
        RuntimeError: Si el comando falla (returncode != 0).
        subprocess.TimeoutExpired: Si se excede el timeout.
    """
    cmd_str = ' '.join(str(c) for c in cmd)
    logger.info(f"  [{case_name}] {step_name}: {cmd_str}")

    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        timeout=timeout_s,
        cwd=str(cwd) if cwd else None,
    )

    stdout = result.stdout.decode('utf-8', errors='replace').strip()
    stderr = result.stderr.decode('utf-8', errors='replace').strip()

    if stdout:
        for line in stdout.split('\n')[-5:]:  # Ultimas 5 lineas
            logger.debug(f"    stdout: {line}")

    if result.returncode != 0:
        logger.error(f"  [{case_name}] {step_name} FALLO (returncode={result.returncode})")
        if stderr:
            for line in stderr.split('\n')[-10:]:
                logger.error(f"    stderr: {line}")
        raise RuntimeError(
            f"{step_name} fallo para caso '{case_name}' "
            f"(returncode={result.returncode}): {stderr[:500]}"
        )

    return result


def cleanup_binaries(case_dir: Path, out_dir: Path = None):
    """
    Elimina archivos binarios pesados (.bi4, .vtk, .bt4) de las carpetas
    del caso y de output.

    Esta funcion se ejecuta en el bloque finally — DEBE ser robusta.
    No lanza excepciones; loggea errores y continua.
    """
    dirs_to_clean = [case_dir]
    if out_dir is not None and out_dir.exists():
        dirs_to_clean.append(out_dir)

    total_freed = 0
    total_files = 0

    for directory in dirs_to_clean:
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix.lower() in CLEANUP_EXTENSIONS:
                try:
                    size = path.stat().st_size
                    path.unlink()
                    total_freed += size
                    total_files += 1
                except Exception as e:
                    logger.warning(f"  No se pudo borrar {path}: {e}")

    if total_files > 0:
        mb = total_freed / (1024 * 1024)
        logger.info(f"  Limpieza: {total_files} archivos eliminados ({mb:.1f} MB liberados)")
    else:
        logger.info(f"  Limpieza: sin archivos binarios que borrar")


def collect_csvs(out_dir: Path, dest_dir: Path):
    """
    Copia los CSVs de Chrono y Gauges desde la carpeta de output a dest_dir.
    Los CSVs se generan automaticamente durante la simulacion en:
      - outdir/  (Run.csv, RunPARTs.csv)
      - outdir/data/  (ChronoExchange, ChronoBody_forces)
      - outdir/  (GaugesVel_*, GaugesMaxZ_*)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    collected = []

    for csv_file in out_dir.rglob('*.csv'):
        dest_path = dest_dir / csv_file.name
        shutil.copy2(str(csv_file), str(dest_path))
        collected.append(csv_file.name)

    if collected:
        logger.info(f"  CSVs recolectados ({len(collected)}): "
                     f"{', '.join(sorted(collected)[:5])}{'...' if len(collected) > 5 else ''}")
    else:
        logger.warning(f"  ATENCION: No se encontraron CSVs en {out_dir}")

    return collected


def verify_outputs(out_dir: Path, case_name: str) -> bool:
    """
    Verifica que la simulacion genero los CSVs esperados.
    Retorna True si al menos ChronoExchange existe y tiene datos.
    """
    # Buscar ChronoExchange en todo el arbol de output
    chrono_csvs = list(out_dir.rglob('ChronoExchange*.csv'))
    if not chrono_csvs:
        logger.error(f"  [{case_name}] No se encontro ChronoExchange CSV")
        return False

    # Verificar que tiene mas de solo el header
    for csv_path in chrono_csvs:
        size = csv_path.stat().st_size
        if size < 100:  # Archivo practicamente vacio
            logger.error(f"  [{case_name}] {csv_path.name} demasiado pequeno ({size} bytes)")
            return False

    return True


def run_case(case_dir: Path, config: dict,
             processed_dir: Path = None, dp: float = None) -> dict:
    """
    Ejecuta la cadena completa para UN caso de DualSPHysics.

    Pasos:
    1. GenCase: XML -> particulas (.bi4)
    2. DualSPHysics GPU: simulacion SPH + Chrono
    3. Verificar outputs (CSVs de Chrono/Gauges)
    4. Recolectar CSVs a processed_dir
    5. FINALLY: limpiar binarios (.bi4, .vtk)

    Args:
        case_dir: Directorio del caso (contiene *_Def.xml + STLs).
        config: Dict de configuracion (de dsph_config.json).
        processed_dir: Directorio donde copiar los CSVs resultantes.
                       Si es None, usa data/processed/{case_name}/.
        dp: Distancia entre particulas (m). Si se proporciona, usa timeout
            adaptativo basado en tabla de convergencia.

    Returns:
        dict con: success (bool), case_name, duration_s, csvs_collected,
                  error_message (str o None).
    """
    # Detectar el XML de definicion
    def_xmls = list(case_dir.glob('*_Def.xml'))
    if len(def_xmls) != 1:
        raise FileNotFoundError(
            f"Se esperaba exactamente 1 archivo *_Def.xml en {case_dir}, "
            f"encontrados: {len(def_xmls)}"
        )

    def_xml = def_xmls[0]
    case_name = def_xml.stem.replace('_Def', '')
    out_dir = case_dir / f"{case_name}_out"

    if processed_dir is None:
        processed_dir = case_dir.parent.parent / 'data' / 'processed' / case_name

    # Timeout adaptativo: prioridad dp explicito > regex en nombre > config default
    timeout_s = config.get('defaults', {}).get('timeout_seconds', 7200)
    if dp is not None:
        timeout_s = get_timeout_for_dp(dp, config)
        if timeout_s is not None:
            logger.info(f"  Timeout adaptativo para dp={dp}: {timeout_s}s ({timeout_s/3600:.1f}h)")
        else:
            logger.info(f"  Timeout: sin limite (dp={dp})")
    else:
        import re
        dp_match = re.search(r'dp(\d+)', case_name)
        if dp_match:
            dp_val = float(f"0.{dp_match.group(1)}")
            timeout_s = get_timeout_for_dp(dp_val, config)
            if timeout_s is not None:
                logger.info(f"  Timeout adaptativo para dp={dp_val}: {timeout_s}s ({timeout_s/3600:.1f}h)")
            else:
                logger.info(f"  Timeout: sin limite (dp={dp_val})")

    result = {
        'case_name': case_name,
        'case_dir': str(case_dir),
        'success': False,
        'duration_s': 0.0,
        'csvs_collected': [],
        'error_message': None,
        'timestamp': datetime.now().isoformat(),
    }

    logger.info(f"=== Ejecutando caso: {case_name} ===")
    start_time = datetime.now()

    try:
        # Limpiar output previo si existe (con retry para locks de Windows)
        if out_dir.exists():
            for attempt in range(3):
                try:
                    shutil.rmtree(str(out_dir))
                    break
                except PermissionError:
                    if attempt < 2:
                        import time as _time
                        logger.warning(f"  rmtree locked, retry {attempt+1}/3...")
                        _time.sleep(2 ** attempt)
                    else:
                        raise

        # --- PASO 1: GenCase ---
        # GenCase necesita cwd=case_dir para resolver paths relativos de STL
        gencase_exe = _get_exe(config, 'gencase')
        out_rel = out_dir.name  # e.g. "test_diego_reference_out"
        gencase_cmd = [
            gencase_exe,
            f"{case_name}_Def",
            f"{out_rel}/{case_name}",
            '-save:all',
        ]
        _run_step(gencase_cmd, 'GenCase', timeout_s=300,
                  case_name=case_name, cwd=case_dir)

        # --- PASO 2: DualSPHysics GPU ---
        # Paths relativos a case_dir (mismo cwd que GenCase)
        dsph_exe = _get_exe(config, 'dualsphysics_gpu')
        gpu_id = config.get('defaults', {}).get('gpu_id', 0)
        dsph_cmd = [
            dsph_exe,
            f'-gpu:{gpu_id}',
            f"{out_rel}/{case_name}",
            out_rel,
        ]
        _run_step(dsph_cmd, 'DualSPHysics', timeout_s=timeout_s,
                  case_name=case_name, cwd=case_dir)

        # --- PASO 3: Verificar outputs ---
        if not verify_outputs(out_dir, case_name):
            result['error_message'] = "CSVs de Chrono no encontrados o vacios"
            logger.error(f"  [{case_name}] Verificacion de outputs FALLIDA")
            return result

        # --- PASO 4: Recolectar CSVs ---
        result['csvs_collected'] = collect_csvs(out_dir, processed_dir)
        result['success'] = True
        logger.info(f"  [{case_name}] Simulacion completada exitosamente")

    except subprocess.TimeoutExpired as e:
        result['error_message'] = f"Timeout ({timeout_s}s) en {e.cmd[0] if e.cmd else '?'}"
        logger.error(f"  [{case_name}] TIMEOUT: {result['error_message']}")

    except RuntimeError as e:
        result['error_message'] = str(e)
        logger.error(f"  [{case_name}] ERROR: {e}")

    except Exception as e:
        result['error_message'] = f"Error inesperado: {type(e).__name__}: {e}"
        logger.error(f"  [{case_name}] ERROR INESPERADO: {e}", exc_info=True)

    finally:
        # === LIMPIEZA BLINDADA ===
        # Se ejecuta SIEMPRE, sin importar si hubo exito, timeout o crash.
        # Nunca debe lanzar una excepcion que enmascare el error original.
        try:
            cleanup_binaries(case_dir, out_dir)
        except Exception as e:
            logger.error(f"  [{case_name}] Error durante limpieza (no critico): {e}")

        duration = (datetime.now() - start_time).total_seconds()
        result['duration_s'] = duration
        logger.info(f"  [{case_name}] Duracion total: {duration:.1f}s")

    return result


def run_batch(cases_dir: Path, config: dict,
              processed_dir: Path = None) -> list:
    """
    Ejecuta todos los casos en un directorio, en serie.
    Cada subcarpeta que contenga un *_Def.xml se considera un caso.

    Args:
        cases_dir: Directorio con subcarpetas de casos.
        config: Dict de configuracion.
        processed_dir: Directorio base para CSVs procesados.

    Returns:
        Lista de dicts con resultados por caso.
    """
    case_dirs = sorted([
        d for d in cases_dir.iterdir()
        if d.is_dir() and list(d.glob('*_Def.xml'))
    ])

    if not case_dirs:
        logger.warning(f"No se encontraron casos en {cases_dir}")
        return []

    logger.info(f"Batch: {len(case_dirs)} casos encontrados en {cases_dir}")
    results = []

    for i, case_dir in enumerate(case_dirs, 1):
        logger.info(f"\n--- Caso {i}/{len(case_dirs)} ---")
        result = run_case(case_dir, config, processed_dir)
        results.append(result)

        status = "OK" if result['success'] else "FALLO"
        logger.info(f"  Resultado: {status}")

    # Resumen
    ok = sum(1 for r in results if r['success'])
    fail = len(results) - ok
    logger.info(f"\n=== BATCH COMPLETO: {ok} exitosos, {fail} fallidos de {len(results)} ===")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'dsph_config.json'
    config = load_config(config_path)

    cases_dir = project_root / 'cases'

    if len(sys.argv) > 1:
        # Ejecutar un caso especifico
        case_name = sys.argv[1]
        case_dir = cases_dir / case_name
        if not case_dir.exists():
            print(f"Error: caso '{case_name}' no encontrado en {cases_dir}")
            sys.exit(1)
        result = run_case(case_dir, config)
        sys.exit(0 if result['success'] else 1)
    else:
        # Ejecutar todos los casos
        results = run_batch(cases_dir, config)
        sys.exit(0 if all(r['success'] for r in results) else 1)
