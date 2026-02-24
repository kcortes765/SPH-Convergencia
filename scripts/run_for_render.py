"""
run_for_render.py — Re-ejecuta dp=0.004 para generar datos de visualizacion

A diferencia del pipeline normal que borra los .bi4, este script:
1. Genera el caso con TimeOut=0.5 (20 snapshots en 10s)
2. Corre GenCase + DualSPHysics GPU
3. Ejecuta PartVTK + IsoSurface + BoundaryVTK (POST-PROCESADO VISUAL)
4. Recolecta CSVs
5. FINALMENTE borra .bi4 (solo despues de tener los VTKs de render)

Los VTK generados se importan en Blender para el render fotorrealista.

Uso (en la workstation UCN):
    python run_for_render.py                # Ejecutar todo
    python run_for_render.py --skip-sim     # Solo post-procesado (si .bi4 ya existen)
    python run_for_render.py --keep-bi4     # NO borrar .bi4 al final

Output:
    data/render/dp004/
        surface/    WaterSurface_XXXX.vtk  (isosurface del agua)
        fluid/      PartFluid_XXXX.vtk     (particulas fluido)
        boulder/    PartBoulder_XXXX.vtk   (particulas boulder)
        channel/    Channel.vtk            (geometria canal)
        csv/        ChronoExchange_*.csv   (cinematica para animar en Blender)

Autor: Kevin Cortes (UCN 2026)
Fecha: 2026-02-21
"""

import sys
import json
import shutil
import logging
import subprocess
import time
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACION DEL RENDER
# ═══════════════════════════════════════════════════════════════════════

RENDER_DP = 0.004
RENDER_TIMEOUT_XML = 0.5    # 1 snapshot cada 0.5s = 20 frames en 10s (~17 GB de .bi4)
CASE_NAME = "render_dp0004"

# Misma fisica que el estudio de convergencia (NO modificar)
BASE_PARAMS = {
    'dam_height': 0.3,
    'boulder_mass': 1.06053,
    'boulder_scale': 0.04,
    'boulder_pos': (8.487, 0.523, 0.124),
    'boulder_rot': (0.0, 0.0, 0.0),
    'material': 'lime-stone',
    'time_max': 10.0,
}


def load_config():
    config_path = PROJECT_ROOT / 'config' / 'dsph_config.json'
    with open(config_path) as f:
        return json.load(f)


def get_exe(config, name):
    return Path(config['dsph_bin']) / config['executables'][name]


def run_cmd(cmd, step_name, timeout_s, cwd=None):
    """Ejecuta comando con logging detallado."""
    cmd_str = ' '.join(str(c) for c in cmd)
    logger.info(f"  [{step_name}] {cmd_str}")

    start = time.time()
    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        timeout=timeout_s,
        cwd=str(cwd) if cwd else None,
    )
    elapsed = time.time() - start

    stdout = result.stdout.decode('utf-8', errors='replace').strip()
    stderr = result.stderr.decode('utf-8', errors='replace').strip()

    if result.returncode != 0:
        logger.error(f"  [{step_name}] FALLO (rc={result.returncode}, {elapsed:.1f}s)")
        if stderr:
            for line in stderr.split('\n')[-15:]:
                logger.error(f"    {line}")
        raise RuntimeError(f"{step_name} fallo: {stderr[:500]}")

    logger.info(f"  [{step_name}] OK ({elapsed:.1f}s)")

    # Mostrar ultimas lineas de stdout para info
    if stdout:
        for line in stdout.split('\n')[-5:]:
            logger.debug(f"    {line}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# FASE 1: GENERAR CASO
# ═══════════════════════════════════════════════════════════════════════

def generate_case(config):
    """Genera el XML del caso con TimeOut para render."""
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from geometry_builder import CaseParams, build_case

    params = CaseParams(
        case_name=CASE_NAME,
        dp=RENDER_DP,
        dam_height=BASE_PARAMS['dam_height'],
        boulder_mass=BASE_PARAMS['boulder_mass'],
        boulder_scale=BASE_PARAMS['boulder_scale'],
        boulder_pos=BASE_PARAMS['boulder_pos'],
        boulder_rot=BASE_PARAMS['boulder_rot'],
        material=BASE_PARAMS['material'],
        time_max=BASE_PARAMS['time_max'],
        time_out=RENDER_TIMEOUT_XML,   # <-- CLAVE: 0.5s entre snapshots
        ft_pause=config['defaults']['ft_pause'],
    )

    template_xml = PROJECT_ROOT / config['paths']['template_xml']
    boulder_stl = PROJECT_ROOT / config['paths']['boulder_stl']
    beach_stl = PROJECT_ROOT / config['paths']['beach_stl']
    materials_xml = PROJECT_ROOT / config['paths']['materials_xml']
    cases_dir = PROJECT_ROOT / config['paths']['cases_dir']

    xml_path = build_case(
        template_xml=template_xml,
        boulder_stl=boulder_stl,
        beach_stl=beach_stl,
        materials_xml=materials_xml,
        output_dir=cases_dir,
        params=params,
    )

    logger.info(f"  Caso generado: {xml_path}")
    logger.info(f"  TimeOut = {RENDER_TIMEOUT_XML}s (20 snapshots en {BASE_PARAMS['time_max']}s)")
    return xml_path


# ═══════════════════════════════════════════════════════════════════════
# FASE 2: SIMULACION
# ═══════════════════════════════════════════════════════════════════════

def run_simulation(config, case_dir):
    """Ejecuta GenCase + DualSPHysics GPU."""
    out_dir = case_dir / f"{CASE_NAME}_out"

    # Limpiar output previo
    if out_dir.exists():
        logger.info(f"  Limpiando output previo: {out_dir}")
        shutil.rmtree(str(out_dir))

    # GenCase
    gencase_exe = get_exe(config, 'gencase')
    out_rel = out_dir.name
    run_cmd(
        [gencase_exe, f"{CASE_NAME}_Def", f"{out_rel}/{CASE_NAME}", '-save:all'],
        'GenCase', timeout_s=600, cwd=case_dir
    )

    # DualSPHysics GPU (timeout 24h para dp=0.004)
    dsph_exe = get_exe(config, 'dualsphysics_gpu')
    gpu_id = config.get('defaults', {}).get('gpu_id', 0)
    run_cmd(
        [dsph_exe, f'-gpu:{gpu_id}', f"{out_rel}/{CASE_NAME}", out_rel],
        'DualSPHysics', timeout_s=86400, cwd=case_dir
    )

    return out_dir


# ═══════════════════════════════════════════════════════════════════════
# FASE 3: POST-PROCESADO VISUAL (ANTES de borrar .bi4)
# ═══════════════════════════════════════════════════════════════════════

def run_visual_postprocess(config, case_dir, out_dir):
    """
    Ejecuta PartVTK, IsoSurface, BoundaryVTK para generar los datos
    que Blender necesita para el render.

    CRITICO: Esto debe ejecutarse ANTES de cleanup_binaries().
    """
    render_dir = PROJECT_ROOT / 'data' / 'render' / 'dp004'

    # Crear subdirectorios
    dirs = {
        'surface': render_dir / 'surface',
        'fluid': render_dir / 'fluid',
        'boulder': render_dir / 'boulder',
        'channel': render_dir / 'channel',
        'csv': render_dir / 'csv',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Paths
    partvtk = get_exe(config, 'partvtk')
    isosurface = get_exe(config, 'isosurface')
    boundaryvtk = get_exe(config, 'boundaryvtk')

    # El XML generado por GenCase esta en out_dir
    case_xml = out_dir / f"{CASE_NAME}.xml"
    if not case_xml.exists():
        # Buscar alternativas
        xmls = list(out_dir.glob("*.xml"))
        if xmls:
            case_xml = xmls[0]
            logger.info(f"  XML encontrado: {case_xml.name}")
        else:
            logger.error(f"  No se encontro XML en {out_dir}")
            return False

    logger.info(f"\n{'='*60}")
    logger.info(f"POST-PROCESADO VISUAL")
    logger.info(f"{'='*60}")

    # --- 3.1: Particulas de fluido (PartVTK) ---
    logger.info("\n  [1/4] Extrayendo particulas de fluido...")
    try:
        run_cmd([
            partvtk,
            f'-dirin', str(out_dir),
            f'-filexml', str(case_xml),
            f'-savevtk', str(dirs['fluid'] / 'PartFluid'),
            '-onlytype:-all,+fluid',
            '-vars:+idp,+vel,+rhop,+press',
        ], 'PartVTK-Fluid', timeout_s=3600)
    except Exception as e:
        logger.error(f"  PartVTK fluido fallo: {e}")

    # --- 3.2: Particulas del boulder (PartVTK) ---
    logger.info("\n  [2/4] Extrayendo particulas del boulder...")
    try:
        run_cmd([
            partvtk,
            f'-dirin', str(out_dir),
            f'-filexml', str(case_xml),
            f'-savevtk', str(dirs['boulder'] / 'PartBoulder'),
            '-onlymk:51',
            '-vars:+idp,+vel',
        ], 'PartVTK-Boulder', timeout_s=1800)
    except Exception as e:
        logger.error(f"  PartVTK boulder fallo: {e}")

    # --- 3.3: Isosurface del agua (IsoSurface) ---
    # ESTE ES EL PASO MAS IMPORTANTE PARA CALIDAD VISUAL
    logger.info("\n  [3/4] Generando isosurface del agua (esto toma tiempo)...")
    try:
        run_cmd([
            isosurface,
            f'-dirin', str(out_dir),
            f'-filexml', str(case_xml),
            f'-saveiso', str(dirs['surface'] / 'WaterSurface'),
            '-onlytype:-all,+fluid',
            '-distnode_dp:1.5',
            '-distinter_2h:1.0',
            '-vars:+vel,+press',
        ], 'IsoSurface', timeout_s=7200)
    except Exception as e:
        logger.error(f"  IsoSurface fallo: {e}")

    # --- 3.4: Geometria del canal (BoundaryVTK) ---
    logger.info("\n  [4/4] Extrayendo geometria del canal...")
    actual_vtk = out_dir / f"{CASE_NAME}__Actual.vtk"
    if not actual_vtk.exists():
        # Buscar en el case_dir
        actual_vtks = list(case_dir.rglob("*__Actual.vtk"))
        if actual_vtks:
            actual_vtk = actual_vtks[0]

    if actual_vtk.exists():
        try:
            run_cmd([
                boundaryvtk,
                f'-loadvtk', str(actual_vtk),
                f'-filexml', str(case_xml),
                f'-savevtk', str(dirs['channel'] / 'Channel'),
                '-onlymk:0',
            ], 'BoundaryVTK', timeout_s=1800)
        except Exception as e:
            logger.error(f"  BoundaryVTK fallo: {e}")
    else:
        logger.warning(f"  No se encontro *__Actual.vtk, saltando BoundaryVTK")
        logger.info(f"  Alternativa: usar el STL directamente en Blender")

    # --- 3.5: Copiar CSVs de Chrono para animacion en Blender ---
    logger.info("\n  Copiando CSVs de Chrono...")
    csv_count = 0
    for csv_file in out_dir.rglob('*.csv'):
        shutil.copy2(str(csv_file), str(dirs['csv'] / csv_file.name))
        csv_count += 1
    logger.info(f"  {csv_count} CSVs copiados a {dirs['csv']}")

    # --- Resumen ---
    logger.info(f"\n{'='*60}")
    logger.info(f"POST-PROCESADO VISUAL COMPLETADO")
    logger.info(f"{'='*60}")

    for name, d in dirs.items():
        files = list(d.glob('*'))
        total_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024*1024)
        logger.info(f"  {name:12s}: {len(files):3d} archivos ({total_mb:.1f} MB)")

    logger.info(f"\n  Directorio de render: {render_dir}")
    logger.info(f"  Proximo paso: importar en ParaView para preview, luego Blender")

    return True


# ═══════════════════════════════════════════════════════════════════════
# FASE 4: LIMPIEZA
# ═══════════════════════════════════════════════════════════════════════

def cleanup_bi4(case_dir, out_dir):
    """Borra .bi4 y .bt4 (NO los .vtk de render, esos estan en data/render/)."""
    total_freed = 0
    total_files = 0
    extensions = {'.bi4', '.bt4'}

    for directory in [case_dir, out_dir]:
        if not directory.exists():
            continue
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix.lower() in extensions:
                try:
                    size = path.stat().st_size
                    path.unlink()
                    total_freed += size
                    total_files += 1
                except Exception as e:
                    logger.warning(f"  No se pudo borrar {path}: {e}")

    gb = total_freed / (1024**3)
    logger.info(f"  Limpieza: {total_files} archivos .bi4/.bt4 eliminados ({gb:.1f} GB liberados)")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]
    skip_sim = '--skip-sim' in args
    keep_bi4 = '--keep-bi4' in args

    config = load_config()
    case_dir = PROJECT_ROOT / config['paths']['cases_dir'] / CASE_NAME
    out_dir = case_dir / f"{CASE_NAME}_out"

    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"  RUN FOR RENDER — dp={RENDER_DP}, TimeOut={RENDER_TIMEOUT_XML}s")
    print(f"  Caso: {CASE_NAME}")
    print(f"  Hardware: GPU:{config['defaults'].get('gpu_id', 0)}")
    print(f"  Opciones: skip_sim={skip_sim}, keep_bi4={keep_bi4}")
    print(f"{'='*60}\n")

    try:
        # FASE 1: Generar caso (XML + STL)
        if not skip_sim:
            logger.info("FASE 1: Generando caso...")
            generate_case(config)
        else:
            logger.info("FASE 1: SALTADA (--skip-sim)")
            if not case_dir.exists():
                logger.error(f"  case_dir no existe: {case_dir}")
                logger.error(f"  No puedes usar --skip-sim sin haber corrido la simulacion antes")
                sys.exit(1)

        # FASE 2: Simulacion GPU
        if not skip_sim:
            logger.info("\nFASE 2: Simulacion GPU...")
            logger.info(f"  Estimado: ~260 min (4.3h) para dp={RENDER_DP}")
            logger.info(f"  Generara ~20 snapshots de ~26.1M particulas (~17 GB de .bi4)")
            out_dir = run_simulation(config, case_dir)
        else:
            logger.info("\nFASE 2: SALTADA (--skip-sim)")
            if not out_dir.exists():
                logger.error(f"  out_dir no existe: {out_dir}")
                sys.exit(1)

        # FASE 3: Post-procesado visual
        logger.info("\nFASE 3: Post-procesado visual...")
        success = run_visual_postprocess(config, case_dir, out_dir)

        if not success:
            logger.warning("Post-procesado visual tuvo errores, revisa los logs")

    finally:
        # FASE 4: Limpieza (solo si no se pidio --keep-bi4)
        if not keep_bi4:
            logger.info("\nFASE 4: Limpieza de binarios...")
            cleanup_bi4(case_dir, out_dir)
        else:
            logger.info("\nFASE 4: SALTADA (--keep-bi4)")

    total_min = (time.time() - total_start) / 60
    total_h = total_min / 60

    print(f"\n{'='*60}")
    print(f"  COMPLETADO en {total_min:.1f} min ({total_h:.1f}h)")
    print(f"  Datos de render en: data/render/dp004/")
    print(f"{'='*60}")
    print(f"\n  Proximo paso:")
    print(f"    1. Abrir ParaView -> verificar data/render/dp004/surface/WaterSurface_*.vtk")
    print(f"    2. Abrir Blender -> importar STLs + isosurface + animar con CSV")
    print(f"    3. Ver PLAN_RENDER.md para el workflow completo")


if __name__ == '__main__':
    log_file = PROJECT_ROOT / 'data' / 'render_dp004.log'
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

    main()
