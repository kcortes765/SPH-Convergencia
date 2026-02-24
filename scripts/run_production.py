"""
run_production.py — Script "Boton Rojo" para campana de produccion masiva

Lee config/param_ranges.json, genera matriz LHS, ejecuta el pipeline
completo en la GPU, y sincroniza status via archivo JSON.

Notificaciones push via ntfy.sh (instalar app ntfy en celular,
suscribirse al topic configurado).

Ejecutar:
    python run_production.py --pilot --dry-run     # Verificar sin GPU
    python run_production.py --pilot --prod         # Correr 50 casos
    python run_production.py --pilot --prod --desde 15  # Recovery

Autor: Kevin Cortes (UCN 2026)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Agregar src/ al path (scripts/ esta un nivel abajo de la raiz)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from main_orchestrator import load_param_ranges, generate_experiment_matrix, run_pipeline_case
from batch_runner import load_config
from data_cleaner import save_to_sqlite
from ml_surrogate import run_surrogate

logger = logging.getLogger(__name__)
STATUS_FILE = PROJECT_ROOT / "data" / "production_status.json"


# ---------------------------------------------------------------------------
# Notificaciones push via ntfy.sh
# ---------------------------------------------------------------------------

NTFY_TOPIC = "sph-kevin-tesis-2026"
NTFY_ENABLED = True


def notify(title: str, message: str, priority: str = "default", tags: str = ""):
    """
    Envia notificacion push via ntfy.sh.

    Prioridades: min, low, default, high, urgent
    Tags: ver https://docs.ntfy.sh/emojis/

    No lanza excepciones — la produccion no debe detenerse por una notificacion fallida.
    """
    if not NTFY_ENABLED:
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode('utf-8'),
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": tags,
            },
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info(f"  [ntfy] {title}")
    except Exception as e:
        logger.warning(f"  [ntfy] Fallo notificacion: {e}")


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Formateo de progreso
# ---------------------------------------------------------------------------

def _fmt_time(seconds):
    """Formatea segundos a string legible."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}min {sec}s"
    elif s < 86400:
        h, rem = divmod(s, 3600)
        m = rem // 60
        return f"{h}h {m}min"
    else:
        d, rem = divmod(s, 86400)
        h = rem // 3600
        return f"{d}d {h}h"


def _progress_banner(i, n_pending, n_total, desde, status, case_durations,
                      campaign_start, row, dp):
    """Genera banner informativo antes de cada caso."""
    case_num = desde + i
    elapsed = (datetime.now() - campaign_start).total_seconds()
    ok = status['completed']
    fail = status['failed']
    done = ok + fail
    remaining = n_pending - i
    pct = done / n_total * 100 if n_total > 0 else 0

    bar_len = 30
    filled = int(bar_len * done / n_total) if n_total > 0 else 0
    bar = '#' * filled + '-' * (bar_len - filled)

    rot_z = row.get('boulder_rot_z', 0)
    lines = [
        f"\n{'='*65}",
        f"  CASO {case_num}/{n_total} [{row['case_id']}]  [{bar}] {pct:.0f}%",
        f"  h={row['dam_height']:.3f}m | M={row['boulder_mass']:.3f}kg | "
        f"theta={rot_z:.1f}deg | dp={dp}",
        f"{'~'*65}",
        f"  Completados: {ok}   Fallidos: {fail}   Quedan: {remaining}",
        f"  Tiempo transcurrido: {_fmt_time(elapsed)}",
    ]

    if case_durations:
        recent = case_durations[-10:]
        avg = sum(recent) / len(recent)
        eta_s = remaining * avg
        fin = (datetime.now() + timedelta(seconds=eta_s)).strftime('%d/%m %H:%M')
        lines.append(f"  Promedio: {avg/60:.1f} min/caso (ultimos {len(recent)})")
        lines.append(f"  ETA: {_fmt_time(eta_s)} restantes  -->  fin ~{fin}")
    elif done == 0:
        lines.append(f"  ETA: calculando tras primer caso...")

    if done > 0:
        tasa = ok / done * 100
        lines.append(f"  Tasa exito: {tasa:.0f}% ({ok}/{done})")

    lines.append(f"{'='*65}")
    return "\n".join(lines)


MAX_FAIL_RATE = 0.30  # Abort si >30% de los casos fallan


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

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

    # 4. Espacio en disco
    import shutil as _shutil
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    usage = _shutil.disk_usage(str(data_dir))
    free_gb = usage.free / (1024**3)
    logger.info(f"  Disco libre: {free_gb:.1f} GB")
    if free_gb < 50:
        logger.warning(f"  ADVERTENCIA: Poco espacio libre ({free_gb:.1f} GB < 50 GB)")

    return checks_passed


# ---------------------------------------------------------------------------
# Produccion
# ---------------------------------------------------------------------------

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
    if args.pilot:
        n_samples = ranges_cfg['sampling'].get('n_samples_pilot', 100)
        logger.info(f"MODO PILOTO: {n_samples} muestras")
    elif args.prod:
        n_samples = ranges_cfg['sampling']['n_samples_prod']
    else:
        n_samples = ranges_cfg['sampling']['n_samples_dev']

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
    campaign_start = datetime.now()
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
        'start_time': campaign_start.isoformat(),
        'dry_run': args.dry_run,
    }
    update_status(status)

    # Pre-flight check
    logger.info("Pre-flight check...")
    if not preflight_check(config):
        logger.error("PRE-FLIGHT FALLIDO. Corregir errores antes de continuar.")
        notify("PRE-FLIGHT FALLIDO", "Ejecutables o archivos faltantes. Revisar log.",
               priority="urgent", tags="x")
        sys.exit(1)
    logger.info("Pre-flight OK.\n")

    logger.info(f"\n{'#'*60}")
    logger.info(f"# PRODUCCION: {n_pending} casos pendientes (de {n_total} total)")
    logger.info(f"# dp={dp}, GPU={config['defaults']['gpu_id']}")
    logger.info(f"# {'DRY RUN' if args.dry_run else 'EJECUCION REAL'}")
    logger.info(f"{'#'*60}\n")

    # Notificacion de inicio
    notify(
        f"PRODUCCION INICIADA: {n_pending} casos",
        f"dp={dp}, modo={'prod' if args.prod else 'dev'}\n"
        f"Estimado: ~{n_pending * 4:.0f}h ({n_pending * 4 / 24:.1f} dias)\n"
        f"Recovery desde: {desde}",
        priority="high",
        tags="rocket",
    )

    all_results = []
    successful_results = []
    case_durations = []  # Para calcular ETA

    for i, (_, row) in enumerate(matrix.iterrows(), 1):
        case_id = row['case_id']
        status['current_case'] = case_id
        status['progress'] = f"{i}/{n_pending}"

        # Calcular ETA con promedio movil
        if case_durations:
            recent = case_durations[-10:]
            avg_duration = sum(recent) / len(recent)
            remaining_s = (n_pending - i) * avg_duration
            eta = datetime.now() + timedelta(seconds=remaining_s)
            status['eta'] = eta.isoformat()
            status['avg_case_seconds'] = round(avg_duration, 1)
            status['eta_human'] = f"{remaining_s/3600:.1f}h ({remaining_s/86400:.1f}d)"
        update_status(status)

        # Banner informativo
        banner = _progress_banner(i, n_pending, n_total, desde, status,
                                   case_durations, campaign_start, row, dp)
        logger.info(banner)

        if args.dry_run:
            logger.info(f"  [DRY RUN] Saltando simulacion GPU")
            continue

        try:
            result = run_pipeline_case(row, PROJECT_ROOT, config, dp)
            all_results.append(result)
            case_durations.append(result['duration_s'])

            if result['success'] and result['result'] is not None:
                successful_results.append(result['result'])
                status['completed'] += 1
                dur = result['duration_s']
                cr = result['result']
                estado = "MOVIMIENTO" if cr.failed else "ESTABLE"
                logger.info(f"\n  >>> {case_id}: {estado} en {_fmt_time(dur)}")
                logger.info(f"      disp={cr.max_displacement:.4f}m  rot={cr.max_rotation:.1f}deg  "
                            f"vel={cr.max_velocity:.3f}m/s  F_sph={cr.max_sph_force:.1f}N")
                if hasattr(cr, 'max_flow_velocity') and cr.max_flow_velocity:
                    logger.info(f"      V_flujo={cr.max_flow_velocity:.3f}m/s  "
                                f"H_agua={cr.max_water_height:.4f}m")

                # Notificar al celular
                elapsed = (datetime.now() - campaign_start).total_seconds()
                eta_str = status.get('eta_human', '?')
                notify(
                    f"{status['completed']}/{n_pending} [{case_id}] {estado}",
                    f"Disp={cr.max_displacement:.3f}m | {_fmt_time(dur)}\n"
                    f"Quedan: {n_pending - i} | ETA: {eta_str}\n"
                    f"Fallidos: {status['failed']} | "
                    f"Total: {_fmt_time(elapsed)}",
                    tags="chart_with_upwards_trend",
                )
            else:
                status['failed'] += 1
                logger.error(f"  FALLO: {result['error']}")
                notify(
                    f"FALLO: {case_id}",
                    f"Error: {result['error']}\n"
                    f"Completados: {status['completed']}, Fallidos: {status['failed']}",
                    priority="high",
                    tags="warning",
                )

        except Exception as e:
            status['failed'] += 1
            logger.error(f"  EXCEPCION: {e}", exc_info=True)
            notify(
                f"EXCEPCION: {case_id}",
                f"{type(e).__name__}: {e}",
                priority="high",
                tags="rotating_light",
            )

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
            notify(
                "ABORT: Tasa de fallos excesiva",
                f"{status['failed']}/{total_run} fallidos ({fail_pct:.0f}%)\n"
                f"Ultimo caso: {case_id}\n"
                f"Usar --desde {desde + i} para recovery",
                priority="urgent",
                tags="octagonal_sign",
            )
            break

        # Guardar a SQLite despues de cada caso exitoso (crash safety)
        if successful_results:
            db_path = PROJECT_ROOT / 'data' / 'results.sqlite'
            save_to_sqlite(successful_results, db_path)
            successful_results = []  # Reset para no duplicar

    # Status final
    elapsed = (datetime.now() - campaign_start).total_seconds()
    status['phase'] = 'completed'
    status['end_time'] = datetime.now().isoformat()
    status['total_elapsed_hours'] = round(elapsed / 3600, 2)
    update_status(status)

    if args.dry_run:
        logger.info(f"\nDRY RUN completado. {n_pending} casos simulados.")
        notify("DRY RUN completado", f"{n_pending} casos verificados OK", tags="white_check_mark")
        return

    # Resumen
    ok = status['completed']
    fail = status['failed']
    logger.info(f"\n{'#'*60}")
    logger.info(f"# PRODUCCION COMPLETADA")
    logger.info(f"# Exitosos: {ok}/{n_pending}")
    logger.info(f"# Fallidos: {fail}/{n_pending}")
    logger.info(f"# Tiempo total: {elapsed/3600:.1f}h ({elapsed/86400:.1f}d)")
    logger.info(f"{'#'*60}")

    # Notificacion final
    notify(
        f"COMPLETADO: {ok}/{n_pending} exitosos",
        f"Fallidos: {fail}\n"
        f"Tiempo total: {elapsed/3600:.1f}h ({elapsed/86400:.1f} dias)\n"
        f"Promedio: {sum(case_durations)/len(case_durations)/60:.1f}min/caso" if case_durations else "",
        priority="high",
        tags="trophy",
    )

    # Re-entrenar surrogate si hay suficientes datos
    if ok >= 10:
        logger.info("\nRe-entrenando GP surrogate con datos frescos...")
        try:
            run_surrogate()
            notify("GP re-entrenado", f"Surrogate actualizado con {ok} datos reales",
                   tags="brain")
        except Exception as e:
            logger.error(f"Error re-entrenando surrogate: {e}")
            notify("Error GP surrogate", str(e), priority="high", tags="warning")


if __name__ == '__main__':
    # Asegurar que data/ existe para el log file
    (PROJECT_ROOT / 'data').mkdir(parents=True, exist_ok=True)

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
    parser.add_argument('--pilot', action='store_true',
                        help='Estudio piloto (n_samples_pilot del JSON)')
    parser.add_argument('--generate', type=int, default=0,
                        help='Solo generar N muestras LHS y salir')
    parser.add_argument('--desde', type=int, default=0,
                        help='Recovery: continuar desde caso N (0-indexed)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simular campana sin ejecutar GPU')
    parser.add_argument('--no-notify', action='store_true',
                        help='Desactivar notificaciones push')
    args = parser.parse_args()

    if args.no_notify:
        NTFY_ENABLED = False

    run_production(args)
