"""
data_cleaner.py — Modulo 3 del Pipeline SPH-IncipientMotion

ETL (Extract, Transform, Load) para CSVs generados por DualSPHysics + Chrono.

Fuentes de datos:
  - ChronoExchange_mkbound_51.csv — cinematica del boulder (pos, vel, omega, accel)
  - ChronoBody_forces.csv — fuerzas SPH + contacto por cuerpo
  - GaugesVel_V**.csv — velocidad del flujo en puntos fijos
  - GaugesMaxZ_hmax**.csv — altura maxima del agua en puntos fijos
  - Run.csv / RunPARTs.csv — metadata de la simulacion

Formato CSV (DualSPHysics):
  - Separador: punto y coma (;)
  - Decimal: punto (.)
  - Sentinel: -3.40282e+38 (float min) en Gauges = sin dato -> NaN

Criterio de falla (incipient motion):
  - Desplazamiento del centro de masa > umbral (% de d_eq)
  - Rotacion neta > umbral (grados)
  - Umbrales pendientes de validacion academica

Autor: Kevin Cortes (UCN 2026)
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sentinel value de DualSPHysics (float min) — indica "sin dato"
SENTINEL = -3.40282e+38
SENTINEL_THRESHOLD = -3.0e+38  # Cualquier valor menor a esto es sentinel


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Resultado procesado de un caso de simulacion."""
    case_name: str

    # Cinematica del boulder
    max_displacement: float      # Desplazamiento maximo del CM (m)
    max_displacement_rel: float  # Desplazamiento relativo (% de d_eq)
    max_rotation: float          # Rotacion maxima neta (grados)
    max_velocity: float          # Velocidad maxima del boulder (m/s)

    # Fuerzas sobre el boulder
    max_sph_force: float         # Fuerza SPH maxima (N)
    max_contact_force: float     # Fuerza de contacto maxima (N)

    # Flujo (en el gauge mas cercano al boulder)
    max_flow_velocity: float     # Velocidad maxima del flujo (m/s)
    max_water_height: float      # Altura maxima del agua (m)

    # Criterio de falla
    moved: bool                  # True si supero umbral de desplazamiento
    rotated: bool                # True si supero umbral de rotacion
    failed: bool                 # True si moved OR rotated

    # Metadata
    sim_time_reached: float      # Tiempo simulado alcanzado (s)
    n_timesteps: int             # Numero de filas en ChronoExchange

    # Parametros de entrada (para ML surrogate)
    dam_height: float = 0.0      # Altura columna de agua (m)
    boulder_mass: float = 0.0    # Masa del boulder (kg)
    dp: float = 0.0              # Distancia entre particulas (m)
    stl_file: str = ""           # Nombre del STL usado


# ---------------------------------------------------------------------------
# Parseo de CSVs
# ---------------------------------------------------------------------------

def parse_chrono_exchange(csv_path: Path) -> pd.DataFrame:
    """
    Parsea ChronoExchange_mkbound_*.csv (cinematica del boulder).

    Columnas: nstep; time [s]; dt [s]; predictor; face.x/y/z [m/s^2];
              fomegaace.x/y/z [rad/s^2]; fvel.x/y/z [m/s];
              fcenter.x/y/z [m]; fomega.x/y/z [rad/s]
    """
    df = pd.read_csv(csv_path, sep=';')

    # Renombrar columnas para facilitar acceso
    rename = {
        'time [s]': 'time',
        'dt [s]': 'dt',
        'face.x [m/s^2]': 'acc_x', 'face.y [m/s^2]': 'acc_y', 'face.z [m/s^2]': 'acc_z',
        'fomegaace.x [rad/s^2]': 'alpha_x', 'fomegaace.y [rad/s^2]': 'alpha_y', 'fomegaace.z [rad/s^2]': 'alpha_z',
        'fvel.x [m/s]': 'vel_x', 'fvel.y [m/s]': 'vel_y', 'fvel.z [m/s]': 'vel_z',
        'fcenter.x [m]': 'cx', 'fcenter.y [m]': 'cy', 'fcenter.z [m]': 'cz',
        'fomega.x [rad/s]': 'omega_x', 'fomega.y [rad/s]': 'omega_y', 'fomega.z [rad/s]': 'omega_z',
    }
    df = df.rename(columns=rename)

    # Convertir 'predictor' de string a bool
    if 'predictor' in df.columns:
        df['predictor'] = df['predictor'].map({'True': True, 'False': False})

    logger.info(f"  ChronoExchange: {len(df)} filas, t=[{df['time'].iloc[0]:.4f}, {df['time'].iloc[-1]:.4f}]s")
    return df


def parse_chrono_forces(csv_path: Path) -> pd.DataFrame:
    """
    Parsea ChronoBody_forces.csv (fuerzas SPH + contacto por cuerpo).

    Header problematico: columnas repetidas (fy, fz, mx...) para cada body.
    Resolvemos renombrando con prefijo del body.

    Columnas originales:
      Time; Body_BLIR_fx;fy;fz;mx;my;mz;cfx;cfy;cfz;cmx;cmy;cmz;
            Body_beach_fx;fy;fz;mx;my;mz;cfx;cfy;cfz;cmx;cmy;cmz;
    """
    # Leer header manualmente para resolver duplicados
    with open(csv_path) as f:
        header_line = f.readline().strip().rstrip(';')

    raw_cols = header_line.split(';')

    # Construir nombres unicos: detectar Body_* como inicio de bloque
    clean_cols = []
    current_body = ''
    body_suffixes = ['fx', 'fy', 'fz', 'mx', 'my', 'mz',
                     'cfx', 'cfy', 'cfz', 'cmx', 'cmy', 'cmz']
    suffix_idx = 0

    for col in raw_cols:
        col = col.strip()
        if col == 'Time':
            clean_cols.append('time')
        elif col.startswith('Body_'):
            # New body block: extract body name
            # "Body_BLIR_fx" -> body="blir", suffix="fx"
            parts = col.split('_', 2)  # ['Body', 'BLIR', 'fx']
            current_body = parts[1].lower()
            suffix = parts[2] if len(parts) > 2 else body_suffixes[0]
            clean_cols.append(f'{current_body}_{suffix}')
            suffix_idx = 1  # Next suffix
        else:
            # Continuation of current body block
            clean_cols.append(f'{current_body}_{col.strip()}')
            suffix_idx += 1

    df = pd.read_csv(csv_path, sep=';', header=0, usecols=range(len(clean_cols)))
    df.columns = clean_cols

    logger.info(f"  ChronoBody_forces: {len(df)} filas, bodies: "
                f"{set(c.split('_')[0] for c in clean_cols if '_' in c and c != 'time')}")
    return df


def parse_gauge_velocity(csv_path: Path) -> pd.DataFrame:
    """
    Parsea GaugesVel_V**.csv (velocidad del flujo en un punto).

    Columnas: time [s]; velx [m/s]; vely [m/s]; velz [m/s]; posx [m]; posy [m]; posz [m]
    """
    df = pd.read_csv(csv_path, sep=';')
    rename = {
        'time [s]': 'time',
        'velx [m/s]': 'velx', 'vely [m/s]': 'vely', 'velz [m/s]': 'velz',
        'posx [m]': 'posx', 'posy [m]': 'posy', 'posz [m]': 'posz',
    }
    df = df.rename(columns=rename)

    # Reemplazar sentinels con NaN
    for col in ['velx', 'vely', 'velz']:
        df.loc[df[col] < SENTINEL_THRESHOLD, col] = np.nan

    return df


def parse_gauge_maxz(csv_path: Path) -> pd.DataFrame:
    """
    Parsea GaugesMaxZ_hmax**.csv (altura maxima del agua).

    Columnas: time [s]; zmax [m]; posx [m]; posy [m]; posz [m]
    """
    df = pd.read_csv(csv_path, sep=';')
    rename = {
        'time [s]': 'time',
        'zmax [m]': 'zmax',
        'posx [m]': 'posx', 'posy [m]': 'posy', 'posz [m]': 'posz',
    }
    df = df.rename(columns=rename)

    # Reemplazar sentinels con NaN
    df.loc[df['zmax'] < SENTINEL_THRESHOLD, 'zmax'] = np.nan

    return df


def parse_all_gauges(case_dir: Path) -> dict:
    """
    Parsea todos los Gauges de un caso.

    Returns:
        dict con claves 'velocity' y 'maxz', cada una es una lista de
        (gauge_id, DataFrame, position).
    """
    result = {'velocity': [], 'maxz': []}

    # Velocity gauges
    for csv_path in sorted(case_dir.glob('GaugesVel_V*.csv')):
        gauge_id = csv_path.stem.replace('GaugesVel_V', '')
        df = parse_gauge_velocity(csv_path)
        pos = (df['posx'].iloc[0], df['posy'].iloc[0], df['posz'].iloc[0])
        result['velocity'].append((gauge_id, df, pos))

    # MaxZ gauges
    for csv_path in sorted(case_dir.glob('GaugesMaxZ_hmax*.csv')):
        gauge_id = csv_path.stem.replace('GaugesMaxZ_hmax', '')
        df = parse_gauge_maxz(csv_path)
        pos = (df['posx'].iloc[0], df['posy'].iloc[0], df['posz'].iloc[0])
        result['maxz'].append((gauge_id, df, pos))

    logger.info(f"  Gauges: {len(result['velocity'])} velocity, {len(result['maxz'])} maxZ")
    return result


# ---------------------------------------------------------------------------
# Analisis de cinematica del boulder
# ---------------------------------------------------------------------------

def compute_displacement(chrono_df: pd.DataFrame) -> pd.Series:
    """Calcula desplazamiento del CM respecto a posicion inicial (magnitud 3D)."""
    cx0 = chrono_df['cx'].iloc[0]
    cy0 = chrono_df['cy'].iloc[0]
    cz0 = chrono_df['cz'].iloc[0]

    return np.sqrt(
        (chrono_df['cx'] - cx0)**2 +
        (chrono_df['cy'] - cy0)**2 +
        (chrono_df['cz'] - cz0)**2
    )


def compute_rotation(chrono_df: pd.DataFrame) -> pd.Series:
    """
    Calcula rotacion acumulada integrando velocidad angular.

    Aproximacion: integral trapezoidal de |omega| sobre el tiempo.
    Resultado en grados.
    """
    omega_mag = np.sqrt(
        chrono_df['omega_x']**2 +
        chrono_df['omega_y']**2 +
        chrono_df['omega_z']**2
    )
    dt = chrono_df['time'].diff().fillna(0)
    # Integral acumulada (trapezoidal)
    rotation_rad = np.cumsum(omega_mag * dt)
    return np.degrees(rotation_rad)


def compute_boulder_velocity(chrono_df: pd.DataFrame) -> pd.Series:
    """Calcula magnitud de velocidad del boulder."""
    return np.sqrt(
        chrono_df['vel_x']**2 +
        chrono_df['vel_y']**2 +
        chrono_df['vel_z']**2
    )


# ---------------------------------------------------------------------------
# Analisis de fuerzas
# ---------------------------------------------------------------------------

def compute_forces(forces_df: pd.DataFrame, body: str = 'blir') -> dict:
    """
    Extrae fuerzas maximas sobre un cuerpo.

    En ChronoBody_forces:
    - fx/fy/fz = fuerzas SPH (presion + viscosidad)
    - cfx/cfy/cfz = fuerzas de contacto (Chrono)
    - mx/my/mz = momentos SPH
    - cmx/cmy/cmz = momentos de contacto
    """
    # Fuerza SPH
    sph_force = np.sqrt(
        forces_df[f'{body}_fx']**2 +
        forces_df[f'{body}_fy']**2 +
        forces_df[f'{body}_fz']**2
    )

    # Fuerza de contacto
    contact_force = np.sqrt(
        forces_df[f'{body}_cfx']**2 +
        forces_df[f'{body}_cfy']**2 +
        forces_df[f'{body}_cfz']**2
    )

    return {
        'max_sph_force': float(sph_force.max()),
        'max_contact_force': float(contact_force.max()),
        'sph_force_series': sph_force,
        'contact_force_series': contact_force,
    }


# ---------------------------------------------------------------------------
# Analisis de flujo
# ---------------------------------------------------------------------------

def find_nearest_gauge(gauges: list, boulder_pos: tuple) -> tuple:
    """
    Encuentra el gauge mas cercano a la posicion del boulder.

    Args:
        gauges: Lista de (gauge_id, df, position).
        boulder_pos: (x, y, z) del boulder.

    Returns:
        (gauge_id, df, distance)
    """
    if not gauges:
        return None, None, float('inf')

    bx, by, bz = boulder_pos
    best = None
    best_dist = float('inf')

    for gauge_id, df, (gx, gy, gz) in gauges:
        dist = np.sqrt((gx - bx)**2 + (gy - by)**2 + (gz - bz)**2)
        if dist < best_dist:
            best_dist = dist
            best = (gauge_id, df)

    return best[0], best[1], best_dist


# ---------------------------------------------------------------------------
# Funcion principal: procesar un caso completo
# ---------------------------------------------------------------------------

def process_case(case_dir: Path, d_eq: float,
                 disp_threshold_pct: float = 5.0,
                 rot_threshold_deg: float = 5.0) -> CaseResult:
    """
    Procesa todos los CSVs de un caso y determina si hubo incipient motion.

    Args:
        case_dir: Directorio con los CSVs del caso (data/processed/case_name/).
        d_eq: Diametro equivalente esferico del boulder (m).
        disp_threshold_pct: Umbral de desplazamiento (% de d_eq).
        rot_threshold_deg: Umbral de rotacion (grados).

    Returns:
        CaseResult con metricas y criterio de falla.
    """
    case_name = case_dir.name
    logger.info(f"Procesando caso: {case_name}")

    # --- 1. ChronoExchange (cinematica) ---
    chrono_csvs = sorted(case_dir.glob('ChronoExchange*.csv'))
    if not chrono_csvs:
        raise FileNotFoundError(f"No se encontro ChronoExchange CSV en {case_dir}")
    chrono_df = parse_chrono_exchange(chrono_csvs[0])

    displacement = compute_displacement(chrono_df)
    rotation = compute_rotation(chrono_df)
    velocity = compute_boulder_velocity(chrono_df)

    max_disp = float(displacement.max())
    max_disp_rel = (max_disp / d_eq) * 100  # porcentaje de d_eq
    max_rot = float(rotation.max())
    max_vel = float(velocity.max())

    logger.info(f"  Desplazamiento max: {max_disp:.6f}m ({max_disp_rel:.2f}% de d_eq={d_eq:.4f}m)")
    logger.info(f"  Rotacion max: {max_rot:.2f} grados")
    logger.info(f"  Velocidad max boulder: {max_vel:.4f} m/s")

    # --- 2. ChronoBody_forces ---
    forces_csv = case_dir / 'ChronoBody_forces.csv'
    max_sph = 0.0
    max_contact = 0.0
    if forces_csv.exists():
        forces_df = parse_chrono_forces(forces_csv)
        forces = compute_forces(forces_df, body='blir')
        max_sph = forces['max_sph_force']
        max_contact = forces['max_contact_force']
        logger.info(f"  Fuerza SPH max: {max_sph:.4f}N, Contacto max: {max_contact:.4f}N")
    else:
        logger.warning(f"  ChronoBody_forces.csv no encontrado")

    # --- 3. Gauges ---
    gauges = parse_all_gauges(case_dir)

    # Posicion inicial del boulder
    boulder_pos = (chrono_df['cx'].iloc[0], chrono_df['cy'].iloc[0], chrono_df['cz'].iloc[0])

    # Velocidad del flujo (gauge mas cercano)
    max_flow_vel = 0.0
    if gauges['velocity']:
        gid, gdf, gdist = find_nearest_gauge(gauges['velocity'], boulder_pos)
        vel_mag = np.sqrt(gdf['velx']**2 + gdf['vely']**2 + gdf['velz']**2)
        max_flow_vel = float(vel_mag.max())
        logger.info(f"  Flujo max (gauge V{gid}, dist={gdist:.2f}m): {max_flow_vel:.4f} m/s")

    # Altura del agua (gauge mas cercano)
    max_water_h = 0.0
    if gauges['maxz']:
        gid, gdf, gdist = find_nearest_gauge(gauges['maxz'], boulder_pos)
        valid_zmax = gdf['zmax'].dropna()
        if len(valid_zmax) > 0:
            max_water_h = float(valid_zmax.max())
        logger.info(f"  Agua max (gauge hmax{gid}, dist={gdist:.2f}m): {max_water_h:.4f}m")

    # --- 4. Criterio de falla ---
    disp_threshold_m = d_eq * (disp_threshold_pct / 100.0)
    moved = max_disp > disp_threshold_m
    rotated = max_rot > rot_threshold_deg
    failed = moved or rotated

    status = "FALLO (movimiento)" if failed else "ESTABLE"
    logger.info(f"  Criterio: {status} (umbral disp={disp_threshold_pct}% d_eq={disp_threshold_m:.4f}m, "
                f"rot={rot_threshold_deg} deg)")

    return CaseResult(
        case_name=case_name,
        max_displacement=max_disp,
        max_displacement_rel=max_disp_rel,
        max_rotation=max_rot,
        max_velocity=max_vel,
        max_sph_force=max_sph,
        max_contact_force=max_contact,
        max_flow_velocity=max_flow_vel,
        max_water_height=max_water_h,
        moved=moved,
        rotated=rotated,
        failed=failed,
        sim_time_reached=float(chrono_df['time'].iloc[-1]),
        n_timesteps=len(chrono_df),
    )


# ---------------------------------------------------------------------------
# Persistencia en SQLite
# ---------------------------------------------------------------------------

def save_to_sqlite(results: list, db_path: Path, table: str = 'results'):
    """
    Guarda lista de CaseResult en SQLite.

    Crea la tabla si no existe. Inserta filas nuevas (no duplica case_name).
    """
    from dataclasses import asdict
    import sqlite3

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    df = pd.DataFrame([asdict(r) for r in results])

    # Eliminar resultados previos de los mismos casos (upsert)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            case_name TEXT PRIMARY KEY,
            max_displacement REAL,
            max_displacement_rel REAL,
            max_rotation REAL,
            max_velocity REAL,
            max_sph_force REAL,
            max_contact_force REAL,
            max_flow_velocity REAL,
            max_water_height REAL,
            moved INTEGER,
            rotated INTEGER,
            failed INTEGER,
            sim_time_reached REAL,
            n_timesteps INTEGER,
            dam_height REAL,
            boulder_mass REAL,
            dp REAL,
            stl_file TEXT
        )
    """)

    # Agregar columnas nuevas si la tabla ya existia sin ellas
    existing_cols = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()}
    for col, coltype in [('dam_height', 'REAL'), ('boulder_mass', 'REAL'),
                         ('dp', 'REAL'), ('stl_file', 'TEXT')]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
            logger.info(f"  SQLite: columna '{col}' agregada a tabla '{table}'")

    for _, row in df.iterrows():
        cursor.execute(f"""
            INSERT OR REPLACE INTO {table} VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            row['case_name'],
            row['max_displacement'], row['max_displacement_rel'],
            row['max_rotation'], row['max_velocity'],
            row['max_sph_force'], row['max_contact_force'],
            row['max_flow_velocity'], row['max_water_height'],
            int(row['moved']), int(row['rotated']), int(row['failed']),
            row['sim_time_reached'], row['n_timesteps'],
            row.get('dam_height', 0.0), row.get('boulder_mass', 0.0),
            row.get('dp', 0.0), row.get('stl_file', ''),
        ))

    conn.commit()
    conn.close()
    logger.info(f"SQLite: {len(results)} resultados guardados en {db_path}:{table}")


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

    # Determinar directorio de CSVs
    if len(sys.argv) > 1:
        case_dir = Path(sys.argv[1])
    else:
        # Default: usar datos de Diego como ejemplo
        case_dir = project_root / 'ENTREGA_KEVIN'

    # d_eq del boulder (de boulder_properties.txt o hardcoded)
    d_eq = 0.1004  # metros, calculado por geometry_builder

    result = process_case(case_dir, d_eq=d_eq)

    print(f"\n{'='*60}")
    print(f"Caso: {result.case_name}")
    print(f"{'='*60}")
    print(f"Desplazamiento max:  {result.max_displacement:.6f}m ({result.max_displacement_rel:.2f}% d_eq)")
    print(f"Rotacion max:        {result.max_rotation:.2f} grados")
    print(f"Velocidad max:       {result.max_velocity:.4f} m/s")
    print(f"Fuerza SPH max:      {result.max_sph_force:.4f} N")
    print(f"Fuerza contacto max: {result.max_contact_force:.4f} N")
    print(f"Flujo max:           {result.max_flow_velocity:.4f} m/s")
    print(f"Agua max:            {result.max_water_height:.4f} m")
    print(f"Tiempo simulado:     {result.sim_time_reached:.4f}s ({result.n_timesteps} steps)")
    print(f"Resultado:           {'FALLO' if result.failed else 'ESTABLE'}")

    # Guardar en SQLite
    db_path = project_root / 'data' / 'results.sqlite'
    save_to_sqlite([result], db_path)
    print(f"\nGuardado en: {db_path}")
