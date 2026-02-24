"""
analisis_completo.py — Script maestro post-simulacion

Ejecuta TODO el pipeline de analisis sobre los resultados de la WS:
  1. Entrenar GP surrogate (3 figuras)
  2. UQ: Monte Carlo + Sobol (6 figuras)
  3. Figuras piloto completas (22 figuras)
  4. Validacion contra caso lhs_001_old (si existe)
  5. Resumen CSV exportable

Total: 31 figuras thesis-quality (PNG 300dpi + PDF vector)

Prerequisito: data/results.sqlite con >= 10 casos reales

Ejecutar:
    python scripts/analisis_completo.py                  # Todo
    python scripts/analisis_completo.py --solo-figuras   # Solo figuras (GP ya entrenado)
    python scripts/analisis_completo.py --validar        # Incluir validacion lhs_001_old
    python scripts/analisis_completo.py --synthetic      # Con datos sinteticos (testing)

Autor: Kevin Cortes (UCN 2026)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / 'data' / 'results.sqlite'
MODEL_PATH = PROJECT_ROOT / 'data' / 'gp_surrogate.pkl'
VALIDATION_CSV = PROJECT_ROOT / 'data' / 'validacion_lhs001_old.csv'


def check_data():
    """Verifica que hay datos suficientes en SQLite."""
    if not DB_PATH.exists():
        print(f"  ERROR: No existe {DB_PATH}")
        print("  Las simulaciones aun no han producido resultados.")
        return 0

    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    try:
        n = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        print(f"  SQLite: {n} casos en results.sqlite")

        if n > 0:
            row = conn.execute(
                "SELECT MIN(dam_height), MAX(dam_height), "
                "MIN(boulder_mass), MAX(boulder_mass), "
                "MIN(max_displacement), MAX(max_displacement), "
                "SUM(failed), COUNT(*) FROM results"
            ).fetchone()
            print(f"  h: [{row[0]:.3f}, {row[1]:.3f}] m")
            print(f"  M: [{row[2]:.3f}, {row[3]:.3f}] kg")
            print(f"  disp: [{row[4]:.4f}, {row[5]:.4f}] m")
            print(f"  Movimiento: {row[6]}/{row[7]} ({row[6]/row[7]*100:.0f}%)")

        return n
    finally:
        conn.close()


def step1_gp_surrogate(force_synthetic=False):
    """Paso 1: Entrenar GP surrogate."""
    print(f"\n{'='*60}")
    print("  PASO 1/4: Entrenamiento GP Surrogate")
    print(f"{'='*60}")

    from ml_surrogate import run_surrogate
    t0 = time.time()
    run_surrogate(force_synthetic=force_synthetic)
    dt = time.time() - t0
    print(f"  Completado en {dt:.1f}s")
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  Figuras: data/figuras_ml/")


def step2_uq_analysis(mc_samples=10000, sobol_n=4096, threshold=0.005,
                       force_synthetic=False):
    """Paso 2: UQ — Monte Carlo + Sobol."""
    print(f"\n{'='*60}")
    print("  PASO 2/4: Analisis de Incertidumbre (UQ)")
    print(f"{'='*60}")

    # Importar desde scripts (no src)
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from run_uq_analysis import run_uq

    t0 = time.time()
    run_uq(mc_samples=mc_samples, sobol_n=sobol_n, threshold=threshold,
            force_synthetic=force_synthetic)
    dt = time.time() - t0
    print(f"  Completado en {dt:.1f}s")
    print(f"  Figuras: data/figuras_uq/")


def step3_figuras_piloto(synthetic=False):
    """Paso 3: 22 figuras thesis-quality."""
    print(f"\n{'='*60}")
    print("  PASO 3/4: Figuras del Estudio Piloto (22)")
    print(f"{'='*60}")

    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from figuras_piloto import main as figuras_main

    t0 = time.time()
    figuras_main(synthetic=synthetic)
    dt = time.time() - t0
    print(f"  Completado en {dt:.1f}s")
    print(f"  Figuras: data/figuras_piloto/")


def step4_validacion():
    """Paso 4: Validar GP contra caso lhs_001_old."""
    print(f"\n{'='*60}")
    print("  PASO 4/4: Validacion contra caso independiente")
    print(f"{'='*60}")

    if not VALIDATION_CSV.exists():
        print("  [SKIP] No existe validacion_lhs001_old.csv")
        return

    if not MODEL_PATH.exists():
        print("  [SKIP] No existe gp_surrogate.pkl")
        return

    import pickle

    # Cargar GP
    with open(MODEL_PATH, 'rb') as f:
        pkg = pickle.load(f)

    # Cargar datos de validacion
    val = pd.read_csv(VALIDATION_CSV)
    print(f"  Caso: {val['case_id'].iloc[0]}")
    print(f"  Params: h={val['dam_height'].iloc[0]:.3f}, "
          f"M={val['boulder_mass'].iloc[0]:.3f}, "
          f"rot={val['boulder_rot_z'].iloc[0]:.1f}")

    # Predecir
    X_val = val[['dam_height', 'boulder_mass', 'boulder_rot_z']].values
    X_s = pkg['scaler_X'].transform(X_val)
    y_s, std_s = pkg['gp'].predict(X_s, return_std=True)
    y_pred = pkg['scaler_y'].inverse_transform(y_s.reshape(-1, 1)).ravel()
    y_std = std_s * pkg['scaler_y'].scale_[0]

    y_real = val['max_displacement'].iloc[0]
    y_hat = y_pred[0]
    y_sig = y_std[0]
    error = abs(y_real - y_hat)
    error_pct = error / y_real * 100 if y_real > 0 else 0

    print(f"\n  Real:      {y_real:.4f} m")
    print(f"  Predicho:  {y_hat:.4f} m (+/- {y_sig:.4f})")
    print(f"  Error:     {error:.4f} m ({error_pct:.1f}%)")
    print(f"  Dentro de 2-sigma: {'SI' if abs(y_real - y_hat) < 2 * y_sig else 'NO'}")

    # Guardar resultado
    result = {
        'case_id': val['case_id'].iloc[0],
        'real_displacement_m': float(y_real),
        'predicted_displacement_m': float(y_hat),
        'prediction_std_m': float(y_sig),
        'error_m': float(error),
        'error_pct': float(error_pct),
        'within_2sigma': bool(abs(y_real - y_hat) < 2 * y_sig),
        'gp_r2_loo': float(pkg['loo_r2']),
        'n_training': int(pkg['n_real']),
    }
    out_path = PROJECT_ROOT / 'data' / 'validacion_resultado.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Guardado: {out_path}")


def export_summary():
    """Exporta resumen CSV con todos los resultados."""
    if not DB_PATH.exists():
        return

    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM results", conn)
    conn.close()

    out = PROJECT_ROOT / 'data' / 'resumen_screening.csv'
    df.to_csv(out, index=False)
    print(f"\n  CSV exportado: {out} ({len(df)} filas)")


def main():
    parser = argparse.ArgumentParser(description='Analisis completo post-simulacion')
    parser.add_argument('--solo-figuras', action='store_true',
                        help='Solo generar figuras (GP ya entrenado)')
    parser.add_argument('--validar', action='store_true',
                        help='Incluir validacion contra lhs_001_old')
    parser.add_argument('--synthetic', action='store_true',
                        help='Usar datos sinteticos (testing sin WS)')
    parser.add_argument('--mc-samples', type=int, default=10000)
    parser.add_argument('--sobol-n', type=int, default=4096)
    parser.add_argument('--threshold', type=float, default=0.005)
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  ANALISIS COMPLETO — SPH-IncipientMotion")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*60}")

    # Verificar datos
    print("\nVerificando datos...")
    n_cases = check_data()

    if n_cases == 0 and not args.synthetic:
        print("\nSin datos reales. Usa --synthetic para testing.")
        sys.exit(1)

    t_total = time.time()

    if not args.solo_figuras:
        # Paso 1: GP
        step1_gp_surrogate(force_synthetic=args.synthetic)

        # Paso 2: UQ
        step2_uq_analysis(
            mc_samples=args.mc_samples,
            sobol_n=args.sobol_n,
            threshold=args.threshold,
            force_synthetic=args.synthetic,
        )

    # Paso 3: Figuras
    step3_figuras_piloto(synthetic=args.synthetic)

    # Paso 4: Validacion
    if args.validar or VALIDATION_CSV.exists():
        step4_validacion()

    # Export
    export_summary()

    dt_total = time.time() - t_total
    print(f"\n{'#'*60}")
    print(f"  ANALISIS COMPLETADO en {dt_total:.0f}s ({dt_total/60:.1f}min)")
    print(f"  Figuras generadas:")
    print(f"    data/figuras_ml/     — 3 figuras GP")
    print(f"    data/figuras_uq/     — 6 figuras UQ")
    print(f"    data/figuras_piloto/ — 22 figuras piloto")
    print(f"  Total: 31 figuras thesis-quality")
    print(f"{'#'*60}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
