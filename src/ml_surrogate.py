"""
ml_surrogate.py — Modulo 4 del Pipeline SPH-IncipientMotion

Gaussian Process Surrogate conectado a datos reales de SQLite.
Entrena un GP Matern nu=2.5 para predecir desplazamiento del boulder
a partir de parametros de entrada (dam_height, boulder_mass, boulder_rot_z).

Si hay menos de MIN_REAL_POINTS datos reales, complementa con datos
sinteticos basados en fisica simplificada (flaggeados como sinteticos).

Genera figuras de tesis y exporta modelo .pkl para UQ downstream.

Ejecutar:
    python src/ml_surrogate.py
    python src/ml_surrogate.py --synthetic  # forzar datos sinteticos

Autor: Kevin Cortes (UCN 2026)
"""

import argparse
import json
import logging
import pickle
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "results.sqlite"
OUT_DIR = PROJECT_ROOT / "data" / "figuras_ml"
MODEL_PATH = PROJECT_ROOT / "data" / "gp_surrogate.pkl"

MIN_REAL_POINTS = 10   # Minimo de datos reales para entrenar sin sinteticos
N_SYNTHETIC = 25       # Puntos sinteticos si no hay suficientes reales
SEED = 42

# Estilo de figuras para tesis
PLT_STYLE = {
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 200, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.2,
    'axes.spines.top': False, 'axes.spines.right': False,
}

FEATURES = ['dam_height', 'boulder_mass', 'boulder_rot_z']
TARGET = 'max_displacement'

FEATURE_LABELS = {
    'dam_height': 'Altura columna $h$ [m]',
    'boulder_mass': 'Masa boulder [kg]',
    'boulder_rot_z': r'$\theta_z$ [grados]',
}

FEATURE_COLORS = {
    'dam_height': '#2166AC',
    'boulder_mass': '#B2182B',
    'boulder_rot_z': '#6A3D9A',
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_real_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Lee resultados reales de SQLite con parametros de entrada."""
    if not db_path.exists():
        logger.warning(f"SQLite no encontrada: {db_path}")
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql("""
        SELECT case_name, dam_height, boulder_mass, boulder_rot_z, dp,
               max_displacement, max_rotation, max_velocity,
               max_sph_force, max_contact_force,
               max_flow_velocity, max_water_height,
               failed
        FROM results
        WHERE dam_height > 0 AND boulder_mass > 0
    """, conn)
    conn.close()

    # Backward compat: si boulder_rot_z es NULL, rellenar con 0
    if 'boulder_rot_z' in df.columns:
        df['boulder_rot_z'] = df['boulder_rot_z'].fillna(0.0)
    else:
        df['boulder_rot_z'] = 0.0

    logger.info(f"Datos reales: {len(df)} casos de {db_path.name}")
    return df


def generate_synthetic(n: int = N_SYNTHETIC, seed: int = SEED) -> pd.DataFrame:
    """Genera datos sinteticos basados en fisica simplificada."""
    rng = np.random.default_rng(seed)

    dam_height = rng.uniform(0.15, 0.50, n)
    boulder_mass = rng.uniform(0.5, 3.0, n)
    boulder_rot_z = rng.uniform(0, 90, n)

    # Modelo fisico simplificado: E ~ rho*g*h^2/2, R ~ m*g*mu
    energy = 1000 * 9.81 * dam_height**2 * 0.5
    resistance = boulder_mass * 9.81 * 0.6
    displacement = 2.0 * (energy / resistance) ** 0.7
    # Rotacion tiene efecto leve (area frontal varia ~10%)
    rot_effect = 1.0 - 0.1 * np.sin(np.radians(boulder_rot_z))
    displacement *= rot_effect
    displacement += rng.normal(0, 0.15, n)
    displacement = np.maximum(displacement, 0)

    df = pd.DataFrame({
        'case_name': [f"synth_{i+1:03d}" for i in range(n)],
        'dam_height': dam_height,
        'boulder_mass': boulder_mass,
        'boulder_rot_z': boulder_rot_z,
        'dp': 0.0,
        'max_displacement': displacement,
        'max_rotation': rng.uniform(0, 90, n),
        'max_velocity': rng.uniform(0, 3, n),
        'max_sph_force': rng.uniform(10, 200, n),
        'max_contact_force': rng.uniform(50, 5000, n),
        'max_flow_velocity': rng.uniform(0.5, 4, n),
        'max_water_height': rng.uniform(0.05, 0.5, n),
        'failed': (displacement > 0.005).astype(int),
        '_synthetic': True,
    })

    logger.info(f"Datos sinteticos: {n} puntos generados")
    return df


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_gp(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Entrena Gaussian Process con kernel Matern nu=2.5.

    Returns:
        (gp, scaler_X, scaler_y)
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 100.0))
        * Matern(length_scale=[1.0] * X.shape[1], nu=2.5,
                 length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=15,
        normalize_y=False,
        random_state=SEED,
    )

    gp.fit(X_scaled, y_scaled)
    logger.info(f"GP entrenado. Kernel: {gp.kernel_}")
    logger.info(f"  Log-marginal likelihood: {gp.log_marginal_likelihood_value_:.3f}")

    return gp, scaler_X, scaler_y


def loo_validation(gp, X_scaled, y_scaled, scaler_y, y_real) -> dict:
    """Leave-One-Out cross-validation."""
    n = len(y_real)
    y_pred = np.zeros(n)
    y_std = np.zeros(n)

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_scaled):
        gp_loo = GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None)
        gp_loo.fit(X_scaled[train_idx], y_scaled[train_idx])
        pred, std = gp_loo.predict(X_scaled[test_idx], return_std=True)
        y_pred[test_idx] = pred
        y_std[test_idx] = std

    y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_std_real = y_std * scaler_y.scale_[0]

    rmse = np.sqrt(np.mean((y_real - y_pred_real) ** 2))
    ss_res = np.sum((y_real - y_pred_real) ** 2)
    ss_tot = np.sum((y_real - y_real.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    logger.info(f"LOO: R²={r2:.4f}, RMSE={rmse:.4f} m")

    return {
        'y_pred': y_pred_real,
        'y_std': y_std_real,
        'rmse': rmse,
        'r2': r2,
    }


# ---------------------------------------------------------------------------
# Rangos de features (para grillas de figuras)
# ---------------------------------------------------------------------------

def _load_feature_ranges() -> dict:
    """Carga rangos de param_ranges.json, con fallback a defaults."""
    ranges_path = PROJECT_ROOT / "config" / "param_ranges.json"
    defaults = {
        'dam_height': (0.10, 0.55),
        'boulder_mass': (0.3, 3.5),
        'boulder_rot_z': (-5, 95),
    }
    if ranges_path.exists():
        with open(ranges_path) as f:
            cfg = json.load(f)
        params = cfg.get('parameters', {})
        for feat in FEATURES:
            if feat in params:
                margin = (params[feat]['max'] - params[feat]['min']) * 0.1
                defaults[feat] = (
                    params[feat]['min'] - margin,
                    params[feat]['max'] + margin,
                )
    return defaults


# ---------------------------------------------------------------------------
# Figuras de tesis
# ---------------------------------------------------------------------------

def plot_2d_slices(gp, scaler_X, scaler_y, X_real, y_real,
                   is_synthetic, out_dir: Path):
    """Fig: slices 2D del GP (un subplot por par de features)."""
    plt.rcParams.update(PLT_STYLE)

    feat_ranges = _load_feature_ranges()
    n_feat = len(FEATURES)
    pairs = list(combinations(range(n_feat), 2))
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 4.5))
    if n_pairs == 1:
        axes = [axes]

    medians = np.median(X_real, axis=0)
    mask_real = ~is_synthetic
    mask_synth = is_synthetic

    for ax, (i, j) in zip(axes, pairs):
        fi, fj = FEATURES[i], FEATURES[j]
        ri = feat_ranges[fi]
        rj = feat_ranges[fj]

        xi = np.linspace(ri[0], ri[1], 50)
        xj = np.linspace(rj[0], rj[1], 50)
        Xi, Xj = np.meshgrid(xi, xj)

        X_grid = np.tile(medians, (Xi.size, 1))
        X_grid[:, i] = Xi.ravel()
        X_grid[:, j] = Xj.ravel()

        X_grid_s = scaler_X.transform(X_grid)
        y_s, _ = gp.predict(X_grid_s, return_std=True)
        y_pred = scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
        Z = y_pred.reshape(Xi.shape)

        cf = ax.contourf(Xi, Xj, Z, levels=20, cmap='RdYlGn_r')
        plt.colorbar(cf, ax=ax, label='Desplaz. [m]')

        if mask_real.any():
            ax.scatter(X_real[mask_real, i], X_real[mask_real, j],
                       c='black', s=25, edgecolors='white', linewidths=0.5,
                       zorder=5, label='SPH reales')
        if mask_synth.any():
            ax.scatter(X_real[mask_synth, i], X_real[mask_synth, j],
                       c='gray', s=15, marker='x', zorder=4, alpha=0.5,
                       label='Sinteticos')

        ax.set_xlabel(FEATURE_LABELS[fi])
        ax.set_ylabel(FEATURE_LABELS[fj])

        # Indicar valor fijo del 3er parametro
        others = [k for k in range(n_feat) if k not in (i, j)]
        fixed_str = ", ".join(
            f"{FEATURES[k]}={medians[k]:.1f}" for k in others
        )
        ax.set_title(f'Fijando {fixed_str}', fontsize=9)
        ax.legend(fontsize=7)

    n_real = mask_real.sum()
    n_synth = mask_synth.sum()
    fig.suptitle(f'GP Surrogate — Slices 2D ({n_real} reales + {n_synth} sint.)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'gp_2d_slices.png')
    fig.savefig(out_dir / 'gp_2d_slices.pdf')
    plt.close(fig)
    logger.info(f"  Figura: gp_2d_slices")


def plot_loo(y_real, loo_result, is_synthetic, out_dir: Path):
    """Fig: LOO validation (predicted vs actual + residuos)."""
    plt.rcParams.update(PLT_STYLE)

    y_pred = loo_result['y_pred']
    y_std = loo_result['y_std']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Predicted vs Actual
    colors = np.where(is_synthetic, '#AAAAAA', '#2166AC')
    for i in range(len(y_real)):
        ax1.errorbar(y_real[i], y_pred[i], yerr=2*y_std[i],
                     fmt='o', markersize=5, color=colors[i],
                     ecolor='#BBDEFB', elinewidth=1, capsize=2, alpha=0.8)

    lims = [min(y_real.min(), y_pred.min()) - 0.3,
            max(y_real.max(), y_pred.max()) + 0.3]
    ax1.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='Perfecto')
    ax1.set_xlabel('Desplazamiento real [m]')
    ax1.set_ylabel('Desplazamiento predicho [m]')
    ax1.set_title(f'(a) LOO: R$^2$ = {loo_result["r2"]:.3f}, RMSE = {loo_result["rmse"]:.3f} m')
    ax1.legend(fontsize=8)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_aspect('equal')

    # Residuos
    residuals = y_real - y_pred
    bar_colors = ['#EF5350' if r > 0 else '#42A5F5' for r in residuals]
    ax2.bar(range(len(y_real)), residuals, color=bar_colors, alpha=0.7, edgecolor='none')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xlabel('Indice de simulacion')
    ax2.set_ylabel('Residuo [m]')
    ax2.set_title('(b) Residuos LOO')

    fig.suptitle('Validacion Leave-One-Out', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'gp_loo_validation.png')
    fig.savefig(out_dir / 'gp_loo_validation.pdf')
    plt.close(fig)
    logger.info(f"  Figura: gp_loo_validation")


def plot_1d_slices(gp, scaler_X, scaler_y, X_real, y_real,
                   is_synthetic, out_dir: Path):
    """Fig: cortes 1D con intervalos de confianza (un panel por feature)."""
    plt.rcParams.update(PLT_STYLE)

    feat_ranges = _load_feature_ranges()
    n_feat = len(FEATURES)
    medians = np.median(X_real, axis=0)

    fig, axes = plt.subplots(1, n_feat, figsize=(5 * n_feat, 4.5))
    if n_feat == 1:
        axes = [axes]

    for idx, (ax, feat) in enumerate(zip(axes, FEATURES)):
        r = feat_ranges[feat]
        x_line = np.linspace(r[0], r[1], 100)

        X_cut = np.tile(medians, (100, 1))
        X_cut[:, idx] = x_line
        X_cut_s = scaler_X.transform(X_cut)

        y_s, std_s = gp.predict(X_cut_s, return_std=True)
        y_cut = scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
        std_cut = std_s * scaler_y.scale_[0]

        color = FEATURE_COLORS[feat]
        ax.plot(x_line, y_cut, color=color, lw=2, label='GP')
        ax.fill_between(x_line, y_cut - 2*std_cut, y_cut + 2*std_cut,
                         alpha=0.2, color=color, label='IC 95%')

        # Scatter datos cercanos a la mediana de las otras features
        tolerances = (np.max(X_real, axis=0) - np.min(X_real, axis=0)) * 0.3
        mask = np.ones(len(X_real), dtype=bool)
        for k in range(n_feat):
            if k != idx:
                mask &= np.abs(X_real[:, k] - medians[k]) < tolerances[k]
        mask_r = mask & ~is_synthetic
        mask_s = mask & is_synthetic
        if mask_r.any():
            ax.scatter(X_real[mask_r, idx], y_real[mask_r], c='red', s=30,
                        zorder=5, edgecolors='white', linewidths=0.5,
                        label='SPH reales')
        if mask_s.any():
            ax.scatter(X_real[mask_s, idx], y_real[mask_s], c='gray', s=15,
                        marker='x', zorder=4, alpha=0.5, label='Sint.')

        ax.set_xlabel(FEATURE_LABELS[feat])
        ax.set_ylabel('Desplazamiento [m]')

        others_str = ", ".join(
            f"{FEATURES[k]}={medians[k]:.1f}" for k in range(n_feat) if k != idx
        )
        ax.set_title(f'Fijando {others_str}', fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle('Cortes 1D del GP con Intervalos de Confianza',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'gp_1d_slices.png')
    fig.savefig(out_dir / 'gp_1d_slices.pdf')
    plt.close(fig)
    logger.info(f"  Figura: gp_1d_slices")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_surrogate(force_synthetic: bool = False) -> dict:
    """
    Pipeline completo del surrogate:
    1. Cargar datos (reales + sinteticos si necesario)
    2. Entrenar GP
    3. LOO validation
    4. Generar figuras
    5. Exportar modelo .pkl

    Returns:
        dict con metricas y paths.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    df_real = pd.DataFrame() if force_synthetic else load_real_data()
    n_real = len(df_real)

    if n_real < MIN_REAL_POINTS:
        logger.info(f"Solo {n_real} datos reales (min={MIN_REAL_POINTS}), "
                     f"complementando con {N_SYNTHETIC} sinteticos")
        df_synth = generate_synthetic(N_SYNTHETIC)
        if n_real > 0:
            df_real['_synthetic'] = False
            df = pd.concat([df_real, df_synth], ignore_index=True)
        else:
            df = df_synth
    else:
        df = df_real
        df['_synthetic'] = False

    is_synthetic = df['_synthetic'].values.astype(bool)
    n_total = len(df)
    n_synth = is_synthetic.sum()

    print(f"\nDatos: {n_total} total ({n_real} reales + {n_synth} sinteticos)")

    # 2. Preparar matrices
    X = df[FEATURES].values
    y = df[TARGET].values

    # 3. Entrenar GP
    gp, scaler_X, scaler_y = train_gp(X, y)

    # 4. LOO validation
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()
    loo_result = loo_validation(gp, X_scaled, y_scaled, scaler_y, y)

    # 5. Figuras
    print("\nGenerando figuras...")
    plot_2d_slices(gp, scaler_X, scaler_y, X, y, is_synthetic, OUT_DIR)
    plot_loo(y, loo_result, is_synthetic, OUT_DIR)
    plot_1d_slices(gp, scaler_X, scaler_y, X, y, is_synthetic, OUT_DIR)

    # 6. Exportar modelo
    model_package = {
        'gp': gp,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': FEATURES,
        'feature_labels': FEATURE_LABELS,
        'target': TARGET,
        'n_real': n_real,
        'n_synthetic': n_synth,
        'loo_r2': loo_result['r2'],
        'loo_rmse': loo_result['rmse'],
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_package, f)
    logger.info(f"Modelo exportado: {MODEL_PATH}")

    # Resumen
    print(f"\n{'='*60}")
    print(f"  RESUMEN GP SURROGATE")
    print(f"{'='*60}")
    print(f"  Datos reales:    {n_real}")
    print(f"  Datos sinteticos: {n_synth}")
    print(f"  Features:        {FEATURES}")
    print(f"  Target:          {TARGET}")
    print(f"  Kernel:          {gp.kernel_}")
    print(f"  R² (LOO):        {loo_result['r2']:.4f}")
    print(f"  RMSE (LOO):      {loo_result['rmse']:.4f} m")
    print(f"  Modelo:          {MODEL_PATH}")
    print(f"  Figuras:         {OUT_DIR}")
    print(f"{'='*60}")

    return {
        'n_real': n_real,
        'n_synthetic': n_synth,
        'r2': loo_result['r2'],
        'rmse': loo_result['rmse'],
        'model_path': str(MODEL_PATH),
        'figures_dir': str(OUT_DIR),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(description='GP Surrogate - Modulo 4')
    parser.add_argument('--synthetic', action='store_true',
                        help='Forzar datos sinteticos (ignorar SQLite)')
    args = parser.parse_args()

    run_surrogate(force_synthetic=args.synthetic)
