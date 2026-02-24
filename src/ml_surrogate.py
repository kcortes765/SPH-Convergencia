"""
ml_surrogate.py — Modulo 4 del Pipeline SPH-IncipientMotion

Gaussian Process Surrogate conectado a datos reales de SQLite.
Entrena un GP Matern nu=2.5 para predecir desplazamiento del boulder
a partir de parametros de entrada (dam_height, boulder_mass).

Si hay menos de MIN_REAL_POINTS datos reales, complementa con datos
sinteticos basados en fisica simplificada (flaggeados como sinteticos).

Genera figuras de tesis y exporta modelo .pkl para el dashboard.

Ejecutar:
    python src/ml_surrogate.py
    python src/ml_surrogate.py --synthetic  # forzar datos sinteticos

Autor: Kevin Cortes (UCN 2026)
"""

import argparse
import logging
import pickle
import sqlite3
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

FEATURES = ['dam_height', 'boulder_mass']
TARGET = 'max_displacement'


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_real_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Lee resultados reales de SQLite con parametros de entrada."""
    if not db_path.exists():
        logger.warning(f"SQLite no encontrada: {db_path}")
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql(f"""
        SELECT case_name, dam_height, boulder_mass, dp,
               max_displacement, max_rotation, max_velocity,
               max_sph_force, max_contact_force,
               max_flow_velocity, max_water_height,
               failed
        FROM results
        WHERE dam_height > 0 AND boulder_mass > 0
    """, conn)
    conn.close()

    logger.info(f"Datos reales: {len(df)} casos de {db_path.name}")
    return df


def generate_synthetic(n: int = N_SYNTHETIC, seed: int = SEED) -> pd.DataFrame:
    """Genera datos sinteticos basados en fisica simplificada."""
    rng = np.random.default_rng(seed)

    dam_height = rng.uniform(0.15, 0.50, n)
    boulder_mass = rng.uniform(0.5, 3.0, n)

    # Modelo fisico simplificado: E ~ rho*g*h^2/2, R ~ m*g*mu
    energy = 1000 * 9.81 * dam_height**2 * 0.5
    resistance = boulder_mass * 9.81 * 0.6
    displacement = 2.0 * (energy / resistance) ** 0.7
    displacement += rng.normal(0, 0.15, n)
    displacement = np.maximum(displacement, 0)

    df = pd.DataFrame({
        'case_name': [f"synth_{i+1:03d}" for i in range(n)],
        'dam_height': dam_height,
        'boulder_mass': boulder_mass,
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
# Figuras de tesis
# ---------------------------------------------------------------------------

def plot_surface(gp, scaler_X, scaler_y, X_real, y_real,
                 is_synthetic, out_dir: Path):
    """Fig: superficie de prediccion GP + incertidumbre."""
    plt.rcParams.update(PLT_STYLE)

    h_grid = np.linspace(0.10, 0.55, 50)
    m_grid = np.linspace(0.3, 3.5, 50)
    H, M = np.meshgrid(h_grid, m_grid)
    X_grid = np.column_stack([H.ravel(), M.ravel()])
    X_grid_s = scaler_X.transform(X_grid)

    y_pred_s, y_std_s = gp.predict(X_grid_s, return_std=True)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_std = y_std_s * scaler_y.scale_[0]

    Z_pred = y_pred.reshape(H.shape)
    Z_std = y_std.reshape(H.shape)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Prediccion
    ax = axes[0]
    cf = ax.contourf(H, M, Z_pred, levels=20, cmap='RdYlGn_r')
    mask_real = ~is_synthetic
    mask_synth = is_synthetic
    if mask_real.any():
        ax.scatter(X_real[mask_real, 0], X_real[mask_real, 1], c='black', s=25,
                   edgecolors='white', linewidths=0.5, zorder=5, label='Datos SPH reales')
    if mask_synth.any():
        ax.scatter(X_real[mask_synth, 0], X_real[mask_synth, 1], c='gray', s=15,
                   marker='x', zorder=4, alpha=0.5, label='Datos sinteticos')
    plt.colorbar(cf, ax=ax, label='Desplaz. max [m]')
    ax.set_xlabel('Altura columna $h$ [m]')
    ax.set_ylabel('Masa boulder [kg]')
    ax.set_title('(a) Prediccion GP')
    ax.legend(fontsize=7.5)

    # Incertidumbre
    ax = axes[1]
    cf2 = ax.contourf(H, M, Z_std * 2, levels=20, cmap='Oranges')
    if mask_real.any():
        ax.scatter(X_real[mask_real, 0], X_real[mask_real, 1], c='black', s=25,
                   edgecolors='white', linewidths=0.5, zorder=5)
    plt.colorbar(cf2, ax=ax, label='Incertidumbre 2$\\sigma$ [m]')
    ax.set_xlabel('Altura columna $h$ [m]')
    ax.set_ylabel('Masa boulder [kg]')
    ax.set_title('(b) Intervalo de confianza 95%')

    n_real = mask_real.sum()
    n_synth = mask_synth.sum()
    fig.suptitle(f'GP Surrogate ({n_real} reales + {n_synth} sinteticos)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'gp_surface.png')
    fig.savefig(out_dir / 'gp_surface.pdf')
    plt.close(fig)
    logger.info(f"  Figura: gp_surface")


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


def plot_slices(gp, scaler_X, scaler_y, X_real, y_real,
                is_synthetic, out_dir: Path):
    """Fig: cortes 1D con intervalos de confianza."""
    plt.rcParams.update(PLT_STYLE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Corte 1: masa fija, variar altura
    m_fixed = float(np.median(X_real[:, 1]))
    h_line = np.linspace(0.10, 0.55, 100)
    X_cut = np.column_stack([h_line, np.full_like(h_line, m_fixed)])
    X_cut_s = scaler_X.transform(X_cut)
    y_s, std_s = gp.predict(X_cut_s, return_std=True)
    y_cut = scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
    std_cut = std_s * scaler_y.scale_[0]

    ax1.plot(h_line, y_cut, color='#2166AC', lw=2, label='GP')
    ax1.fill_between(h_line, y_cut - 2*std_cut, y_cut + 2*std_cut,
                     alpha=0.2, color='#2166AC', label='IC 95%')
    mask = np.abs(X_real[:, 1] - m_fixed) < 0.5
    mask_r = mask & ~is_synthetic
    mask_s = mask & is_synthetic
    if mask_r.any():
        ax1.scatter(X_real[mask_r, 0], y_real[mask_r], c='red', s=30,
                    zorder=5, edgecolors='white', linewidths=0.5, label='SPH reales')
    if mask_s.any():
        ax1.scatter(X_real[mask_s, 0], y_real[mask_s], c='gray', s=15,
                    marker='x', zorder=4, alpha=0.5, label='Sinteticos')
    ax1.set_xlabel('Altura columna $h$ [m]')
    ax1.set_ylabel('Desplazamiento [m]')
    ax1.set_title(f'(a) Masa fija = {m_fixed:.2f} kg')
    ax1.legend(fontsize=7.5)

    # Corte 2: altura fija, variar masa
    h_fixed = float(np.median(X_real[:, 0]))
    m_line = np.linspace(0.3, 3.5, 100)
    X_cut2 = np.column_stack([np.full_like(m_line, h_fixed), m_line])
    X_cut2_s = scaler_X.transform(X_cut2)
    y_s2, std_s2 = gp.predict(X_cut2_s, return_std=True)
    y_cut2 = scaler_y.inverse_transform(y_s2.reshape(-1, 1)).ravel()
    std_cut2 = std_s2 * scaler_y.scale_[0]

    ax2.plot(m_line, y_cut2, color='#B2182B', lw=2, label='GP')
    ax2.fill_between(m_line, y_cut2 - 2*std_cut2, y_cut2 + 2*std_cut2,
                     alpha=0.2, color='#B2182B', label='IC 95%')
    mask2 = np.abs(X_real[:, 0] - h_fixed) < 0.05
    mask2_r = mask2 & ~is_synthetic
    mask2_s = mask2 & is_synthetic
    if mask2_r.any():
        ax2.scatter(X_real[mask2_r, 1], y_real[mask2_r], c='blue', s=30,
                    zorder=5, edgecolors='white', linewidths=0.5, label='SPH reales')
    if mask2_s.any():
        ax2.scatter(X_real[mask2_s, 1], y_real[mask2_s], c='gray', s=15,
                    marker='x', zorder=4, alpha=0.5, label='Sinteticos')
    ax2.set_xlabel('Masa boulder [kg]')
    ax2.set_ylabel('Desplazamiento [m]')
    ax2.set_title(f'(b) Altura fija = {h_fixed:.2f} m')
    ax2.legend(fontsize=7.5)

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
    plot_surface(gp, scaler_X, scaler_y, X, y, is_synthetic, OUT_DIR)
    plot_loo(y, loo_result, is_synthetic, OUT_DIR)
    plot_slices(gp, scaler_X, scaler_y, X, y, is_synthetic, OUT_DIR)

    # 6. Exportar modelo
    model_package = {
        'gp': gp,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': FEATURES,
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
