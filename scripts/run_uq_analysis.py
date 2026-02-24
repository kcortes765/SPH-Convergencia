"""
run_uq_analysis.py — Cuantificacion de Incertidumbre sobre GP Surrogate

Carga un GP entrenado (gp_surrogate.pkl), ejecuta:
  1. Monte Carlo: 10,000 muestras → distribucion de desplazamiento + CI 95%
  2. Sobol (Saltelli): indices S1 y ST para cada parametro
  3. 6 figuras thesis-quality (PNG 300dpi + PDF vector, espanol)

Prerequisito: haber corrido src/ml_surrogate.py para generar el .pkl

Ejecutar:
    python scripts/run_uq_analysis.py                      # Default
    python scripts/run_uq_analysis.py --mc-samples 50000   # Mas MC
    python scripts/run_uq_analysis.py --sobol-n 8192       # Mas Sobol
    python scripts/run_uq_analysis.py --threshold 0.01     # Umbral movimiento
    python scripts/run_uq_analysis.py --synthetic           # GP sintetico

Autor: Kevin Cortes (UCN 2026)
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol as SobolSampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "gp_surrogate.pkl"
RANGES_PATH = PROJECT_ROOT / "config" / "param_ranges.json"
OUT_DIR = PROJECT_ROOT / "data" / "figuras_uq"

PLT_STYLE = {
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 200, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.2,
    'axes.spines.top': False, 'axes.spines.right': False,
}

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
# Carga
# ---------------------------------------------------------------------------

def load_model(model_path: Path = MODEL_PATH) -> dict:
    """Carga GP entrenado desde .pkl."""
    if not model_path.exists():
        logger.error(f"Modelo no encontrado: {model_path}")
        logger.error("Ejecutar primero: python src/ml_surrogate.py")
        sys.exit(1)
    with open(model_path, 'rb') as f:
        pkg = pickle.load(f)
    logger.info(f"Modelo cargado: {model_path.name} "
                f"(R²={pkg['loo_r2']:.3f}, {pkg['n_real']} reales)")
    return pkg


def load_param_ranges(ranges_path: Path = RANGES_PATH) -> dict:
    """Carga rangos de parametros como {feature: (min, max)}."""
    with open(ranges_path) as f:
        cfg = json.load(f)
    ranges = {}
    for feat, info in cfg['parameters'].items():
        ranges[feat] = (info['min'], info['max'])
    return ranges


def predict_gp(pkg: dict, X: np.ndarray) -> tuple:
    """Predice con GP en unidades reales. Returns (y_mean, y_std)."""
    X_s = pkg['scaler_X'].transform(X)
    y_s, std_s = pkg['gp'].predict(X_s, return_std=True)
    y_mean = pkg['scaler_y'].inverse_transform(y_s.reshape(-1, 1)).ravel()
    y_std = std_s * pkg['scaler_y'].scale_[0]
    return y_mean, y_std


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo(pkg: dict, param_ranges: dict,
                n_samples: int = 10000, seed: int = 42,
                threshold: float = 0.005) -> dict:
    """
    Monte Carlo sobre GP: muestrea uniformemente en rangos,
    predice desplazamiento, calcula estadisticas.
    """
    features = pkg['features']
    d = len(features)
    rng = np.random.default_rng(seed)

    # Muestreo uniforme
    X_mc = np.zeros((n_samples, d))
    for i, feat in enumerate(features):
        lo, hi = param_ranges[feat]
        X_mc[:, i] = rng.uniform(lo, hi, n_samples)

    # Prediccion
    y_mean, y_std = predict_gp(pkg, X_mc)

    # Estadisticas
    ci_lo = np.percentile(y_mean, 2.5)
    ci_hi = np.percentile(y_mean, 97.5)
    prob_motion = np.mean(y_mean > threshold)

    result = {
        'X_mc': X_mc,
        'y_mean': y_mean,
        'y_std': y_std,
        'mean': float(np.mean(y_mean)),
        'median': float(np.median(y_mean)),
        'std': float(np.std(y_mean)),
        'ci_95': (float(ci_lo), float(ci_hi)),
        'prob_motion': float(prob_motion),
        'threshold': threshold,
        'n_samples': n_samples,
    }

    logger.info(f"MC: media={result['mean']:.4f}m, "
                f"CI95=[{ci_lo:.4f}, {ci_hi:.4f}]m, "
                f"P(mov)={prob_motion:.3f}")
    return result


# ---------------------------------------------------------------------------
# Sobol (algoritmo de Saltelli)
# ---------------------------------------------------------------------------

def sobol_indices(pkg: dict, param_ranges: dict,
                  n_base: int = 4096, seed: int = 42) -> dict:
    """
    Calcula indices de Sobol S1 y ST via Saltelli/Jansen.

    Genera N*(2d+2) evaluaciones del GP.
    """
    features = pkg['features']
    d = len(features)

    # Generar matrices A y B con Sobol quasi-random
    sampler = SobolSampler(d=d, scramble=True, seed=seed)
    A_unit = sampler.random(n_base)  # (N, d) en [0,1]

    sampler_b = SobolSampler(d=d, scramble=True, seed=seed + 1000)
    B_unit = sampler_b.random(n_base)

    # Escalar a rangos reales
    bounds = np.array([param_ranges[f] for f in features])
    A = bounds[:, 0] + A_unit * (bounds[:, 1] - bounds[:, 0])
    B = bounds[:, 0] + B_unit * (bounds[:, 1] - bounds[:, 0])

    # Evaluar GP en A y B
    fA, _ = predict_gp(pkg, A)
    fB, _ = predict_gp(pkg, B)

    # Para cada parametro i, crear AB_i (A con columna i de B)
    fABi = np.zeros((d, n_base))
    for i in range(d):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        fABi[i], _ = predict_gp(pkg, AB_i)

    # Varianza total
    f_all = np.concatenate([fA, fB])
    var_total = np.var(f_all)

    if var_total < 1e-12:
        logger.warning("Varianza total ~ 0. Indices no calculables.")
        return {
            'S1': {f: 0.0 for f in features},
            'ST': {f: 0.0 for f in features},
            'S1_ci': {f: (0.0, 0.0) for f in features},
            'ST_ci': {f: (0.0, 0.0) for f in features},
            'var_total': float(var_total),
            'n_base': n_base,
        }

    # Jansen estimator
    S1 = {}
    ST = {}
    for i, feat in enumerate(features):
        # S1_i = (1/N) * sum(fB * (fABi - fA)) / Var
        S1[feat] = float(np.mean(fB * (fABi[i] - fA)) / var_total)
        # ST_i = (1/2N) * sum((fA - fABi)^2) / Var
        ST[feat] = float(np.mean((fA - fABi[i])**2) / (2 * var_total))

    # Bootstrap CI
    n_bootstrap = 1000
    rng = np.random.default_rng(seed + 2000)
    S1_boot = {f: [] for f in features}
    ST_boot = {f: [] for f in features}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_base, n_base)
        fA_b = fA[idx]
        fB_b = fB[idx]
        f_all_b = np.concatenate([fA_b, fB_b])
        var_b = np.var(f_all_b)
        if var_b < 1e-12:
            continue
        for i, feat in enumerate(features):
            fABi_b = fABi[i][idx]
            S1_boot[feat].append(np.mean(fB_b * (fABi_b - fA_b)) / var_b)
            ST_boot[feat].append(np.mean((fA_b - fABi_b)**2) / (2 * var_b))

    S1_ci = {}
    ST_ci = {}
    for feat in features:
        if S1_boot[feat]:
            S1_ci[feat] = (float(np.percentile(S1_boot[feat], 2.5)),
                           float(np.percentile(S1_boot[feat], 97.5)))
            ST_ci[feat] = (float(np.percentile(ST_boot[feat], 2.5)),
                           float(np.percentile(ST_boot[feat], 97.5)))
        else:
            S1_ci[feat] = (0.0, 0.0)
            ST_ci[feat] = (0.0, 0.0)

    total_evals = n_base * (2 + d)
    logger.info(f"Sobol: {total_evals} evaluaciones GP")
    for feat in features:
        logger.info(f"  {feat}: S1={S1[feat]:.3f} [{S1_ci[feat][0]:.3f}, {S1_ci[feat][1]:.3f}], "
                     f"ST={ST[feat]:.3f} [{ST_ci[feat][0]:.3f}, {ST_ci[feat][1]:.3f}]")

    return {
        'S1': S1, 'ST': ST,
        'S1_ci': S1_ci, 'ST_ci': ST_ci,
        'var_total': float(var_total),
        'n_base': n_base,
        'total_evals': total_evals,
    }


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def plot_mc_histogram(mc: dict, out_dir: Path):
    """Fig 1: Histograma de desplazamiento MC."""
    plt.rcParams.update(PLT_STYLE)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    y = mc['y_mean']
    ax.hist(y, bins=50, color='#2166AC', alpha=0.7, edgecolor='white',
            density=True, label='Distribucion MC')

    # Lineas verticales
    ax.axvline(mc['mean'], color='#B2182B', lw=2, ls='-',
               label=f'Media = {mc["mean"]:.3f} m')
    ax.axvline(mc['ci_95'][0], color='#FF7F00', lw=1.5, ls='--',
               label=f'CI 95% = [{mc["ci_95"][0]:.3f}, {mc["ci_95"][1]:.3f}] m')
    ax.axvline(mc['ci_95'][1], color='#FF7F00', lw=1.5, ls='--')
    ax.axvline(mc['threshold'], color='#2CA02C', lw=1.5, ls=':',
               label=f'Umbral = {mc["threshold"]} m')

    ax.set_xlabel('Desplazamiento maximo [m]')
    ax.set_ylabel('Densidad')
    ax.set_title(f'Monte Carlo: {mc["n_samples"]:,} muestras — '
                 f'P(movimiento) = {mc["prob_motion"]:.1%}')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / 'mc_histograma_desplazamiento.png')
    fig.savefig(out_dir / 'mc_histograma_desplazamiento.pdf')
    plt.close(fig)
    logger.info("  Figura: mc_histograma_desplazamiento")


def plot_prob_vs_params(mc: dict, features: list,
                        param_ranges: dict, out_dir: Path):
    """Fig 2: P(movimiento) vs cada parametro (binned)."""
    plt.rcParams.update(PLT_STYLE)

    n_feat = len(features)
    fig, axes = plt.subplots(1, n_feat, figsize=(5 * n_feat, 4.5))
    if n_feat == 1:
        axes = [axes]

    X = mc['X_mc']
    y = mc['y_mean']
    threshold = mc['threshold']

    for idx, (ax, feat) in enumerate(zip(axes, features)):
        lo, hi = param_ranges[feat]
        bins = np.linspace(lo, hi, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        probs = []

        for b_lo, b_hi in zip(bins[:-1], bins[1:]):
            mask = (X[:, idx] >= b_lo) & (X[:, idx] < b_hi)
            if mask.sum() > 0:
                probs.append(np.mean(y[mask] > threshold))
            else:
                probs.append(0.0)

        color = FEATURE_COLORS.get(feat, '#333333')
        ax.bar(bin_centers, probs, width=(hi - lo) / 10 * 0.85,
               color=color, alpha=0.7, edgecolor='white')
        ax.set_xlabel(FEATURE_LABELS.get(feat, feat))
        ax.set_ylabel('P(movimiento)')
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)

    fig.suptitle('Probabilidad de movimiento incipiente vs parametros',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'mc_probabilidad_vs_parametros.png')
    fig.savefig(out_dir / 'mc_probabilidad_vs_parametros.pdf')
    plt.close(fig)
    logger.info("  Figura: mc_probabilidad_vs_parametros")


def plot_sobol_tornado(sobol: dict, features: list, out_dir: Path):
    """Fig 3: Barras horizontales S1 y ST."""
    plt.rcParams.update(PLT_STYLE)

    fig, ax = plt.subplots(figsize=(7, 3 + 0.5 * len(features)))

    y_pos = np.arange(len(features))
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    s1_vals = [sobol['S1'][f] for f in features]
    st_vals = [sobol['ST'][f] for f in features]
    s1_err = [
        [sobol['S1'][f] - sobol['S1_ci'][f][0] for f in features],
        [sobol['S1_ci'][f][1] - sobol['S1'][f] for f in features],
    ]
    st_err = [
        [sobol['ST'][f] - sobol['ST_ci'][f][0] for f in features],
        [sobol['ST_ci'][f][1] - sobol['ST'][f] for f in features],
    ]

    bar_h = 0.35
    ax.barh(y_pos + bar_h/2, s1_vals, bar_h, xerr=s1_err,
            color='#2166AC', alpha=0.8, label='$S_1$ (primer orden)',
            capsize=3)
    ax.barh(y_pos - bar_h/2, st_vals, bar_h, xerr=st_err,
            color='#B2182B', alpha=0.8, label='$S_T$ (orden total)',
            capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Indice de Sobol')
    ax.set_title('Analisis de Sensibilidad Global')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(left=-0.05)
    ax.axvline(0, color='black', lw=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / 'sobol_indices.png')
    fig.savefig(out_dir / 'sobol_indices.pdf')
    plt.close(fig)
    logger.info("  Figura: sobol_indices")


def plot_probability_frontier(pkg: dict, param_ranges: dict,
                              threshold: float, out_dir: Path):
    """Fig 4: Contour 2D de P(movimiento) para h vs m."""
    plt.rcParams.update(PLT_STYLE)
    features = pkg['features']

    # Usar los dos primeros parametros (h vs m)
    if len(features) < 2:
        logger.warning("Se necesitan al menos 2 features para frontera.")
        return

    f0, f1 = features[0], features[1]
    r0 = param_ranges[f0]
    r1 = param_ranges[f1]

    x0 = np.linspace(r0[0], r0[1], 60)
    x1 = np.linspace(r1[0], r1[1], 60)
    X0, X1 = np.meshgrid(x0, x1)

    # Fijar otros parametros en su mediana
    d = len(features)
    medians = np.array([(param_ranges[f][0] + param_ranges[f][1]) / 2
                        for f in features])

    X_grid = np.tile(medians, (X0.size, 1))
    X_grid[:, 0] = X0.ravel()
    X_grid[:, 1] = X1.ravel()

    y_mean, y_std = predict_gp(pkg, X_grid)

    # P(displacement > threshold) asumiendo distribucion normal GP
    from scipy.stats import norm
    p_motion = np.zeros_like(y_mean)
    for i in range(len(y_mean)):
        if y_std[i] > 1e-10:
            p_motion[i] = 1.0 - norm.cdf(threshold, loc=y_mean[i], scale=y_std[i])
        else:
            p_motion[i] = 1.0 if y_mean[i] > threshold else 0.0

    P = p_motion.reshape(X0.shape)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cf = ax.contourf(X0, X1, P, levels=np.linspace(0, 1, 21),
                     cmap='RdYlGn_r')
    plt.colorbar(cf, ax=ax, label='P(movimiento)')

    # Contorno 50%
    cs = ax.contour(X0, X1, P, levels=[0.5], colors='black', linewidths=2)
    ax.clabel(cs, fmt='P=%.1f', fontsize=9)

    # Contorno 95%
    cs95 = ax.contour(X0, X1, P, levels=[0.95], colors='white',
                      linewidths=1.5, linestyles='--')
    ax.clabel(cs95, fmt='P=%.2f', fontsize=8)

    ax.set_xlabel(FEATURE_LABELS[f0])
    ax.set_ylabel(FEATURE_LABELS[f1])

    others = [f for f in features if f not in (f0, f1)]
    if others:
        fixed = ", ".join(f"{f}={medians[features.index(f)]:.1f}" for f in others)
        ax.set_title(f'Frontera de estabilidad (umbral={threshold}m, {fixed})')
    else:
        ax.set_title(f'Frontera de estabilidad (umbral={threshold}m)')

    fig.tight_layout()
    fig.savefig(out_dir / 'frontera_probabilidad.png')
    fig.savefig(out_dir / 'frontera_probabilidad.pdf')
    plt.close(fig)
    logger.info("  Figura: frontera_probabilidad")


def plot_ci_bands(pkg: dict, param_ranges: dict, mc: dict, out_dir: Path):
    """Fig 5: Sweep 1D por parametro con bandas CI del MC."""
    plt.rcParams.update(PLT_STYLE)

    features = pkg['features']
    n_feat = len(features)
    d = len(features)
    medians = np.array([(param_ranges[f][0] + param_ranges[f][1]) / 2
                        for f in features])

    fig, axes = plt.subplots(1, n_feat, figsize=(5 * n_feat, 4.5))
    if n_feat == 1:
        axes = [axes]

    for idx, (ax, feat) in enumerate(zip(axes, features)):
        lo, hi = param_ranges[feat]
        x_line = np.linspace(lo, hi, 100)

        X_sweep = np.tile(medians, (100, 1))
        X_sweep[:, idx] = x_line

        y_mean, y_std = predict_gp(pkg, X_sweep)

        color = FEATURE_COLORS.get(feat, '#333333')
        ax.plot(x_line, y_mean, color=color, lw=2, label='GP media')
        ax.fill_between(x_line, y_mean - 2*y_std, y_mean + 2*y_std,
                        alpha=0.15, color=color, label='GP $\\pm 2\\sigma$')

        # Banda MC: para cada bin del parametro, calcular percentiles
        X_mc = mc['X_mc']
        y_mc = mc['y_mean']
        n_bins = 15
        bins = np.linspace(lo, hi, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        p05, p50, p95 = [], [], []

        for b_lo, b_hi in zip(bins[:-1], bins[1:]):
            mask = (X_mc[:, idx] >= b_lo) & (X_mc[:, idx] < b_hi)
            if mask.sum() > 5:
                p05.append(np.percentile(y_mc[mask], 5))
                p50.append(np.percentile(y_mc[mask], 50))
                p95.append(np.percentile(y_mc[mask], 95))
            else:
                p05.append(np.nan)
                p50.append(np.nan)
                p95.append(np.nan)

        p05 = np.array(p05)
        p50 = np.array(p50)
        p95 = np.array(p95)
        valid = ~np.isnan(p50)

        ax.fill_between(bin_centers[valid], p05[valid], p95[valid],
                        alpha=0.2, color='#FF7F00', label='MC 90% CI')
        ax.plot(bin_centers[valid], p50[valid], color='#FF7F00',
                lw=1.5, ls='--', label='MC mediana')

        ax.set_xlabel(FEATURE_LABELS.get(feat, feat))
        ax.set_ylabel('Desplazamiento [m]')
        others_str = ", ".join(
            f"{features[k]}={medians[k]:.1f}" for k in range(d) if k != idx
        )
        ax.set_title(f'Fijando {others_str}', fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle('Bandas de Confianza: GP vs Monte Carlo',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'ci_bandas_parametros.png')
    fig.savefig(out_dir / 'ci_bandas_parametros.pdf')
    plt.close(fig)
    logger.info("  Figura: ci_bandas_parametros")


def plot_sobol_convergence(pkg: dict, param_ranges: dict, out_dir: Path):
    """Fig 6: Convergencia de indices Sobol vs N_base."""
    plt.rcParams.update(PLT_STYLE)

    features = pkg['features']
    n_values = [256, 512, 1024, 2048, 4096, 8192]

    results = {feat: {'S1': [], 'ST': []} for feat in features}

    for n in n_values:
        sob = sobol_indices(pkg, param_ranges, n_base=n, seed=42)
        for feat in features:
            results[feat]['S1'].append(sob['S1'][feat])
            results[feat]['ST'].append(sob['ST'][feat])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for feat in features:
        color = FEATURE_COLORS.get(feat, '#333333')
        label = FEATURE_LABELS.get(feat, feat)
        ax1.plot(n_values, results[feat]['S1'], 'o-', color=color,
                 lw=1.5, markersize=5, label=label)
        ax2.plot(n_values, results[feat]['ST'], 'o-', color=color,
                 lw=1.5, markersize=5, label=label)

    for ax, title in [(ax1, '$S_1$ (primer orden)'),
                      (ax2, '$S_T$ (orden total)')]:
        ax.set_xlabel('$N_{base}$ (muestras Sobol)')
        ax.set_ylabel('Indice')
        ax.set_title(title)
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=8)

    fig.suptitle('Convergencia de Indices de Sobol',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'sobol_convergencia.png')
    fig.savefig(out_dir / 'sobol_convergencia.pdf')
    plt.close(fig)
    logger.info("  Figura: sobol_convergencia")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_uq(mc_samples: int = 10000, sobol_n: int = 4096,
           threshold: float = 0.005, force_synthetic: bool = False):
    """Pipeline completo de UQ."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Cargar GP y rangos
    if force_synthetic:
        # Entrenar GP sintetico primero
        sys.path.insert(0, str(PROJECT_ROOT / 'src'))
        from ml_surrogate import run_surrogate
        run_surrogate(force_synthetic=True)

    pkg = load_model()
    param_ranges = load_param_ranges()
    features = pkg['features']

    print(f"\n{'='*60}")
    print(f"  UQ ANALYSIS")
    print(f"{'='*60}")
    print(f"  Features:     {features}")
    print(f"  MC muestras:  {mc_samples:,}")
    print(f"  Sobol N_base: {sobol_n}")
    print(f"  Umbral:       {threshold} m")
    print(f"  GP R² (LOO):  {pkg['loo_r2']:.4f}")
    print(f"{'='*60}\n")

    # 1. Monte Carlo
    print("1/3 Monte Carlo...")
    mc = monte_carlo(pkg, param_ranges, n_samples=mc_samples, threshold=threshold)

    # 2. Sobol
    print("2/3 Sobol indices...")
    sobol = sobol_indices(pkg, param_ranges, n_base=sobol_n)

    # 3. Figuras
    print("3/3 Generando figuras...")
    plot_mc_histogram(mc, OUT_DIR)
    plot_prob_vs_params(mc, features, param_ranges, OUT_DIR)
    plot_sobol_tornado(sobol, features, OUT_DIR)
    plot_probability_frontier(pkg, param_ranges, threshold, OUT_DIR)
    plot_ci_bands(pkg, param_ranges, mc, OUT_DIR)
    plot_sobol_convergence(pkg, param_ranges, OUT_DIR)

    # 4. Resumen JSON
    summary = {
        'monte_carlo': {
            'n_samples': mc_samples,
            'threshold_m': threshold,
            'mean_displacement_m': mc['mean'],
            'median_displacement_m': mc['median'],
            'std_displacement_m': mc['std'],
            'ci_95_m': mc['ci_95'],
            'prob_motion': mc['prob_motion'],
        },
        'sobol': {
            'n_base': sobol_n,
            'total_evaluations': sobol['total_evals'],
            'var_total': sobol['var_total'],
            'S1': sobol['S1'],
            'ST': sobol['ST'],
            'S1_ci_95': sobol['S1_ci'],
            'ST_ci_95': sobol['ST_ci'],
        },
        'gp_model': {
            'features': features,
            'loo_r2': pkg['loo_r2'],
            'loo_rmse': pkg['loo_rmse'],
            'n_real': pkg['n_real'],
            'n_synthetic': pkg['n_synthetic'],
        },
    }

    summary_path = OUT_DIR / 'uq_summary.json'

    def _json_safe(obj):
        """Convierte numpy types a Python nativos para JSON."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_safe)

    # Imprimir resumen
    print(f"\n{'='*60}")
    print(f"  RESULTADOS UQ")
    print(f"{'='*60}")
    print(f"  Desplazamiento: {mc['mean']:.3f} +/- {mc['std']:.3f} m")
    print(f"  CI 95%:         [{mc['ci_95'][0]:.3f}, {mc['ci_95'][1]:.3f}] m")
    print(f"  P(movimiento):  {mc['prob_motion']:.1%}")
    print(f"\n  Indices de Sobol (ST):")
    for feat in features:
        label = FEATURE_LABELS.get(feat, feat)
        s1 = sobol['S1'][feat]
        st = sobol['ST'][feat]
        bar = '#' * int(st * 40)
        print(f"    {label:30s}  S1={s1:.3f}  ST={st:.3f}  {bar}")
    print(f"\n  Figuras:  {OUT_DIR}")
    print(f"  Resumen:  {summary_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(
        description='UQ Analysis — Monte Carlo + Sobol sobre GP Surrogate')
    parser.add_argument('--mc-samples', type=int, default=10000,
                        help='Numero de muestras Monte Carlo (default: 10000)')
    parser.add_argument('--sobol-n', type=int, default=4096,
                        help='Tamano base para Sobol/Saltelli (default: 4096)')
    parser.add_argument('--threshold', type=float, default=0.005,
                        help='Umbral de movimiento en metros (default: 0.005)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Usar GP sintetico (para testing)')
    args = parser.parse_args()

    run_uq(
        mc_samples=args.mc_samples,
        sobol_n=args.sobol_n,
        threshold=args.threshold,
        force_synthetic=args.synthetic,
    )
