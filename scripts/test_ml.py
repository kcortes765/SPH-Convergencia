"""
test_ml.py — Prueba del Gaussian Process Regressor para SPH-IncipientMotion

Genera datos sintéticos que simulan el comportamiento de un boulder bajo
flujo tipo tsunami, entrena un GP con kernel Matern, y visualiza las
predicciones con intervalos de confianza.

Esto es el prototipo del Módulo 4 (ml_surrogate.py). Cuando lleguen
datos reales de la RTX 5090, se reemplaza la generación sintética por
la lectura de results.sqlite.

Conceptos clave:
- Gaussian Process: modelo bayesiano no-paramétrico
- Kernel Matern (nu=2.5): flexible, común en ingeniería
- Intervalos de confianza: ±2σ = 95% de probabilidad
- Normalización: StandardScaler para que el GP converja bien

Ejecutar:
    python test_ml.py

Autor: Kevin Cortes (UCN 2026)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# 1. GENERAR DATOS SINTETICOS
# ═══════════════════════════════════════════════════════════════════════

np.random.seed(42)

N_TRAIN = 30    # puntos de entrenamiento (simula 30 simulaciones SPH)
N_TEST = 200    # puntos para predicción continua

# Variables de entrada (rangos realistas del estudio)
dam_height = np.random.uniform(0.15, 0.50, N_TRAIN)     # Altura de la columna [m]
boulder_mass = np.random.uniform(0.5, 3.0, N_TRAIN)     # Masa del boulder [kg]

# Función "real" (desconocida): desplazamiento = f(dam_height, mass)
# Más agua + menos masa → más desplazamiento (físicamente correcto)
def true_displacement(h, m):
    """Modelo sintético basado en física simplificada."""
    # Energía del flujo proporcional a h^2, resistencia proporcional a m
    energy = 1000 * 9.81 * h**2 * 0.5   # E = rho*g*h^2/2 (por unidad)
    resistance = m * 9.81 * 0.6          # F = m*g*mu (fricción)
    disp = 2.0 * (energy / resistance) ** 0.7
    return disp

displacement = true_displacement(dam_height, boulder_mass)
# Agregar ruido (simula variabilidad numérica del SPH)
noise_std = 0.15
displacement += np.random.normal(0, noise_std, N_TRAIN)
displacement = np.maximum(displacement, 0)  # No negativos

# Matriz de entrada
X_train = np.column_stack([dam_height, boulder_mass])
y_train = displacement

print(f"Datos de entrenamiento: {N_TRAIN} simulaciones sintéticas")
print(f"  dam_height: [{dam_height.min():.2f}, {dam_height.max():.2f}] m")
print(f"  boulder_mass: [{boulder_mass.min():.2f}, {boulder_mass.max():.2f}] kg")
print(f"  displacement: [{displacement.min():.2f}, {displacement.max():.2f}] m")


# ═══════════════════════════════════════════════════════════════════════
# 2. NORMALIZAR DATOS
# ═══════════════════════════════════════════════════════════════════════

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_train)
y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()


# ═══════════════════════════════════════════════════════════════════════
# 3. DEFINIR Y ENTRENAR EL GAUSSIAN PROCESS
# ═══════════════════════════════════════════════════════════════════════

# Kernel: Constante * Matern + WhiteKernel (ruido)
# Matern nu=2.5: dos veces diferenciable, bueno para funciones físicas
kernel = (
    ConstantKernel(1.0, constant_value_bounds=(1e-3, 100.0))
    * Matern(length_scale=[1.0, 1.0], nu=2.5,
             length_scale_bounds=(1e-2, 10.0))
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=15,
    normalize_y=False,     # Ya normalizamos manualmente
    random_state=42,
)

print("\nEntrenando Gaussian Process...")
gp.fit(X_scaled, y_scaled)

print(f"  Kernel optimizado: {gp.kernel_}")
print(f"  Log-marginal likelihood: {gp.log_marginal_likelihood_value_:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# 4. VALIDACION LEAVE-ONE-OUT
# ═══════════════════════════════════════════════════════════════════════

print("\nValidación Leave-One-Out (LOO)...")
loo = LeaveOneOut()
y_pred_loo = np.zeros(N_TRAIN)
y_std_loo = np.zeros(N_TRAIN)

for train_idx, test_idx in loo.split(X_scaled):
    gp_loo = GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None)
    gp_loo.fit(X_scaled[train_idx], y_scaled[train_idx])
    pred, std = gp_loo.predict(X_scaled[test_idx], return_std=True)
    y_pred_loo[test_idx] = pred
    y_std_loo[test_idx] = std

# Desnormalizar
y_pred_loo_real = scaler_y.inverse_transform(y_pred_loo.reshape(-1, 1)).ravel()
y_std_loo_real = y_std_loo * scaler_y.scale_[0]

rmse_loo = np.sqrt(np.mean((y_train - y_pred_loo_real) ** 2))
r2_loo = 1 - np.sum((y_train - y_pred_loo_real)**2) / np.sum((y_train - y_train.mean())**2)

print(f"  RMSE (LOO): {rmse_loo:.4f} m")
print(f"  R² (LOO):   {r2_loo:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# 5. PREDICCION EN GRILLA 2D
# ═══════════════════════════════════════════════════════════════════════

h_grid = np.linspace(0.10, 0.55, 50)
m_grid = np.linspace(0.3, 3.5, 50)
H, M = np.meshgrid(h_grid, m_grid)
X_grid = np.column_stack([H.ravel(), M.ravel()])
X_grid_scaled = scaler_X.transform(X_grid)

y_pred_scaled, y_std_scaled = gp.predict(X_grid_scaled, return_std=True)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_std = y_std_scaled * scaler_y.scale_[0]

Z_pred = y_pred.reshape(H.shape)
Z_std = y_std.reshape(H.shape)
Z_true = true_displacement(H, M)


# ═══════════════════════════════════════════════════════════════════════
# 6. FIGURAS
# ═══════════════════════════════════════════════════════════════════════

OUT = Path("C:/Seba/Tesis/data/figuras_ml")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.2,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ── Fig A: Superficie de predicción GP ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# Predicción GP
ax = axes[0]
cf = ax.contourf(H, M, Z_pred, levels=20, cmap='RdYlGn_r')
ax.scatter(dam_height, boulder_mass, c='black', s=15, edgecolors='white',
           linewidths=0.5, zorder=5, label='Datos SPH')
plt.colorbar(cf, ax=ax, label='Desplaz. [m]')
ax.set_xlabel('Altura columna $h$ [m]')
ax.set_ylabel('Masa boulder [kg]')
ax.set_title('(a) Predicción GP')
ax.legend(fontsize=8, loc='upper right')

# Incertidumbre (±2σ)
ax = axes[1]
cf2 = ax.contourf(H, M, Z_std * 2, levels=20, cmap='Oranges')
ax.scatter(dam_height, boulder_mass, c='black', s=15, edgecolors='white',
           linewidths=0.5, zorder=5)
plt.colorbar(cf2, ax=ax, label='Incertidumbre ±2σ [m]')
ax.set_xlabel('Altura columna $h$ [m]')
ax.set_ylabel('Masa boulder [kg]')
ax.set_title('(b) Intervalo de confianza 95%')

# Error absoluto vs función real
ax = axes[2]
error = np.abs(Z_pred - Z_true)
cf3 = ax.contourf(H, M, error, levels=20, cmap='Reds')
ax.scatter(dam_height, boulder_mass, c='black', s=15, edgecolors='white',
           linewidths=0.5, zorder=5)
plt.colorbar(cf3, ax=ax, label='Error absoluto [m]')
ax.set_xlabel('Altura columna $h$ [m]')
ax.set_ylabel('Masa boulder [kg]')
ax.set_title('(c) Error vs función real')

fig.suptitle('Gaussian Process Surrogate — Predicción de Desplazamiento del Boulder',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT / 'gp_surface_prediction.png')
fig.savefig(OUT / 'gp_surface_prediction.pdf')
print(f"\nOK: gp_surface_prediction")
plt.close(fig)


# ── Fig B: LOO Validation ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Predicted vs Actual
ax1.errorbar(y_train, y_pred_loo_real, yerr=2*y_std_loo_real,
             fmt='o', markersize=5, color='#2166AC', ecolor='#BBDEFB',
             elinewidth=1, capsize=2, alpha=0.8, label='Predicción ± 2σ')
lims = [min(y_train.min(), y_pred_loo_real.min()) - 0.3,
        max(y_train.max(), y_pred_loo_real.max()) + 0.3]
ax1.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='Perfecto')
ax1.set_xlabel('Desplazamiento real [m]')
ax1.set_ylabel('Desplazamiento predicho [m]')
ax1.set_title(f'(a) LOO: R² = {r2_loo:.3f}, RMSE = {rmse_loo:.3f} m')
ax1.legend(fontsize=8)
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_aspect('equal')

# Residuals
residuals = y_train - y_pred_loo_real
ax2.bar(range(N_TRAIN), residuals, color=['#EF5350' if r > 0 else '#42A5F5' for r in residuals],
        alpha=0.7, edgecolor='none')
ax2.axhline(0, color='black', lw=0.8)
ax2.axhline(2*noise_std, color='gray', ls='--', lw=0.7, label=f'±2σ ruido ({2*noise_std:.2f} m)')
ax2.axhline(-2*noise_std, color='gray', ls='--', lw=0.7)
ax2.set_xlabel('Índice de simulación')
ax2.set_ylabel('Residuo [m]')
ax2.set_title('(b) Residuos LOO')
ax2.legend(fontsize=8)

fig.suptitle('Validación Leave-One-Out del Gaussian Process',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT / 'gp_loo_validation.png')
fig.savefig(OUT / 'gp_loo_validation.pdf')
print(f"OK: gp_loo_validation")
plt.close(fig)


# ── Fig C: Corte 1D con intervalo de confianza ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Corte: masa fija = 1.5 kg, variar dam_height
m_fixed = 1.5
h_line = np.linspace(0.10, 0.55, 100)
X_cut1 = np.column_stack([h_line, np.full_like(h_line, m_fixed)])
X_cut1_s = scaler_X.transform(X_cut1)
y_cut1_s, std_cut1_s = gp.predict(X_cut1_s, return_std=True)
y_cut1 = scaler_y.inverse_transform(y_cut1_s.reshape(-1, 1)).ravel()
std_cut1 = std_cut1_s * scaler_y.scale_[0]

ax1.plot(h_line, y_cut1, color='#2166AC', lw=2, label='Predicción GP')
ax1.fill_between(h_line, y_cut1 - 2*std_cut1, y_cut1 + 2*std_cut1,
                 alpha=0.2, color='#2166AC', label='IC 95%')
ax1.plot(h_line, true_displacement(h_line, m_fixed), 'k--', lw=1, alpha=0.5, label='Función real')
# Puntos cercanos
mask_near = np.abs(boulder_mass - m_fixed) < 0.3
ax1.scatter(dam_height[mask_near], displacement[mask_near], c='red', s=30,
            zorder=5, edgecolors='white', linewidths=0.5, label='Datos cercanos')
ax1.set_xlabel('Altura columna $h$ [m]')
ax1.set_ylabel('Desplazamiento [m]')
ax1.set_title(f'(a) Masa fija = {m_fixed} kg')
ax1.legend(fontsize=7.5, loc='upper left')

# Corte: dam_height fija = 0.3 m, variar masa
h_fixed = 0.3
m_line = np.linspace(0.3, 3.5, 100)
X_cut2 = np.column_stack([np.full_like(m_line, h_fixed), m_line])
X_cut2_s = scaler_X.transform(X_cut2)
y_cut2_s, std_cut2_s = gp.predict(X_cut2_s, return_std=True)
y_cut2 = scaler_y.inverse_transform(y_cut2_s.reshape(-1, 1)).ravel()
std_cut2 = std_cut2_s * scaler_y.scale_[0]

ax2.plot(m_line, y_cut2, color='#B2182B', lw=2, label='Predicción GP')
ax2.fill_between(m_line, y_cut2 - 2*std_cut2, y_cut2 + 2*std_cut2,
                 alpha=0.2, color='#B2182B', label='IC 95%')
ax2.plot(m_line, true_displacement(h_fixed, m_line), 'k--', lw=1, alpha=0.5, label='Función real')
mask_near2 = np.abs(dam_height - h_fixed) < 0.05
ax2.scatter(boulder_mass[mask_near2], displacement[mask_near2], c='blue', s=30,
            zorder=5, edgecolors='white', linewidths=0.5, label='Datos cercanos')
ax2.set_xlabel('Masa boulder [kg]')
ax2.set_ylabel('Desplazamiento [m]')
ax2.set_title(f'(b) Altura fija = {h_fixed} m')
ax2.legend(fontsize=7.5, loc='upper right')

fig.suptitle('Cortes 1D del Gaussian Process con Intervalos de Confianza',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT / 'gp_1d_slices.png')
fig.savefig(OUT / 'gp_1d_slices.pdf')
print(f"OK: gp_1d_slices")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 7. RESUMEN
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  RESUMEN DEL GAUSSIAN PROCESS")
print(f"{'='*60}")
print(f"  Datos:    {N_TRAIN} simulaciones (sintéticas)")
print(f"  Inputs:   dam_height [m], boulder_mass [kg]")
print(f"  Output:   max_displacement [m]")
print(f"  Kernel:   {gp.kernel_}")
print(f"  R² LOO:   {r2_loo:.4f}")
print(f"  RMSE LOO: {rmse_loo:.4f} m")
print(f"  Figuras:  {OUT}")
print(f"{'='*60}")
print(f"\nPróximo paso: reemplazar datos sintéticos por results.sqlite")
print(f"cuando la campaña de producción en RTX 5090 esté completa.")
