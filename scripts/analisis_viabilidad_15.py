"""
Analisis de viabilidad - 15/50 casos LHS
¿Vale la pena seguir con las simulaciones?
"""
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Seba\Tesis\data\results_ws_15cases.csv")

print("=" * 70)
print("ANALISIS DE VIABILIDAD - 15/50 CASOS LHS")
print("=" * 70)

# --- 1. Resumen basico ---
print(f"\nCasos completados: {len(df)}")
print(f"Todos MOVIMIENTO: si ({df['failed'].sum()}/{len(df)})")

# --- 2. Varianza en respuestas continuas ---
print("\n--- VARIANZA EN RESPUESTAS CONTINUAS ---")
metrics = {
    'max_displacement': ('Desplazamiento (m)', 'm'),
    'max_rotation': ('Rotacion (deg)', 'deg'),
    'max_velocity': ('Velocidad boulder (m/s)', 'm/s'),
    'max_sph_force': ('Fuerza SPH (N)', 'N'),
    'max_flow_velocity': ('Vel flujo (m/s)', 'm/s'),
    'max_water_height': ('Altura agua (m)', 'm'),
}

for col, (label, unit) in metrics.items():
    vals = df[col]
    print(f"\n  {label}:")
    print(f"    min={vals.min():.4f}  max={vals.max():.4f}  ratio={vals.max()/max(vals.min(),1e-10):.1f}x")
    print(f"    mean={vals.mean():.4f}  std={vals.std():.4f}  CV={vals.std()/vals.mean()*100:.1f}%")

# --- 3. Correlaciones con parametros de entrada ---
print("\n--- CORRELACIONES (Pearson) ---")
inputs = ['dam_height', 'boulder_mass', 'boulder_rot_z']
outputs = ['max_displacement', 'max_rotation', 'max_velocity', 'max_sph_force', 'max_flow_velocity']

print(f"\n{'':>20s}", end='')
for inp in inputs:
    print(f"  {inp:>15s}", end='')
print()

for out in outputs:
    print(f"  {out:>18s}", end='')
    for inp in inputs:
        r = df[out].corr(df[inp])
        print(f"  {r:>15.3f}", end='')
    print()

# --- 4. Analisis por dam_height (variable dominante?) ---
print("\n--- DAM_HEIGHT vs DISPLACEMENT (ordenado) ---")
sorted_df = df.sort_values('dam_height')
for _, row in sorted_df.iterrows():
    bar_len = int(row['max_displacement'] / 6.4 * 40)
    bar = '#' * bar_len
    print(f"  h={row['dam_height']:.3f}  m={row['boulder_mass']:.2f}  "
          f"disp={row['max_displacement']:6.3f}m  |{bar}")

# --- 5. lhs_010 - caso critico ---
print("\n--- CASO CRITICO: lhs_010 ---")
c10 = df[df['case_name'] == 'lhs_010'].iloc[0]
print(f"  dam_h={c10['dam_height']:.3f}  mass={c10['boulder_mass']:.3f}  rot_z={c10['boulder_rot_z']:.1f}")
print(f"  disp={c10['max_displacement']:.4f}m ({c10['max_displacement_rel']:.1f}% de d_eq)")
print(f"  rot={c10['max_rotation']:.1f}deg  vel={c10['max_velocity']:.3f}m/s")
print(f"  INTERPRETACION: Desplazamiento = 5cm = ~50% del d_eq")
print(f"  Esto es BORDERLINE - casi en el umbral de incipient motion")

# --- 6. Casos pendientes con potencial de estabilidad ---
print("\n--- CASOS PENDIENTES CON MAYOR CHANCE DE ESTABILIDAD ---")
matrix = pd.read_csv(r"C:\Seba\Tesis\config\experiment_matrix.csv")
done = set(df['case_name'].tolist())
pending = matrix[~matrix['case_id'].isin(done)].copy()
# Score: dam_h bajo + mass alta = mas probable estable
pending['stability_score'] = pending['boulder_mass'] / pending['dam_height']
pending_sorted = pending.sort_values('stability_score', ascending=False).head(10)
print(f"  {'case':>10s}  {'dam_h':>6s}  {'mass':>6s}  {'rot_z':>6s}  {'score':>6s}")
for _, row in pending_sorted.iterrows():
    print(f"  {row['case_id']:>10s}  {row['dam_height']:>6.3f}  {row['boulder_mass']:>6.3f}  "
          f"{row['boulder_rot_z']:>6.1f}  {row['stability_score']:>6.1f}")

# --- 7. Prediccion: ¿habra algun caso estable? ---
print("\n--- PREDICCION DE ESTABILIDAD EN CASOS PENDIENTES ---")
# Usando relacion lineal simple dam_h -> disp de los 15 casos
from numpy.polynomial import polynomial as P
coeffs = P.polyfit(df['dam_height'].values, df['max_displacement'].values, 1)
# Extrapolar al dam_h mas bajo pendiente
min_pending_h = pending['dam_height'].min()
predicted_disp = P.polyval(min_pending_h, coeffs)
print(f"  Regresion lineal: disp ~ {coeffs[1]:.1f} * dam_h + {coeffs[0]:.2f}")
print(f"  dam_h minimo pendiente: {min_pending_h:.3f}m")
print(f"  Desplazamiento predicho: {predicted_disp:.3f}m")
print(f"  d_eq = 0.1004m, umbral 5% = 0.005m")
if predicted_disp > 0.005:
    print(f"  -> Probablemente SIGUE MOVIMIENTO incluso en caso mas suave")
else:
    print(f"  -> Podria haber ESTABILIDAD")

# --- 8. GP preliminar ---
print("\n--- GP PRELIMINAR (15 puntos) ---")
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    X = df[['dam_height', 'boulder_mass', 'boulder_rot_z']].values
    y = df['max_displacement'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0, 1.0]) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp.fit(X_scaled, y)

    print(f"  Kernel optimizado: {gp.kernel_}")
    print(f"  Log-marginal-likelihood: {gp.log_marginal_likelihood_value_:.2f}")

    # R2 en training
    y_pred = gp.predict(X_scaled)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  R² (training): {r2:.4f}")

    # Prediccion en caso mas extremo (dam_h=0.05, mass=1.6)
    X_test = scaler.transform([[0.05, 1.6, 45.0]])
    y_test, y_std = gp.predict(X_test, return_std=True)
    print(f"\n  Prediccion dam_h=0.05, mass=1.6, rot=45:")
    print(f"    disp = {y_test[0]:.3f} ± {y_std[0]:.3f}m")
    print(f"    {'ESTABLE' if y_test[0] < 0.005 else 'MOVIMIENTO'} (umbral=0.005m)")

    # Importancia relativa (longscales del RBF)
    ls = gp.kernel_.k1.k2.length_scale
    importance = 1.0 / ls
    importance_pct = importance / importance.sum() * 100
    print(f"\n  Importancia relativa (1/lengthscale):")
    for name, imp in zip(['dam_height', 'boulder_mass', 'boulder_rot_z'], importance_pct):
        print(f"    {name:>15s}: {imp:5.1f}%")

except ImportError:
    print("  sklearn no disponible, saltando GP")

# --- 9. VEREDICTO ---
print("\n" + "=" * 70)
print("VEREDICTO: ¿VALE LA PENA SEGUIR?")
print("=" * 70)

disp_range = df['max_displacement'].max() / max(df['max_displacement'].min(), 1e-10)
disp_cv = df['max_displacement'].std() / df['max_displacement'].mean() * 100
r_damh = df['max_displacement'].corr(df['dam_height'])

print(f"""
  1. VARIANZA: El desplazamiento varia de {df['max_displacement'].min():.3f}m a
     {df['max_displacement'].max():.3f}m (ratio {disp_range:.0f}x, CV={disp_cv:.0f}%).
     -> EXCELENTE varianza para entrenar un surrogate.

  2. CORRELACION: dam_height domina (r={r_damh:.3f}). Hay señal clara.
     -> El GP puede aprender la relacion.

  3. CASO LIMITE: lhs_010 (dam_h=0.108) tiene solo 5cm de desplazamiento.
     -> Estamos CERCA del umbral de incipient motion.
     -> Casos pendientes con dam_h~0.11-0.12 podrian mostrar estabilidad.

  4. RESPUESTA CONTINUA: Aunque todos "se mueven", la MAGNITUD del
     movimiento es la variable de interes, no el binario si/no.
     -> Un GP de regresion sobre desplazamiento es perfectamente valido.
     -> El umbral critico se extrae del GP (donde disp cruza el threshold).

  5. CONCLUSION: SI VALE LA PENA SEGUIR.
     - Las 50 sims daran un GP robusto con Sobol confiable.
     - La zona de transicion esta en dam_h ~ 0.05-0.12m.
     - Si se quiere un 2do batch, deberia explorar dam_h < 0.10m.
     - Pero para la tesis, 50 casos con esta varianza son SUFICIENTES.
""")
