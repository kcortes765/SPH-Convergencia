"""
AUDITOR FORENSE - Mision 1: Geometria del boulder BLIR3.stl
Analiza volumen, densidad implicita, y propiedades geometricas.
"""
import trimesh
import numpy as np

# --- Cargar STL original ---
mesh = trimesh.load("ENTREGA_KEVIN/BLIR3.stl")

print("=" * 60)
print("MISION FORENSE 1: GEOMETRIA DEL BOULDER")
print("=" * 60)

# Diagnostico de malla
print(f"\n--- STL ORIGINAL (sin escalar) ---")
print(f"  Vertices:        {len(mesh.vertices)}")
print(f"  Caras:           {len(mesh.faces)}")
print(f"  Es watertight:   {mesh.is_watertight}")
print(f"  Es manifold:     {mesh.is_volume}")
print(f"  Bounding box:    {mesh.bounds[0]} -> {mesh.bounds[1]}")
bb_orig = mesh.bounds[1] - mesh.bounds[0]
print(f"  Dimensiones:     {bb_orig[0]:.3f} x {bb_orig[1]:.3f} x {bb_orig[2]:.3f} (unidades STL)")
print(f"  Volumen (crudo): {mesh.volume:.6f} unidades^3")
print(f"  Centro de masa:  {mesh.center_mass}")

# --- Aplicar escala x0.04 (como en el XML de Diego) ---
scale = 0.04
mesh_scaled = mesh.copy()
mesh_scaled.apply_scale(scale)

print(f"\n--- STL ESCALADO x{scale} (como en el XML) ---")
bb_scaled = mesh_scaled.bounds[1] - mesh_scaled.bounds[0]
print(f"  Bounding box:    {mesh_scaled.bounds[0]} -> {mesh_scaled.bounds[1]}")
print(f"  Dimensiones:     {bb_scaled[0]:.4f} x {bb_scaled[1]:.4f} x {bb_scaled[2]:.4f} metros")
print(f"  Volumen:         {mesh_scaled.volume:.8f} m^3")
print(f"  Volumen (L):     {mesh_scaled.volume * 1000:.4f} litros")
print(f"  Centro de masa:  {mesh_scaled.center_mass}")

# --- Calculo de densidad implicita ---
masa_diego = 1.06053  # kg (del XML)
vol_m3 = mesh_scaled.volume
densidad = masa_diego / vol_m3

print(f"\n--- ANALISIS DE DENSIDAD ---")
print(f"  Masa impuesta:   {masa_diego} kg")
print(f"  Volumen real:    {vol_m3:.8f} m^3")
print(f"  Densidad = M/V:  {densidad:.1f} kg/m^3")
print()

# Contexto de densidades
print(f"  Referencia de densidades:")
print(f"    Agua:          1000 kg/m^3")
print(f"    PVC rigido:    1300-1450 kg/m^3")
print(f"    PVC espumado:  500-800 kg/m^3")
print(f"    Roca caliza:   2300-2700 kg/m^3")
print(f"    Roca granito:  2600-2800 kg/m^3")
print(f"    Roca basalto:  2800-3100 kg/m^3")
print()

# El comentario de Diego dice "800 kg/m^3"
densidad_diego_comment = 800
vol_implicito_comment = masa_diego / densidad_diego_comment
print(f"  Diego comenta '800 kg/m^3' en el XML:")
print(f"    Vol implicito:  {vol_implicito_comment:.8f} m^3")
print(f"    Vol trimesh:    {vol_m3:.8f} m^3")
print(f"    Ratio:          {vol_m3 / vol_implicito_comment:.3f}")

# --- Diametro equivalente ---
d_eq = (6 * vol_m3 / np.pi) ** (1/3)
print(f"\n--- DIAMETRO EQUIVALENTE ---")
print(f"  d_eq = (6V/pi)^(1/3) = {d_eq:.4f} m = {d_eq*100:.2f} cm")

# --- Inercia ---
print(f"\n--- TENSOR DE INERCIA (escalado, densidad uniforme) ---")
inertia = mesh_scaled.moment_inertia
# trimesh.moment_inertia asume densidad=1 (no 1000). Para masa real: I = I_trimesh * rho_real
inertia_density1 = inertia  # con densidad=1
inertia_real = inertia * densidad  # con densidad real

print(f"  Trimesh (densidad=1, crudo):")
for i in range(3):
    print(f"    [{inertia_density1[i][0]:.10f}, {inertia_density1[i][1]:.10f}, {inertia_density1[i][2]:.10f}]")

print(f"\n  Trimesh (densidad={densidad:.0f} kg/m3, corregido):")
for i in range(3):
    print(f"    [{inertia_real[i][0]:.8f}, {inertia_real[i][1]:.8f}, {inertia_real[i][2]:.8f}]")

# Verificacion cruzada: calcular inercia usando masa real directamente
# I = (M / M_trimesh) * I_trimesh_density1, donde M_trimesh = rho=1 * V
M_trimesh = 1.0 * mesh_scaled.volume  # masa con densidad=1
inertia_by_mass = inertia_density1 * (masa_diego / M_trimesh)
print(f"\n  Trimesh (escalado por masa real {masa_diego} kg):")
for i in range(3):
    print(f"    [{inertia_by_mass[i][0]:.8f}, {inertia_by_mass[i][1]:.8f}, {inertia_by_mass[i][2]:.8f}]")

# Inercia de Diego (del XML generado por GenCase)
print(f"\n  Diego (del XML generado):")
print(f"    [0.00405562,  0.00020416, -7.44909e-05]")
print(f"    [0.00020416,  0.0047619,  -1.93125e-05]")
print(f"    [-7.44909e-05, -1.93125e-05, 0.00749323]")

# --- Particulas a distintos dp ---
print(f"\n--- ESTIMACION DE PARTICULAS POR dp ---")
for dp in [0.05, 0.02, 0.01, 0.005, 0.002]:
    vol_particula = dp ** 3
    n_approx = vol_m3 / vol_particula
    n_dim_min = min(bb_scaled) / dp  # particulas en la dimension mas chica
    print(f"  dp={dp:.3f}m: ~{n_approx:.0f} particulas, dim_min/dp={n_dim_min:.1f}")

print(f"\n  Diego con dp=0.05: 31 particulas (del XML generado)")
print(f"  Regla practica: minimo ~10 particulas en la dimension menor del solido")
print(f"  Dimension menor del boulder: {min(bb_scaled):.4f} m")
print(f"  dp necesario para 10 part en dim_min: {min(bb_scaled)/10:.4f} m")
