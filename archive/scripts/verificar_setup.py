"""
verificar_setup.py — Correr ANTES del estudio de convergencia.

Verifica que el PC tenga todo lo necesario:
  1. Python + dependencias
  2. Ejecutables de DualSPHysics
  3. Archivos del proyecto (STLs, XMLs, config)

Uso: python verificar_setup.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ok_count = 0
fail_count = 0


def check(label, condition, detail=""):
    global ok_count, fail_count
    if condition:
        print(f"  [OK] {label}")
        ok_count += 1
    else:
        print(f"  [!!] {label} — {detail}")
        fail_count += 1


print(f"Python: {sys.version}")
print(f"Proyecto: {PROJECT_ROOT}\n")

# --- 1. Dependencias Python ---
print("=== DEPENDENCIAS PYTHON ===")
for mod in ['numpy', 'pandas', 'trimesh', 'lxml', 'scipy', 'sqlalchemy']:
    try:
        __import__(mod)
        check(mod, True)
    except ImportError:
        check(mod, False, f"pip install {mod}")

# --- 2. Config ---
print("\n=== CONFIGURACION ===")
config_path = PROJECT_ROOT / 'config' / 'dsph_config.json'
check("dsph_config.json", config_path.exists(), "Falta archivo de config")

if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)

    dsph_bin = Path(config['dsph_bin'])
    check(f"dsph_bin: {dsph_bin}", dsph_bin.exists(),
          "Directorio de ejecutables no encontrado")

    # Ejecutables criticos
    print("\n=== EJECUTABLES DUALSPHYSICS ===")
    for name in ['gencase', 'dualsphysics_gpu']:
        exe = dsph_bin / config['executables'][name]
        check(config['executables'][name], exe.exists(),
              f"No encontrado en {exe}")

# --- 3. Archivos del proyecto ---
print("\n=== ARCHIVOS DEL PROYECTO ===")
files_needed = [
    'config/template_base.xml',
    'config/Floating_Materials.xml',
    'models/BLIR3.stl',
    'models/Canal_Playa_1esa20_750cm.stl',
    'src/geometry_builder.py',
    'src/batch_runner.py',
    'src/data_cleaner.py',
    'run_convergence.py',
]
for f in files_needed:
    fp = PROJECT_ROOT / f
    check(f, fp.exists(), "FALTA")

# --- 4. GPU ---
print("\n=== GPU ===")
try:
    import subprocess
    r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                        '--format=csv,noheader'],
                       capture_output=True, text=True, timeout=5)
    if r.returncode == 0:
        print(f"  [OK] {r.stdout.strip()}")
        ok_count += 1
    else:
        check("nvidia-smi", False, "No se pudo consultar GPU")
except FileNotFoundError:
    check("nvidia-smi", False, "nvidia-smi no encontrado en PATH")

# --- Resumen ---
print(f"\n{'='*40}")
if fail_count == 0:
    print(f"TODO OK ({ok_count} checks). Listo para correr:")
    print(f"  python run_convergence.py")
else:
    print(f"{fail_count} PROBLEMAS encontrados. Resolver antes de correr.")
