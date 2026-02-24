# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SPH-IncipientMotion** — Academic capstone research (UCN 2026) to numerically determine critical movement thresholds of coastal boulders under tsunami-type flows using Smoothed Particle Hydrodynamics (SPH).

The goal is to build an automated **Hydraulic Data Refinery**: a Python pipeline that takes simulation parameters, generates DualSPHysics cases, runs them in batch, extracts kinematics, and trains a Gaussian Process surrogate model to predict boulder stability without simulating every scenario.

See `PLAN.md` for the full implementation plan with unknowns, risks, and decisions pending.
See `EDUCACION_TECNICA.md` for detailed explanations of SPH physics, DualSPHysics parameters, XML structure, and project concepts.

**Scope:** This capstone is 100% numerical simulation — no physical laboratory work. The "canal" and "tank" are computational domains defined in XML, not physical structures.

## Development Environment

**Python 3.10+ required.** Core dependencies:
```bash
pip install numpy pandas trimesh lxml scipy sqlalchemy scikit-learn matplotlib
```

**External engine:** DualSPHysics **v5.4.355** (confirmed from workstation Run.csv). GenCase **v5.4.354.01**. A v6.0 beta exists (early 2026) — **DO NOT USE** for this thesis. Python calls `.exe` files via `subprocess.run()`. DualSPHysics has **no native Python API** — all interaction is CLI-based.

**CRITICAL — ProjectChrono Integration:** The simulation uses `RigidAlgorithm=3` (Chrono), NOT basic SPH floating bodies. This means:
- Chrono handles rigid body dynamics, collisions, and friction
- CSVs are generated AUTOMATICALLY during simulation (no post-processing needed)
- **FloatingInfo and ComputeForces are NOT used** — Chrono outputs replace them entirely

Key executables:
- `GenCase_win64.exe` — XML → particle geometry (.bi4)
- `DualSPHysics5.4_win64.exe` — GPU-accelerated SPH solver + Chrono coupling
- Other post-processing tools (PartVTK, BoundaryVTK, etc.) are NOT needed for the pipeline

**Primary data sources (generated automatically by Chrono during simulation):**
- `ChronoExchange_mkbound_51.csv` — boulder kinematics (position, velocity, angular velocity, accelerations). Separator: `;`
- `ChronoBody_forces.csv` — SPH forces + contact forces per body. Separator: `;`
- `GaugesVel_V**.csv` — flow velocity at fixed gauge points. Separator: `;`
- `GaugesMaxZ_hmax**.csv` — max water height at gauge points. Separator: `;`
- `Run.csv` / `RunPARTs.csv` — simulation metadata

**Hardware contexts:**
- Development (laptop i7-14650HX + RTX 4060 8GB): geometry scripts, low-resolution tests, ETL pipeline
- Production (workstation RTX 5090 32GB VRAM): high-fidelity batch runs (20M+ particles)

## Forensic Audit Findings (2026-02-20)

See `AUDITORIA_FISICA.md` for the full report. Key corrections applied:
- **Inertia:** GenCase overestimates by 2-3x at coarse dp. Pipeline injects trimesh-computed inertia via `<inertia>` XML tag.
- **Density:** Diego comments 800 kg/m³ but real density is 2000 kg/m³ (he used bounding box volume instead of mesh volume).
- **FtPause:** Set to 0.5s (Diego had 0.0 — no gravity settling before wave impact).
- **dp=0.05 is unacceptable** (31 particles, <1 in min dimension). Minimum dp ≤ 0.004m for 10 particles in thinnest axis.

## Architecture: 4-Module Pipeline

```
Geometry Builder → Batch Runner → ETL / Data Cleaner → ML Surrogate
(STL+XML→N XMLs)   (GenCase→DSPH→Post)  (CSV→SQLite)      (GP Regression)
```

### Module 1 — `geometry_builder.py`

Takes an irregular `.stl` boulder + DualSPHysics XML template. Uses `trimesh` to compute center of mass, bounding box, volume, and inertia tensor. Injects into XML:
- `<drawfilestl>` with optional `<drawscale>`, `<drawmove>`, `<drawrotate>` children (transformations applied in-place, no need to generate separate rotated STL files)
- `<fillbox>` with `<modefill>void</modefill>` to solve the hollow shell problem (seed point = centroid, size = bounding box)
- `<floatings>` section with `<massbody>`, `<center>`, `<inertia>` (3x3 tensor)

**Critical DualSPHysics constraints:**
- STL imports (`drawfilestl`) only create surface particles — interior is hollow by default. The `fillbox void` fill is mandatory.
- `autofill="true"` on `drawfilestl` is unreliable for complex geometry. Always use explicit `fillbox`.
- The fillbox seed point must be inside the STL mesh, at least 2h from boundary particles (h = coefh * sqrt(3 * dp^2) in 3D).
- If filling fails, likely cause is inverted STL normals. Fix: Blender → Edit Mode → F3 → "Flip Normals".
- Always use `<massbody value="X"/>` (explicit mass in kg) instead of `<rhopbody>` (density) because particle discretization introduces volume errors.
- GenCase auto-computes inertia from particles; override with `<inertia>` only if trimesh values are more accurate.

### Module 2 — `batch_runner.py`

Execution chain per case (simplified — no post-processing tools needed):
```
GenCase Case_Def  outdir/Case  -save:all
  → DualSPHysics5.4_win64.exe  -gpu  outdir/Case  outdir
  → (Chrono CSVs + Gauge CSVs generated automatically during simulation)
  → cleanup .bi4 files in try/finally
```

**FloatingInfo and ComputeForces are NOT used.** Chrono outputs replace them.

Key DualSPHysics CLI flags:
- `-gpu[:id]` — GPU mode (required for production)
- `-symplectic` — Symplectic time integration (preferred over Verlet)
- `-wendland` — Wendland kernel (preferred for floating bodies)
- `-viscoart:0.1` — artificial viscosity
- `-deltasph:0.1` — Delta-SPH density diffusion
- `-posdouble:1` — double precision positions (recommended for large domains)
- `-partbegin:N dir` — restart from checkpoint Part000N.bi4 (critical for crash recovery)
- `FtPause` XML parameter — freezes floating bodies for N seconds at start (lets fluid settle)
- `RigidAlgorithm` XML parameter — 0=collision-free, 1=SPH, 2=DEM, 3=Chrono

**Disk cleanup is critical** — 10M particles can generate ~64 GB per simulation. Cleanup of `.bi4`/`.vtk` MUST be in a `try/finally` block to guarantee deletion even on crash. Verify output CSVs exist before deleting binaries.

Watchdog: `subprocess.run(timeout=...)` + try/except to skip crashed cases and continue batch.

### Module 3 — `data_cleaner.py` (ETL)

**Primary data source is `ChronoExchange_mkbound_51.csv`**, NOT FloatingInfo. Chrono outputs kinematics (position, velocity, angular velocity, accelerations) and forces automatically during simulation.

**CSV format (DualSPHysics Chrono + Gauges):**
- Separator: **semicolon (`;`)** — hardcoded. `-csvsep:1` flag does NOT apply to Chrono outputs.
- Decimal: **point (`.`)** always
- Precision: 6 significant digits (`%g` format)
- Headers include units in brackets: `time [s]`, `fvel.x [m/s]`, etc.
- Sentinel value `-3.40282e+38` (float min) in Gauges = no data → replace with NaN
- Read with: `pd.read_csv(path, sep=';')`

**Failure criterion** (thresholds pending academic validation with Dr. Moris):
- Displacement of center of mass > X% of equivalent diameter (d_eq = (6V/pi)^(1/3))
- Net rotation > Y degrees
- The exact percentages and whether velocity-based criteria are needed is TBD.

**Storage cleanup:** Delete `.bi4` and `.vtk` files only AFTER verifying that FloatingInfo/ComputeForces CSVs exist and contain valid data. Output sizes: ~32 bytes/particle/snapshot (1M particles × 200 snapshots ≈ 6.4 GB per simulation).

### Module 4 — `ml_surrogate.py`

Trains `GaussianProcessRegressor` (scikit-learn) on SQLite results. Inputs: mass, wave height, angle, shape descriptors → Outputs: movement yes/no, max displacement/force. Shape encoded as geometric descriptors (axis ratios, Wadell sphericity), not raw geometry. This module is last to implement — requires real simulation data.

### `main_orchestrator.py` — Entry Point

Reads experiment matrix CSV (generated via `scipy.stats.qmc.LatinHypercube`), loops through all cases, runs the full pipeline, stores results in SQLite, cleans up.

## DualSPHysics XML Structure Reference

```xml
<case>
  <casedef>
    <constantsdef>
      <gravity x="0" y="0" z="-9.81" />
      <rhop0 value="1000" />        <!-- fluid reference density kg/m3 -->
      <cflnumber value="0.2" />
      <coefh value="1.0" />          <!-- h = coefh * sqrt(3*dp^2) in 3D -->
    </constantsdef>
    <mkconfig boundcount="240" fluidcount="10" />
    <geometry>
      <definition dp="0.0085">       <!-- inter-particle distance (m) -->
        <pointmin x="..." y="..." z="..." />
        <pointmax x="..." y="..." z="..." />
      </definition>
      <commands>
        <mainlist>
          <!-- Tank boundaries -->
          <setmkbound mk="0" />
          <drawbox><boxfill>bottom|left|right|front|back</boxfill>...</drawbox>

          <!-- Fluid (dam break column) -->
          <setmkfluid mk="0" />
          <fillbox x="seed_x" y="seed_y" z="seed_z">
            <modefill>void</modefill>
            <point x="min_x" y="min_y" z="min_z" />
            <size x="dx" y="dy" z="dz" />
          </fillbox>

          <!-- Boulder from STL -->
          <setmkbound mk="1" />
          <drawfilestl file="boulder.stl">
            <drawscale x="0.001" y="0.001" z="0.001" />  <!-- mm→m if needed -->
            <drawmove x="px" y="py" z="pz" />
            <drawrotate angx="0" angy="0" angz="45" />
          </drawfilestl>

          <!-- Fill boulder interior -->
          <fillbox x="cx" y="cy" z="cz">
            <modefill>void</modefill>
            <point x="xmin" y="ymin" z="zmin" />
            <size x="width" y="height" z="depth" />
          </fillbox>
        </mainlist>
      </commands>
    </geometry>
    <floatings>
      <floating mkbound="1">
        <massbody value="150.0" />    <!-- real mass in kg (NOT rhopbody) -->
        <center x="cx" y="cy" z="cz" />
        <inertia>
          <values v11="Ixx" v12="0" v13="0" />
          <values v21="0" v22="Iyy" v23="0" />
          <values v31="0" v32="0" v33="Izz" />
        </inertia>
      </floating>
    </floatings>
  </casedef>
  <execution>
    <parameters>
      <parameter key="StepAlgorithm" value="2" />    <!-- 1=Verlet, 2=Symplectic -->
      <parameter key="Kernel" value="2" />            <!-- 1=Cubic, 2=Wendland -->
      <parameter key="ViscoTreatment" value="1" />    <!-- 1=Artificial -->
      <parameter key="Visco" value="0.1" />
      <parameter key="DensityDT" value="2" />         <!-- 0=None, 1=Molteni, 2=Fourtakas -->
      <parameter key="RigidAlgorithm" value="3" />    <!-- 0=free, 1=SPH, 2=DEM, 3=Chrono (MUST be 3) -->
      <parameter key="FtPause" value="0.0" />         <!-- freeze floatings N sec -->
      <parameter key="TimeMax" value="2.0" />
      <parameter key="TimeOut" value="0.01" />
    </parameters>
  </execution>
</case>
```

## Key Design Rules

1. **No hardcoded XML** — all case variations injected procedurally via `lxml` from a template.
2. **Hollow shell fix is mandatory** — every STL import needs `fillbox void` + `massbody`.
3. **Failure criterion is algorithmic** — no visual inspection in ParaView. Python reads FloatingInfo CSV and applies numerical thresholds.
4. **Bypass Excel entirely** — DualSPHysics (with `-csvsep:1`) → Pandas → SQLite.
5. **Academic validity** — SPH parameters (h, CFL, viscosity, dp) must be justified. A mesh convergence study (varying dp) is required before the parametric sweep.
6. **Use ChronoExchange CSV for boulder kinematics**, not FloatingInfo. Chrono outputs are richer and generated automatically.
7. **Use `massbody` not `rhopbody`** — particle discretization makes density-based mass inaccurate.
8. **Inject inertia from trimesh** — GenCase overestimates inertia at coarse dp (2-3x error at dp=0.05). Always use `<inertia>` tag.
9. **Disk cleanup in `try/finally`** — binary deletion must be guaranteed even on error. Never leave `.bi4` files undeleted.
10. **No v6.0 beta** — stick to v5.4.x stable for the thesis.
11. **FtPause >= 0.5** — always allow gravity settling before wave impact.

## Starter Files Needed from Diego

The project depends on inheriting these base files from the previous researcher:
- `template_base.xml` — working DualSPHysics case definition
- `boulder_irregular.stl` — 3D model of the test boulder
- `run_command.bat` — exact CLI commands and executable names
- `sample_output.csv` — real FloatingInfo/MeasureTool output to validate column names

**Do not write the geometry builder or XML parser without inspecting the real `.xml` and `.stl` first.** Use DualSPHysics example cases (01_DamBreak, 07_DambreakCubes/CaseSolidsCHRONO) as temporary stand-ins for development.

[DesignSPHysics](https://github.com/DualSPHysics/DesignSPHysics) (official FreeCAD plugin, written in Python) can be used as reference for XML generation patterns — not as a runtime dependency.

## Planned Directory Structure

```
Tesis/
├── main_orchestrator.py
├── geometry_builder.py
├── batch_runner.py
├── data_cleaner.py
├── ml_surrogate.py
├── config/
│   ├── template_base.xml       # Base DualSPHysics case (from Diego)
│   └── experiment_matrix.csv   # LHS-generated parameter cases
├── data/
│   ├── raw/                    # Simulation outputs (temporary, deleted after ETL)
│   ├── processed/              # Cleaned CSVs from FloatingInfo/ComputeForces
│   └── results.sqlite          # Failure/success results DB
├── models/
│   └── boulder_irregular.stl   # Source 3D geometry
├── PLAN.md                     # Full implementation plan, risks, unknowns
└── CLAUDE.md                   # This file
```
