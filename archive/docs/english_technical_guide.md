# Technical English Guide — SPH-IncipientMotion
## Interview Prep for MSc Applications (Canada)

---

## A. Core Vocabulary

### SPH & Simulation
| Term | Pronunciation | Meaning |
|------|--------------|---------|
| Smoothed Particle Hydrodynamics (SPH) | "ess-pee-aitch" | Meshless method where fluid is represented by particles |
| Particle spacing (dp) | "dee-pee" | Distance between adjacent particles (resolution) |
| Kernel function | "KER-nel" | Weighting function for particle interactions (Wendland, Cubic) |
| Smoothing length (h) | "aitch" | Support radius of the kernel (h = coefh * sqrt(3*dp^2)) |
| CFL condition | "see-eff-ell" | Courant-Friedrichs-Lewy: stability constraint on time step |
| Symplectic integrator | "sim-PLEC-tic" | Time integration scheme (energy-conserving) |
| Weakly Compressible SPH (WCSPH) | "double-you-see-ess-pee-aitch" | SPH variant using equation of state for pressure |
| Delta-SPH | "DELL-tah ess-pee-aitch" | Density diffusion term to reduce noise (Fourtakas) |
| Artificial viscosity | "ar-tih-FISH-al vis-KOSS-ih-tee" | Numerical dissipation to stabilize the scheme |

### Rigid Body Dynamics
| Term | Pronunciation | Meaning |
|------|--------------|---------|
| ProjectChrono | "pro-JECT KROH-no" | Physics engine for rigid body dynamics |
| Fluid-Structure Interaction (FSI) | "eff-ess-eye" | Coupling between fluid and solid bodies |
| Incipient motion | "in-SIP-ee-ent" | Threshold at which a body starts to move |
| Boulder transport | "BOWL-der" | Movement of large rocks by fluid forces |
| Inertia tensor | "in-ER-sha TEN-sor" | 3x3 matrix describing resistance to rotation |
| Mesh convergence | "konVER-jens" | Solution stabilizes as resolution increases |
| GCI (Grid Convergence Index) | "jee-see-eye" | Formal uncertainty estimate from Richardson extrapolation |

### Machine Learning
| Term | Pronunciation | Meaning |
|------|--------------|---------|
| Gaussian Process (GP) | "GOW-see-an" | Bayesian non-parametric regression model |
| Surrogate model | "SUR-oh-git" | Fast approximation trained on expensive simulations |
| Latin Hypercube Sampling (LHS) | "LAT-in HY-per-kyoob" | Space-filling experimental design method |
| Matern kernel | "mah-TERN" | Covariance function with adjustable smoothness (nu) |
| Confidence interval | "KON-fih-dens" | Range where prediction lies with given probability |

---

## B. Interview Questions & Model Answers

### Q1: "Can you describe your thesis in 2-3 sentences?"

**Answer:**
"My thesis uses GPU-accelerated Smoothed Particle Hydrodynamics to simulate tsunami-type flows impacting coastal boulders. I built an automated pipeline in Python that generates hundreds of simulation cases, extracts kinematics from the Chrono physics engine, and trains a Gaussian Process surrogate model to predict boulder stability without running every simulation. The goal is to establish critical movement thresholds — incipient motion criteria — for irregular boulders under different wave conditions."

### Q2: "Why SPH instead of mesh-based methods like OpenFOAM?"

**Answer:**
"SPH is naturally suited for this problem because of the violent free-surface flow with large deformations — the dam-break wave creates splash, fragmentation, and complex topology changes that would require expensive mesh adaptation in Eulerian methods. SPH handles this inherently since particles carry the fluid properties without needing a mesh. Additionally, the coupling with ProjectChrono for rigid body dynamics is straightforward in DualSPHysics. The trade-off is that SPH is more computationally expensive per particle than a mesh cell, but GPU acceleration makes it feasible — my convergence study ran on an RTX 5090 with up to 26 million particles."

### Q3: "How did you validate your simulations?"

**Answer:**
"I performed a formal mesh convergence study with 7 particle resolutions, from dp=0.020 down to dp=0.003 meters. I tracked five metrics: boulder displacement, rotation, velocity, hydrodynamic force, and contact force. Displacement converged to within 6.4% between the two finest resolutions, and SPH forces converged to within 1%. I found that contact forces from the Chrono NSC solver do NOT converge monotonically — I identified five mechanisms causing this, including kernel truncation effects and aliasing of impulsive events. This is actually a known limitation in coupled SPH-DEM systems, and I argue in my thesis that displacement is the physically meaningful convergence metric, not peak contact force."

### Q4: "Explain Latin Hypercube Sampling and why you chose it."

**Answer:**
"Latin Hypercube Sampling is a stratified sampling technique that ensures each input variable is evenly represented across its range. Unlike random sampling, LHS divides each dimension into N equal intervals and places exactly one sample per interval, then shuffles them. This gives better space coverage with fewer samples — critical when each simulation takes hours on a GPU. I used scipy's quasi-Monte Carlo implementation with a fixed seed for reproducibility. For my parametric study, the variables are dam height, boulder mass, and boulder orientation."

### Q5: "How does the Gaussian Process surrogate work?"

**Answer:**
"A Gaussian Process is a Bayesian regression model that provides not just a prediction but also an uncertainty estimate — a confidence interval around every prediction. I use a Matern kernel with nu=2.5, which is appropriate for functions that are twice differentiable, as we expect from physical systems. The GP is trained on the simulation results stored in an SQLite database: inputs are wave height, boulder mass, and shape descriptors; output is maximum displacement. In my prototype with synthetic data, the GP achieved R-squared of 0.999 with Leave-One-Out cross-validation. The key advantage over neural networks is that with small datasets — typical in computational physics where each sample costs hours of GPU time — GPs outperform deep learning and provide calibrated uncertainty."

### Q6: "What were the main challenges?"

**Answer:**
"Three main challenges. First, I inherited a simulation setup from a previous researcher that had several errors — incorrect density, missing gravity settling period, and inadequate resolution. I conducted a forensic audit and corrected everything based on trimesh geometry calculations. Second, the contact forces from the Chrono solver don't converge with mesh refinement — I had to diagnose this and justify why displacement is the correct convergence metric. Third, automating the pipeline end-to-end required handling DualSPHysics's CSV format quirks — semicolon separators, sentinel values, and coordinate frame conventions that aren't well documented."

### Q7: "What would you do differently or improve?"

**Answer:**
"Three things: First, I would add an adaptive resolution approach — using variable particle spacing to concentrate resolution near the boulder while keeping coarse particles far away. DualSPHysics doesn't support this natively, but frameworks like SPHinXsys do. Second, I would replace the WCSPH formulation with an incompressible SPH variant to eliminate the pressure oscillations that affect force measurements. Third, for the surrogate model, I would explore Bayesian Optimization to actively choose which simulations to run next, rather than a static Latin Hypercube design — this would converge faster to the critical threshold with fewer expensive simulations."

---

## C. Key Phrases for Academic Emails

**Cold email to a PI:**
- "I am completing my undergraduate thesis on GPU-accelerated SPH simulation of boulder transport under tsunami loading."
- "My work combines computational fluid dynamics with machine learning surrogate modeling."
- "I have hands-on experience with high-performance GPU computing (RTX 5090, 32GB VRAM) and automated simulation pipelines in Python."
- "I am particularly interested in your research on [specific topic from their publications]."

**Describing your skills:**
- "Proficient in Python, DualSPHysics, ProjectChrono, and scientific data analysis."
- "Experience with Latin Hypercube Sampling, Gaussian Process regression, and mesh convergence analysis."
- "Built end-to-end automated pipelines: geometry generation, batch GPU simulation, ETL, and machine learning."

---

## D. Practice Exercise

Record yourself answering Q1, Q2, and Q5 out loud. Focus on:
1. Pronouncing "SPH", "Gaussian", "Chrono", "incipient", "surrogate" correctly
2. Speaking in complete sentences, not bullet points
3. Keeping each answer under 90 seconds
4. Sounding confident but not arrogant — use "I built", "I found", "I identified"

Then ask Claude or ChatGPT Voice to play the role of a UBC professor and quiz you.
