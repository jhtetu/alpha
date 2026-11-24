# The Known Laws of the Universe

A catalog of the established laws of nature, organized by field. Each law has a plain-language statement and, where useful, its equation.

**Scope.** This file covers the fundamental laws and the major established laws of physics, chemistry and biology. Science also has thousands of narrower empirical rules, so no list can be truly complete.

**Tiers.** Each law is tagged:
- **F (Fundamental):** as far as we know, holds everywhere and always.
- **A (Approximation):** accurate within its range and replaced by a deeper law outside it, the way Newton's gravity gives way to Einstein's.
- **E (Empirical):** a pattern observed to hold reliably, without a deeper derivation.

---

## 1. Symmetries and Conservation Laws

These are the deepest "unchanging laws." Noether's theorem (1918) shows that every continuous symmetry of nature produces a conserved quantity.

| Law | Tier | Statement | Comes from |
|---|---|---|---|
| Noether's theorem | F | Every continuous symmetry corresponds to a conserved quantity | Mathematics of physical laws |
| Conservation of energy | F | Total energy of a closed system never changes | Laws are the same at all times |
| Conservation of linear momentum | F | Total momentum of a closed system never changes | Laws are the same at all places |
| Conservation of angular momentum | F | Total spin and rotation of a closed system never changes | Laws are the same in all directions |
| Conservation of electric charge | F | Total charge never changes | Gauge symmetry of electromagnetism |
| CPT symmetry | F | Physics is unchanged if charge, mirror image and time direction are all flipped together | Relativity + quantum field theory |
| Conservation of baryon and lepton number | A | The counts of heavy and light matter particles stay the same in all observed processes | Possibly broken at extreme energies |

---

## 2. Classical Mechanics

| Law | Tier | Statement | Equation |
|---|---|---|---|
| Newton's first law (inertia) | A | An object keeps its state of motion unless a force acts on it | — |
| Newton's second law | A | Force equals mass times acceleration | F = m·a |
| Newton's third law | A | Every action has an equal and opposite reaction | F₁₂ = −F₂₁ |
| Newton's law of universal gravitation | A | Every mass attracts every other mass | F = G·m₁·m₂ / r² |
| Kepler's first law | A | Planets orbit in ellipses with the Sun at one focus | — |
| Kepler's second law | A | A planet sweeps out equal areas in equal times | — |
| Kepler's third law | A | Orbital period squared is proportional to orbit size cubed | T² ∝ a³ |
| Principle of least action | F | Nature takes the path that makes the "action" stationary; most laws of physics can be derived from it | δS = 0 |
| Hooke's law | E | A spring's force is proportional to its stretch | F = −k·x |
| Galilean relativity | A | The laws of motion are the same for all observers moving at constant speed | — |

---

## 3. Thermodynamics and Statistical Physics

| Law | Tier | Statement | Equation |
|---|---|---|---|
| Zeroth law | F | Two systems each in balance with a third are in balance with each other, which defines temperature | — |
| First law | F | Energy is conserved: heat added = change in internal energy + work done | ΔU = Q − W |
| Second law | F | Entropy of an isolated system never decreases; heat flows from hot to cold | dS/dt ≥ 0 |
| Third law | F | Absolute zero can't be reached; entropy approaches a minimum there | S → S₀ as T → 0 |
| Ideal gas law | A | Relates pressure, volume and temperature of a gas | P·V = n·R·T |
| Boyle's law | A | At fixed temperature, pressure × volume is constant | P·V = constant |
| Charles's law | A | At fixed pressure, volume is proportional to temperature | V ∝ T |
| Gay-Lussac's law | A | At fixed volume, pressure is proportional to temperature | P ∝ T |
| Boltzmann's entropy formula | F | Entropy counts the number of possible microscopic arrangements | S = k_B·ln W |
| Planck's law | F | The spectrum of light emitted by a perfect hot body | — |
| Stefan–Boltzmann law | F | Total radiated power rises with the fourth power of temperature | P = σ·A·T⁴ |
| Wien's displacement law | F | Hotter objects glow at shorter wavelengths | λ_max·T = b |
| Fourier's law | A | Heat flow is proportional to the temperature gradient | q = −k·∇T |
| Newton's law of cooling | E | Cooling rate is proportional to the temperature difference | dT/dt = −k(T − T_env) |

---

## 4. Electromagnetism

**Maxwell's equations** (F): four laws that together describe all electric, magnetic and light phenomena.

| Law | Tier | Statement |
|---|---|---|
| Gauss's law | F | Electric charge produces electric fields that spread outward from it |
| Gauss's law for magnetism | F | No magnetic "charges" exist; magnetic field lines always form closed loops |
| Faraday's law of induction | F | A changing magnetic field creates an electric field |
| Ampère–Maxwell law | F | Electric currents and changing electric fields create magnetic fields |

| Law | Tier | Statement | Equation |
|---|---|---|---|
| Coulomb's law | A | Charges attract or repel in proportion to their size, falling off with distance squared | F = k·q₁·q₂ / r² |
| Lorentz force law | F | Force on a charge moving through electric and magnetic fields | F = q(E + v×B) |
| Lenz's law | F | An induced current opposes the change that created it | — |
| Biot–Savart law | A | Magnetic field produced by a steady current | — |
| Ohm's law | E | Current is proportional to voltage in many conductors | V = I·R |
| Kirchhoff's current law | F | Current flowing into a junction equals current flowing out | ΣI = 0 |
| Kirchhoff's voltage law | F | Voltages around any closed loop sum to zero | ΣV = 0 |
| Law of reflection | F | Angle of incidence equals angle of reflection | θᵢ = θᵣ |
| Snell's law | F | Light bends predictably when it crosses between materials | n₁·sin θ₁ = n₂·sin θ₂ |

---

## 5. Relativity

| Law | Tier | Statement | Equation |
|---|---|---|---|
| Principle of relativity | F | The laws of physics are the same for all observers moving at constant velocity | — |
| Constancy of the speed of light | F | Light in a vacuum moves at the same speed for every observer | c = 299,792,458 m/s |
| Mass–energy equivalence | F | Mass is a form of energy | E = m·c² |
| Time dilation | F | Moving clocks run slow relative to stationary ones | Δt′ = γ·Δt |
| Length contraction | F | Moving objects shorten in their direction of motion | L′ = L / γ |
| Equivalence principle | F | Gravity and acceleration are locally indistinguishable | — |
| Einstein field equations | F | Mass and energy curve spacetime, and curved spacetime tells matter how to move | G_μν + Λg_μν = (8πG/c⁴)·T_μν |
| Gravitational time dilation | F | Clocks run slower in stronger gravity | — |

---

## 6. Quantum Mechanics

| Law | Tier | Statement | Equation |
|---|---|---|---|
| Planck–Einstein relation | F | Light comes in packets (photons) whose energy depends on frequency | E = h·f |
| de Broglie relation | F | All matter also behaves as a wave | λ = h / p |
| Schrödinger equation | F | How a quantum system's state changes over time | iħ·∂ψ/∂t = Ĥψ |
| Superposition principle | F | A quantum system can be in a combination of states at once | — |
| Born rule | F | The chance of a measurement result equals the squared size of its wave amplitude | P = \|ψ\|² |
| Heisenberg uncertainty principle | F | Position and momentum can't both be known exactly | Δx·Δp ≥ ħ/2 |
| Pauli exclusion principle | F | No two identical matter particles (fermions) can share the same quantum state | — |
| Quantum entanglement | F | Particles can share a single state, so measurements on them stay correlated at any distance | — |

---

## 7. The Fundamental Forces (Standard Model)

All known forces reduce to four. The first three are described by the Standard Model of particle physics.

| Force | Tier | What it does | Theory |
|---|---|---|---|
| Electromagnetism | F | Light, chemistry, electricity, magnetism | Quantum electrodynamics (QED) |
| Strong nuclear force | F | Binds quarks into protons and neutrons, and holds nuclei together | Quantum chromodynamics (QCD) |
| Weak nuclear force | F | Radioactive decay; powers the Sun's fusion | Electroweak theory |
| Gravity | F | Attracts all mass and energy | General relativity (not yet unified with the other three) |

| Law | Tier | Statement |
|---|---|---|
| Higgs mechanism | F | Fundamental particles get their mass by interacting with the Higgs field |
| Electroweak unification | F | At very high energies, electromagnetism and the weak force are one force |
| Color confinement | F | Quarks are never found alone, only in bound groups |

---

## 8. Cosmology

| Law | Tier | Statement | Equation |
|---|---|---|---|
| Cosmological principle | A | On the largest scales, the universe looks the same everywhere and in every direction | — |
| Hubble–Lemaître law | A | Distant galaxies recede at speeds proportional to their distance | v = H₀·d |
| Friedmann equations | F | How the universe's expansion rate depends on what it contains | — |
| Accelerating expansion | E | The expansion of the universe is speeding up, attributed to "dark energy" | Λ > 0 |

---

## 9. Fluids and Waves

| Law | Tier | Statement | Equation |
|---|---|---|---|
| Archimedes' principle | F | Buoyant force equals the weight of displaced fluid | — |
| Pascal's law | F | Pressure applied to an enclosed fluid spreads equally in all directions | — |
| Bernoulli's principle | A | In a moving fluid, higher speed means lower pressure | P + ½ρv² + ρgh = constant |
| Continuity equation | F | Fluid isn't created or destroyed as it flows | — |
| Navier–Stokes equations | A | Govern the motion of liquids and gases | — |
| Inverse-square law | F | Light, gravity and sound from a point source weaken with distance squared | I ∝ 1/r² |
| Doppler effect | F | Frequency shifts when the source and observer move relative to each other | — |
| Huygens' principle | A | Every point on a wave front acts as a source of new waves | — |
| Malus's law | F | Intensity of polarized light through a filter | I = I₀·cos²θ |

---

## 10. Chemistry

| Law | Tier | Statement |
|---|---|---|
| Conservation of mass | A | Mass is neither created nor destroyed in chemical reactions (exact form: conservation of energy) |
| Law of definite proportions | E | A compound always has the same elements in the same mass ratio |
| Law of multiple proportions | E | When elements form several compounds, their mass ratios are small whole numbers |
| Periodic law | E | Element properties repeat periodically with atomic number |
| Avogadro's law | A | Equal volumes of gas at the same temperature and pressure contain equal numbers of molecules |
| Dalton's law of partial pressures | A | Total gas pressure is the sum of each gas's own pressure |
| Henry's law | A | The amount of gas dissolved in a liquid is proportional to its pressure above the liquid |
| Raoult's law | A | A solution's vapor pressure depends on the fraction of each component |
| Hess's law | F | Total energy change of a reaction doesn't depend on the path taken |
| Le Chatelier's principle | A | A system in balance shifts to counteract any change imposed on it |
| Rate law | E | Reaction speed depends on reactant concentrations raised to set powers |

---

## 11. Information and Probability

Not laws of matter, but they constrain every physical system, including any AI.

| Law | Tier | Statement |
|---|---|---|
| Landauer's principle | F | Erasing one bit of information costs a minimum amount of energy (k_B·T·ln 2) |
| Shannon's source coding theorem | F | Data can't be compressed below its information content (entropy) without loss |
| Shannon–Hartley theorem | F | Every communication channel has a maximum error-free data rate |
| Bekenstein bound | F | A finite region of space can hold only a finite amount of information |
| Law of large numbers | F | Averages of many trials converge to the expected value |
| Central limit theorem | F | Sums of many independent random effects tend toward a bell curve |

---

## 12. Biology

| Law | Tier | Statement |
|---|---|---|
| Natural selection | F | Traits that improve survival and reproduction become more common over generations |
| Mendel's law of segregation | A | Each parent passes on one of its two copies of each gene |
| Mendel's law of independent assortment | A | Genes for different traits are inherited independently (unless they sit close together on a chromosome) |
| Mendel's law of dominance | A | Some gene versions mask others |
| Hardy–Weinberg principle | A | Gene frequencies stay constant without selection, mutation, migration or chance effects |
| Cell theory | F | All living things are made of cells, and all cells come from existing cells |
| Kleiber's law | E | An animal's metabolic rate scales with its mass to the ¾ power |

---

## 13. Fundamental Constants

The fixed numbers that set the scale of the laws above. Since 2019, several have exact defined values.

| Constant | Symbol | Value | Status |
|---|---|---|---|
| Speed of light | c | 299,792,458 m/s | Exact |
| Planck constant | h | 6.62607015 × 10⁻³⁴ J·s | Exact |
| Elementary charge | e | 1.602176634 × 10⁻¹⁹ C | Exact |
| Boltzmann constant | k_B | 1.380649 × 10⁻²³ J/K | Exact |
| Avogadro constant | N_A | 6.02214076 × 10²³ per mole | Exact |
| Gravitational constant | G | ≈ 6.674 × 10⁻¹¹ N·m²/kg² | Measured |
| Fine-structure constant | α | ≈ 1/137.036 | Measured, dimensionless |
| Cosmological constant | Λ | ≈ 1.1 × 10⁻⁵² m⁻² | Measured, approximate |

---

## 14. Open Problems (Where the Laws Run Out)

- **Quantum gravity:** general relativity and quantum mechanics haven't been unified into one theory.
- **Dark matter:** about 85% of the matter in the universe is unseen and unexplained.
- **Dark energy:** what drives the accelerating expansion is unknown.
- **Matter–antimatter asymmetry:** why the universe holds matter rather than equal parts antimatter.
- **Quantum measurement:** why measurement produces a single definite outcome is still debated.
- **Arrow of time:** why the early universe had such low entropy.

---

## 15. Connection to the Framework

**Axiom:** The contents of the universe are in constant change, governed by laws that appear unchanging, for as long as there's usable energy to drive that change.

- Almost every dynamic law above has the form **dx/dt = f(x)**: Newton's second law, Maxwell's equations, the Schrödinger equation and the Einstein field equations all describe how a state changes under a fixed rule.
- The **conservation laws** (Section 1) are the invariants: what stays the same while everything else changes.
- The **second law of thermodynamics** is the energy clause: entropy rises until usable energy runs out and change winds down.
- The **constants** (Section 13) are the fixed parameters inside f, and as far as anyone has measured, they have not changed.
