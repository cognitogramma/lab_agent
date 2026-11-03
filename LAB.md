# LAB.md — Continuum Spectranalytics, Tulane

## Project mission

Continuum Spectranalytics is a research workspace for literature review and
storing research ideas. Its lasting value is a connected record of sources,
critical synthesis, open questions, hypotheses, and the next experiments to try.

Review original sources when available, cite the material actually read, and
distinguish published findings from lab observations and proposed ideas. Save
ideas with their rationale, evidence, source links, author, and next step so
future sessions can retrieve and build on them. A hypothesis stays a hypothesis
until evidence supports it. Light-scattering analysis remains the project's
deterministic measurement foundation.

Instrument context and house conventions. This file is always in the agent's
context, whole. Swapping it swaps the lab; nothing is retrained.

Values marked **UNCONFIRMED** must not be used to compute a reported result
until someone with access to the instrument's calibration record confirms them.
The pipeline asks for each of these at intake rather than defaulting them.

## Instruments

### Fluence / Yokogawa ARGEN

| Property | Value | Status |
|---|---|---|
| Technique | Static light scattering | confirmed |
| Cells | 16 independent (`ls1`…`ls16`) — **not** 16 angles | confirmed |
| Working channel | `ls8` | house convention |
| Detection angle | 90° | confirmed |
| Wavelength λ₀ | 660 nm | confirmed |
| Sampling | ~1 Hz | confirmed |
| Detector full-scale | **UNCONFIRMED** | see below |
| `ls8` units | **UNCONFIRMED** | see below |

### Brookhaven NanoBrook Omni

| Property | Value | Status |
|---|---|---|
| Technique | Dynamic light scattering | confirmed |
| Wavelength λ₀ | 640 nm | confirmed |

## Constants

| Key | Value | Source |
|---|---|---|
| `n_s` (solvent RI, water) | 1.333 | paper |
| `N_A` | 6.02214076e23 | SI definition |
| `dn_dc` — protein/peptide | 0.185 cm³/g | paper |
| `dn_dc` — polysaccharide | 0.150 cm³/g | paper |
| `R_toluene` @ 660 nm | **DISPUTED** — 1.19e-5 or 9.44e-6 cm⁻¹ | see below |
| `R_toluene` @ 640 nm | 1.34e-5 cm⁻¹ | paper |
| ACD cuvette volume `V` | **DISPUTED** — 2 or 1 cm³ | see below |
| ACD flow rate `Q` | 0.001 cm³/s | paper |

## Open items

**`R_toluene` at 660 nm is disputed.** The specification cites 1.19e-5 cm⁻¹; the
value carried in the code was 9.44e-6. They differ by about 26%. The choice
scales Mw and A₂ by that ratio and leaves A₂·Mw and Mw/M₀ untouched, so **no
internal check will catch the wrong one.** Confirm against the instrument's own
calibration record.

**The ACD cuvette volume is disputed** — 2 cm³ per the specification, 1 cm³ in
the code's default. V sets the Q/V decay rate, so it bends the whole
concentration axis.

**The detector ceiling is not known.** It must come from the instrument or ADC
documentation. It is never read off a trace: the maximum of a railed trace *is*
the rail, so any trace would confirm any ceiling. Without it, saturation masking
is blocked and the pipeline says so.

**`ls8` units are not known.** The observed [0.027, 1.0] range confirms the
channel is not already an absolute Rayleigh ratio, which would be ~1e-5 cm⁻¹.
It does not distinguish a detector count rate from an already-normalised
channel. Blocks `I_rayleigh`.

**Smoothing may need to be cycle-aware.** DLS-linked acquisitions cycle ~20 s
stir → ~40 s settle → ~60 s measure. Whether `ls8_smoothed` should be one value
per quiescent window or a continuous trace over the full series is unresolved.
The pipeline runs continuous and flags detected cycling.

## House conventions

- Only `ls8` is analysed. The other fifteen channels are dropped at load.
- `experiment_type` is declared by a person every run. It is never inferred from
  the shape of the trace.
- Rolling-median and moving-average smoothers are prohibited — they attenuate the
  real transitions the experiment exists to measure.
- Savitzky-Golay polynomial order 2, ceiling 3.
- Survivors are never re-indexed after excision.
- Baseline correction defaults to none and is applied only by explicit commit.
- Saturated samples are censored, not capped: masked to NaN with the timestamp
  left in place.

## References

- Jarand, McLeod & Reed (2024), *Biomacromolecules* **25**, 5198–5211.
- Jarand & Reed (2026), *Meas. Sci. Technol.* — primary reference for the
  measurement model.
