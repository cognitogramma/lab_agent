# Data labelling

The catalogue metadata attached to a stored run: whose data it is, how it was
measured, what it is, and what question it answers.

Implemented in [`src/polyrmc/labeling.py`](../src/polyrmc/labeling.py), attached
to a run as `RunConfig.labels`, and recorded in the provenance sidecar. Labels
are catalogue metadata only — no label changes a number the pipeline computes,
and nothing in `tier0/` reads them.

Source: handwritten labelling notes, 29 July 2026.

## Why the vocabularies are closed

A label that accepts anything cannot be grouped on, and grouping is the point.
Free text is allowed only where the value is genuinely open — company name,
buffer, sample name, instrument. Everything else is an enum, and a value outside
it fails at construction rather than at analysis time.

Three checks do real work beyond spelling:

- **A method's observables must belong to that method.** A DLS record claiming
  `kc_over_ir` is a transcription error.
- **Blank-requiring methods must record their blank.** Fluorimetry, UV/Vis,
  Raman and CD readings are meaningless without one, and its absence is not
  recoverable from the trace later. `blank_measured` has no default — `False` is
  a valid, informative answer; unknown is not.
- **An ACD label must agree with `experiment_type`.** See below.

## I. Ownership

| Field | Confidential | Non-confidential |
| --- | --- | --- |
| `confidentiality` | `confidential` | `non_confidential` |
| `institution` | — | required (e.g. Tulane) |
| `company` | required | — |
| `project_type` | required | — |

Confidential runs must name the company and project because an unattributable
confidential run cannot be returned to its owner or deleted on request.

Examples from the notes: Bachem — liraglutides; Moderna — mRNA lipid
nanoparticles.

### Redaction

The provenance sidecar serializes the whole `RunConfig` to disk beside the
processed CSV, so a client name recorded here would travel with the data.
`write_sidecar` therefore redacts `company` and `project_type` for confidential
runs **by default**:

```python
write_sidecar(state, config, out)                            # identity removed
write_sidecar(state, config, out, redact_confidential=False) # internal archive
```

Redaction keeps the confidentiality flag — a redacted record must stay
distinguishable from a genuinely non-confidential one — and keeps every
scientific label. What the sample was and how it was measured is not the
confidential part.

## II. Methods

At least one, each at most once.

| Method | Observables | Blank |
| --- | --- | --- |
| `dls` | `d_h`, `g2`, `kcps`, `diffusion_coefficient` | — |
| `sls` | `i_r`, `kc_over_ir` | — |
| `viscometry` | `raw`, `reduced_viscosity` | — |
| `fluorimetry` | `intensity` | required |
| `uv_vis` | `absorbance` | required |
| `raman` | `intensity` | required |
| `circular_dichroism` | `ellipticity` | required |

## III. Sample

`name` (free text) and `sample_class`, one of `synthetic_polymer`, `rna`,
`protein`, `vaccine`, `other`.

`protein_subclass` — `monoclonal_antibody`, `globular`, `non_globular` — applies
only when `sample_class` is `protein`, and is rejected otherwise.

Solution conditions, all optional: `concentration_g_per_cm3`, `ph`, `buffer`,
`temperature_c`. They are part of the label because a scattering result is only
interpretable against them — A₂ and k_D are properties of the macromolecule *and
its solvent*, not of the macromolecule alone.

## IV. Goal

| Goal | Modes |
| --- | --- |
| `equilibrium_characterization` | `acd`, `manual` |
| `kinetics` | `aggregation`, `hydrolysis`, `dissolution` |
| `thermal` | `temperature_scan`, `isothermal` |
| `dialysis` | `nacl`, `guanidine_hcl`, `ph` |
| `isochemical` | `ionic_strength`, `ph`, `denaturant` |

Modes are scoped per goal rather than pooled into one list: `ph` means a buffer
exchange under `dialysis` and a titration under `isochemical`, and collapsing
them would merge two different experiments.

### Agreement with `RunConfig`

`equilibrium_characterization` / `acd` *is* the continuous dilution the pipeline
models as `c(t) = c0·exp(−Qt/V)`, so it requires
`experiment_type="dilution_trajectory"`. A run labelled ACD but configured as
`fixed_concentration` would be analysed with no dilution trajectory at all — and
would still produce numbers — so the mismatch is refused at construction.

Every other goal is compatible with either acquisition, and the config stays the
authority: `experiment_type` is supplied, never inferred from the label.
