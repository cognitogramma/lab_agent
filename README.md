# PolyRMC Light-Scattering Processing Pipeline

An auditable analysis pipeline for automatic continuous dilution (ACD)
light-scattering data, with subjective analysis decisions encoded as explicit,
reviewable logic rather than left in an analyst's head.

Tulane University · PolyRMC · Advisor: Prof. Wayne F. Reed

Turns a raw ARGEN trace into weight-average molar mass, second virial
coefficient A₂, and the diffusion interaction parameter k_D — recording, for
every number, how it was produced.

## The architecture invariant

**Models judge, they never compute.**

```
agent/   orchestrator            the model drives; a person approves every commit
tier1/   bounded evaluator loops propose (deterministic) → judge (model)
tier0/   measurement layer       fully local math, zero model dependency
```

No function in `tier0/` may accept a model client or import a provider package.
That is not a convention — [test_tier0_purity.py](tests/test_tier0_purity.py)
fails the build on any violation. A model outage, a bad response, or a provider
change can cost you a suboptimal smoothing window inside an already-validated
safe band. It cannot produce a wrong molar mass.

Tier 1 loops propose deterministically and judge subjectively. The proposer
generates a candidate set that is safe by construction; the judge selects among
candidates or requests a re-proposal with a direction hint, and **cannot
construct a candidate** — an out-of-range selection is refused and the
conservative fallback is taken. Loops are capped. Rejected candidates are logged
alongside accepted ones, because for fit-range selection *which points were
excluded* is the scientific claim.

## The model drives

There is no fixed route through the pipeline. The agent sees where the run
stands, decides what to do next, and calls tools to do it — load the file,
sweep the windows, search what the same sample did last time, stop and ask.
The route is the model's; the guarantees are not, because the guarantees were
never in the route:

**Validation lives in tools, not in prompts.** A prompt asking a model not to do
something is a request. A tool that rejects it is a guarantee, and it keeps
working when the model is having a bad day, is a different model, or is replaced
by a script. Every rejection states what failed, by how much, and what would be
valid instead, and it goes back to the model as the result of its call.

**Commits need a person.** Every tool is tiered. `read` and `compute` run
freely — they cannot change a run, and making someone approve each one trains
them to approve without reading. Every `commit` stops and asks, showing the
arguments in full.

**The loop is capped and the transcript is generated.** What the agent reports
having done is rendered from the calls it actually made, so its account cannot
drift from its behaviour.

```
polyrmc agent                                    # a session; ask about past work
polyrmc agent --raw <file> --declarations <f>    # a session over a nominated run
```

`--provider gemini` (default, uses `GEMINI_API_KEY`) or `--provider claude`
(needs `ANTHROPIC_API_KEY`). Without a key the session still opens and the
slash commands still work; the deterministic path never needed a model at all.

Slash commands: `/state`, `/tools`, `/history`, `/runs [query]`, `/notes
[query]`, `/always <tool>`, `/auto`, `/help`, `/exit`.

## What a session costs

It runs on your key, so it tells you what it spent. Every turn ends with a line,
and `/cost` gives the session total:

```
[this turn: 3 request(s), 35,420 in / 320 out  |  session: 3 request(s), ...]
```

Two figures because two different limits bite.

**Requests** are what the Gemini free tier rations — `gemini-3.6-flash` allows
5 per minute and **20 per day**, and it counts requests, not tokens. An
orchestrator spends one per tool round, so a single question can cost six. Three
questions can exhaust a day. This is the limit you will actually hit; either set
`--model` to something with a larger allowance, enable billing, or use
`--provider claude`.

**Tokens** are what a paid key is billed on, and input dominates: the transcript
is re-sent every round, so a three-round question paid for its history three
times — 35,420 in against 320 out above. Sessions are expensive because they are
long, not because the model is verbose. Shorter sessions on focused questions
cost far less than one long one.

Ceilings, checked *before* each call rather than reported after:

```bash
--budget-requests 20     # or POLYRMC_BUDGET_REQUESTS
--budget-tokens 200000   # or POLYRMC_BUDGET_TOKENS
```

Cost in currency is opt-in — set `POLYRMC_RATES="in=0.30,out=2.50"` (per million
tokens, from your provider's current pricing). Token counts are measured and are
always right. A price hardcoded here would be a claim about a vendor's pricing
page on one particular day: wrong within a quarter, and believed anyway.

## Memory across experiments

The agent searches what this lab has already processed. `data/runs` holds a
`.provenance.json` sidecar per run, and `search_past_runs`, `get_past_run`, and
`compare_to_past_runs` read them — so "have we run this sample before, and did
it look like this?" is a question the agent answers from the record rather than
from recall.

Alongside that is a notebook it writes to itself (`data/memory/notes.json`):
the gotchas, quirks, and reasons behind a decision that no sidecar records.
Notes are kept deliberately apart from run provenance. A sidecar has to stay
reconstructible from the run itself; a note is somebody's opinion, and filing
one inside the audit trail would make the audit trail unfalsifiable. Every note
carries an author, for the same reason every declaration does.

## Layout

```
src/polyrmc/
  config.py       run configuration and physical constants (pydantic)
  state.py        typed state: arrays, anomalies, splice, loops, provenance
  io_csv.py       the six-column CSV that is the only Part 1 → Part 2 interface
  provenance.py   per-run sidecar; replay a historical result without a model
  pipeline.py     run_part1 (clean) and run_part2 (extract)
  cli.py          command-line entry point

  tier0/  argen_io    dynamic header, time reconstruction, scoped nulls,
                      ls8-only channel selection
          detect      derivative/Hampel/saturation/variance detectors, PELT
          classify    deterministic anomaly typing; escalates ambiguous cases
          splice      excision without re-indexing
          baseline    drift subtraction (AsLS, robust polynomial)
          smoothing   adaptive Savitzky-Golay, safe-by-construction windows
          dilution    c(t) = c0·exp(−Qt/V), aggregation index
          optics      optical constant K, Rayleigh ratio, alpha calibration
          virial      Kc/I_R fit, A₂·Mw, Mw, A₃, k_D
          fit_range   linear-region candidates and linearity gates

  tier1/  loop        the bounded propose → judge → re-propose harness
          judge       ModelJudge / RecordedJudge / StaticJudge
          fit_range_selector, smoothing_selector

  tier2/  orchestrator  staging + human confirmation gate

  agent/  orchestrator  the agent loop: model → tool calls → results → repeat
          toolspec      tool schemas, dispatch, and compact result rendering
          toolcall      provider-neutral tool-calling transport
          lanes         the two-lane numeric/visual judge
          lab           LAB.md, loaded whole into every session

  memory/ registry    index over past runs' provenance sidecars
          notes       what the agent concluded, kept between sessions
```

## Setup

```bash
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add an API key only if you want the model
judge; the pipeline runs end to end without one.

## Run

```bash
.venv\Scripts\python.exe -m polyrmc.cli agent
.venv\Scripts\python.exe -m polyrmc.cli run --config examples/run_config.example.json
```

`--judge static` (the default) takes the conservative option deterministically
and makes no network calls. `--judge model` uses the pinned Gemini model
(`gemini-3.6-flash`) and requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
Pro models need a paid Google plan; the free tier returns a zero quota for them.

```bash
.venv\Scripts\python.exe -m pytest
```

387 tests, no API key or instrument file required. The agent loop is exercised
end to end against a scripted provider, including the paths that matter most:
a tool that rejects, a commit the researcher declines, a handler that raises,
and the step cap holding.

## The scattering channel

The instrument writes `ls1`…`ls16`. **Only `ls8` is the measurement**, and it is
the only one this pipeline analyses. The others are dropped at load, so no later
stage can read one by accident, and null handling keys on `ls8` alone — a bad
`ls3` reading never discards a row whose `ls8` value is fine. `RunConfig.channel`
carries the choice and it is recorded in the output CSV and the sidecar.

## What alpha does and does not affect

The instrument reports scattering in its own units, so `I_meas = α · I_R`.
Substituting into the virial expansion:

```
Kc/I_meas = (1/α)·(1/Mw + 2·A₂·c + 3·A₃·c² + …)

  intercept       = 1/(α·Mw)
  slope           = 2·A₂/α
  slope/intercept = 2·A₂·Mw          ← α cancels
```

So **A₂·Mw is calibration-free**, as is Mw/M₀ (a ratio within one file). Only
separating A₂ and Mw into absolute units needs α. Uncalibrated runs report
A₂·Mw and k_D and explicitly withhold Mw, A₂, and A₃ rather than reporting
numbers that silently carry a factor of α.

Fit-range selection is scale-invariant, so it does not wait on calibration.

## Status

Implemented and tested against synthetic data with known answers: the full
Part 1 chain, the full Part 2 chain, both tier-1 loops, provenance and replay,
and the orchestrator's confirmation gate. An end-to-end uncalibrated run
recovers a known A₂·Mw to within 0.5%.

Not yet done, in the order the design calls for:

1. **Measure the toluene reference and pin α.** A bench measurement.
   `alpha_from_toluene` and `alpha_from_known_mw` are implemented and agree with
   each other in tests; neither has seen a real instrument reading.
2. **Run the published-value validation gates.** The regression targets (latex
   ≈ 0, lysozyme 3.76 cm³/g, BSA 5.00 cm³/g, and the calibration-dependent Mw
   and A₂ values) are not yet encoded as tests, because that needs the real
   files and a pinned c₀ for each. Per the design, the gates must be run
   repeatedly with the **spread** inside tolerance, not just the mean.
3. **Exercise the model judge on real traces.** All tests use `StaticJudge`.

### Known limitations

- **The ARGEN reader is written to the documented format, not to a real file.**
  Header length is read from the declared value in the preamble; the recognised
  spellings are in `HEADER_COUNT_PATTERNS` and will likely need one more entry
  for this instrument. A file that declares nothing raises rather than guessing.
- **Smooth, long-timescale drift is not separable from the ACD decay** by the
  detectors alone — both look like a trend. Baseline correction therefore runs
  only when a `BASELINE_DRIFT` region was actually classified; on a dilution
  trajectory the exponential decay *is* the signal, and fitting a flexible
  baseline to it would subtract the measurement.
- **Change-point search decimates above 8,000 points.** A 22-hour acquisition at
  1 Hz does not carry change points needing single-sample resolution, and the
  exact search is superlinear in practice.
- **Ambiguous anomaly regions are left in the trace**, not excised, and raise a
  warning. Escalating them to a tier-1 classifier is designed but not wired in.

## References

- Jarand C W & Reed W F (2026). Automatic, single-sweep, cuvette-based
  determination of A₂ (B₂₂), k_D, and other equilibrium properties for
  macromolecules. *Meas. Sci. Technol.*, in press.
  DOI: 10.1088/1361-6501/ae8923 — primary reference for the measurement model.
- Jarand C W, McLeod M J & Reed W F (2024). *Biomacromolecules* 25, 5198–5211.
- Jarand A R, Jarand C W & Reed W F (2026). *Meas. Sci. Technol.* 37, 045104.
