# Continuum Spectranalytics

A Windows research workspace for finding and reviewing papers, keeping a
searchable notebook of research ideas, and analyzing ARGEN light-scattering
data with a record of how each result was produced.

## Download for Windows

1. On a 64-bit Windows PC, download the [Windows executable](https://github.com/cognitogramma/lab_agent/releases/download/v0.2.0/Continuum-Spectranalytics.exe)
   (about 120 MB).
2. In File Explorer, double-click `Continuum-Spectranalytics.exe`. The app opens
   directly and includes Python and its dependencies.

The executable is unsigned. If Windows blocks it on a Tulane-managed computer,
contact Tulane IT for help. [Release notes and checksum](https://github.com/cognitogramma/lab_agent/releases/tag/v0.2.0)

## Quick start

1. Open **Literature** and search for `light scattering second virial coefficient`
   using **Find related papers**.
2. Select a result and choose **Read page** or **Open original** to inspect the
   source. Choose **Save paper** to add it to your reading list.
3. Open **Ideas & notes > New research idea** to record a hypothesis, source
   links, and a next step.

These steps work without a model API key. To use the assistant or create a
specialist, open **Research assistant > Model settings**, choose Gemini, Claude,
or OpenAI, enter your provider API key, and select **Use this key**. Provider
usage charges may apply. Try asking, “Find two public papers on the second
virial coefficient in light scattering, compare their methods, and cite your
sources.”

For instrument data, press **Ctrl+O** to open a raw ARGEN file. The
[run guide](#opening-a-run-and-what-you-have-to-declare) lists the experimental
details needed for each calculation.

The app stores settings, notes, the reading list, specialist definitions, and
run data under `%LOCALAPPDATA%\Continuum Spectranalytics`.

## Specialized subagents

With a model connected, ask the **Research assistant** to build a specialist.
For example, “Create a literature subagent called Citation Reviewer. Have it
compare reported methods and cite a source for each finding.” Review the proposed
instructions and approve the save. Ask “List my subagents” to see saved roles or
“Ask Citation Reviewer to compare light-scattering methods in recent papers” to
delegate a task. The terminal `continuum-spectranalytics agent` also accepts
these requests.

Choose a scope when creating a specialist.

| Scope | What the specialist can use |
| --- | --- |
| `general` | The text supplied with its task |
| `literature` | Public source search and page reading |
| `run_review` | Read-only access to the current run, run history, and notes |

Saved roles are available across sessions. A subagent uses the current model
provider and shares the parent turn's budget and stop controls. It returns a
report to the Research assistant for review.

## Build the Windows app

Use a Windows x64 Python installation with Tcl/Tk. From the checkout, run

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[windows-build]"
.\Build-Continuum-Spectranalytics.ps1
```

The build produces `dist\Continuum Spectranalytics.exe`. It bundles the default
lab context and agent prompts. User settings, credentials, and research data
are stored separately. For source development, run
`continuum-spectranalytics gui` or `./Start-Continuum-Spectranalytics.ps1`.

The Python package is `continuum_spectranalytics`. Set
`CONTINUUM_SPECTRANALYTICS_ROOT` to point the app at an existing workspace.
Legacy `POLYRMC_*` settings still load.

Agent access to local files can be configured in `filesystem-permissions.json`.
See [filesystem access](docs/filesystem-access.md) for setup, exclusions, backups,
and the GUI's **Menu > Session > File permissions...** command.

An auditable analysis pipeline for automatic continuous dilution (ACD)
light-scattering data, with subjective analysis decisions encoded as explicit,
reviewable logic rather than left in an analyst's head.

Tulane University · Continuum Spectranalytics · Advisor: Prof. Wayne F. Reed

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
continuum-spectranalytics agent                                    # a session; ask about past work
continuum-spectranalytics agent --raw <file>                       # a session over a nominated run
continuum-spectranalytics gui --raw <file>                         # the same session, in a window
```

## Opening a run, and what you have to declare

**A raw file is enough.** There is no declarations file to prepare and no
interview to sit through first. The run opens undeclared, and undeclared is a
state the pipeline holds rather than a failure it reports:

| you have declared | you can get |
|---|---|
| *nothing* | the trace loaded and cleaned, anomalies found and decided, the safe smoothing band, a chosen window, the figures, and an emitted CSV |
| `experiment_type`, `c0` (and `Q`, `V` on a dilution) | a concentration axis, and therefore **A₂·Mw** and **k_D** |
| `sample_class` (which sets `dn_dc`) | **Mw**, **A₂** and **A₃** separately |
| the toluene reference (sets α) | the absolute Rayleigh ratio |

Each row buys the row below it. Nothing is guessed to fill a gap, and nothing
stalls waiting for an answer it does not need: `concentration_column` and its
neighbours have always had an explicit branch for the unanswered case, which
costs that column and records the reason. Ask for a number that is out of reach
and you get the reason, not a stack trace:

```
no concentration axis in this file. The virial expansion is fitted against
concentration, so it needs a run whose concentration is known -- see the
OPEN-1 path recorded in the sidecar for why this column is blocked.
```

You answer by telling the agent, in the session, whenever you happen to know.
`set_declaration` is a commit, so it stops and asks you, and what it records
carries your name and the time.

Two things this deliberately does **not** do. It does not infer a declaration
from the shape of the trace — a rising channel is suggestive of aggregation and
never conclusive, because a dilution whose sample aggregates faster than it
dilutes rises too. And it does not fall back on a default: `dn_dc` is 0.185 for
a protein and 0.150 for a polysaccharide, which is a factor of 1.52 in K and a
third in Mw, and the two traces look identical. An invented constant produces a
number that is wrong and completely plausible, which is the one failure this
pipeline exists to prevent.

`continuum-spectranalytics intake` still exists and still writes a declarations file. It is now a
convenience — answer everything up front, once, for a run you already
understand — rather than a gate you have to pass to begin.

Slash commands: `/state`, `/tools`, `/history`, `/runs [query]`, `/notes
[query]`, `/always <tool>`, `/auto`, `/cost`, `/providers`, `/log [n]`, `/help`,
`/exit`.

**Ctrl+C ends the turn, not the process.** It asks the run to stop at its next
boundary and reports where that left it, the same way the step cap does. Press
it again and you get your terminal back, for when a provider has hung and there
is no next boundary coming.

**What it prints is an account, not a trace.** The session shows what the model
said and one plain line per action:

```
  Let me see where this run stands before touching anything.

  . checking where the run stands
  . loading and cleaning the raw file
  . sweeping the smoothing windows

  The run is loaded and the safe band is computed. Nothing is committed yet.
```

Everything underneath that — call arguments, whole tool outputs, per-turn token
counts, warnings raised inside a provider's SDK, and the provider's own error
text — goes to a session log under `data/logs/`, named on the way in and on the
way out. `/log` shows where it is and its last few lines; `--verbose` puts the
lot back on screen.

Two things are never summarised. A **refusal** is printed whole, because the
tool surface writes its rejections as recovery instructions and the band it
names is the useful part. An **approval prompt** is printed whole, because a
prompt that hides what is about to change trains you to approve without
looking.

## The window

```bash
continuum-spectranalytics gui
```

The same session — same orchestrator, same tool surface, same log — with the
run beside it instead of recalled on demand.

The desktop has an ink-blue and copper research-studio design, a compact
workspace rail, and independent workflow panels for data, reading, and memory.
Source search sits in the workspace header and session status stays in a slim
footer. **Research assistant** opens a contextual drawer when needed. The rail opens
Workbench, Literature, Visuals, Experiments, Ideas & notes, and Toolbox. Research puts sources
beside their preview on wide screens and stacks them at narrower widths.
Times New Roman headings and body text use installed serif fallbacks.

Install the desktop controls and web search (also included in `.[all]`):

```powershell
python -m pip install -e ".[gui,search]"
```

The presentation lives in `gui/theme.py`, `gui/layout.py`, and `gui/shell.py`,
so later visual changes do not require changes to the measurement or search logic.

Use **Menu > Session > Open a run...** to open an experiment or **Model settings**
inside the assistant to connect a model.
**Ctrl+J** shows or hides the assistant, **Ctrl+L** opens it and focuses the
question box, **Ctrl+O** opens a run, and **Ctrl+K** focuses research. The
**Menu** dropdown contains Session, View, and Help; **View > Reset workspace**
closes the assistant. Session details remain available from the overview when
no trace is loaded.

### Internet research

The **Literature** workspace is an experiment-aware reading desk. **Use experiment**
fills an editable public research query from the declared experiment and flagged
topics. **Find related papers** combines one web or arXiv search with matching
lab notes, indexed experiments, and saved papers. The proposed query is visible;
raw notes, file paths, and private run identifiers are not automatically sent to
the search service. Changing a run refreshes the context without starting a search.

**Found**, **Memory**, and **Saved** keep published-source candidates, local
observations, and the reading list easy to browse. Each match explains its topic
overlap. Relevance is a discovery aid, not proof that a source explains the current
measurement. Local notes remain attributed lab observations; arXiv results remain
preprints, and web search can also return pages other than papers.

The reader uses large Times New Roman text, generous line spacing, adjustable
type size, and a focus mode. **Read page** fetches available public text into the
reader; **Open source** opens the original. Search summaries are labelled
separately from extracted page text. PDFs, paywalled pages, and JavaScript-only
sites may need the original page. **Save paper** keeps bounded citation metadata
in `data/literature/reading-list.json`, a local, gitignored reading list reused
in later searches. It does not save the full article.

**Ask assistant to review** uses the configured model to compare sources with
the experiment and memory, cite what it actually reads, and identify unresolved
questions. It uses the normal agent budgets and approval gates. Discovery,
reading, and saving work without a model key.

The workspace searches the web or arXiv without spending a model request.
Install general web search with the optional extra (included in `.[all]`):

```powershell
python -m pip install -e ".[search]"
```

Web search uses DDGS and needs no search API key. arXiv uses its public HTTPS
API and works without that extra. Search terms go to the selected search
service. The date filter applies to both sources; for arXiv it uses submission
dates. An optional hostname, such as `nist.gov`, restricts web results to that
domain and its subdomains. Web date filters are supplied by the search engines;
check dates at the original source. The API enforces result limits, removes duplicate
links and common tracking parameters, and reports connection or service errors.

Select a result to inspect its summary, domain, available publication date,
and authors when supplied. **Open source** opens the original page; **Copy
citation** copies its title, available authors/date, and URL. Missing dates are
shown as unknown. arXiv records are preprints, and a search summary does not
verify a source's claims or peer-review status.

Search runs in the background. **Cancel**, or **Escape** while Research is
selected, ignores a pending search's results; a network request already sent
finishes under its timeout. Cancelled and superseded results cannot replace
the current search. arXiv requests are spaced by at least three seconds.

The model can use the same read-only `search_internet` tool when asked for
external references. Its sources also appear in Research, and Markdown source
links in the conversation are clickable. External snippets are treated as
untrusted source material. Searches cannot change a run or supply missing
experimental declarations. `/search [query]` opens research in the GUI;
without a query it focuses the search box without making a request.

### The internet-enabled agent harness

Internet access is part of the agent's tool set. `review_literature` combines
experiment context, local memory, saved papers, and one internet search. In a
conversation, the model can choose it or `search_internet`, inspect the results, call `read_web_page` on a
source, follow relevant returned links, refine its search, and answer with
citations. A supplied URL can be read directly. The existing orchestrator
passes each result back to the model to choose its next action; it does not
impose a search-and-read pipeline.

For example, connect a model in **Research assistant > Model settings**, then ask:

> Find two public references on the second virial coefficient in light
> scattering. Read the original sources, compare their explanations, and cite
> the pages you used.

The terminal uses the same harness and tools:

```powershell
continuum-spectranalytics agent --max-web-calls 6 --budget-requests 8 --prompt "Find public references on light scattering, read the original sources, and cite the explanation."
```

Each question allows eight internet tool calls by default, including calls
requested in one batch. Set `--max-web-calls` or `CONTINUUM_SPECTRANALYTICS_MAX_WEB_CALLS` to change
that allowance; zero disables the agent's internet requests. The manual Research
panel is independent. Model request/token budgets and the Stop button still
apply. A failed search or page read is a failed tool result, so the model can
explain it or choose another source without ending the conversation.

`read_web_page` retrieves public HTML, text, Markdown, and JSON. It returns
bounded source text, resolved links, the final URL, retrieval time, a source
hash, and whether text was truncated. Requests have a 15-second deadline,
response-size limits, and redirect checks. Private/local addresses are rejected
and connections use the validated public address. No browser login, cookies,
JavaScript execution, form submission, or PDF extraction is performed.

Page content remains untrusted evidence. It cannot authorize changes to an
experiment or replace a researcher's declarations. Every scientific commit
still uses the existing approval gate. A configured model provider is required
for autonomous tool selection; the search and page-reading tools themselves do
not require a model key.

```
+------------------------------------------------------------------------+
| Continuum Spectranalytics                     Research assistant Menu |
+---------------+--------------------------------------------------------+
| Workspaces    |  Workbench                      [ Find sources ]      |
| Workbench     |  Work with the evidence.                               |
| Literature    |  Open an experiment                                    |
| Visuals       |  Explore the literature                                |
| Experiments   |  Revisit an experiment                                 |
| Notebook      |                     Assistant drawer opens when needed |
+---------------+--------------------------------------------------------+
| Current session                 Activity              Model and log   |
+------------------------------------------------------------------------+
```

It exists for the figures. This pipeline draws them on every processed run —
the raw trace with its anomalies marked, the smoothing window against the trace
it tracked, the excess Rayleigh ratio, the residuals — and a terminal can only
tell you the path it wrote them to. So the decision those figures exist to
inform gets taken from a number in a transcript, with the picture of it
unopened two directories away, which is the decision this whole pipeline exists
to stop being taken by eye. Every caveat is printed *into* the image by
`figures.py`, so what you see is the whole claim rather than a thumbnail of it.

**You can stop it.** The Ask button becomes Stop while a turn is running, and
Escape does the same thing. It is a cooperative stop, and it says so: a model
call is a blocking request inside a vendor SDK and a tool is numpy over eighty
thousand points, so neither can be cut off without leaving the run in a state
nobody can describe. The button therefore reads *stopping*, not *stopped*, until
the thing already in flight returns — usually a second or two, occasionally the
length of one model call. A button that claimed otherwise would have people
pulling the plug when the window "ignored" them.

What a stop costs is the rest of the turn, never the run and never the
conversation. Tools behind the stop are answered rather than dropped — a
provider handed an assistant turn that asked for three tools needs three
results back or the next request is malformed — so the model is told which
calls did not run and why. Anything that had already completed stands. A stop
leaves the run exactly where the step cap would: wherever the last completed
tool left it.

Beside that, two smaller things. The run state is available in Overview,
rendered from the same `render_state` the model gets from `get_state`, so what
you are looking at and what the agent is reasoning over are one text. And the
commit gate is a dialog rather than a line in a stream — a prompt with three
sensible answers, sitting in the same scroll as everything else, is a prompt
answered by muscle memory.

**The window adds no route.** There are no stage buttons, no wizard, no
"Part 1 → Part 2" — because the model still decides what happens next, and a
window is exactly where a fixed route would creep back in as a row of buttons
that runs the stages in order. The window directly invokes `search_past_runs`,
`read_notes`, and `search_internet` for its read-only panels; none can change a
run. Compute and commit tools are reached only by the
model choosing to call them, past the approval gate. That is asserted, not
promised: `test_the_window_offers_no_route_of_its_own` fails the build if a
panel directly invokes anything beyond the read tier.

Nothing else is weakened either. It is one loop: commits go through the same
`ApprovalPolicy`, refusals arrive whole, the cap holds, and the log under
`data/logs/` is written by the same narrator, so the file a colleague opens is
the same file whichever surface produced it. Closing the window on a waiting
commit **refuses** it — an unanswered prompt must never become an approval
nobody gave.

Markdown in the model's reply is rendered rather than shown. A model writes
`###` and `**` whatever you ask of it; a console cannot honour that, but a
window has no excuse — displaying a formatting instruction instead of obeying it
is the same mistake as printing call syntax. Web links are clickable; tables and LaTeX are left
verbatim, and underscores are never emphasis, because this lab writes
`c0_g_per_cm3` in ordinary prose and a key italicised through the middle is a
key nobody can search for.

The window costs no new dependency: Tk ships with CPython, and matplotlib is
already on the measurement path. On Linux, Tk is sometimes a separate package
(`python3-tk`); without it, `continuum-spectranalytics gui` says so and names `continuum-spectranalytics agent`,
which does everything the window does apart from showing you the picture.

## Bring your own tokens

**You supply the brain.** This package ships no key and brokers no model access.
What it is worth is below the model — the ARGEN ingest, the declarations that
configure a run, the provenance sidecars, the checks that catch a plausible
wrong number — and every bit of that runs with no model at all. Which brain
drives a session is your purchase, and you can change it per session:

| `--provider` | key | where |
|---|---|---|
| `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | <https://aistudio.google.com/apikey> |
| `claude` | `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> |
| `openai` | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> |

With no `--provider`, the session uses whichever one you have keyed. With no key
at all it asks for one at the prompt, reads it without echoing it, and offers to
save it to `.env` (which is gitignored). `/providers` shows what is keyed.
Decline and the session still opens: the slash commands work and the whole
deterministic path is there, because it never needed a model.

These are **API keys, billed per token**. A ChatGPT Plus/Pro or Claude Pro
subscription is a separate product and does not grant API access — those are
spent through the vendors' own apps, not here.

Model pins (`--model`, or `GEMINI_MODEL` / `ANTHROPIC_MODEL` / `OPENAI_MODEL`)
exist so a run can record which brain judged it. They go stale; a rejected id is
reported as a rejected id, with the override named.

**A provider that fails ends the session.** Out of quota, dead key, stale model
id — it says which, prints what was spent, and stops. It does not fail over onto
another of your keys, because that is your money and the decision to spend it is
yours. Nothing committed is lost; the next session's `/state` shows where the
run stands.

A stop is a banner, not a message from the model, and it never reads like one:

```
------------------------------------------------------------------
  OUT OF QUOTA
------------------------------------------------------------------
  That key is out of quota or over its rate limit. Spent before it stopped: 3
  request(s), 14,600 in, 270 out.

  What to do: This is a buy-your-own-tokens setup, so the fix is on the account
  the key belongs to -- top it up, wait out the window, or start again on
  another --provider you hold a key for.

  full detail: data/logs/session-20260818-234923.log
------------------------------------------------------------------
```

**Whatever already ran is still reported.** A key that expires on round four
does not erase rounds one to three: the tools that ran, ran, and the session
prints them before the banner.

A non-interactive session exits `0` only when it answered. A dead provider, a
budget ceiling and the step cap all exit `2` — a script that asked a question
and got a banner has not been answered, and a zero there is how a caller ends up
treating a spent key as a result. At a terminal only a dead provider ends the
session; a ceiling is something you raise and ask again.

## What a session costs

It runs on your key, so it tells you what it spent — once at the end of the
session, and on demand from `/cost`, rather than after every turn:

```
spent this session: 3 request(s), 35,420 in, 320 out
```

The per-turn figures are in the session log, where a question that turned out to
be expensive can be traced back to the round that made it so.

Two figures because two different limits bite.

**Requests** are what a free tier rations — Gemini's, for instance, counts
requests rather than tokens and allows only a few dozen a day. An orchestrator
spends one per tool round, so a single question can cost six, and three
questions can exhaust a day. This is the limit you will actually hit on a free
key; set `--model` to something with a larger allowance, enable billing on the
account, or switch `--provider` to a key you have paid for.

**Tokens** are what a paid key is billed on, and input dominates: the transcript
is re-sent every round, so a three-round question paid for its history three
times — 35,420 in against 320 out above. Sessions are expensive because they are
long, not because the model is verbose. Shorter sessions on focused questions
cost far less than one long one.

Ceilings, checked *before* each call rather than reported after:

```bash
--budget-requests 20     # or CONTINUUM_SPECTRANALYTICS_BUDGET_REQUESTS
--budget-tokens 200000   # or CONTINUUM_SPECTRANALYTICS_BUDGET_TOKENS
```

### The limit you will hit first is per *minute*

A free tier rations requests per minute as well as per day — Gemini's flash
models allow **5 a minute and 20 a day, per model id**. An agent spends one
request per tool round, so a session doing real work trips the per-minute
ceiling around round five with its daily allowance barely touched.

These are opposite problems and the session tells them apart. A daily quota is
spent and the fix costs money. A per-minute limit is only impatience: the
session waits the number of seconds the provider itself named and carries on.
That is the one place it retries, and it costs nothing — a rate-limited request
was *refused*, not served, so there are no tokens to pay for twice.

Cheaper than being refused is not asking too fast in the first place:

```bash
--min-request-interval 13    # or CONTINUUM_SPECTRANALYTICS_MIN_REQUEST_INTERVAL; paces a 5/min limit
```

Two more things follow from the per-model daily cap. A full end-to-end session
costs 12–19 rounds, so one model id is roughly one session a day on a free key —
and `--model` moves you onto a different allowance. And prompting the model to
**batch independent steps** into one turn genuinely helps: asked to decide seven
flagged regions one at a time it spends seven requests, asked to decide them
together it spends one.

Cost in currency is opt-in — set `CONTINUUM_SPECTRANALYTICS_RATES="in=0.30,out=2.50"` (per million
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
src/continuum_spectranalytics/
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
          transcript    what a person reads; everything else to data/logs/
          lanes         the two-lane numeric/visual judge
          lab           LAB.md, loaded whole into every session

  gui/    app         the window: panels, the event pump, the menus
          session     the turn thread, and the approval handshake across it
          widgets     transcript, state panel, figure viewer, the commit dialog
          theme       one palette; four things that must not be skimmed

  memory/ registry    index over past runs' provenance sidecars
          notes       what the agent concluded, kept between sessions
```

## Setup

```bash
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

That is the whole measurement path — no model libraries, no vendor SDKs. Add one
only for the provider you actually buy tokens from:

```bash
.venv\Scripts\python.exe -m pip install -e ".[gemini]"   # or [claude], [openai], [all]
.venv\Scripts\python.exe -m continuum_spectranalytics.cli setup            # take a key, check it works
```

`continuum-spectranalytics setup` is the first thing to run if you want the agent, and the last
thing you need if you do not: `intake`, `process`, `plot` and `--judge static`
run end to end with no key at all.

Three documents, depending on where you are starting:

| | |
|---|---|
| [docs/continuum-spectranalytics-setup.pdf](docs/continuum-spectranalytics-setup.pdf) | A new machine with no Python on it, assuming this repository is all you have. Covers the offline case. |
| [docs/LAB-SETUP.md](docs/LAB-SETUP.md) | Day-to-day use from a working checkout. |
| [docs/continuum-spectranalytics-architecture.pdf](docs/continuum-spectranalytics-architecture.pdf) | What the system is, what each tier may not do, the KaleidaGraph formats, and the known gaps. |

Note there is **no sample data in this repository** — everything under `data/`
is lab data and excluded. `pytest` is how you verify an install; it writes its
own synthetic acquisitions.

## Run

```bash
.venv\Scripts\python.exe -m continuum_spectranalytics.cli agent
.venv\Scripts\python.exe -m continuum_spectranalytics.cli gui
.venv\Scripts\python.exe -m continuum_spectranalytics.cli run --config examples/run_config.example.json
.venv\Scripts\python.exe -m continuum_spectranalytics.cli plot --file <any data file>
```

`plot` opens a file and tells you what is in it — ARGEN acquisitions,
KaleidaGraph `.QDA` and `.QPC`, processed CSVs, and ordinary delimited exports —
then draws any column against any other. It needs no key and no declared run.

`--judge static` (the default) takes the conservative option deterministically
and makes no network calls. `--judge model` uses the pinned Gemini model
(`gemini-3.6-flash`) and requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
Pro models need a paid Google plan; the free tier returns a zero quota for them.

Unlike the agent session, the tier-1 judge is **still Gemini-only** — it predates
the provider table and has not been moved onto it. `--judge static` is the
offline path either way, and it is what every test runs.

```bash
.venv\Scripts\python.exe -m pytest
```

The regression suite needs no API key or instrument file. The agent loop is exercised
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
