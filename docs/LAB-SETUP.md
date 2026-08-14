# PolyRMC — setting it up in the lab

Everything here is Windows PowerShell, because that is what the instrument
machines run. Nothing in this document needs an API key until the section that
says it does.

> **Standing up a brand-new machine?** Read
> [`docs/polyrmc-setup.pdf`](polyrmc-setup.pdf) instead. It starts from a
> computer with no Python on it and assumes this repository is all you have,
> including the case where the machine cannot reach the internet. This document
> assumes you already have a working checkout.

---

## 1. Install

You need **Python 3.11 or newer**. Check with `py --version`.

```powershell
cd <wherever you keep the project>
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

That installs the measurement path and nothing else: numpy, scipy, pandas,
pydantic, matplotlib, plus pytest so you can verify the result. No model
libraries, no vendor SDKs. Plain `pip install -e .` also gives a working
installation — it just leaves out pytest, so there would be nothing to run the
check with.

Check it worked:

```powershell
.\.venv\Scripts\python.exe -m polyrmc.cli plot --file examples\run_config.example.json
```

It should refuse that file politely. A refusal with a readable sentence means the
install is fine.

> **Note the `.\.venv\Scripts\python.exe` prefix.** A bare `python` is usually not
> on PATH on these machines, and activating the venv is one more thing to
> forget. Every command below uses the full path so it works from a cold
> terminal.

> **Ignore `requirements.txt`.** It predates the packaging in `pyproject.toml`,
> lists the orchestration libraries and all three vendor SDKs as though they
> were required, and installing from it pulls in SDKs for providers you will
> never key. `pip install -e .` is the supported route.

### There is no sample data in the repository

A fresh copy contains **no instrument files and no processed runs** —
everything under `data/` is excluded, because traces and sidecars are lab data
and this repository is public. So there is nothing in the box to open, and the
first real file you point it at will be one of your own. This is also why
`pytest` rather than "open the example file" is how you verify an install: the
tests write their own synthetic acquisitions.

### If you want the agent session as well

Only needed to *talk* to the system. Pick the vendor you actually buy tokens
from — there is no reason to install an SDK for one you will never key:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gemini]"   # or [claude], [openai], [all]
```

---

## 2. The part that needs no key

**This is most of the system.** If you only want to process data and make
figures, you are already finished installing.

| Command | What it does | Key needed |
|---|---|---|
| `polyrmc plot --file <f>` | See what is in a data file | no |
| `polyrmc plot --file <f> --x A --y B` | Make a PNG | no |
| `polyrmc intake` | Declare an experiment | no |
| `polyrmc process` | Run the pipeline | no |
| `polyrmc run --judge static` | The whole path, deterministically | no |
| `polyrmc agent` | Talk to it | **yes** |

---

## 3. Looking at a file

Point it at anything and it tells you what is inside:

```powershell
.\.venv\Scripts\python.exe -m polyrmc.cli plot --file "C:\data\conductivity.QDA"
```

```
conductivity.QDA: kaleidagraph, 24000 rows, 10 columns

plottable columns:
  time (sec)
  conductivity (uS/cm)
  ...
```

It reads **ARGEN acquisitions**, **KaleidaGraph `.QDA` files**, processed CSVs
this pipeline wrote, and ordinary tab- or comma-delimited exports. It tells you
which kind it decided the file was, and says so when it had to guess.

### Making a figure

```powershell
.\.venv\Scripts\python.exe -m polyrmc.cli plot `
    --file "C:\data\conductivity.QDA" `
    --x "time (sec)" --y "conductivity (uS/cm)"
```

The PNG lands **beside the data file**. Add `--out <path>` to put it elsewhere,
`--style scatter` for points, `--logy` for a log axis.

Column names with spaces need quotes. If you get the name wrong it tells you
what the real ones are, so you can just fix it and re-run.

> Figures made this way are stamped **"exploratory plot"** in the corner. That
> is deliberate: you chose those columns by hand and nothing was processed or
> reviewed. Figures produced by the pipeline are not stamped, because they were.

### KaleidaGraph graph files

A `.QPC` is a *graph*, and it contains **no data** — the numbers are in the
`.QDA` it was built from. Opening one tells you what was plotted and which curve
fit was used:

```powershell
.\.venv\Scripts\python.exe -m polyrmc.cli plot --file "C:\data\Stress Plot.QPC"
```

There is **no `.QDR` format**. KaleidaGraph writes `.QDA` (data), `.QPC` (graph),
`.QPL` (layout), `.QSC` (script), `.QST` (style) and `.QCT` (colour table). If
somebody asks you for a `.QDR`, they mean one of those two.

---

## 4. Processing a run

```powershell
.\.venv\Scripts\python.exe -m polyrmc.cli intake --raw "C:\data\run.txt"
.\.venv\Scripts\python.exe -m polyrmc.cli run --config <the config intake wrote>
```

Intake asks you to **declare** the experiment — what the sample is, what
`dn_dc` to use, what kind of run it was. It will not infer these from the trace.
`dn_dc` is 0.185 for a protein and 0.150 for a polysaccharide, and guessing it
processes your run under a physical model nobody chose while producing a number
that looks perfectly reasonable.

Each processed run writes three things next to each other:

```
data/runs/<run>_ls8.csv                  the processed data
data/runs/<run>_ls8.provenance.json      every decision taken to produce it
data/runs/figures/<run>_*.png            the figure set
```

**The sidecar is the point.** Open it when you need to know why a number is what
it is: which smoothing window was used, what was excised and why, whether a
baseline was subtracted, what the declarations were, and what the run was
blocked from computing.

---

## 5. If you want the agent session

Only now do you need a key. These are **API keys, billed per token** — a ChatGPT
Plus or Claude Pro subscription is a different product and does not work here.

```powershell
.\.venv\Scripts\python.exe -m polyrmc.cli setup
```

It lists the providers, shows which are already keyed, and gives you the URL to
buy one. Paste a key and it spends one tiny call to confirm the key actually
works — better to find out now than three questions into a session.

The key is stored in `.env` at the project root, which is gitignored. `polyrmc
setup` prints the exact path it used.

| Provider | Key | Where |
|---|---|---|
| gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | <https://aistudio.google.com/apikey> |
| claude | `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> |
| openai | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> |

Then:

```powershell
.\.venv\Scripts\python.exe -m polyrmc.cli agent
```

Every turn prints what it spent. `/cost` gives the session total. Anything that
would **change** a run stops and asks you first; reading and computing do not,
because approving every read just trains you to click yes without looking.

---

## 6. Where files live

By default everything hangs off the project root — the folder with
`pyproject.toml` in it — no matter which directory you launch from. So you can
run `polyrmc` from the folder holding your raw data and it will still find the
run history and your saved key.

To keep data somewhere else entirely:

```powershell
$env:POLYRMC_ROOT = "D:\lab\polyrmc-data"
```

---

## 7. Checking the install is sound

```powershell
.\.venv\Scripts\python.exe -m pytest
```

497 tests, no API key, no network, no instrument file required. Takes several
minutes because it renders real figures. If these pass, the measurement path on
this machine is behaving.

---

## Troubleshooting

**`python` is not recognised** — use `.\.venv\Scripts\python.exe`, not `python`.

**A column name is rejected** — the error lists the real column names. In an
ARGEN file the instrument's own time column is text; the numeric one is
`seconds`.

**"could not be read as a table"** — the file has no header row, or its numbers
are quoted. Open it in a text editor and look at the first two lines.

**A `.QDA` column reads as empty** — check the notes printed underneath. A
column stored as text (an ARGEN `elapsedtime`, for instance) is reported by name
rather than silently skipped.

**The agent says a provider failed** — it names the category. Out of quota is a
billing matter; a rejected key wants `polyrmc setup`; a stale model id wants
`--model`. It will not quietly fail over onto another of your keys, because that
is your money.

---

## Further reading

`docs/polyrmc-architecture.pdf` — what the system is, what each tier may and may
not do, the reverse-engineered KaleidaGraph format, and the known gaps. Rebuild
it after changing anything it describes:

```powershell
.\.venv\Scripts\python.exe scripts\build_architecture_pdf.py
```
