# polyrmc-agentic

An agentic AI research assistant for the lab, built with LangChain/LangGraph
and Claude (Opus 4.8). A single agent with tools for:

- **Literature search** — queries the arXiv API for papers.
- **Data analysis** — lists/reads files in `data/` and computes summary
  statistics for CSV datasets.
- **Lab notes** — saves and recalls persistent notes in `data/notes/`
  (they survive across sessions).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then paste your Anthropic API key into .env
```

## Run

From the project root:

```powershell
python -m src.main
```

Example prompts:

- "Find recent papers on reverse Monte Carlo modeling of polymers."
- "What data files do we have? Summarize measurements.csv."
- "Save a note: the RMC run converged after 5e6 steps with chi^2 = 1.2."
- "What notes do we have from earlier?"

## Tests

Offline tests (no API key needed):

```powershell
python -m pytest tests -v
```

## Project layout

```
src/
  config.py   # model choice, paths, system prompt
  tools.py    # the agent's tools (arXiv, data files, CSV stats, notes)
  agent.py    # builds the LangGraph agent
  main.py     # interactive CLI chat loop
data/         # your datasets; data/notes/ holds the agent's saved notes
tests/        # offline tool tests
```

## Extending

Add a new capability by writing a function in `src/tools.py`, decorating it
with `@tool` (docstring becomes the tool description the model sees), and
appending it to `ALL_TOOLS`. The agent picks it up automatically.
