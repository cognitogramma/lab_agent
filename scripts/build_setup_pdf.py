"""Build the setup guide as a PDF, for standing up PolyRMC on a new machine.

Written on the assumption that the repository is the only thing the reader has:
no Python yet, no virtual environment, no data, and nobody sitting next to them.
That assumption is what makes this document different from ``docs/LAB-SETUP.md``,
which is the browsable copy for somebody already working in a checkout.

Two facts about a fresh clone shape the whole document, and both were nearly got
wrong:

* **There is no data in it.** Every path under ``data/`` is gitignored, because
  this repository is public and instrument traces are lab data. So the
  install cannot be verified by opening the example file that exists on a
  developer's machine and nowhere else. ``pytest`` is the verification, and it
  works precisely because it writes its own synthetic instrument files.
* **Installing needs the network.** numpy, scipy, pandas, pydantic, matplotlib
  and python-dotenv come from PyPI. A machine with no route to it needs the
  wheels carried across, which is covered at the end rather than left to be
  discovered.

Rendered with the same layout code as the architecture document::

    .venv\\Scripts\\python.exe scripts/build_setup_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from build_architecture_pdf import CSS, render  # noqa: E402
from polyrmc import paths  # noqa: E402


def document() -> str:
    return """
<h1>PolyRMC &mdash; setting up a new machine</h1>
<p class="sub">From a computer with nothing on it to a working installation,
assuming this repository is all you have.</p>

<p>Commands are Windows PowerShell, because that is what the instrument machines
run. Nothing here needs an API key until section 6, and most labs never get
there: the measurement path &mdash; opening files, processing runs, making
figures &mdash; runs with no model and no key at all.</p>

<h2>1. Install Python</h2>

<p>You need <b>Python 3.11 or newer</b>. Check whether it is already there:</p>

<pre>py --version</pre>

<p>If that prints a version of 3.11 or above, skip to section 2. Otherwise
install it from <b>python.org/downloads</b> and, at the first screen of the
installer, tick <b>&ldquo;Add python.exe to PATH&rdquo;</b>. Close and reopen
PowerShell afterwards, then check <code>py --version</code> again.</p>

<p class="rule">The Microsoft Store build of Python also works, but installs
into a sandboxed location that makes paths harder to reason about later. The
python.org installer is the easier one to support.</p>

<h2>2. Put the code somewhere</h2>

<p>Copy or clone the repository to a path <b>without spaces or OneDrive in
it</b>. <code>C:\\lab\\polyrmc</code> is a good choice;
<code>C:\\Users\\you\\OneDrive\\My Documents\\</code> is not &mdash; OneDrive
syncs files out from under a running process, and spaces make every later
command need quoting.</p>

<pre>cd C:\\lab\\polyrmc
dir</pre>

<p>You should see <code>pyproject.toml</code>, <code>src</code>,
<code>tests</code>, <code>docs</code> and <code>examples</code>. If
<code>pyproject.toml</code> is missing you are in the wrong folder, or one level
too high.</p>

<h2>3. Create the environment and install</h2>

<pre>py -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -e ".[test]"</pre>

<p>This needs a working internet connection: it fetches numpy, scipy, pandas,
pydantic, matplotlib and python-dotenv from PyPI, plus pytest so that section 4
can verify the result. It installs <b>no model libraries and no vendor SDKs</b>
&mdash; those are optional and covered in section 6.</p>

<p class="rule">Plain <code>pip install -e .</code> also works and gives you a
working installation; it just leaves out pytest, so the check in the next
section would have nothing to run.</p>

<p class="rule"><b>Note the <code>.\\.venv\\Scripts\\python.exe</code>
prefix.</b> A bare <code>python</code> is usually not on PATH on these machines,
and activating the environment is one more thing to forget between sessions.
Every command in this guide uses the full path, so it works from a cold
terminal.</p>

<p class="rule">Ignore <code>requirements.txt</code> if you find it. It predates
the packaging in <code>pyproject.toml</code>, lists the orchestration libraries
and all three vendor SDKs as though they were required, and installing from it
pulls in SDKs for providers you will never key. <code>pip install -e .</code> is
the supported route.</p>

<h2>4. Check the installation</h2>

<pre>.\\.venv\\Scripts\\python.exe -m pytest</pre>

<p><b>This is the verification.</b> Just under five hundred tests, needing no
API key, no network and no instrument file &mdash; they generate their own
synthetic acquisitions with known answers, so a pass means the arithmetic on
this machine is right, not merely that the imports resolved. It takes several
minutes, mostly rendering real figures.</p>

<p>Then check the command itself responds:</p>

<pre>.\\.venv\\Scripts\\python.exe -m polyrmc.cli plot</pre>

<p>It should answer <code>plot needs --file &lt;path&gt;</code>. A readable
sentence rather than a traceback means the entry point is wired up.</p>

<h3>There is no sample data, and that is deliberate</h3>

<p>A fresh copy of this repository contains <b>no instrument files and no
processed runs</b>. Everything under <code>data/</code> is excluded, because
traces and sidecars are lab data and this repository is public. So there is
nothing in the box to open, and the first real file you point it at will be one
of your own.</p>

<h2>5. Open your first file</h2>

<p>Point it at any data file with no other arguments and it describes what is
inside:</p>

<pre>.\\.venv\\Scripts\\python.exe -m polyrmc.cli plot --file "D:\\data\\run.txt"</pre>

<p>It reads ARGEN acquisitions, KaleidaGraph <code>.QDA</code> files, processed
CSVs this pipeline wrote, and ordinary delimited exports. It reports which kind
it decided the file was, which columns can go on an axis, and whether it had to
guess at the format.</p>

<p>Then draw something:</p>

<pre>.\\.venv\\Scripts\\python.exe -m polyrmc.cli plot `
    --file "D:\\data\\run.txt" --x seconds --y ls8</pre>

<p>The PNG lands beside the data file. <code>--out</code> puts it elsewhere,
<code>--style scatter</code> draws points, <code>--logy</code> gives a log axis.
Column names containing spaces need quotes; if you get one wrong, the error
lists the real ones.</p>

<p>Figures made this way carry an <b>exploratory</b> stamp in the corner, because
you chose those columns by hand and nothing was processed or reviewed. Figures
the pipeline produces are not stamped, because they were.</p>

<h2>6. Optional: the agent session</h2>

<p>Only needed to <i>talk</i> to the system. Install the SDK for the provider you
actually buy tokens from &mdash; there is no reason to install one for a vendor
you will never key:</p>

<pre>.\\.venv\\Scripts\\python.exe -m pip install -e ".[gemini]"
.\\.venv\\Scripts\\python.exe -m polyrmc.cli setup</pre>

<p>Substitute <code>[claude]</code>, <code>[openai]</code>, or <code>[all]</code>.
<code>polyrmc setup</code> lists the providers, shows which are keyed, gives you
the URL to buy a key, and spends one tiny call confirming the key works &mdash;
better to learn that now than three questions into a session. The key is stored
in <code>.env</code> at the project root, which is gitignored; setup prints the
exact path.</p>

<table>
<tr><th>Provider</th><th>Environment variable</th><th>Where to buy</th></tr>
<tr><td>gemini</td><td><code>GEMINI_API_KEY</code> or
<code>GOOGLE_API_KEY</code></td><td>aistudio.google.com/apikey</td></tr>
<tr><td>claude</td><td><code>ANTHROPIC_API_KEY</code></td>
<td>console.anthropic.com/settings/keys</td></tr>
<tr><td>openai</td><td><code>OPENAI_API_KEY</code></td>
<td>platform.openai.com/api-keys</td></tr>
</table>

<p class="warn">These are <b>API keys, billed per token</b>. A ChatGPT Plus or
Claude Pro subscription is a separate product and does not grant API access.</p>

<pre>.\\.venv\\Scripts\\python.exe -m polyrmc.cli agent</pre>

<p>Every turn prints what it spent; <code>/cost</code> gives the session total.
Anything that would <b>change</b> a run stops and asks you first. Reading and
computing do not, because approving every read only teaches you to click yes
without looking.</p>

<h2>7. Where files live</h2>

<p>Everything hangs off the project root &mdash; the folder holding
<code>pyproject.toml</code> &mdash; whichever directory you launch from. So you
can run these commands while sitting in the folder with your raw files and it
will still find the run history and your saved key.</p>

<p>To keep data somewhere else entirely, on a data drive or a share:</p>

<pre>$env:POLYRMC_ROOT = "D:\\lab\\polyrmc-data"</pre>

<p>Set it in the same terminal before running, or add it to the machine's
environment variables to make it permanent.</p>

<h2>8. A machine with no internet</h2>

<p>The install in section 3 is the only step that needs the network. On a
machine that cannot reach PyPI, fetch the wheels on one that can &mdash;
matching the target's Python version and Windows architecture &mdash; carry
them across, and install from the folder:</p>

<pre># on a connected machine
py -m pip download numpy scipy pandas pydantic matplotlib python-dotenv pytest -d wheels

# on the instrument machine, with wheels\\ copied alongside
.\\.venv\\Scripts\\python.exe -m pip install --no-index --find-links wheels -e .</pre>

<p>Nothing else in the system reaches the network unless you key a provider and
start a session.</p>

<h2>Troubleshooting</h2>

<table>
<tr><th>Symptom</th><th>What it means</th></tr>
<tr><td><code>py</code> is not recognised</td><td>Python is not installed, or was
installed without &ldquo;Add to PATH&rdquo;. Reopen PowerShell first; if it
still fails, re-run the installer and choose Modify.</td></tr>
<tr><td><code>python</code> is not recognised</td><td>Expected. Use
<code>.\\.venv\\Scripts\\python.exe</code>, not a bare
<code>python</code>.</td></tr>
<tr><td><code>No module named polyrmc</code></td><td>The install in section 3
did not run, or ran against a different environment. Re-run it from the folder
containing <code>pyproject.toml</code>.</td></tr>
<tr><td>Running scripts is disabled on this system</td><td>You tried to activate
the environment. You do not need to &mdash; call
<code>.\\.venv\\Scripts\\python.exe</code> directly.</td></tr>
<tr><td>&ldquo;could not be read as a table&rdquo;</td><td>The file has no header
row, or its numbers are quoted. Open the first two lines in a text
editor.</td></tr>
<tr><td>A column name is rejected</td><td>The message lists the real names. In an
ARGEN file the instrument's own time column is text; the numeric one is
<code>seconds</code>.</td></tr>
<tr><td>A <code>.QDA</code> column reads as empty</td><td>Read the notes printed
underneath. A column stored as text is reported by name rather than silently
skipped.</td></tr>
<tr><td>Somebody asks for a <code>.QDR</code></td><td>No such format exists.
KaleidaGraph writes <code>.QDA</code> for data and <code>.QPC</code> for a
graph; they mean one of those.</td></tr>
<tr><td>A provider failed</td><td>The message names the category. Out of quota is
a billing matter; a rejected key wants <code>polyrmc setup</code>; a stale model
id wants <code>--model</code>. It will not quietly fail over onto another of your
keys, because that is your money.</td></tr>
</table>

<h2>What to read next</h2>

<p><b><code>docs/polyrmc-architecture.pdf</code></b> &mdash; what the system is,
what each tier may and may not do, the reverse-engineered KaleidaGraph formats,
and the known gaps.</p>

<p><b><code>docs/LAB-SETUP.md</code></b> &mdash; the same ground as this
document, browsable in a text editor or on the repository page.</p>

<p>Both this guide and the architecture document are generated from the code
they describe. After changing anything they cover, rebuild:</p>

<pre>.\\.venv\\Scripts\\python.exe scripts\\build_setup_pdf.py
.\\.venv\\Scripts\\python.exe scripts\\build_architecture_pdf.py</pre>
"""


def main() -> int:
    out_path = paths.project_root() / "docs" / "polyrmc-setup.pdf"
    pages = render(document(), out_path)
    print(f"wrote {paths.relative_to_root(out_path)} ({pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
