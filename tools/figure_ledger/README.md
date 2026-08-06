# Figure ledger

Label figure images one at a time. Writes `ledger.json` into the image folder.

## Running it

Open a terminal, paste one line, press Enter:

```bash
python tools/figure_ledger/serve.py "C:\path\to\your\figures"
```

A browser opens on the tool. **Leave the terminal window alone** — closing it
stops the tool. When you are finished, click back to the terminal and press
`Ctrl-C`; it saves on the way out and tells you where.

If your `python` is the project venv:

```bash
.\.venv\Scripts\python.exe tools\figure_ledger\serve.py "C:\path\to\your\figures"
```

Nothing to install. Standard library only — no `pip install`, no `npm install`.

If the browser does not open by itself, the terminal prints a link. If the port
is busy (another copy still running), add `--port 8766`.

## Using it

| | |
|---|---|
| **Left / right arrows** | previous / next image |
| **Copy previous** | carries the whole previous record forward — then change the two fields that differ |
| **Clear** | empties the current record |
| **CSV** | downloads `ledger.csv` |
| Filter box | narrows the sidebar by filename |
| All / To do / Done | narrows by whether the record is logged |

Typing saves by itself after about 400 ms. The indicator next to the filename
says `editing…` then `saved`. A green dot appears in the sidebar once a record
has **a method, a sample type, and a goal**.

## Where your work goes

`ledger.json`, in the folder you pointed at. Real file, on disk, one record per
image — never browser storage.

Every write goes to a temporary file first and is then renamed over the target.
A crash or a power cut mid-write leaves you with either the old ledger or the
new one, never a half-written file.

An existing `ledger.json` is **read and resumed**, so the 13 figures you have
already labelled stay labelled. If that file is ever unreadable the tool refuses
to start rather than overwriting it.

## About your images

Files out of `pdfimages` are frequently Netpbm data wearing a `.png` extension —
`pdfimages` writes PPM/PGM/PBM unless you pass `-png`, and renaming converts
nothing. Every file here is identified by its **magic bytes**, and Netpbm P1–P6
is decoded and converted so it displays. You do not need to re-extract anything.

Where a file genuinely cannot be shown, the tool names the format: *"This is
TIFF (little-endian)"* — so you know what to convert rather than just that
something failed.

## Adding vocabulary

Open `app.html` and edit the `VOCAB` object at the top of the `<script>` block.
Add a method, a measured quantity, a goal, a sub-option — it appears in the form
with no other change. Field names are deliberately stable so panel-level records
can extend them later without a migration.

## Checking it still works

```bash
python tools/figure_ledger/test_figure_ledger.py
```

45 tests, no pytest needed.

## Why a server rather than a single HTML file

A single file using the File System Access API would have been nicer, and it was
the first thing I tried to make work. It cannot: `showDirectoryPicker` requires
a secure context, and a page opened by double-clicking an HTML file runs on
`file://`, where the call throws. Every workaround ends in serving the page from
somewhere — which is a server. It is also Chromium-only, and decoding Netpbm in
the browser would mean shipping a JavaScript decoder to solve a problem Python
handles in forty lines.

So: a server, but a small one. No dependencies, no build step, three routes —
the page, the images, the ledger. The page arrives with its data already in it,
so the first paint has content rather than an empty shell.
