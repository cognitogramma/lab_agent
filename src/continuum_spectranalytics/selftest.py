"""Check that a build can do what the lab relies on, without a model key.

Run by the packaged app as ``"Continuum Spectranalytics.exe" --self-test
<report.json>`` (the Windows build runs it after every build), or from source
as ``python -m continuum_spectranalytics.selftest <report.json>``.

A frozen executable fails in ways a source checkout never does: a native
library or a package's data file left out of the bundle only shows up when the
feature that needs it is used. So this uses each one -- every attachment reader,
drag-and-drop, and each model provider's client -- and writes what happened.
Exit code 0 means everything passed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable


def _samples(folder: Path) -> dict[str, Path]:
    import docx
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.backends.backend_pdf import PdfPages
    from pptx import Presentation
    from pptx.util import Inches

    files: dict[str, Path] = {}
    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [1, 3, 2])
    files["image"] = folder / "plot.png"
    figure.savefig(files["image"])
    plt.close(figure)

    files["pdf"] = folder / "paper.pdf"
    with PdfPages(files["pdf"]) as pdf:
        page = plt.figure()
        page.text(0.1, 0.8, "Zimm analysis of lysozyme. A2 and Mw.")
        pdf.savefig(page)
        plt.close(page)

    files["word"] = folder / "protocol.docx"
    document = docx.Document()
    document.add_heading("Protocol", 1)
    document.add_paragraph("Filter the buffer.")
    document.add_picture(str(files["image"]))
    document.save(files["word"])

    files["powerpoint"] = folder / "talk.pptx"
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Results"
    slide.shapes.add_picture(str(files["image"]), Inches(1), Inches(2))
    deck.save(files["powerpoint"])

    files["excel"] = folder / "series.xlsx"
    pd.DataFrame({"conc": [0.1, 0.2], "KcR": [1.1, 1.2]}).to_excel(files["excel"], index=False)
    files["csv"] = folder / "export.csv"
    files["csv"].write_text("time,signal\n0,1.0\n1,1.5\n")
    files["text"] = folder / "notes.md"
    files["text"].write_text("# Notes\n")
    return files


def _attachments(folder: Path) -> str:
    from continuum_spectranalytics.attachments import AttachmentShelf
    from continuum_spectranalytics.filesystem import FileAccess

    expected = {"image": "image", "pdf": "pdf", "word": "word", "powerpoint": "powerpoint",
                "excel": "excel", "csv": "delimited", "text": "text"}
    shelf = AttachmentShelf(FileAccess())
    for name, path in _samples(folder).items():
        attachment = shelf.add(path)
        if attachment.kind != expected[name]:
            raise AssertionError(f"{path.name}: recognised as {attachment.kind} ({attachment.reason})")
        if attachment.kind != "image":
            shelf.read(attachment.id)
        if attachment.kind in {"image", "pdf", "word", "powerpoint"}:
            shelf.read(attachment.id, view=None if attachment.kind == "image" else 1)
    images = shelf.take_pending_images()
    if len(images) != 4:
        raise AssertionError(f"expected 4 viewable pictures, got {len(images)}")
    return f"{len(expected)} kinds read, {len(images)} pictures rendered"


def _drag_and_drop() -> str:
    import tkinter as tk

    from continuum_spectranalytics.gui import dropfiles

    root = tk.Tk()
    root.withdraw()
    try:
        if not dropfiles.enable(root):
            raise AssertionError("tkdnd did not load")
        frame = tk.Frame(root)
        dropfiles.register(root, [frame], lambda paths: None)
        if not root.tk.call("bind", frame._w, "<<Drop>>"):
            raise AssertionError("drop target was not registered")
        return f"tkdnd {root.tk.call('package', 'present', 'tkdnd')}"
    finally:
        root.destroy()


def _gui_imports() -> str:
    import customtkinter  # noqa: F401

    from continuum_spectranalytics.gui import app, clipboard_files  # noqa: F401

    return "customtkinter and the window modules import"


def _providers() -> str:
    from continuum_spectranalytics.agent.toolcall import SPECS, _client_for

    built = []
    for spec in SPECS.values():
        _client_for(spec, spec.default_model, "self-test-not-a-key", 0.2)
        built.append(spec.name)
    return "clients built: " + ", ".join(built)


def _prompts() -> str:
    from continuum_spectranalytics.agent.orchestrator import default_system_prompt
    from continuum_spectranalytics.agent.toolspec import build_registry

    prompt = default_system_prompt()
    if "Files the researcher gives you" not in prompt:
        raise AssertionError("the assistant's file instructions are missing from the bundle")
    tools = build_registry()
    for name in ("read_attachment", "copy_file", "open_file"):
        if name not in tools:
            raise AssertionError(f"tool {name} is missing")
    return f"{len(tools)} tools, prompt {len(prompt):,} characters"


def run() -> dict[str, dict[str, str]]:
    checks: dict[str, Callable[[], str]] = {
        "gui": _gui_imports,
        "drag_and_drop": _drag_and_drop,
        "providers": _providers,
        "prompts": _prompts,
    }
    results: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="continuum-selftest-") as scratch:
        checks["attachments"] = lambda: _attachments(Path(scratch))
        for name, check in checks.items():
            try:
                results[name] = {"ok": "yes", "detail": check()}
            except BaseException as error:  # noqa: BLE001 - a report, not a crash
                results[name] = {"ok": "no", "detail": f"{type(error).__name__}: {error}",
                                 "trace": traceback.format_exc()}
    return results


def startup_probe() -> int:
    """Do what opening the window does -- load the app and start Tk -- then quit.

    Timed by the Windows build, so startup is measured on every build rather
    than guessed at. Exit code 0 on success.
    """
    import tkinter as tk

    from continuum_spectranalytics import cli  # noqa: F401 - the real import cost
    from continuum_spectranalytics.gui import app  # noqa: F401

    root = tk.Tk()
    root.withdraw()
    root.update()
    root.destroy()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    results = run()
    text = json.dumps(results, indent=2)
    if argv:
        Path(argv[0]).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if all(r["ok"] == "yes" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
