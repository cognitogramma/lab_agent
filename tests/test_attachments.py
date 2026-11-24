"""Every kind of file a researcher drops is recognised, summarised and readable."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from continuum_spectranalytics import filesystem
from continuum_spectranalytics.agent.orchestrator import ApprovalPolicy, Orchestrator, allow_all
from continuum_spectranalytics.agent.toolcall import (
    AssistantTurn, LangChainToolProvider, ScriptedProvider, ToolCall,
)
from continuum_spectranalytics.agent.toolspec import ToolContext, build_registry
from continuum_spectranalytics.attachments import PART_CHARS, AttachmentShelf
from continuum_spectranalytics.filesystem import FileAccess
from continuum_spectranalytics.memory.notes import NoteBook
from continuum_spectranalytics.state import RunState
from continuum_spectranalytics.tools import Rejected, ToolSurface
from conftest import write_argen_file

pytest.importorskip("pypdfium2")
pytest.importorskip("docx")
pytest.importorskip("pptx")
pytest.importorskip("openpyxl")
pytest.importorskip("PIL")


@pytest.fixture(autouse=True)
def no_policy(monkeypatch, tmp_path):
    """The shipped default: no filesystem-permissions.json."""
    monkeypatch.setattr(filesystem, "policy_path", lambda: tmp_path / "absent" / filesystem.POLICY_NAME)


@pytest.fixture(scope="module")
def samples(tmp_path_factory) -> dict[str, Path]:
    import docx
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.backends.backend_pdf import PdfPages
    from pptx import Presentation
    from pptx.util import Inches

    out = tmp_path_factory.mktemp("lab files")
    files = {"argen": write_argen_file(out / "argen run 12.txt")}

    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [1, 3, 2])
    axes.set_ylabel("LS8 (V)")
    files["image"] = out / "kinetics plot.png"
    figure.savefig(files["image"])
    plt.close(figure)

    files["pdf"] = out / "paper.pdf"
    with PdfPages(files["pdf"]) as pdf:
        for page in (1, 2, 3):
            sheet = plt.figure()
            sheet.text(0.1, 0.8, f"Zimm analysis of lysozyme, page {page}. A2 and Mw.", fontsize=8)
            pdf.savefig(sheet)
            plt.close(sheet)

    files["word"] = out / "protocol.docx"
    document = docx.Document()
    document.add_heading("Filtration protocol", 1)
    document.add_paragraph("Filter the buffer through 0.22 um before every run.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "conc", "dn/dc"
    table.cell(1, 0).text, table.cell(1, 1).text = "0.5", "0.185"
    document.add_picture(str(files["image"]))
    document.save(files["word"])

    files["powerpoint"] = out / "group meeting.pptx"
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Aggregation results"
    slide.shapes.add_picture(str(files["image"]), Inches(1), Inches(2))
    deck.save(files["powerpoint"])

    files["excel"] = out / "series.xlsx"
    with pd.ExcelWriter(files["excel"]) as writer:
        pd.DataFrame({"conc": [0.1, 0.2], "KcR": [1.1, 1.2]}).to_excel(writer, sheet_name="Data", index=False)

    files["csv"] = out / "export.csv"
    files["csv"].write_text("time,signal\n0,1.0\n1,1.5\n2,1.7\n")
    files["text"] = out / "ideas.md"
    files["text"].write_text("# Ideas\nTry a lower stir speed.\n" + "more notes. " * 1500)
    files["legacy"] = out / "old notes.doc"
    files["legacy"].write_bytes(b"\xd0\xcf\x11\xe0 not really")
    files["binary"] = out / "dump.bin"
    files["binary"].write_bytes(bytes(range(256)) * 8)
    files["broken_pdf"] = out / "broken.pdf"
    files["broken_pdf"].write_bytes(b"%PDF-1.4 truncated")
    return files


@pytest.fixture
def shelf() -> AttachmentShelf:
    return AttachmentShelf(FileAccess())


EXPECTED = {
    "argen": ("argen", "ls8"),
    "image": ("image", "pixels"),
    "pdf": ("pdf", "3 page(s)"),
    "word": ("word", "Filtration protocol"),
    "powerpoint": ("powerpoint", "Aggregation results"),
    "excel": ("excel", "conc, KcR"),
    "csv": ("delimited", "time, signal"),
    "text": ("text", "# Ideas"),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_kind_is_recognised_and_summarised(samples, shelf, name):
    kind, phrase = EXPECTED[name]
    attachment = shelf.add(samples[name])
    assert attachment.kind == kind
    assert phrase in attachment.brief()
    assert 'read_attachment(id="a1"' in attachment.brief()


@pytest.mark.parametrize("name, phrase", [
    ("argen", "ls8"), ("pdf", "--- page 2 ---"), ("word", "# Filtration protocol"),
    ("powerpoint", "--- slide 1 ---"), ("excel", "0.1,1.1"), ("csv", "time,signal"),
])
def test_each_kind_reads_as_text(samples, shelf, name, phrase):
    attachment = shelf.add(samples[name])
    assert phrase in "".join(shelf.read(attachment.id, part) for part in range(1, attachment.parts() + 1))


def test_word_tables_keep_their_cells_together(samples, shelf):
    attachment = shelf.add(samples["word"])
    assert "| 0.5 | 0.185 |" in shelf.read(attachment.id)


def test_long_files_come_in_parts(samples, shelf):
    attachment = shelf.add(samples["text"])
    assert attachment.parts() > 1
    first = shelf.read(attachment.id, 1)
    assert f"part 1 of {attachment.parts()}" in first
    assert 'part=2' in first
    assert len(first) < PART_CHARS + 400
    assert "[end of file]" in shelf.read(attachment.id, attachment.parts())
    with pytest.raises(Rejected, match="part"):
        shelf.read(attachment.id, attachment.parts() + 1)


@pytest.mark.parametrize("name, view, label", [
    ("pdf", 2, "paper.pdf, page 2"),
    ("word", 1, "protocol.docx, picture 1"),
    ("powerpoint", 1, "group meeting.pptx, picture 1"),
    ("image", None, "kinetics plot.png"),
])
def test_pages_and_figures_can_be_viewed_as_images(samples, shelf, name, view, label):
    attachment = shelf.add(samples[name])
    reply = shelf.read(attachment.id, view=view)
    assert "shown to you" in reply
    images = shelf.take_pending_images()
    assert [image.label for image in images] == [label]
    assert images[0].media_type in {"image/png", "image/jpeg"}
    assert shelf.take_pending_images() == []


@pytest.mark.parametrize("name, reason", [
    ("legacy", "Save As"),
    ("binary", "not one Continuum reads"),
    ("broken_pdf", "could not be opened"),
])
def test_unreadable_files_say_why_and_what_to_do(samples, shelf, name, reason):
    attachment = shelf.add(samples[name])
    assert attachment.kind == "unsupported"
    assert reason in attachment.reason
    with pytest.raises(Rejected, match="cannot be read"):
        shelf.read(attachment.id)


def test_big_images_are_shrunk_before_sending(tmp_path, shelf):
    from PIL import Image

    path = tmp_path / "photo of notebook.jpg"
    Image.new("RGB", (5000, 3000), "white").save(path)
    shelf.add(path)
    _, images = shelf.take_for_message()
    from base64 import b64decode
    from io import BytesIO

    sent = Image.open(BytesIO(b64decode(images[0].data)))
    assert max(sent.size) == 1568


def test_attaching_the_same_file_twice_reuses_its_id(samples, shelf):
    first = shelf.add(samples["csv"])
    shelf.take_for_message()
    again = shelf.add(samples["csv"])
    assert again.id == first.id
    assert shelf.unsent() == [again]


def test_unknown_ids_list_what_is_attached(samples, shelf):
    shelf.add(samples["csv"])
    with pytest.raises(Rejected, match="a1 \\(export.csv\\)"):
        shelf.read("a9")
    assert "time,signal" in shelf.read("export.csv")  # a model passing the name still works


def test_folders_and_credentials_are_refused(tmp_path, shelf):
    with pytest.raises(Rejected, match="folder"):
        shelf.add(tmp_path)
    secret = tmp_path / ".env"
    secret.write_text("KEY=1")
    with pytest.raises(Rejected, match="protected"):
        shelf.add(secret)


def test_excel_also_works_with_the_plotting_tools(samples, run_config, tmp_path):
    context = _context(run_config, tmp_path)
    reply = build_registry()["open_data_file"].handler(context, {"path": str(samples["excel"])})
    assert "excel" in reply and "conc" in reply


# --- the hand-over to the assistant -------------------------------------------


def _context(run_config, tmp_path) -> ToolContext:
    surface = ToolSurface(state=RunState(run_id="t", source_file=str(run_config.source_file)), config=run_config)
    return ToolContext(surface=surface, notes=NoteBook(tmp_path / "memory"), runs_dir=tmp_path / "runs")


def test_attachments_go_with_the_next_message_only(samples, run_config, tmp_path):
    context = _context(run_config, tmp_path)
    context.attachments.add(samples["pdf"])
    context.attachments.add(samples["image"])
    provider = ScriptedProvider([AssistantTurn(text="read it"), AssistantTurn(text="ok")])
    agent = Orchestrator(context=context, provider=provider, registry=build_registry(),
                         approvals=ApprovalPolicy(ask=allow_all))

    agent.ask("what does this paper say?")
    first = provider.calls[0][1][-1]
    assert first.text.startswith("what does this paper say?")
    assert "[a1] paper.pdf -- PDF document" in first.text
    assert "[a2] kinetics plot.png -- image" in first.text
    assert [image.label for image in first.images] == ["kinetics plot.png"]

    agent.ask("thanks")
    second = provider.calls[1][1]
    assert second[-1].text == "thanks" and not second[-1].images
    # The earlier picture is not resent on later questions.
    assert not second[0].images and "not resent" in second[0].text


def test_a_viewed_page_reaches_the_model_after_the_tool_round(samples, run_config, tmp_path):
    context = _context(run_config, tmp_path)
    context.attachments.add(samples["pdf"])
    provider = ScriptedProvider([
        AssistantTurn(tool_calls=[ToolCall("c1", "read_attachment", {"id": "a1", "view": 1})]),
        AssistantTurn(text="Figure 1 shows..."),
    ])
    agent = Orchestrator(context=context, provider=provider, registry=build_registry(),
                         approvals=ApprovalPolicy(ask=allow_all))
    agent.ask("what is in the first figure?")

    sent = provider.calls[1][1]
    assert [m.role for m in sent[-3:]] == ["assistant", "tool", "user"]
    assert "shown to you" in sent[-2].results[0].content
    assert [image.label for image in sent[-1].images] == ["paper.pdf, page 1"]


def test_images_are_sent_in_langchains_standard_block():
    from continuum_spectranalytics.agent.toolcall import Message
    from continuum_spectranalytics.attachments import ImagePart

    converted = LangChainToolProvider._to_langchain(
        [Message(role="user", text="look", images=[ImagePart("image/png", "AAAA", "plot.png")])]
    )
    blocks = converted[0].content
    assert blocks[0] == {"type": "text", "text": "look"}
    assert blocks[-1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


# --- the window ---------------------------------------------------------------------


@pytest.fixture
def app(gui_app):
    yield gui_app
    for attachment in gui_app.session.context.attachments:
        gui_app.session.context.attachments.remove(attachment.id)
    gui_app.render_attachments()
    gui_app.entry.delete("1.0", "end")
    gui_app.transcript.clear()


def _settle(app, seconds=0.2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update()
        time.sleep(0.005)


def _chip_texts(app) -> list[str]:
    import customtkinter as ctk

    return [" ".join(str(w.cget("text")) for w in chip.winfo_children())
            for chip in app.attach_bar.winfo_children() if isinstance(chip, ctk.CTkFrame)]


def test_an_empty_bar_says_how_to_attach(app):
    import customtkinter as ctk

    app.render_attachments()
    tips = [w.cget("text") for w in app.attach_bar.winfo_children() if isinstance(w, ctk.CTkLabel)]
    assert any("Ctrl+V" in tip for tip in tips)


def test_dropped_files_become_chips_and_an_argen_file_offers_open_as_run(app, samples):
    app.attach_files([str(samples["argen"]), str(samples["pdf"]), str(samples["legacy"])])
    _settle(app)
    chips = _chip_texts(app)
    assert any("argen run 12.txt" in c and "Open as run" in c for c in chips)
    assert any("paper.pdf" in c and "Open as run" not in c for c in chips)
    shown = app.transcript.plain()
    assert "attached paper.pdf -- PDF document" in shown
    assert "Save As" in shown  # the .doc refusal says how to fix it


def test_the_cross_on_a_chip_removes_the_file(app, samples):
    app.attach_files([str(samples["csv"])])
    _settle(app)
    (attachment,) = app.session.context.attachments.unsent()
    app._unattach(attachment.id)
    _settle(app)
    assert _chip_texts(app) == []
    assert not app.session.context.attachments.unsent()
