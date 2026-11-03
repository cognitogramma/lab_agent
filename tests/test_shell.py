"""Navigation and home actions use real callbacks and real library totals."""

from __future__ import annotations

import pytest

from polyrmc.gui.shell import HomePanel, LibraryRail


@pytest.fixture
def shell_factory(gui_app):
    from polyrmc.gui.theme import build_fonts

    created = []
    fonts = build_fonts(gui_app)

    def make(kind, **callbacks):
        widget = kind(gui_app, fonts, **callbacks)
        created.append(widget)
        return widget

    yield make
    for widget in created:
        widget.destroy()
    gui_app.update_idletasks()


def test_library_buttons_select_the_page_then_invoke_its_callback(shell_factory):
    calls = []
    rail = shell_factory(LibraryRail, on_select=lambda key: calls.append((key, rail.selected)),
                         on_open=lambda: None, on_settings=lambda: None)
    for key in ("research", "figures", "runs", "notes", "tools", "run"):
        rail.buttons[key].invoke()
        assert rail.selected == key
        assert calls[-1] == (key, key)


def test_programmatic_navigation_does_not_reenter_the_callback(shell_factory):
    calls = []
    rail = shell_factory(LibraryRail, on_select=calls.append,
                         on_open=lambda: None, on_settings=lambda: None)
    rail.select("notes")
    assert rail.selected == "notes"
    assert calls == []
    with pytest.raises(ValueError, match="Unknown library page"):
        rail.select("missing")
    assert rail.selected == "notes"


def test_library_open_and_settings_use_separate_real_actions(shell_factory):
    calls = []
    rail = shell_factory(LibraryRail, on_select=calls.append,
                         on_open=lambda: calls.append("open"), on_settings=lambda: calls.append("settings"))
    rail.open_button.invoke()
    rail.settings_button.invoke()
    assert calls == ["open", "settings"]


def home(shell_factory, calls):
    return shell_factory(HomePanel, on_open=lambda: calls.append("open"),
                         on_research=lambda: calls.append("research"),
                         on_runs=lambda: calls.append("runs"), on_notes=lambda: calls.append("notes"))


def test_home_cards_and_notes_are_working_entry_points(shell_factory):
    calls = []
    panel = home(shell_factory, calls)
    for key in ("open", "research", "runs"):
        panel.action_buttons[key].invoke()
    panel.notes_button.invoke()
    panel.runs_button.invoke()
    assert calls == ["open", "research", "runs", "notes", "runs"]


def test_home_never_invents_recent_runs_or_library_totals(shell_factory):
    panel = home(shell_factory, [])
    assert panel.library_counts == {"runs": None, "notes": None}
    panel.set_library(runs=12, notes=1)
    assert panel.library_counts == {"runs": 12, "notes": 1}
    assert panel.runs_button.cget("text") == "12 experiments in your library  →"
    assert panel.notes_button.cget("text") == "1 saved lab note  →"
    panel.set_library(runs=0, notes=0)
    assert panel.runs_button.cget("text").startswith("0 experiments")


@pytest.mark.parametrize("runs,notes", [(-1, 2), (1, -1), (True, 1), (1, "3")])
def test_invalid_counts_do_not_replace_a_known_library_total(shell_factory, runs, notes):
    panel = home(shell_factory, [])
    panel.set_library(4, 5)
    with pytest.raises(ValueError, match="non-negative integers"):
        panel.set_library(runs, notes)
    assert panel.library_counts == {"runs": 4, "notes": 5}
