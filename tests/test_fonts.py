"""Headings and prose resolve the requested serif family with installed fallbacks."""

import tkinter.font as tkfont

import pytest

from continuum_spectranalytics.gui import theme


@pytest.fixture(scope="module")
def root(gui_app):
    return gui_app


def test_available_heading_and_reading_fonts_are_selected_independently(root):
    fonts = theme.build_fonts(root)
    installed = {name.casefold() for name in tkfont.families(root)}
    expected_body = next((name for name in theme.UI_FAMILIES if name.casefold() in installed), None)
    expected_heading = next((name for name in theme.HEADING_FAMILIES if name.casefold() in installed), None)
    if expected_body:
        assert fonts.body[0] == expected_body
        assert tkfont.Font(root, font=fonts.body).actual("family").casefold() == expected_body.casefold()
    if expected_heading:
        assert fonts.heading[0] == expected_heading
        assert fonts.title[0] == expected_heading
        assert tkfont.Font(root, font=fonts.heading).actual("family").casefold() == expected_heading.casefold()
    reading = tkfont.Font(root, font=fonts.body)
    title = tkfont.Font(root, font=fonts.title)
    assert title.metrics("linespace") > reading.metrics("linespace") >= 14
    assert 0 < reading.measure("Search papers, sources, and past work") < 600


def test_missing_requested_family_uses_an_installed_serif_fallback(root, monkeypatch):
    monkeypatch.setattr(theme.tkfont, "families", lambda _root: ("Liberation Serif", "Consolas"))
    fonts = theme.build_fonts(root)
    assert fonts.heading[0] == "Liberation Serif"
    assert fonts.title[0] == "Liberation Serif"
    assert fonts.body[0] == "Liberation Serif"
    assert fonts.mono[0] == "Consolas"
