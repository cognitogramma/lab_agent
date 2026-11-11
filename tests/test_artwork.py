"""Artwork follows its actual display size and releases its Tk resources."""

from __future__ import annotations

import pytest

from continuum_spectranalytics.gui.shell import _Artwork
from continuum_spectranalytics.gui.theme import build_fonts


@pytest.mark.parametrize("kind", ["open", "research", "runs", "hero"])
def test_artwork_resizes_without_stretching_caches_and_releases_photo(gui_app, kind):
    art = _Artwork(gui_app, kind, fonts=build_fonts(gui_app))
    try:
        art.place(x=0, y=0, width=220, height=120)
        gui_app.update_idletasks()
        art._draw()
        assert (art._photo.width(), art._photo.height()) == (220, 120)
        assert len(art.find_withtag("illustration")) == 1
        first_photo = art._photo
        art._draw()
        assert art._photo is first_photo
        del first_photo

        art.place_configure(width=330, height=150)
        gui_app.update_idletasks()
        # The session root is intentionally withdrawn; hidden windows do not
        # receive the normal display <Configure> notification.
        art._schedule_draw()
        gui_app.update_idletasks()
        assert (art._photo.width(), art._photo.height()) == (330, 150)
        assert len(art.find_withtag("illustration")) == 1
        photo_name = str(art._photo)
        art._schedule_draw()
    finally:
        art.destroy()
    assert art._photo is None and art._redraw_id is None
    assert photo_name not in gui_app.tk.call("image", "names")
    gui_app.update_idletasks()


@pytest.mark.parametrize("kind", ["open", "research", "runs", "hero"])
def test_illustration_labels_use_the_resolved_theme_families(gui_app, kind):
    fonts = build_fonts(gui_app)
    art = _Artwork(gui_app, kind, fonts=fonts)
    try:
        art.place(x=0, y=0, width=760 if kind == "hero" else 220, height=155)
        gui_app.update_idletasks()
        art._schedule_draw()
        gui_app.update_idletasks()
        labels = art.find_withtag("label")
        assert len(labels) == (3 if kind == "hero" else 1)
        for label in labels:
            family = gui_app.tk.splitlist(art.itemcget(label, "font"))[0]
            assert family in {fonts.body[0], fonts.heading[0]}
    finally:
        art.destroy()
