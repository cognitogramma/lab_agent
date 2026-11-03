"""The window: the agent session, with the run and its figures beside it.

The same session as ``polyrmc agent``, driven from the same orchestrator over
the same tool surface, writing the same log. Not a second implementation of the
agent -- there is one loop, one approval policy, one place a model can reach a
run, and this package is a second reader of it.

What a window is *for*, here, is the figures. This pipeline draws them on every
processed run and the terminal can only tell you where they were written, so the
decision they exist to inform gets taken from a number while the picture of it
sits unopened. Beside that: the run state is permanently on screen rather than
recalled with ``/state``, and the commit gate is a dialog that has to be
answered rather than a prompt that can be typed past.

Nothing in here is required to process data. The window uses Tk, CustomTkinter,
and matplotlib. Install its optional controls with ``pip install -e .[gui]``.
If Tk is missing,
:func:`launch` says so and names the terminal session, which does everything the
window does apart from showing you the picture.

    polyrmc gui
    polyrmc gui --raw <file> --declarations <file>
"""

from __future__ import annotations


def launch(args) -> int:
    """Open the window. Returns a process exit code.

    Tk is imported here rather than at module scope so that importing
    :mod:`polyrmc.gui` -- which ``polyrmc --help`` does not, but a test
    collector might -- never fails on a headless machine. A missing Tk is
    reported as the one thing it actually costs you.
    """
    try:
        from polyrmc.gui.app import launch as _launch
    except ImportError as error:  # tkinter absent from this Python
        if error.name == "customtkinter":
            print("Install the desktop controls with: python -m pip install -e .[gui]")
            return 2
        print(
            f"the window needs tkinter, which this Python does not have: {error}\n\n"
            "On Linux it is usually a separate package (python3-tk). Everything "
            "the window does apart from drawing the figures is in the terminal "
            "session:\n  polyrmc agent"
        )
        return 2
    return _launch(args)


__all__ = ["launch"]
