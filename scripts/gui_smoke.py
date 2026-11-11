"""Build the window, draw it once, and close it.

A layout error -- a bad grid weight, a widget packed into the wrong parent, a
style name that does not exist -- only shows up when Tk is asked to realise the
window. This asks it to, without a person having to sit and look, so a broken
window fails here rather than in front of a researcher.

    .venv\\Scripts\\python.exe scripts/gui_smoke.py [--hold]

``--hold`` leaves it open, for when a person *does* want to look.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from dotenv import load_dotenv

from continuum_spectranalytics import paths

DEFAULTS = dict(
    provider=None,
    model=None,
    max_steps=12,
    auto_approve=False,
    budget_requests=0,
    budget_tokens=0,
    min_request_interval=0.0,
    verbose=False,
    raw=None,
    declarations=None,
    config=None,
    run_id=None,
    channel=None,
    runs_dir=None,
    notes_dir=None,
    author="smoke",
    out=None,
    no_prompt=False,
    prompt=None,
)


def main(argv: list[str]) -> int:
    load_dotenv(paths.env_file())
    from continuum_spectranalytics.gui.app import Workbench

    window = Workbench(SimpleNamespace(**DEFAULTS))
    # Pumped rather than driven: the session opens itself from a scheduled
    # callback, and calling it from here as well would report the greeting twice
    # and hide the fact that the scheduling works.
    for _ in range(60):
        window.update()

    if "--hold" in argv:
        window.mainloop()
        return 0

    print("window built:", window.title())
    print("state panel:", window.state_panel.text.get("1.0", "2.0").strip())
    print("figures found:", len(window.figures.paths))
    print("tools:", len(window.registry))
    print("brain:", window.model_label.cget("text"))
    print("log:", window.log.location)
    window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
