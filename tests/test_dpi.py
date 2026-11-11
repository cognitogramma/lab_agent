"""The real desktop window opts out of Windows bitmap scaling in a fresh process."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPI awareness is platform-specific")
def test_workbench_process_and_native_window_are_per_monitor_dpi_aware(tmp_path):
    pytest.importorskip("tkinter")
    pytest.importorskip("customtkinter")
    if not hasattr(ctypes.windll.user32, "GetWindowDpiAwarenessContext"):
        pytest.skip("Native window DPI-context inspection requires Windows 10 or newer")

    from continuum_spectranalytics.agent.toolcall import SPECS
    from continuum_spectranalytics import paths

    environment = os.environ.copy()
    # A previous GUI fixture may have redirected the project root. Resolve the
    # checked-in example before installing this child's isolated data root.
    environment.pop(paths.ROOT_ENV, None)
    for spec in SPECS.values():
        for key in spec.env_keys:
            environment.pop(key, None)
    environment["TEMP"] = str(tmp_path)
    environment["TMP"] = str(tmp_path)
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = textwrap.dedent("""
        import ctypes
        from ctypes import wintypes
        import json
        import os
        from pathlib import Path
        import sys
        from types import SimpleNamespace

        from continuum_spectranalytics import paths
        from scripts.gui_smoke import DEFAULTS

        example = paths.example_config()
        assert example.is_file(), f"Missing example configuration: {example}"
        data_root = Path(sys.argv[1])
        data_root.mkdir(parents=True, exist_ok=True)
        os.environ[paths.ROOT_ENV] = str(data_root)

        from continuum_spectranalytics.gui.app import Workbench

        values = dict(DEFAULTS)
        values.update(config=str(example), run_id="dpi-check",
                      runs_dir=str(data_root / "runs"), notes_dir=str(data_root / "notes"))
        window = Workbench(SimpleNamespace(**values))
        window.withdraw()
        try:
            window.update_idletasks()
            process_awareness = ctypes.c_int(-1)
            status = ctypes.windll.shcore.GetProcessDpiAwareness(
                None, ctypes.byref(process_awareness)
            )
            user32 = ctypes.windll.user32
            user32.GetWindowDpiAwarenessContext.argtypes = [wintypes.HWND]
            user32.GetWindowDpiAwarenessContext.restype = wintypes.HANDLE
            user32.GetAwarenessFromDpiAwarenessContext.argtypes = [wintypes.HANDLE]
            user32.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
            native_context = user32.GetWindowDpiAwarenessContext(
                wintypes.HWND(window.winfo_id())
            )
            result = {
                "process_status": status,
                "process_awareness": process_awareness.value,
                "window_awareness": user32.GetAwarenessFromDpiAwarenessContext(native_context),
            }
        finally:
            window.close()
        print("DPI_RESULT=" + json.dumps(result))
    """)
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "project")],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    line = next((line for line in process.stdout.splitlines() if line.startswith("DPI_RESULT=")), None)
    assert line is not None, process.stdout + process.stderr
    result = json.loads(line.removeprefix("DPI_RESULT="))
    assert result == {"process_status": 0, "process_awareness": 2, "window_awareness": 2}
