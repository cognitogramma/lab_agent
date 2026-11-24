"""Launch the desktop workbench from the Windows executable."""

import multiprocessing
import sys


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if sys.argv[1:2] == ["--startup-probe"]:
        from continuum_spectranalytics.selftest import startup_probe

        raise SystemExit(startup_probe())

    if sys.argv[1:2] == ["--self-test"]:
        from continuum_spectranalytics.selftest import main as self_test

        raise SystemExit(self_test(sys.argv[2:]))

    from continuum_spectranalytics.cli import main

    raise SystemExit(main(["gui", *sys.argv[1:]]))
