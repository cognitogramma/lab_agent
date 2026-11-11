"""Launch the desktop workbench from the Windows executable."""

import multiprocessing
import sys


if __name__ == "__main__":
    multiprocessing.freeze_support()

    from continuum_spectranalytics.cli import main

    raise SystemExit(main(["gui", *sys.argv[1:]]))
