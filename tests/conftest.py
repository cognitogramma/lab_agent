"""Shared synthetic data. No test in this suite requires an instrument file,
a network connection, or an API key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyrmc.config import (
    DilutionConfig,
    FitRangeConfig,
    LoopConfig,
    OpticalConfig,
    RunConfig,
    SmoothingConfig,
)
from polyrmc.tier0.optics import optical_constant

# Values chosen to sit in the same range as the published validation set.
TRUE_MW = 66430.0  # g/mol, BSA
TRUE_A2 = 5.0e-5  # cm^3 mol / g^2
TRUE_A3 = 8.0e-2  # large enough to bend the curve at high c
C0 = 4.0e-3  # g/cm^3
OVERLAP_C = 2.0e-3  # above this the virial expansion is visibly failing


def seconds_to_wrapping_mmss(seconds: float) -> str:
    """Format as MM:SS.s, resetting to zero every hour like the instrument does."""
    within_hour = seconds % 3600.0
    minutes = int(within_hour // 60)
    remainder = within_hour - minutes * 60
    return f"{minutes:02d}:{remainder:04.1f}"


@pytest.fixture
def optical() -> OpticalConfig:
    return OpticalConfig(wavelength_nm=660.0, dn_dc=0.185, solvent_refractive_index=1.333)


@pytest.fixture
def dilution() -> DilutionConfig:
    # V/Q = 1000 s, so concentration falls by e over ~17 minutes.
    return DilutionConfig(c0_g_per_cm3=C0, flow_rate_cm3_per_s=1.0e-3, cell_volume_cm3=1.0)


def ideal_intensity(
    concentration: np.ndarray, k: float, alpha: float = 1.0
) -> np.ndarray:
    """Intensity implied by the virial expansion, scaled by an instrument alpha.

    Kc/I_R = 1/Mw + 2*A2*c + 3*A3*c^2, so I_meas = alpha * K*c / (that).
    """
    ordinate = 1.0 / TRUE_MW + 2.0 * TRUE_A2 * concentration
    curvature = 3.0 * TRUE_A3 * concentration**2
    # Curvature only bites once chains start to overlap.
    ordinate = ordinate + np.where(concentration > OVERLAP_C, curvature, 0.0)
    return alpha * k * concentration / ordinate


@pytest.fixture
def virial_curve(optical: OpticalConfig, dilution: DilutionConfig):
    """A clean synthetic dilution trajectory with a known linear region."""
    k = optical_constant(optical)
    time_s = np.arange(0, 4000, 1.0)
    concentration = dilution.c0_g_per_cm3 * np.exp(
        -dilution.flow_rate_cm3_per_s / dilution.cell_volume_cm3 * time_s
    )
    intensity = ideal_intensity(concentration, k)
    return time_s, concentration, intensity, k


@pytest.fixture
def noisy_trace(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A decaying trace carrying a spike, a saturation run, and slow drift."""
    time_s = np.arange(0, 2000, 1.0)
    clean = 5000.0 * np.exp(-time_s / 900.0) + 500.0
    drift = 0.06 * time_s
    signal = clean + drift + rng.normal(0, 8.0, time_s.size)

    signal[700] += 900.0  # single-point transient spike
    signal[1200:1215] = 65535.0  # detector pinned at its rail
    return time_s, signal


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260728)


def write_argen_file(
    path: Path,
    n_points: int = 4000,
    dt: float = 1.0,
    n_channels: int = 16,
    seed: int = 7,
    null_channel_index: int = 7,  # ls8 -- the analysed channel
    null_channel_rows: tuple[int, ...] = (10, 11),
    null_temp_rows: tuple[int, ...] = (20, 21, 22),
) -> Path:
    """Write a synthetic ARGEN-style file with the awkward parts intact.

    Includes an hour-wrapping time field, nulls in a scattering channel (which
    must drop rows) and nulls in temperature (which must not).
    """
    generator = np.random.default_rng(seed)
    time_s = np.arange(n_points, dtype=float) * dt
    channels = [
        4000.0 * np.exp(-time_s / 1200.0) + 300.0 + generator.normal(0, 5.0, n_points)
        for _ in range(n_channels)
    ]

    # Declared count includes the column-name row, as the instrument does.
    header = [
        "# Number of header rows: 5",
        "# Instrument: ARGEN (synthetic)",
        "# Sample: test fixture",
        "# Note: elapsed time wraps hourly",
    ]
    column_names = ["Elapsed Time", "Temperature", "Stir Speed", "ND Filter"] + [
        f"ls{i + 1}" for i in range(n_channels)
    ]

    lines = list(header)
    lines.append("\t".join(column_names))
    for row in range(n_points):
        temperature = "" if row in null_temp_rows else f"{25.0:.2f}"
        values = [
            seconds_to_wrapping_mmss(time_s[row]),
            temperature,
            "300",
            "1.0",
        ]
        for index, channel in enumerate(channels):
            if index == null_channel_index and row in null_channel_rows:
                values.append("")
            else:
                values.append(f"{channel[row]:.3f}")
        lines.append("\t".join(values))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def argen_file(tmp_path: Path) -> Path:
    return write_argen_file(tmp_path / "synthetic_argen.txt")


def write_virial_argen_file(
    path: Path,
    optical_config: OpticalConfig,
    dilution_config: DilutionConfig,
    n_points: int = 4000,
    seed: int = 11,
) -> Path:
    """An ARGEN-style file whose channels follow the virial model exactly.

    Lets an end-to-end run be checked against a known A2*Mw rather than only
    against "it produced a number".
    """
    generator = np.random.default_rng(seed)
    k = optical_constant(optical_config)
    time_s = np.arange(n_points, dtype=float)
    decay = dilution_config.flow_rate_cm3_per_s / dilution_config.cell_volume_cm3
    concentration = dilution_config.c0_g_per_cm3 * np.exp(-decay * time_s)

    # Scale into instrument-like count magnitudes; alpha cancels downstream.
    alpha = 1.0e9
    base = ideal_intensity(concentration, k, alpha=alpha)

    lines = [
        "# Number of header rows: 4",
        "# Instrument: ARGEN (synthetic virial)",
        "# Sample: known A2 and Mw",
    ]
    column_names = ["Elapsed Time", "Temperature"] + [f"ls{i + 1}" for i in range(16)]
    lines.append("\t".join(column_names))

    channels = [base * (1.0 + generator.normal(0, 3e-4, n_points)) for _ in range(16)]
    for row in range(n_points):
        values = [seconds_to_wrapping_mmss(time_s[row]), "25.00"]
        values.extend(f"{channel[row]:.6f}" for channel in channels)
        lines.append("\t".join(values))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def virial_argen_file(tmp_path: Path, optical, dilution) -> Path:
    return write_virial_argen_file(tmp_path / "virial_argen.txt", optical, dilution)


@pytest.fixture
def run_config(argen_file: Path, optical: OpticalConfig, dilution: DilutionConfig):
    return RunConfig(
        run_id="test-run",
        source_file=argen_file,
        experiment_type="dilution_trajectory",
        optical=optical,
        dilution=dilution,
        smoothing=SmoothingConfig(min_window=5, max_window=101, n_candidates=6),
        fit_range=FitRangeConfig(min_points=30, n_candidates=6),
        loop=LoopConfig(max_iterations=2),
    )
