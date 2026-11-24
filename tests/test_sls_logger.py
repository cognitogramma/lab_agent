"""Exports from the lab's Basler Dual-Channel SLS Logger: header kept, problems flagged.

The fixture is synthetic but has the logger's exact layout: ``#,key,value``
header lines (some holding JSON), then a CSV with polarized and depolarized
camera means. Found with a real poly-A dialysis export, where the header was
thrown away as comments and a saturated channel went unnoticed.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from continuum_spectranalytics.tabular import column_summary, open_table


def _q(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def write_logger_export(path, seconds=1200, planned=84000.0, saturate_from=1000, spike_at=300):
    camera = {"camera": "Basler acA640-750um (25478979)", "exposure_us": 50000.0, "actual_exposure_us": 50000.0,
              "sample_interval_s": 1.0, "pixel_format": "Mono10"}
    laser = {"controller_model": "OBIS MINI-CONTROLLER", "wavelength_nm": 662.0, "setpoint_mw": 35.0,
             "measured_power_mw": 35.01, "emission": "ON"}
    rois = {"polarized": {"x": 192, "y": 210, "radius": 20}, "depolarized": {"x": 564, "y": 415, "radius": 12}}
    header = [
        ("app", _q('"Basler Dual-Channel SLS Logger v0.4.2"')),
        ("camera", _q(json.dumps(camera))),
        ("laser", _q(json.dumps(laser))),
        ("rois", _q(json.dumps(rois))),
        ("automatic_laser_attenuation", _q(json.dumps({"enabled_at_recording_start": False}))),
        ("notes", _q('"history_scale records corrections applied to each row."')),
        ("record_duration_s", str(planned)),
    ]
    rng = np.random.default_rng(3)
    t = np.arange(seconds, dtype=float)
    pol = np.clip(0.05 + 0.5 * t / seconds + rng.normal(0, 0.003, seconds), 0, 1)
    pol[saturate_from:] = 1.0
    pol[spike_at] = 1.0
    pol[spike_at + 100] += 0.2  # a dust spike below the ceiling
    dep = 0.0045 + rng.normal(0, 0.0002, seconds)
    lines = [f"#,{key},{value}" for key, value in header]
    lines.append("time_s,timestamp_unix_s,pol_mean,dep_mean,temperature_c,pol_raw_mean,dep_raw_mean,history_scale")
    for i in range(seconds):
        temp = "42.000" if i % 3 else ""
        lines.append(f"{t[i]:.6f},{1792598526 + t[i]:.6f},{pol[i]:.6f},{dep[i]:.6f},{temp},{pol[i]:.6f},{dep[i]:.6f},1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def export(tmp_path):
    return write_logger_export(tmp_path / "poly-A NaCl dialysis.csv")


def test_the_header_is_kept_and_described(export):
    table = open_table(export)
    assert table.kind == "sls_logger"
    assert table.columns[:4] == ["time_s", "timestamp_unix_s", "pol_mean", "dep_mean"]
    assert len(table.frame) == 1200
    notes = " | ".join(table.notes)
    assert "Basler Dual-Channel SLS Logger v0.4.2" in notes
    assert "662.0 nm, set 35.0 mW (measured 35.01 mW)" in notes
    assert "polarized centre (192, 210) radius 20 px" in notes
    assert "attenuation OFF" in notes


def test_a_saturated_plateau_is_told_apart_from_a_spike(export):
    warnings = [line for line in column_summary(open_table(export).frame) if line.startswith("WARNING")]
    pinned = [w for w in warnings if w.startswith("WARNING: pol_mean is pinned")]
    assert pinned and "from 1,000 s to the end" in pinned[0]
    assert "1 other time(s) in single spikes" in pinned[0]
    assert any("pol_mean has" in w and "spikes" in w for w in warnings)
    assert not any(w.startswith("WARNING: dep_mean is pinned") for w in warnings)


def test_the_assistant_is_told_how_the_file_was_made_and_what_is_wrong(export, tmp_path, monkeypatch):
    from continuum_spectranalytics import filesystem
    from continuum_spectranalytics.attachments import AttachmentShelf
    from continuum_spectranalytics.filesystem import FileAccess

    monkeypatch.setattr(filesystem, "policy_path", lambda: tmp_path / "absent.json")
    brief = AttachmentShelf(FileAccess()).add(export).brief()
    assert "SLS logger export" in brief
    assert "recording stopped early: 1,199 s of a planned 84,000 s" in brief
    assert "pinned at its maximum" in brief
    assert "1,200 rows x 8 columns" in brief


def test_ordinary_csvs_are_unaffected(tmp_path):
    plain = tmp_path / "plain.csv"
    plain.write_text("time,signal\n0,1.0\n1,1.5\n2,1.7\n")
    assert open_table(plain).kind == "delimited"
