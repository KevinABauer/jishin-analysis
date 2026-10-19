import csv
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import eew_diagnose as D
import eewlib as L
import make_synthetic as S


def run(capsys, *args):
    data = D.main([str(a) for a in args])
    capsys.readouterr()
    return data


@pytest.fixture(scope="module")
def case_2026():
    from conftest import CASE
    return D.main(["picks", "--picks", str(CASE / "picks_synthetic.csv"), "--alert", str(CASE / "alert.json"),
                   "--reference", str(CASE / "usgs_event.geojson")])


def _picks_rows(case):
    return list(csv.DictReader(L.read_text(case / "picks_synthetic.csv").splitlines()))


def _write(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return path


def test_2026_failure_classes(case_2026):
    fc = set(case_2026["failure_classes"])
    assert {"OUT_OF_NETWORK", "CUTOFF_NOT_APPLIED", "STAGE_HYPOCENTRE_MISMATCH", "INTENSITY_FROM_BAD_INPUTS"} <= fc
    assert "OUTLIER_PICKS" not in fc
    assert case_2026["solution"]["sep_from_reference_km"] < 60


def test_control_case_flags_nothing(capsys, tmp_path):
    """A correct alert for an in-network event must come back clean."""
    rng = np.random.default_rng(7)
    tt = L.TravelTimes(quiet=True)
    H = L.Hypo(36.30, 141.20, 40.0, datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc))
    names = list(S.STATIONS)
    la = np.array([S.STATIONS[n][0] for n in names]); lo = np.array([S.STATIONS[n][1] for n in names])
    d = L.gc_km(H.lat, H.lon, la, lo)
    tp = tt.p(d, H.depth_km) + rng.normal(0, 0.3, d.size)
    lA = L.RELATIONS["eew_illustrative"].predict_logA(6.0, d, H.depth_km) + rng.normal(0, 0.2, d.size)
    I = L.pgv_to_shindo(L.predict_pgv(6.0, H.depth_km, d, "interplate", 1.4, finite=True)) + rng.normal(0, 0.3, d.size)
    rows = [{"station": f"S{i:02d}", "lat": la[i], "lon": lo[i],
             "p_time": (H.time + timedelta(seconds=float(tp[i]))).isoformat(),
             "amp_um": f"{10 ** lA[i]:.4g}", "observed_shindo": L.shindo_class(float(I[i]))} for i in range(len(names))]
    p = _write(rows, tmp_path / "picks.csv")
    (tmp_path / "alert.json").write_text(json.dumps({"magnitude": 6.0, "hypocenter": {"lat": 36.3, "lon": 141.2, "depth_km": 40}}))
    (tmp_path / "ref.json").write_text(json.dumps({"lat": 36.3, "lon": 141.2, "depth_km": 40, "time": H.time.isoformat(), "mag": 6.0}))
    out = run(capsys, "picks", "--picks", p, "--alert", tmp_path / "alert.json", "--reference", tmp_path / "ref.json",
              "--fault-type", "interplate")
    assert out["failure_classes"] == []


def test_alert_without_hypocentre_does_not_crash(capsys, case, tmp_path):
    a = tmp_path / "a.json"; a.write_text('{"magnitude": 9.9}')
    out = run(capsys, "picks", "--picks", case / "picks_synthetic.csv", "--alert", a, "--reference", case / "usgs_event.geojson")
    codes = {(c["code"], c["status"]) for c in out["checks"]}
    assert ("D2", "N/A") in codes and ("D3", "N/A") in codes and ("D6", "N/A") in codes
    assert "IMPLAUSIBLE_MAGNITUDE" in out["failure_classes"]


def test_alert_without_magnitude_does_not_crash(capsys, case, tmp_path):
    a = tmp_path / "a.json"; a.write_text(json.dumps({"hypocenter": {"lat": 48.2, "lon": 153.4, "depth_km": 140}}))
    out = run(capsys, "picks", "--picks", case / "picks_synthetic.csv", "--alert", a)
    assert ("D5", "N/A") in {(c["code"], c["status"]) for c in out["checks"]}


def test_zero_amplitude_is_ignored_not_fatal(capsys, case, tmp_path):
    rows = _picks_rows(case)
    amp_key = next(k for k in rows[0] if k in ("amp_um",) or "μm" in k or "um" in k)
    rows[0][amp_key] = "0"; rows[1][amp_key] = "-3"
    p = _write(rows, tmp_path / "picks.csv")
    out = run(capsys, "picks", "--picks", p, "--alert", case / "alert.json")
    assert "failure_classes" in out


def test_d8_is_capped_and_reference_catches_early_picks(capsys, case, tmp_path):
    """The reported failure: 4 of 5 picks 113–425 s early. MAD must not swallow them."""
    rows = [_picks_rows(case)[i] for i in (0, 3, 9, 14, 21)]
    t_key = next(k for k in rows[0] if "time" in k or "時刻" in k)
    for r, off in zip(rows, (0, -113, -240, -330, -425)):
        r[t_key] = (L.parse_time(r[t_key]) + timedelta(seconds=off)).isoformat()
    p = _write(rows, tmp_path / "picks.csv")
    out = run(capsys, "picks", "--picks", p, "--alert", case / "alert.json", "--reference", case / "usgs_event.geojson")
    assert out["D8"]["threshold_s"] <= D.D8_CAP_S
    assert len(out["D8"]["outliers_vs_reference"]) == 4
    assert "OUTLIER_PICKS" in out["failure_classes"]


def test_fetch_defaults_to_earthscope():
    a = D.build_parser().parse_args(["fetch", "--stations", "IU.MAJO", "--origin", "2026-10-16T03:35:31.9"])
    assert a.provider == "EARTHSCOPE"
