import csv
import json

import pytest

import alert_triangulate as T


def run(capsys, *args):
    data = T.main([str(a) for a in args])
    capsys.readouterr()
    return data


def test_2026_case_is_a_processing_fault(capsys, case):
    d = run(capsys, "--alert", case / "alert.json", "--authority", case / "authority_jma.json",
            "--receipts", case / "receipts_synthetic.csv", "--catalog", case / "usgs_event.geojson")
    assert d["verdict"]["code"] == "PROCESSING_FAULT"
    assert d["verdict"]["axes"]["footprint"] == "NATIONAL"
    assert abs(d["timing"]["lag_s"]) < 5


def test_usgs_fixture_keeps_sub_second_origin(case):
    t = json.loads((case / "usgs_event.geojson").read_text(encoding="utf-8"))["features"][0]["properties"]["time"]
    assert t % 1000 == 900                     # 03:35:31.9 UTC, not truncated to :31.0


def test_magnitude_only_alert_is_not_an_all_clear(capsys, case, tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"magnitude": 9.9, "detected_at": "2026-10-16T12:38:10.1+09:00"}), encoding="utf-8")
    d = run(capsys, "--alert", a, "--catalog", case / "usgs_event.geojson")
    assert d["verdict"]["axes"]["physics"] == "REAL_EVENT_WRONG_PARAMETERS"
    assert d["verdict"]["code"] != "CONSISTENT"
    assert d["discrepancy"]["compared"] == ["magnitude"]


def test_alert_with_nothing_to_compare_is_not_checked(capsys, case, tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"detected_at": "2026-10-16T12:38:10.1+09:00"}), encoding="utf-8")
    d = run(capsys, "--alert", a, "--catalog", case / "usgs_event.geojson")
    assert d["verdict"]["axes"]["physics"] == "NOT_CHECKED"
    assert d["verdict"]["code"] != "CONSISTENT"


def _first_receipts(spoof, tmp_path, n):
    rows = list(csv.reader((spoof / "receipts.csv").read_text(encoding=_enc(spoof)).splitlines()))
    out = tmp_path / f"r{n}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows[: n + 1])
    return out


def _enc(spoof):
    raw = (spoof / "receipts.csv").read_bytes()
    try:
        raw.decode("utf-8-sig"); return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp932"


@pytest.mark.parametrize("n,expected", [(1, "INDETERMINATE"), (9, "INDETERMINATE"), (10, "LOCAL")])
def test_local_verdict_needs_enough_receipts(capsys, spoof, tmp_path, n, expected):
    d = run(capsys, "--alert", spoof / "alert.json", "--receipts", _first_receipts(spoof, tmp_path, n))
    assert d["verdict"]["axes"]["footprint"] == expected
    if expected != "LOCAL":
        assert d["verdict"]["code"] != "LOCAL_INJECTION"


def test_spoof_demo_locates_the_transmitter(capsys, spoof):
    import eewlib as L
    d = run(capsys, "--alert", spoof / "alert.json", "--receipts", spoof / "receipts.csv",
            "--catalog", spoof / "usgs_empty.geojson")
    assert d["verdict"]["code"] == "LOCAL_INJECTION"
    tx = d["transmitter_estimate"]
    assert float(L.gc_km(tx["lat"], tx["lon"], 35.6595, 139.7005)) < 0.2
