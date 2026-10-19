import csv
from pathlib import Path

import pytest

import eew_diagnose as D
import eewlib as L
import make_synthetic as S
from synthetic_waveforms import make_fetch_like_dir

STATIONS = {f"S{i:02d}": v for i, v in enumerate(list(S.STATIONS.values())[:8])}


@pytest.fixture(scope="module")
def fetch_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fdsn_data")
    truth = make_fetch_like_dir(out, S.TRUE, 6.9, STATIONS)
    return out, truth


def test_only_waveform_files_are_read(fetch_dir):
    out, _ = fetch_dir
    files, skipped = D.waveform_files(out)
    assert [f.name for f in files] == ["waveforms.mseed"]
    assert "stations.xml" in skipped


def test_one_channel_per_station_prefers_00_BH(fetch_dir):
    from obspy import read
    out, _ = fetch_dir
    st = read(str(out / "waveforms.mseed"))
    z, group = D.choose_channels([tr for tr in st if tr.stats.station == "S00"])
    assert z.id == "SY.S00.00.BHZ"
    assert {tr.stats.channel for tr in group} == {"BHZ", "BHN", "BHE"}


def test_picks_land_on_p_despite_pre_event_glitches(capsys, fetch_dir, case, tmp_path):
    out, truth = fetch_dir
    picks = tmp_path / "picks.csv"
    D.main(["waveforms", "--data", str(out), "--alert", str(case / "alert.json"),
            "--reference", str(case / "usgs_event.geojson"), "--picks-out", str(picks)])
    capsys.readouterr()
    rows = list(csv.DictReader(L.read_text(picks).splitlines()))
    assert len(rows) == len(STATIONS)
    for r in rows:
        key = next(k for k in r if k in ("station", "観測点"))
        tkey = next(k for k in r if k in ("p_time", "P波到達時刻"))
        akey = next(k for k in r if k.startswith("amp") or k.startswith("最大変位"))
        tp, amp = truth[r[key]]
        assert abs(L.parse_time(r[tkey]).timestamp() - tp.timestamp) < 2.0, r
        assert float(r[akey]) == pytest.approx(amp, rel=0.15)
