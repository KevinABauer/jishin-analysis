"""日本語環境への対応（このブランチ固有）"""
import subprocess
import sys

import pytest

import eewlib as L
from conftest import ROOT


def test_shift_jis_csv_with_japanese_headers_is_read(tmp_path):
    p = tmp_path / "受信記録.csv"
    p.write_bytes("受信時刻,緯度,経度,事業者,経路\r\n2026/10/16 12:40:03,35.0,139.0,ドコモ,エリアメール\r\n".encode("cp932"))
    r = L.read_csv_rows(p)[0]
    assert r["lat"] == "35.0" and r["carrier"] == "ドコモ"
    assert L.norm_carrier(r["carrier"]) == "docomo" and L.norm_channel(r["channel"]) == "cbs"


def test_times_without_zone_are_jst():
    want = L.parse_time("2026-10-16T03:40:02.6+00:00")
    assert L.parse_time("2026/10/16 12:40:02.6") == want
    assert L.parse_time("2026年10月16日12時40分02.6秒") == want
    assert L.parse_time("２０２６/１０/１６ １２:４０:０２.６") == want


@pytest.mark.parametrize("s,v", [("5弱", 4.75), ("５弱", 4.75), ("震度5弱", 4.75), ("5-", 4.75), ("6強", 6.25), ("", None)])
def test_shindo_notations(s, v):
    got = L.parse_shindo(s)
    assert (got != got) if v is None else got == v


def test_fullwidth_magnitude_in_json(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"magnitude": "９．９"}', encoding="utf-8")
    assert L.load_alert(p).magnitude == 9.9


def test_help_and_errors_are_japanese():
    out = subprocess.run([sys.executable, str(ROOT / "eew_diagnose.py"), "picks", "--alert", "x"],
                         capture_output=True, text=True)
    assert "次の引数は必須です" in out.stderr
    h = subprocess.run([sys.executable, str(ROOT / "alert_triangulate.py"), "-h"], capture_output=True, text=True).stdout
    assert "使い方" in h and "このヘルプを表示して終了します" in h


def test_reports_are_shift_jis_safe():
    for f in list((ROOT / "cases").rglob("*.txt")) + [ROOT / "README.md"]:
        f.read_text(encoding="utf-8").encode("cp932")
