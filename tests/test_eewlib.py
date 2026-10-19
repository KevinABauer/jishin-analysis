import json

import pytest

import eewlib as L

XML = """<Report xmlns="http://xml.kishou.go.jp/jmaxml1/">
<Head xmlns="http://xml.kishou.go.jp/jmaxml1/informationBasis1/"><Serial>1</Serial><InfoType>発表</InfoType></Head>
<Body xmlns="http://xml.kishou.go.jp/jmaxml1/body/seismology1/" xmlns:jmx_eb="http://xml.kishou.go.jp/jmaxml1/elementBasis1/">
<Earthquake><Hypocenter><Area><Name>千島列島</Name>
<jmx_eb:Coordinate description="北緯４８．２度　東経１５３．４度　深さ１４０ｋｍ">+48.2+153.4-140000/</jmx_eb:Coordinate>
</Area></Hypocenter>{mag}</Earthquake></Body></Report>"""


def test_json_alert_with_bom_is_read_as_json(tmp_path):
    p = tmp_path / "alert.json"
    p.write_text(json.dumps({"magnitude": 9.9, "hypocenter": {"lat": 48.2, "lon": 153.4, "depth_km": 140}}),
                 encoding="utf-8-sig")
    a = L.load_alert(p)
    assert a.magnitude == 9.9 and a.h_int.lat == 48.2


def test_jma_magnitude_condition_unknown_is_none(tmp_path):
    p = tmp_path / "vxse.xml"
    p.write_text(XML.format(mag='<jmx_eb:Magnitude type="Mj" condition="不明" description="Ｍ不明">NaN</jmx_eb:Magnitude>'),
                 encoding="utf-8")
    a = L.load_alert(p)
    assert a.magnitude is None
    assert a.h_int.depth_km == pytest.approx(140)
    assert any("不明" in n for n in a.notes)


def test_jma_magnitude_normal(tmp_path):
    p = tmp_path / "vxse.xml"
    p.write_text(XML.format(mag='<jmx_eb:Magnitude type="Mj" description="Ｍ９．９">9.9</jmx_eb:Magnitude>'), encoding="utf-8")
    assert L.load_alert(p).magnitude == 9.9


def test_json_nan_magnitude_is_none(tmp_path):
    p = tmp_path / "alert.json"
    p.write_text('{"magnitude": NaN}', encoding="utf-8")
    assert L.load_alert(p).magnitude is None


def test_coordinate_parser():
    assert L.parse_jma_coordinate("+48.2+153.4-140000/") == (48.2, 153.4, 140.0)
    assert L.parse_jma_coordinate("/") is None


def test_scalar_travel_time_is_a_float():
    t = float(L.TravelTimes(quiet=True).p(1261.0, 135.0))
    assert 150 < t < 165


def test_history_page_parser(case):
    h = L.parse_jma_history(L.read_text(case / "jma_history_excerpt.txt"))
    assert [r["serial"] for r in h["serials"]] == [17, 19]
    assert h["serials"][0]["magnitude"] == 9.9
    assert h["detected_at"].isoformat().startswith("2026-10-16T03:38:10.1")
