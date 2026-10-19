#!/usr/bin/env python3
"""
make_synthetic.py ― 2つのツールの検証用データを作る

  cases/2026-10-16/   2026年10月16日の事案。発表された速報の値と USGS の震源は【実データ】、
                      観測点の読み取り値・振幅・震度と受信記録は【合成データ】（端末ごとの公開データは存在しない）。
  cases/spoof-demo/   偽基地局を想定した架空の事例。送信源の真の位置が分かっているので、
                      位置推定がそれを復元できるかを確かめられる。受信記録は Excel 既定の Shift_JIS で保存する。

観測点の座標は、気象庁・Hi-net の観測点に見立てたおおよその市町村位置です。
合成振幅は坪井（1954）式の外挿を「真値」の代わりに使っています。診断を実行する前に決めたもので、
M9.9 を再現するように調整したものではありません。
"""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import eewlib as L

rng = np.random.default_rng(20261016)

STATIONS = {  # 名称: (緯度, 経度) ― おおよその位置、北海道から東海へ
    "羅臼": (44.02, 145.19), "根室": (43.33, 145.58), "中標津": (43.55, 144.97), "網走": (44.02, 144.27),
    "釧路": (42.98, 144.38), "紋別": (44.36, 143.35), "帯広": (42.92, 143.20), "えりも": (41.93, 143.25),
    "旭川": (43.77, 142.37), "稚内": (45.42, 141.67), "浦河": (42.17, 142.77), "札幌": (43.06, 141.35),
    "苫小牧": (42.63, 141.60), "室蘭": (42.32, 140.97), "函館": (41.77, 140.73), "むつ": (41.29, 141.18),
    "八戸": (40.51, 141.49), "青森": (40.82, 140.74), "宮古": (39.64, 141.95), "盛岡": (39.70, 141.15),
    "秋田": (39.72, 140.10), "仙台": (38.27, 140.87), "福島": (37.76, 140.47), "新潟": (37.90, 139.02),
    "水戸": (36.37, 140.47), "銚子": (35.73, 140.83), "東京": (35.68, 139.77), "長野": (36.65, 138.19),
}

TRUE = L.Hypo(51.985, 157.270, 135.0, datetime(2026, 10, 16, 3, 35, 31, 900000, tzinfo=timezone.utc), "USGS")
M_TRUE = 6.9
tt = L.TravelTimes()
truth = L.RELATIONS["tsuboi"]


def case_2026():
    out = Path("cases/2026-10-16"); out.mkdir(parents=True, exist_ok=True)
    names = list(STATIONS)
    la = np.array([STATIONS[n][0] for n in names]); lo = np.array([STATIONS[n][1] for n in names])
    d = L.gc_km(TRUE.lat, TRUE.lon, la, lo)
    tp = tt.p(d, TRUE.depth_km) + rng.normal(0, 0.3, d.size)
    logA = truth.predict_logA(M_TRUE, d, TRUE.depth_km) + rng.normal(0, 0.2, d.size)
    I = L.pgv_to_shindo(L.predict_pgv(M_TRUE, TRUE.depth_km, d, "intraplate", 1.4)) + rng.normal(0, 0.3, d.size)
    L.write_csv_rows(out / "picks_synthetic.csv", ["観測点", "緯度", "経度", "P波到達時刻", "最大変位(μm)", "観測震度"], [
        {"観測点": n, "緯度": la[i], "経度": lo[i],
         "P波到達時刻": (TRUE.time + timedelta(seconds=float(tp[i]))).astimezone(L.JST).isoformat(timespec="milliseconds"),
         "最大変位(μm)": f"{10**logA[i]:.4g}", "観測震度": L.shindo_class(float(I[i]))} for i, n in enumerate(names)])

    # M 計算の震源: 気象庁（16日 22時40分の報道発表と記者会見）は、M を実際より日本から約680km
    # 遠い震源で計算したと説明。座標は公表されていないため、根室→真の震央の方位に沿って再構成した。
    brg = float(L.bearing_deg(43.33, 145.58, TRUE.lat, TRUE.lon))
    mlat, mlon = L.destination(TRUE.lat, TRUE.lon, brg, 680.0)
    alert = {
        "source": "気象庁", "serial": 17, "info_type": "発表", "region": "千島列島",
        "issued_at": "2026-10-16T12:40:02.6+09:00", "detected_at": "2026-10-16T12:38:10.1+09:00",
        "magnitude": 9.9,
        "hypocenter": {"lat": 48.2, "lon": 153.4, "depth_km": 140},
        "hypocenter_magnitude": {"lat": round(mlat, 2), "lon": round(mlon, 2), "depth_km": 140},
        "notes": ["第17報、発表時刻、地震波検知時刻、震源（北緯48.2度・東経153.4度・深さ140km）、M9.9 は、"
                  "気象庁「緊急地震速報（警報）発表状況」の本事案ページに公表された値",
                  "hypocenter_magnitude は【再構成】。気象庁は M を真の震源より日本から約680km 遠い位置で"
                  "計算したと説明しているが、座標は公表されていない"]}
    (out / "alert.json").write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")
    auth = {k: alert[k] for k in ("source", "serial", "issued_at", "detected_at", "magnitude", "hypocenter", "region")}
    auth["notes"] = ["気象庁が公表した 2026年10月16日の緊急地震速報（第17報）の記録を書き起こしたもの"]
    (out / "authority_jma.json").write_text(json.dumps(auth, ensure_ascii=False, indent=2), encoding="utf-8")
    usgs = {"type": "FeatureCollection", "metadata": {"note": "USGS GeoJSON 形式の検証用データ。値は USGS の公表による"},
            "features": [{"type": "Feature", "id": "fixture-kamchatka-20261016",
                          "properties": {"mag": 6.9, "magType": "mww", "place": "76 km NE of Ozernovskiy, Russia",
                                         "time": int(TRUE.time.timestamp() * 1000)},
                          "geometry": {"type": "Point", "coordinates": [TRUE.lon, TRUE.lat, TRUE.depth_km]}}]}
    (out / "usgs_event.geojson").write_text(json.dumps(usgs, ensure_ascii=False, indent=2), encoding="utf-8")
    # 気象庁の発表履歴ページのうち、パーサの検証に要る数値だけをページと同じ並びで抜き出したもの
    (out / "jma_history_excerpt.txt").write_text(
        "発生した地震の概要（速報値） 令和 8年10月16日12時36分 千島列島 48.2 153.4 140km 9.9 ---\n"
        "地震波 検知時刻 12時38分10.1秒 -- -- -- -- -- --\n"
        "17 12時40分02.6秒 112.5 48.2 153.4 140km 9.9 ※1\n"
        "19 12時40分41.6秒 151.5 48.2 153.4 140km 9.9 ※2\n", encoding="utf-8")

    # 合成の受信記録（全国型）。ドコモは意図的に含めていない ― ドコモは自社システムの不具合で、
    # この速報のエリアメールが配信されなかったと発表している。
    t_issue = L.parse_time(alert["issued_at"])
    rows = []
    for _ in range(400):
        ch = rng.choice(["エリアメール"] * 8 + ["テレビ", "Jアラート"])
        rows.append({"受信時刻": (t_issue + timedelta(seconds=float(abs(rng.normal(2.0, 1.0))))).astimezone(L.JST)
                     .strftime("%Y/%m/%d %H:%M:%S.%f")[:-3],
                     "緯度": f"{rng.uniform(34.6, 45.3):.4f}", "経度": f"{rng.uniform(137.5, 145.5):.4f}",
                     "事業者": rng.choice(["au", "ソフトバンク", "楽天モバイル"]) if ch == "エリアメール" else "",
                     "経路": ch, "受信強度_dbm": "", "セルID": "", "本文ハッシュ": "sha256:7f3a…eew17"})
    L.write_csv_rows(out / "receipts_synthetic.csv", list(rows[0]), rows)
    print(f"cases/2026-10-16 を作成。M 計算の震源（再構成）: {L.latlon_ja(mlat, mlon, 2)}")


def case_spoof():
    out = Path("cases/spoof-demo"); out.mkdir(parents=True, exist_ok=True)
    tx = (35.6595, 139.7005)     # 架空の偽基地局（渋谷スクランブル交差点付近）
    rows = []
    for _ in range(60):
        r_km = abs(rng.normal(0, 0.9)); th = rng.uniform(0, 2 * math.pi)
        lat, lon = L.from_enu_km(r_km * math.cos(th), r_km * math.sin(th), *tx)
        rssi = -30 - 10 * 3.1 * math.log10(max(r_km * 1000, 15)) + rng.normal(0, 4)
        t = datetime(2026, 10, 20, 18, 15, 2) + timedelta(seconds=float(rng.uniform(0, 4)))
        rows.append([t.strftime("%Y/%m/%d %H:%M:%S"), f"{float(lat):.6f}", f"{float(lon):.6f}", "ドコモ",
                     "エリアメール", f"{rssi:.1f}", "440-10-99999-1", "sha256:bad1"])
    # Excel から書き出した想定で Shift_JIS（cp932）・日本語の列名にする
    with open(out / "receipts.csv", "w", newline="", encoding="cp932") as f:
        f.write("受信時刻,緯度,経度,事業者,経路,受信強度_dbm,セルID,本文ハッシュ\r\n")
        for r in rows:
            f.write(",".join(r) + "\r\n")
    alert = {"source": "受信端末", "issued_at": "2026-10-20T18:15:00+09:00", "magnitude": 8.1, "region": "東京湾",
             "hypocenter": {"lat": 35.5, "lon": 139.9, "depth_km": 30}, "notes": ["検証用に作った架空の速報"]}
    (out / "alert.json").write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "usgs_empty.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    (out / "ground_truth.txt").write_text(f"送信源の真の位置: 北緯{tx[0]} 東経{tx[1]}（距離減衰係数 n=3.1、雑音 4dB）\n", encoding="utf-8")
    print(f"cases/spoof-demo を作成。送信源の真の位置: {L.latlon_ja(*tx, 4)}")


if __name__ == "__main__":
    L.setup_console()
    case_2026()
    case_spoof()
