#!/usr/bin/env python3
"""
make_synthetic.py — build test inputs for both tools.

  cases/2026-10-16/   the 16 Oct 2026 event: REAL published alert values and the USGS
                      solution, with SYNTHETIC station picks/amplitudes/intensities and
                      SYNTHETIC receipts (no public per-handset data exists).
  cases/spoof-demo/   a fabricated rogue-base-station scenario with a known transmitter
                      location, to verify the multilateration recovers it.

Station coordinates are approximate town locations standing in for JMA/Hi-net sites.
Synthetic amplitudes use Tsuboi (1954) extrapolated — a stand-in for ground truth, chosen
before running the diagnostics and not tuned to reproduce M9.9.
"""
import csv, json, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import eewlib as L

rng = np.random.default_rng(20261016)
JST = L.JST

STATIONS = {  # name: (lat, lon) — approximate, Hokkaido → Tokai
 "RAUSU": (44.02,145.19), "NEMURO": (43.33,145.58), "NAKASHIBETSU": (43.55,144.97), "ABASHIRI": (44.02,144.27),
 "KUSHIRO": (42.98,144.38), "MONBETSU": (44.36,143.35), "OBIHIRO": (42.92,143.20), "ERIMO": (41.93,143.25),
 "ASAHIKAWA": (43.77,142.37), "WAKKANAI": (45.42,141.67), "URAKAWA": (42.17,142.77), "SAPPORO": (43.06,141.35),
 "TOMAKOMAI": (42.63,141.60), "MURORAN": (42.32,140.97), "HAKODATE": (41.77,140.73), "MUTSU": (41.29,141.18),
 "HACHINOHE": (40.51,141.49), "AOMORI": (40.82,140.74), "MIYAKO": (39.64,141.95), "MORIOKA": (39.70,141.15),
 "AKITA": (39.72,140.10), "SENDAI": (38.27,140.87), "FUKUSHIMA": (37.76,140.47), "NIIGATA": (37.90,139.02),
 "MITO": (36.37,140.47), "CHOSHI": (35.73,140.83), "TOKYO": (35.68,139.77), "NAGANO": (36.65,138.19),
}

TRUE = L.Hypo(51.985, 157.270, 135.0, datetime(2026, 10, 16, 3, 35, 31, 900000, tzinfo=timezone.utc), "USGS")
M_TRUE = 6.9
tt = L.TravelTimes()
truth = L.RELATIONS["tsuboi"]

def case_2026():
    out = Path("cases/2026-10-16"); out.mkdir(parents=True, exist_ok=True)
    names = list(STATIONS); la = np.array([STATIONS[n][0] for n in names]); lo = np.array([STATIONS[n][1] for n in names])
    d = L.gc_km(TRUE.lat, TRUE.lon, la, lo)
    tp = tt.p(d, TRUE.depth_km) + rng.normal(0, 0.3, d.size)
    logA = truth.predict_logA(M_TRUE, d, TRUE.depth_km) + rng.normal(0, 0.2, d.size)
    I = L.pgv_to_shindo(L.predict_pgv(M_TRUE, TRUE.depth_km, d, "intraplate", 1.4)) + rng.normal(0, 0.3, d.size)
    with open(out / "picks_synthetic.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["station", "lat", "lon", "p_time", "amp_um", "observed_shindo"])
        for i, n in enumerate(names):
            w.writerow([n, la[i], lo[i], (TRUE.time + timedelta(seconds=float(tp[i]))).isoformat(),
                        f"{10**logA[i]:.4g}", L.shindo_class(float(I[i]))])

    # Magnitude hypocentre: JMA (22:40 report + press conference) says M was computed as if the
    # source were ~680 km farther from Japan than it was. Coordinates were not published; this is
    # a reconstruction along the bearing Nemuro → true epicentre.
    brg = float(L.bearing_deg(43.33, 145.58, TRUE.lat, TRUE.lon))
    mlat, mlon = L.destination(TRUE.lat, TRUE.lon, brg, 680.0)
    alert = {
        "source": "JMA", "serial": 17, "info_type": "発表", "region": "千島列島",
        "issued_at": "2026-10-16T12:40:02.6+09:00", "detected_at": "2026-10-16T12:38:10.1+09:00",
        "magnitude": 9.9,
        "hypocenter": {"lat": 48.2, "lon": 153.4, "depth_km": 140},
        "hypocenter_magnitude": {"lat": round(mlat, 2), "lon": round(mlon, 2), "depth_km": 140},
        "notes": ["serial/issue/detection time, hypocentre 48.2N 153.4E 140 km and M9.9 are as published on "
                  "JMA's 緊急地震速報(警報)発表状況 page for this event",
                  "hypocenter_magnitude is a RECONSTRUCTION: JMA stated M was computed ~680 km farther from Japan "
                  "than the true source; exact coordinates not published"]}
    (out / "alert.json").write_text(json.dumps(alert, ensure_ascii=False, indent=2))
    auth = {k: alert[k] for k in ("source", "serial", "issued_at", "detected_at", "magnitude", "hypocenter", "region")}
    auth["notes"] = ["transcribed from JMA's published EEW record for 2026-10-16 (serial 17)"]
    (out / "authority_jma.json").write_text(json.dumps(auth, ensure_ascii=False, indent=2))
    usgs = {"type": "FeatureCollection", "metadata": {"note": "fixture in USGS GeoJSON shape; values from USGS reporting"},
            "features": [{"type": "Feature", "id": "fixture-kamchatka-20261016",
                          "properties": {"mag": 6.9, "magType": "mww", "place": "76 km NE of Ozernovskiy, Russia",
                                         "time": int(TRUE.time.timestamp() * 1000)},
                          "geometry": {"type": "Point", "coordinates": [TRUE.lon, TRUE.lat, TRUE.depth_km]}}]}
    (out / "usgs_event.geojson").write_text(json.dumps(usgs, indent=2))
    # Excerpt of the facts on JMA's history page (numbers only), in the page's own layout.
    (out / "jma_history_excerpt.txt").write_text(
        "発生した地震の概要（速報値） 令和 8年10月16日12時36分 千島列島 48.2 153.4 140km 9.9 ---\n"
        "地震波 検知時刻 12時38分10.1秒 -- -- -- -- -- --\n"
        "17 12時40分02.6秒 112.5 48.2 153.4 140km 9.9 ※1\n"
        "19 12時40分41.6秒 151.5 48.2 153.4 140km 9.9 ※2\n", encoding="utf-8")

    # Synthetic receipts: national footprint. docomo deliberately absent — docomo reported that its
    # Area Mail was not delivered for this alert because of a fault in its own system.
    t_issue = L.parse_time(alert["issued_at"])
    rows = []
    for _ in range(400):
        lat = rng.uniform(34.6, 45.3); lon = rng.uniform(137.5, 145.5)
        ch = rng.choice(["cbs"] * 8 + ["tv", "jalert"])
        car = rng.choice(["au", "softbank", "rakuten"]) if ch == "cbs" else ""
        t = t_issue + timedelta(seconds=float(abs(rng.normal(2.0, 1.0))))
        rows.append([t.isoformat(), f"{lat:.4f}", f"{lon:.4f}", car, ch, "", "", "sha256:7f3a…eew17"])
    with open(out / "receipts_synthetic.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["received_at", "lat", "lon", "carrier", "channel", "rssi_dbm", "cell_id", "message_digest"])
        w.writerows(rows)
    print("cases/2026-10-16 written; reconstructed magnitude hypocentre", round(mlat, 2), round(mlon, 2))

def case_spoof():
    out = Path("cases/spoof-demo"); out.mkdir(parents=True, exist_ok=True)
    tx = (35.6595, 139.7005)     # fictional rogue cell near Shibuya crossing
    rows = []
    for _ in range(60):
        r_km = abs(rng.normal(0, 0.9)); th = rng.uniform(0, 2 * math.pi)
        lat, lon = L.from_enu_km(r_km * math.cos(th), r_km * math.sin(th), *tx)
        d_m = max(r_km * 1000, 15)
        rssi = -30 - 10 * 3.1 * math.log10(d_m) + rng.normal(0, 4)
        rows.append([(datetime(2026, 10, 20, 9, 15, 2, tzinfo=timezone.utc) + timedelta(seconds=float(rng.uniform(0, 4)))).isoformat(),
                     f"{float(lat):.6f}", f"{float(lon):.6f}", "docomo", "cbs", f"{rssi:.1f}", "440-10-99999-1", "sha256:bad1…"])
    with open(out / "receipts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["received_at", "lat", "lon", "carrier", "channel", "rssi_dbm", "cell_id", "message_digest"])
        w.writerows(rows)
    alert = {"source": "received-handset", "issued_at": "2026-10-20T09:15:00Z", "magnitude": 8.1, "region": "東京湾",
             "hypocenter": {"lat": 35.5, "lon": 139.9, "depth_km": 30}, "notes": ["fabricated demo alert"]}
    (out / "alert.json").write_text(json.dumps(alert, ensure_ascii=False, indent=2))
    (out / "usgs_empty.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    (out / "TRUTH.txt").write_text(f"true transmitter {tx[0]} {tx[1]} (path-loss n=3.1, 4 dB noise)\n")
    print("cases/spoof-demo written; true transmitter", tx)

if __name__ == "__main__":
    case_2026(); case_spoof()
