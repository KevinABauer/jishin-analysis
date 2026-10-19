#!/usr/bin/env python3
"""
alert_triangulate.py — where did this alert actually come from?

"Triangulates" an earthquake alert along three independent axes, each of which can
separate an upstream processing error from a forged or injected message:

  A. AUTHORITY   Does the received alert match the issuing agency's own record?
                 (JMA 防災情報XML, the JMA EEW history page, or a JSON transcription.)
  B. FOOTPRINT   Who received it, where, over which carriers and channels?
                 A rogue LTE base station spoofing cell broadcast (ETWS/CMAS) reaches a
                 few km on the carriers it impersonates. An upstream alert reaches every
                 carrier, TV, radio and J-ALERT across the warned prefectures at once.
                 If the footprint is local, the transmitter is multilaterated from RSSI.
  C. PHYSICS     Is there a real earthquake behind it? Compared against an independent
                 catalogue (USGS by default), and — the strongest single test — whether
                 the agency's "wave detected" time equals the moment the real event's
                 P wave first reached the agency's network.

Usage
  python alert_triangulate.py --alert alert.json [--authority jma.xml ...]
         [--receipts receipts.csv] [--catalog usgs.geojson | --fetch-usgs]
         [--stations stations.csv] [--json report.json]

receipts.csv columns: received_at,lat,lon,carrier,channel[,rssi_dbm,cell_id,message_digest]
  channel ∈ cbs | tv | radio | jalert | app | other

No judgement here is certain; each verdict states what evidence would overturn it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

import eewlib as L


# ------------------------------------------------------------- A. authority
def check_authority(alert: L.Alert, auth_paths: list[str], rep: L.Report) -> str:
    rep.h("A · Authority — does the agency's own record contain this alert?")
    if not auth_paths:
        rep.flag("A0", "N/A", "No authoritative record supplied (--authority). JMA publishes EEW (warning) "
                 "content on its 緊急地震速報(警報)発表状況 pages and in 防災情報XML (VXSE43).")
        return "NOT_CHECKED"
    best, best_score = None, -1e9
    for p in auth_paths:
        try:
            a = L.load_alert(p)
        except Exception as e:  # noqa: BLE001
            rep.flag("A1", "INFO", f"could not parse {p}: {e}")
            continue
        score = 0.0
        if alert.event_id and a.event_id and alert.event_id == a.event_id: score += 5
        if alert.serial is not None and a.serial == alert.serial: score += 2
        if alert.magnitude is not None and a.magnitude is not None:
            score -= abs(alert.magnitude - a.magnitude) * 5
        if alert.h_int and a.h_int:
            score -= L.gc_km(alert.h_int.lat, alert.h_int.lon, a.h_int.lat, a.h_int.lon) / 20
        if alert.issued_at and a.issued_at:
            score -= abs((alert.issued_at - a.issued_at).total_seconds()) / 30
        if score > best_score:
            best, best_score = a, score
    if best is None:
        rep.flag("A1", "FLAG", "No parseable authoritative record.")
        return "ABSENT"

    diffs = []
    if alert.magnitude is not None and best.magnitude is not None and abs(alert.magnitude - best.magnitude) > 0.05:
        diffs.append(f"M {alert.magnitude} vs agency {best.magnitude}")
    if alert.h_int and best.h_int:
        sep = float(L.gc_km(alert.h_int.lat, alert.h_int.lon, best.h_int.lat, best.h_int.lon))
        if sep > 15 or abs(alert.h_int.depth_km - best.h_int.depth_km) > 15:
            diffs.append(f"hypocentre differs by {sep:.0f} km / {abs(alert.h_int.depth_km - best.h_int.depth_km):.0f} km depth")
    if alert.issued_at and best.issued_at:
        dt = (alert.issued_at - best.issued_at).total_seconds()
        if abs(dt) > 60:
            diffs.append(f"issue time differs by {dt:+.0f} s")
    rep.data["authority_record"] = best.to_dict()
    if diffs:
        rep.flag("A2", "FLAG", "Alert does NOT match the agency record: " + "; ".join(diffs))
        return "MISMATCH"
    rep.flag("A2", "PASS", f"Alert matches the agency record (M{best.magnitude}, {best.h_int}, "
             f"issued {L.fmt_time(best.issued_at)}). The agency itself published this content.")
    return "MATCH"


# ------------------------------------------------------------- B. footprint
def load_receipts(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": L.parse_time(r["received_at"]), "lat": float(r["lat"]), "lon": float(r["lon"]),
                "carrier": (r.get("carrier") or "").strip().lower(),
                "channel": (r.get("channel") or "cbs").strip().lower(),
                "rssi": float(r["rssi_dbm"]) if r.get("rssi_dbm") not in (None, "") else None,
                "cell": (r.get("cell_id") or "").strip(),
                "digest": (r.get("message_digest") or "").strip(),
            })
    return rows


def multilaterate_rssi(rows, rep: L.Report):
    """
    Log-distance path loss: RSSI = P0 − 10·n·log10(d / 1 m). For every candidate
    transmitter position on a grid, (P0, n) is solved by linear least squares
    (n constrained to 1.6–4.5); the best position minimises the residual. The 95 %
    region uses Δχ² ≤ 5.99 (2 d.o.f.) with σ² estimated from the best fit.
    Phone-reported RSSI is noisy: treat the result as a search area for a
    direction-finding team, not a fix.
    """
    pts = [r for r in rows if r["rssi"] is not None]
    if len(pts) < 5:
        rep.flag("B5", "N/A", f"Only {len(pts)} receipts carry RSSI; need ≥5 to multilaterate. "
                 "Weighted centroid only.")
        return None
    lat0 = float(np.mean([r["lat"] for r in pts])); lon0 = float(np.mean([r["lon"] for r in pts]))
    x, y = L.to_enu_km([r["lat"] for r in pts], [r["lon"] for r in pts], lat0, lon0)
    rssi = np.array([r["rssi"] for r in pts])
    span = max(np.ptp(x), np.ptp(y), 1.0)
    g = np.linspace(-span, span, 241)
    GX, GY = np.meshgrid(g, g)
    d_m = np.sqrt((GX[..., None] - x) ** 2 + (GY[..., None] - y) ** 2) * 1000 + 10.0
    X = -10 * np.log10(d_m)                                      # regressor for n
    Xm, Ym = X.mean(-1, keepdims=True), rssi.mean()
    n = ((X - Xm) * (rssi - Ym)).sum(-1) / np.maximum(((X - Xm) ** 2).sum(-1), 1e-9)
    n = np.clip(n, 1.6, 4.5)
    P0 = Ym - n * Xm[..., 0]
    ssr = ((rssi - (P0[..., None] + n[..., None] * X)) ** 2).sum(-1)
    k = np.unravel_index(np.argmin(ssr), ssr.shape)
    sigma2 = ssr[k] / max(len(pts) - 4, 1)
    region = (ssr - ssr[k]) / max(sigma2, 1e-6) <= 5.99
    ry, rx = GY[region], GX[region]
    blat, blon = L.from_enu_km(GX[k], GY[k], lat0, lon0)
    radius = max(float(np.sqrt(((rx - GX[k]) ** 2 + (ry - GY[k]) ** 2).max())) if rx.size else 0.0, float(g[1] - g[0]))
    res = {"lat": float(blat), "lon": float(blon), "radius95_km": radius, "path_loss_exponent": float(n[k]),
           "p0_dbm_at_1m": float(P0[k]), "rms_db": float(math.sqrt(sigma2)), "n_used": len(pts)}
    rep.flag("B5", "INFO", f"Transmitter estimate {blat:.5f}°N {blon:.5f}°E, 95 % region radius ≈ {radius:.2f} km "
             f"(path-loss exponent {n[k]:.2f}, fit RMS {math.sqrt(sigma2):.1f} dB, {len(pts)} receipts).")
    return res


def check_footprint(rows, rep: L.Report, min_local: int = 10) -> str:
    rep.h("B · Footprint — what does the distribution pattern say about the origin?")
    if not rows:
        rep.flag("B0", "N/A", "No receipts supplied (--receipts).")
        return "NOT_CHECKED"
    lat = np.array([r["lat"] for r in rows]); lon = np.array([r["lon"] for r in rows])
    idx = np.random.default_rng(0).choice(len(rows), size=min(len(rows), 1500), replace=False)
    D = L.gc_km(lat[idx, None], lon[idx, None], lat[None, idx], lon[None, idx])
    extent = float(D.max())
    carriers = sorted({r["carrier"] for r in rows if r["carrier"]})
    channels = sorted({r["channel"] for r in rows})
    digests = sorted({r["digest"] for r in rows if r["digest"]})
    ts = sorted(r["t"] for r in rows if r["t"])
    spread = (ts[int(0.95 * (len(ts) - 1))] - ts[int(0.05 * (len(ts) - 1))]).total_seconds() if len(ts) > 2 else 0.0
    rep.data["footprint"] = {"receipts": len(rows), "extent_km": extent, "carriers": carriers,
                             "channels": channels, "distinct_digests": len(digests), "t_spread_p5_p95_s": spread,
                             "min_local_receipts": min_local}
    rep.flag("B1", "INFO", f"{len(rows)} receipts spanning {extent:.0f} km; carriers {carriers or '—'}; "
             f"channels {channels}; 90 % of receipts within {spread:.1f} s.")
    if len(digests) > 1:
        rep.flag("B2", "FLAG", f"{len(digests)} different message texts in circulation — content diverged "
                 "somewhere in distribution. Compare each against the agency text.")
    # Asymmetry on purpose: a WIDE footprint is positive evidence and a handful of receipts can show it.
    # A NARROW footprint is an absence of spread, and every small sample looks narrow — so a local
    # verdict needs a minimum number of receipts.
    non_cbs = [c for c in channels if c not in ("cbs", "")]
    if extent >= 150 or (len(carriers) >= 2 and extent >= 50) or non_cbs:
        why = []
        if extent >= 150: why.append(f"{extent:.0f} km extent")
        if len(carriers) >= 2: why.append(f"{len(carriers)} independent carriers")
        if non_cbs: why.append(f"non-cellular channels {non_cbs}")
        rep.flag("B3", "PASS", "NATIONAL pattern (" + ", ".join(why) + "). A radio-layer spoof cannot produce "
                 "this: it would need simultaneous rogue cells on every carrier across hundreds of km, plus "
                 "control of broadcast TV/J-ALERT. The common point is upstream of the carriers.")
        return "NATIONAL"
    if extent <= 30 and len(carriers) <= 1 and set(channels) <= {"cbs"}:
        if len(rows) < min_local:
            rep.flag("B3", "INFO", f"Only {len(rows)} receipt(s), all close together — too few to call the footprint "
                     f"local (need ≥{min_local}). A small sample always looks local. Collect more receipts, "
                     "including from people on other carriers nearby who did NOT receive it.")
            return "INDETERMINATE"
        rep.flag("B3", "FLAG", f"LOCAL pattern: {len(rows)} receipts within {extent:.1f} km, single carrier, cell "
                 "broadcast only. Consistent with a rogue base station (fake eNB/gNB) broadcasting ETWS/CMAS.")
        cells = sorted({r['cell'] for r in rows if r['cell']})
        if cells:
            rep.flag("B4", "INFO", f"Cell IDs reported: {cells[:12]}{' …' if len(cells) > 12 else ''}. "
                     "IDs absent from the carrier's own cell list are a strong fake-cell indicator.")
        tx = multilaterate_rssi(rows, rep)
        if tx is None:
            rep.flag("B5", "INFO", f"Weighted centroid of receipts: {lat.mean():.5f}°N {lon.mean():.5f}°E "
                     "(weak — the transmitter can sit well outside the receipt cloud).")
        rep.data["transmitter_estimate"] = tx
        return "LOCAL"
    rep.flag("B3", "INFO", "Footprint neither clearly national nor clearly local. Collect more receipts "
             "(other carriers, TV/J-ALERT logs) before concluding.")
    return "INDETERMINATE"
def fetch_usgs(alert: L.Alert, minmag=4.0) -> dict:
    t0 = alert.origin_time or alert.detected_at or alert.issued_at
    if t0 is None:
        raise SystemExit("--fetch-usgs needs an origin/detection/issue time in the alert")
    q = {"format": "geojson", "minmagnitude": minmag,
         "starttime": (t0 - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S"),
         "endtime": (t0 + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")}
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def load_catalog(path) -> dict:
    return json.loads(L.read_text(path))


def catalog_events(geo: dict) -> list[dict]:
    out = []
    for ft in geo.get("features", []):
        pr, (lon, lat, dep) = ft["properties"], ft["geometry"]["coordinates"][:3]
        out.append({"id": ft.get("id"), "mag": pr.get("mag"), "magType": pr.get("magType"),
                    "place": pr.get("place"),
                    "time": datetime.fromtimestamp(pr["time"] / 1000, tz=timezone.utc),
                    "hypo": L.Hypo(lat, lon, dep, datetime.fromtimestamp(pr["time"] / 1000, tz=timezone.utc), "catalog")})
    return out


def check_physics(alert: L.Alert, geo: dict | None, points, tt: L.TravelTimes, rep: L.Report) -> tuple[str, str]:
    rep.h("C · Physics — is there a real earthquake behind the alert?")
    if geo is None:
        rep.flag("C0", "N/A", "No independent catalogue (--catalog or --fetch-usgs).")
        return "NOT_CHECKED", "NOT_CHECKED"
    evs = catalog_events(geo)
    ref_t = alert.origin_time or alert.detected_at or alert.issued_at
    cands = [e for e in evs if ref_t and timedelta(minutes=-20) <= (e["time"] - ref_t) <= timedelta(minutes=10)]
    if not cands:
        rep.flag("C1", "FLAG", "No M≥4 event in the independent catalogue within −20/+10 min. Either nothing "
                 "happened (sensor/telemetry glitch, or fabrication), or the catalogue lags — re-query later.")
        return "NO_PHYSICAL_EVENT", "NOT_CHECKED"

    # Pick the candidate whose P wave reached the network closest to the agency's detection time;
    # fall back to the largest event.
    def p_arrival(e):
        _, _, t = L.first_arrival_in_network(e["hypo"], tt, points)
        return e["time"] + timedelta(seconds=t)
    if alert.detected_at:
        ev = min(cands, key=lambda e: abs((p_arrival(e) - alert.detected_at).total_seconds()))
    else:
        ev = max(cands, key=lambda e: e["mag"] or 0)
    rep.data["catalog_event"] = {k: v for k, v in ev.items() if k != "hypo"} | {"hypo": ev["hypo"].to_dict()}
    rep.flag("C1", "INFO", f"Independent catalogue: M{ev['mag']} {ev['magType'] or ''} — {ev['place']} — "
             f"{ev['hypo']} at {L.fmt_time(ev['time'])} ({ev['id']}).")

    # Compare whatever the alert actually carries. Nothing compared → NOT_CHECKED, never "consistent".
    physics, compared, disagree = "NOT_CHECKED", [], []
    disc = {}
    if alert.magnitude is not None and ev["mag"] is not None:
        dM = alert.magnitude - ev["mag"]
        disc |= {"dM": dM, "energy_ratio": 10 ** (1.5 * dM)}
        compared.append("magnitude")
        if abs(dM) >= 1.0:
            disagree.append(f"ΔM = {dM:+.1f} (≈{10 ** (1.5 * dM):,.0f}× the energy)")
    if alert.h_int:
        sep = float(L.gc_km(alert.h_int.lat, alert.h_int.lon, ev["hypo"].lat, ev["hypo"].lon))
        disc["epicentre_sep_km"] = sep
        compared.append("location")
        if sep > 100:
            disagree.append(f"epicentre {sep:.0f} km away")
    rep.data["discrepancy"] = disc | {"compared": compared}
    if disagree:
        physics = "REAL_EVENT_WRONG_PARAMETERS"
        rep.flag("C2", "FLAG", "Alert parameters disagree with the real event: " + "; ".join(disagree) +
                 ". The event is real; the alert's description of it is not.")
    elif compared:
        physics = "CONSISTENT"
        rep.flag("C2", "PASS", f"Alert agrees with the catalogue on {' and '.join(compared)}"
                 + ("" if len(compared) == 2 else f" (alert carries no {'location' if 'location' not in compared else 'magnitude'})") + ".")
    else:
        rep.flag("C2", "N/A", "Alert carries neither a magnitude nor a hypocentre — nothing to compare.")
    if alert.magnitude is not None and alert.magnitude >= 9.6:
        rep.flag("C3", "FLAG", f"M{alert.magnitude} exceeds the largest instrumentally recorded earthquake "
                 "(Chile 1960, Mw 9.5). Physically possible in principle, but as a prior it marks an estimator "
                 "operating outside its envelope.")

    timing = "NOT_CHECKED"
    if alert.detected_at:
        name, dkm, tP = L.first_arrival_in_network(ev["hypo"], tt, points)
        pred = ev["time"] + timedelta(seconds=tP)
        lag = (alert.detected_at - pred).total_seconds()
        rep.data["timing"] = {"first_point": name, "distance_km": dkm, "p_travel_s": tP,
                              "predicted_first_p": pred, "agency_detected": alert.detected_at, "lag_s": lag}
        ok = -5.0 <= lag <= 20.0
        timing = "CONSISTENT" if ok else "INCONSISTENT"
        rep.flag("C4", "PASS" if ok else "FLAG",
                 f"Agency detection {L.fmt_time(alert.detected_at)} vs. real event's P wave first reaching the "
                 f"network ({name}, {dkm:.0f} km, {tP:.1f} s travel) at {L.fmt_time(pred)}: lag {lag:+.1f} s. "
                 + ("The trigger WAS that earthquake's P wave." if ok else
                    "The trigger does not line up with this event's P arrival — look for another cause."))
        if not tt.exact:
            rep.flag("C4", "INFO", "Travel times from the crude fallback model — install obspy to rely on C4.")
    return physics, timing
VERDICTS = {
    "LOCAL_INJECTION": "LOCAL RADIO-LAYER INJECTION — candidate",
    "DISTRIBUTION_ALTERED": "CONTENT ALTERED IN DISTRIBUTION — investigate the chain between agency and receivers",
    "PROCESSING_FAULT": "AUTHENTIC UPSTREAM ALERT, WRONG CONTENT — processing-fault class",
    "UPSTREAM_FALSE_TRIGGER": "UPSTREAM FALSE TRIGGER (no real event) — sensor/telemetry fault or upstream compromise",
    "CONSISTENT": "CONSISTENT ALERT — no anomaly found in what was checked",
    "INDETERMINATE": "INDETERMINATE — the evidence supplied does not support a conclusion",
}


def verdict(auth, foot, phys, timing, rep: L.Report, compared=()):
    rep.h("Verdict")
    code, why, overturn = "INDETERMINATE", [], []
    if foot == "LOCAL" and auth in ("ABSENT", "MISMATCH", "NOT_CHECKED"):
        code = "LOCAL_INJECTION"
        why = ["enough receipts, all confined to a small area on one carrier, cell broadcast only"]
        overturn = ["the agency record contains the same alert", "receipts from other carriers/channels or regions"]
    elif auth == "MISMATCH" and foot in ("NATIONAL", "NOT_CHECKED", "INDETERMINATE"):
        code = "DISTRIBUTION_ALTERED"
        why = ["what people received differs from what the agency published"]
        overturn = ["the difference is a transcription error in the received copy"]
    elif auth in ("MATCH", "NOT_CHECKED") and phys == "REAL_EVENT_WRONG_PARAMETERS" and timing in ("CONSISTENT", "NOT_CHECKED"):
        code = "PROCESSING_FAULT"
        why = ["a real earthquake occurred", "the alert describes it wrongly"]
        if timing == "CONSISTENT": why.append("the agency's trigger time equals that earthquake's P arrival")
        if auth == "MATCH": why.append("the agency's own record carries the same content")
        if foot == "NATIONAL": why.append("distribution pattern is upstream/national, not radio-layer")
        overturn = ["evidence the estimator's inputs (station data, telemetry) were tampered with — "
                    "which needs station-level data; run eew_diagnose.py",
                    "unexplained changes to the EEW software/configuration before the event (change-management logs)"]
    elif phys == "NO_PHYSICAL_EVENT" and auth in ("MATCH", "NOT_CHECKED"):
        code = "UPSTREAM_FALSE_TRIGGER"
        why = ["agency issued it", "no earthquake in an independent catalogue"]
        overturn = ["a late catalogue entry", "station-level data showing a single faulty sensor "
                    "(cf. JMA 2016-08-01 Futtsu power-supply fault; 2013-08-08 seafloor timestamp fault)"]
    elif phys == "CONSISTENT" and timing != "INCONSISTENT" and auth != "MISMATCH":
        code = "CONSISTENT"
        why = [f"agrees with an independent catalogue on {' and '.join(compared) or '—'}"]
        if "location" not in compared: overturn.append("the alert's location, once known, disagrees")
    else:
        why = [f"authority={auth}, footprint={foot}, physics={phys}, timing={timing}"]
        if timing == "INCONSISTENT":
            why.append("the agency's trigger time does not match the catalogue event's P arrival")
        overturn = ["more evidence on whichever axis is NOT_CHECKED or INDETERMINATE"]
    rep.p(f"  {VERDICTS[code]}")
    for w in why: rep.p(f"    because: {w}")
    for o in overturn: rep.p(f"    would change if: {o}")
    rep.data["verdict"] = {"code": code, "verdict": VERDICTS[code], "because": why, "would_change_if": overturn,
                           "axes": {"authority": auth, "footprint": foot, "physics": phys, "timing": timing}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alert", required=True, help="alert as received (normalised JSON or JMA XML)")
    ap.add_argument("--authority", nargs="*", default=[], help="agency record(s): JMA XML or JSON")
    ap.add_argument("--receipts", help="CSV of receipt observations")
    ap.add_argument("--min-local-receipts", type=int, default=10,
                    help="receipts needed before a footprint can be called local (default 10)")
    ap.add_argument("--catalog", help="independent catalogue, USGS GeoJSON")
    ap.add_argument("--fetch-usgs", action="store_true", help="query the USGS FDSN event service")
    ap.add_argument("--stations", help="CSV station,lat,lon — the agency's sensors (default: reference points)")
    ap.add_argument("--json", help="write machine-readable report here")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    alert = L.load_alert(a.alert)
    rep = L.Report()
    rep.p(f"Alert: {alert.source} {alert.region} M{alert.magnitude if alert.magnitude is not None else '?'} at "
          f"{alert.h_int or 'no hypocentre'} — issued {L.fmt_time(alert.issued_at)} (serial {alert.serial}, "
          f"detected {L.fmt_time(alert.detected_at)})")
    for n in alert.notes:
        rep.p(f"  note: {n}")
    rep.data["alert"] = alert.to_dict()

    auth = check_authority(alert, a.authority, rep)
    foot = check_footprint(load_receipts(a.receipts) if a.receipts else [], rep, a.min_local_receipts)
    geo = load_catalog(a.catalog) if a.catalog else (fetch_usgs(alert) if a.fetch_usgs else None)
    points = L.load_points_csv(a.stations) if a.stations else L.JAPAN_REFERENCE_POINTS
    tt = L.TravelTimes(quiet=False)
    phys, timing = check_physics(alert, geo, points, tt, rep)
    verdict(auth, foot, phys, timing, rep, rep.data.get("discrepancy", {}).get("compared", ()))

    sys.stdout.write(rep.text())
    if a.json:
        L.write_json(a.json, rep.data)
    return rep.data


if __name__ == "__main__":
    main()
