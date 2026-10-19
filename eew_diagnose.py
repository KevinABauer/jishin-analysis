#!/usr/bin/env python3
"""
eew_diagnose.py — given seismograph data, why did the early-warning estimate go wrong?

Independently re-derives location and magnitude from the station data, then tests the
alert against the failure classes JMA's EEW has actually exhibited:

  D1  mislocation            alert hypocentre vs. what the P picks support
  D2  distance cutoff        magnitude needs stations beyond N km of the hypocentre
                             (JMA has excluded stations >700 km since 2020-09-07)
  D3  stage inconsistency    magnitude and intensity computed from different hypocentres
  D4  out-of-network event   azimuthal gap / distance to nearest sensor — location is then
                             under-determined along the back-azimuth
  D5  magnitude plausibility M vs. physical prior and vs. observed amplitudes
  D6  intensity over-predict predicted vs. observed shindo, and whether the intensity
                             stage was faithful to the inputs it was given
  D7  single-sensor control  leave-one-out: does one station drive M or location?
  D8  outlier picks          overlapping events / timestamp faults

Subcommands
  picks       analyse a picks/amplitudes CSV         (station,lat,lon,p_time[,amp_um,observed_shindo])
  waveforms   pick P and measure amplitudes from miniSEED/SAC, then analyse
  fetch       download waveforms + StationXML from an FDSN data centre (needs network)
  jma-history parse a saved JMA 緊急地震速報 history page into the serial-by-serial sequence

The magnitude relation is a parameter (--relation / --relation-coeffs). The agency's
operational coefficients are not public; findings that do not depend on them
(D2 station counts, D3, D4, D6-faithfulness, timing) are marked as relation-independent.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import eewlib as L


# ===================================================================== data
class Picks:
    def __init__(self, rows):
        self.sta = [r["station"] for r in rows]
        self.lat = np.array([float(r["lat"]) for r in rows])
        self.lon = np.array([float(r["lon"]) for r in rows])
        t = [L.parse_time(r["p_time"]) for r in rows]
        self.t_ref = min(t)
        self.t = np.array([(x - self.t_ref).total_seconds() for x in t])
        self.logA = np.array([math.log10(float(r["amp_um"])) if r.get("amp_um") not in (None, "") else np.nan for r in rows])
        self.obs_I = np.array([L.SHINDO_NUM.get(str(r.get("observed_shindo", "")).strip(), np.nan)
                               if r.get("observed_shindo") not in (None, "") else np.nan for r in rows])

    def __len__(self):
        return len(self.sta)

    def subset(self, mask):
        new = object.__new__(Picks)
        new.sta = [s for s, m in zip(self.sta, mask) if m]
        new.lat, new.lon, new.t = self.lat[mask], self.lon[mask], self.t[mask]
        new.logA, new.obs_I, new.t_ref = self.logA[mask], self.obs_I[mask], self.t_ref
        return new

    @classmethod
    def from_csv(cls, path):
        with open(path, newline="", encoding="utf-8") as f:
            return cls(list(csv.DictReader(f)))


# ================================================================== locator
def misfit_grid(p: Picks, tt: L.TravelTimes, lats, lons, depths, fixed_t0=None):
    """L1 misfit over a lat×lon×depth grid; origin time solved per node (median) unless fixed."""
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    D = L.gc_km(LA[..., None], LO[..., None], p.lat, p.lon)            # (nlat,nlon,nsta)
    best = (np.inf, None)
    cube = np.empty((lats.size, lons.size, depths.size))
    for k, z in enumerate(depths):
        T = tt.p(D, z)
        if fixed_t0 is None:
            t0 = np.median(p.t - T, axis=-1)
        else:
            t0 = np.full(T.shape[:-1], fixed_t0)
        r = p.t - T - t0[..., None]
        m = np.mean(np.abs(r), axis=-1)
        cube[..., k] = m
        i, j = np.unravel_index(np.argmin(m), m.shape)
        if m[i, j] < best[0]:
            best = (m[i, j], (i, j, k, t0[i, j]))
    return cube, best


def locate(p: Picks, tt: L.TravelTimes, center=None, half_deg=20.0, fixed_t0=None):
    if center is None:
        center = (float(np.mean(p.lat)), float(np.mean(p.lon)))
    clat, clon, cz = center[0], center[1], None
    stages = [(half_deg, 1.0, np.arange(0, 701, 20.0)), (2.0, 0.1, None), (0.25, 0.02, None)]
    for hw, step, depths in stages:
        lats = np.arange(max(clat - hw, -89), min(clat + hw, 89) + 1e-9, step)
        lons = np.arange(clon - hw, clon + hw + 1e-9, step)
        if depths is None:
            depths = np.arange(max(cz - 60, 0), min(cz + 60, 700) + 1e-9, 5.0)
        cube, (mis, (i, j, k, t0)) = misfit_grid(p, tt, lats, lons, depths, fixed_t0)
        clat, clon, cz = float(lats[i]), float(lons[j]), float(depths[k])
        if hw == stages[0][0]:
            coarse = (cube, lats, lons, depths)
    h = L.Hypo(clat, clon, cz, p.t_ref + timedelta(seconds=float(t0)), "data")
    # Uncertainty from the coarse cube: nodes within misfit ≤ best + max(0.5 s, 50 %).
    cube, lats, lons, depths = coarse
    thr = mis + max(0.5, 0.5 * mis)
    ii, jj, kk = np.nonzero(cube <= thr)
    region_km = float(L.gc_km(clat, clon, lats[ii], lons[jj]).max()) if ii.size else float("nan")
    region_az = None
    if ii.size > 3:
        az = L.bearing_deg(clat, clon, lats[ii], lons[jj])
        far = L.gc_km(clat, clon, lats[ii], lons[jj]) > 0.5 * region_km
        if far.any():
            region_az = float(np.median(az[far]) % 180)
    return h, float(mis), region_km, region_az


def residuals(p: Picks, tt: L.TravelTimes, h: L.Hypo, fixed_t0=True):
    d = L.gc_km(h.lat, h.lon, p.lat, p.lon)
    T = tt.p(d, h.depth_km)
    if fixed_t0 and h.time is not None:
        t0 = (h.time - p.t_ref).total_seconds()
    else:
        t0 = float(np.median(p.t - T))
    return p.t - T - t0, t0


# =============================================================== magnitudes
def station_M(p: Picks, h: L.Hypo, rel: L.AmpRelation):
    d = L.gc_km(h.lat, h.lon, p.lat, p.lon)
    return rel.invert_M(p.logA, d, h.depth_km), d, L.hypo_km(d, h.depth_km)


def network_M(p: Picks, h: L.Hypo, rel: L.AmpRelation, cutoff_km: float | None):
    M, d, R = station_M(p, h, rel)
    ok = np.isfinite(M)
    if cutoff_km is not None:
        ok &= R <= cutoff_km
    return (float(np.median(M[ok])) if ok.any() else None), int(ok.sum()), float(R.min())


def _ref_mag(ref):
    if ref is not None and "M" in (ref.label or ""):
        try:
            return float(ref.label.split("M")[-1])
        except ValueError:
            return None
    return None


# ================================================================ analysis
def analyse(p: Picks, alert: L.Alert, rel: L.AmpRelation, cutoff: float, tt: L.TravelTimes,
            reference: L.Hypo | None, fault_type: str, site_amp: float, rep: L.Report):
    rep.p(f"Stations: {len(p)}   P-time reference: {L.fmt_time(p.t_ref)}")
    rep.p(f"Relation: {rel.describe()}")
    if rel.note:
        rep.p(f"          {rel.note}")
    rep.p(f"Travel times: {'IASP91 (TauP)' if tt.exact else 'CRUDE FALLBACK — install obspy'}")
    hM, hI = alert.h_mag, alert.h_int
    rep.p(f"Alert: M{alert.magnitude}  intensity hypocentre {hI}  magnitude hypocentre {hM}"
          + ("  (same)" if alert.hypocenter_magnitude is None else ""))

    # ---- independent solution
    rep.h("Independent location from P picks")
    h, mis, reg_km, reg_az = locate(p, tt)
    rep.p(f"  {h}  origin {L.fmt_time(h.time)}  mean |residual| {mis:.2f} s")
    rep.p(f"  near-best region spans {reg_km:.0f} km" + (f", elongated along bearing {reg_az:.0f}°/{reg_az+180:.0f}°" if reg_az is not None else ""))
    gap = L.azimuthal_gap(h.lat, h.lon, p.lat, p.lon)
    dmin = float(L.gc_km(h.lat, h.lon, p.lat, p.lon).min())
    baz = float(np.median(L.bearing_deg(h.lat, h.lon, p.lat, p.lon)))
    rep.data["solution"] = {"hypo": h.to_dict(), "mean_abs_res_s": mis, "region_km": reg_km,
                            "region_bearing": reg_az, "azimuthal_gap": gap, "nearest_station_km": dmin}
    if reference:
        sep = float(L.gc_km(h.lat, h.lon, reference.lat, reference.lon))
        rep.p(f"  reference catalogue: {reference} → independent solution is {sep:.0f} km from it")
        rep.data["solution"]["sep_from_reference_km"] = sep

    Mdat, nM, _ = network_M(p, h, rel, None)
    if Mdat is not None and dmin <= rel.valid_max_km:
        rep.p(f"  magnitude at this location under {rel.name}: M{Mdat:.1f}")
    else:
        Mdat = None
        rep.p(f"  magnitude: not computed — nearest station {dmin:.0f} km is beyond {rel.name}'s calibrated "
              f"{rel.valid_max_km:.0f} km. Use the catalogue/teleseismic Mw for this event.")

    # ---- D1 mislocation
    rep.h("D1 · Location — do the picks support the alert's hypocentres?")
    rows = [("independent", h)] + ([("reference", reference)] if reference else []) + \
           [("alert (intensity)", hI)] + ([("alert (magnitude)", hM)] if alert.hypocenter_magnitude else [])
    base = None
    for name, hh in rows:
        if hh is None:
            continue
        r_free, _ = residuals(p, tt, hh, fixed_t0=False)
        mf = float(np.mean(np.abs(r_free)))
        base = base or mf
        line = f"  {name:20s} {str(hh):30s} mean|res| (free t0) {mf:6.2f} s"
        if hh.time is not None:
            r_fix, _ = residuals(p, tt, hh, fixed_t0=True)
            line += f"   (stated t0) {float(np.mean(np.abs(r_fix))):7.2f} s"
        rep.p(line)
    for name, hh in rows[1:]:
        if hh is None or name == "reference":
            continue
        r_free, _ = residuals(p, tt, hh, fixed_t0=False)
        mf = float(np.mean(np.abs(r_free)))
        sep = float(L.gc_km(h.lat, h.lon, hh.lat, hh.lon))
        if sep <= 30:
            rep.flag("D1", "PASS", f"{name} within {sep:.0f} km of the independent solution.")
        elif mf <= max(1.5 * base, base + 0.5):
            rep.flag("D1", "FLAG", f"{name}: {sep:.0f} km from the independent solution yet fits the picks almost as "
                     f"well ({mf:.2f} s vs {base:.2f} s). The picks cannot tell these apart — geometry, not data, "
                     "decides the location. This is how an EEW locator gets pulled along the back-azimuth.")
        else:
            rep.flag("D1", "FLAG", f"{name}: {sep:.0f} km away; the FULL pick set rejects it ({mf:.2f} s vs {base:.2f} s). "
                     "An alert computed from only the earliest picks can still land there — see below.")

    # EEW decides on the first few picks, not the full set. Where can an early solution land?
    order = np.argsort(p.t)
    alert_h = [(n, x) for n, x in (("alert (intensity)", hI), ("alert (magnitude)", hM if alert.hypocenter_magnitude else None))
               if x and float(L.gc_km(h.lat, h.lon, x.lat, x.lon)) > 30]
    rep.p("")
    rep.p("  Early solutions (first N picks only — the information an EEW system has when it decides):")
    early = []
    for n in (3, 4, 5, 8):
        if n >= len(p):
            break
        mask = np.zeros(len(p), bool); mask[order[:n]] = True
        q = p.subset(mask)
        he, me, rk, raz = locate(q, tt, center=(h.lat, h.lon), half_deg=20.0)
        thr = me + max(0.5, 0.5 * me)
        permits = []
        for name, hh in alert_h:
            rr, _ = residuals(q, tt, hh, fixed_t0=False)
            if float(np.mean(np.abs(rr))) <= thr:
                permits.append(name)
        early.append({"n": n, "hypo": he.to_dict(), "region_km": rk, "permits": permits})
        rep.p(f"    N={n}: best {he}  ({float(L.gc_km(h.lat, h.lon, he.lat, he.lon)):.0f} km from full solution); "
              f"near-best region {rk:.0f} km" + (f"; ALSO FITS {', '.join(permits)}" if permits else ""))
    rep.data["early_solutions"] = early
    if any(e["permits"] for e in early):
        nmax = max(e["n"] for e in early if e["permits"])
        rep.flag("D1", "FLAG", f"With the first {nmax} pick(s) the alert's hypocentre is as compatible with the data as the "
                 "true one: an early-stage EEW solution could legitimately land there. Mislocation is explained by "
                 "geometry; the failure is that nothing downstream caught it.")

    # ---- D4 geometry
    rep.h("D4 · Geometry — was the event inside the network?")
    out = gap > 270 or dmin > 300
    edge = (not out) and gap > 180
    rep.flag("D4", "FLAG" if out else ("INFO" if edge else "PASS"),
             f"azimuthal gap {gap:.0f}°, nearest station {dmin:.0f} km, stations lie toward bearing {baz:.0f}°. "
             + ("OUT-OF-NETWORK: distance along the back-azimuth trades off against origin time; "
                "relations calibrated on in-network events are being extrapolated." if out else
                ("Edge of network (typical offshore event): depth/offshore distance less constrained." if edge
                 else "In-network geometry.")))

    # ---- D2 distance cutoff
    rep.h(f"D2 · Distance cutoff ({cutoff:.0f} km from the hypocentre) — relation-independent part")
    cand = [("independent", h)] + ([("reference", reference)] if reference else []) + \
           [("alert (magnitude)", hM)] + ([("alert (intensity)", hI)] if alert.hypocenter_magnitude else [])
    d2_flag = False
    tab = []
    for name, hh in cand:
        if hh is None:
            continue
        Mc, nc, rmin = network_M(p, hh, rel, cutoff)
        Mn, nn, _ = network_M(p, hh, rel, None)
        tab.append({"hypo": name, "stations_within_cutoff": nc, "nearest_R_km": rmin, "M_with_cutoff": Mc, "M_no_cutoff": Mn})
        rep.p(f"  {name:20s} stations ≤{cutoff:.0f} km: {nc:3d}   nearest {rmin:6.0f} km   "
              f"M(cutoff) {'—  no estimate' if Mc is None else f'{Mc:4.1f}'}   M(no cutoff)* {('—' if Mn is None else f'{Mn:4.1f}')}")
    rep.p("  * relation-dependent; shown to illustrate extrapolation, not as an estimate of the event")
    rep.data["D2"] = tab
    mag_row = next(t for t in tab if t["hypo"] == "alert (magnitude)")
    if mag_row["stations_within_cutoff"] == 0:
        d2_flag = True
        rep.flag("D2", "FLAG", f"No station lies within {cutoff:.0f} km of the alert's magnitude hypocentre "
                 f"(nearest {mag_row['nearest_R_km']:.0f} km). With the cutoff working, NO magnitude could have been "
                 "computed and no warning issued. The published M can only have come from stations the cutoff "
                 "should have excluded — the safeguard was not applied.")
    elif mag_row["M_with_cutoff"] is not None and alert.magnitude is not None and mag_row["M_no_cutoff"] is not None and \
            abs(mag_row["M_no_cutoff"] - alert.magnitude) < abs(mag_row["M_with_cutoff"] - alert.magnitude) - 0.5:
        d2_flag = True
        rep.flag("D2", "FLAG", "Alert M is reproduced only when far stations are included — cutoff likely not applied.")
    else:
        rep.flag("D2", "PASS", "Stations exist inside the cutoff; far-station contamination not required to explain M.")

    # relation-dependent sensitivity
    near = int(np.argmin(L.gc_km(hM.lat, hM.lon, p.lat, p.lon)))
    dnear = float(L.gc_km(hM.lat, hM.lon, p.lat[near], p.lon[near]))
    g = float(rel.dM_dR(dnear, hM.depth_km))
    rep.flag("D2", "INFO", f"[relation-dependent] at {dnear:.0f} km, {rel.name} gives ∂M/∂R = {g*100:.2f} per 100 km "
             f"of distance error. Every 100 km the hypocentre is pushed away adds that much to M; at 700 km "
             f"the same relation gives {float(rel.dM_dR(700, hM.depth_km))*100:.2f}/100 km.")
    if alert.magnitude is not None and mag_row["M_no_cutoff"] is not None:
        rep.flag("D2", "INFO", f"[relation-dependent] {rel.name} at the alert's magnitude hypocentre with no cutoff → "
                 f"M{mag_row['M_no_cutoff']:.1f} (alert said M{alert.magnitude}). Agreement or disagreement here says "
                 "more about the relation's coefficients than about the event — supply the operational relation to test it.")

    # ---- D3 stage consistency
    rep.h("D3 · Stage consistency — same hypocentre for magnitude and intensity?")
    if alert.hypocenter_magnitude is None:
        rep.flag("D3", "N/A", "Alert carries only one hypocentre; if the agency reports that the magnitude stage used "
                 "another, add it as hypocenter_magnitude.")
    else:
        s = float(L.gc_km(hM.lat, hM.lon, hI.lat, hI.lon))
        rep.flag("D3", "FLAG" if s > 30 else "PASS",
                 f"magnitude computed at {hM}, intensity predicted from {hI}: {s:.0f} km apart. "
                 + ("A magnitude inflated by a far hypocentre was then applied to a near one — the errors compound."
                    if s > 30 else ""))

    # ---- D5 magnitude plausibility
    rep.h("D5 · Magnitude plausibility")
    if alert.magnitude is not None:
        if alert.magnitude >= 9.6:
            rep.flag("D5", "FLAG", f"M{alert.magnitude} > Mw 9.5 (Chile 1960, largest instrumentally recorded). "
                     "JMA introduced a seismologically plausible M ceiling after 2016-08-01; ask what its value is "
                     "and whether it applies on this code path.")
        Mr = _ref_mag(reference)
        if Mr is not None:
            # Differential: the same relation evaluated for the alert's event and the real one, at the same
            # stations. Distance terms largely cancel, so this is far less sensitive to the coefficients.
            pa = rel.predict_logA(alert.magnitude, L.gc_km(hI.lat, hI.lon, p.lat, p.lon), hI.depth_km)
            pr = rel.predict_logA(Mr, L.gc_km(reference.lat, reference.lon, p.lat, p.lon), reference.depth_km)
            ex = float(np.median(pa - pr))
            rep.flag("D5", "FLAG" if ex > 1 else "PASS",
                     f"if the alert were right, the network would have recorded ≈10^{ex:.1f} = {10**ex:,.0f}× the ground "
                     f"motion the real M{Mr} event produces at the same stations (differential, {rel.name}).")
        else:
            rep.flag("D5", "N/A", "No reference magnitude; amplitude check against raw recordings would depend "
                     "entirely on the relation's coefficients at these distances, so it is not attempted.")

    # ---- D6 intensity
    rep.h("D6 · Intensity — predicted vs observed shindo (Si & Midorikawa 1999)")
    ok = np.isfinite(p.obs_I)
    faithful = False
    epi_I = L.gc_km(hI.lat, hI.lon, p.lat, p.lon)
    I_pt = L.pgv_to_shindo(L.predict_pgv(alert.magnitude, hI.depth_km, epi_I, fault_type, site_amp, finite=False))
    I_alert = L.pgv_to_shindo(L.predict_pgv(alert.magnitude, hI.depth_km, epi_I, fault_type, site_amp, finite=True))
    Lf = L.fault_length_km(alert.magnitude)
    rep.p(f"  M{alert.magnitude} at {hI}, point source:  max {L.shindo_class(float(I_pt.max()))} (I={float(I_pt.max()):.1f}); "
          f"{int((I_pt >= 4.5).sum())}/{len(p)} stations ≥ 5-")
    rep.p(f"  M{alert.magnitude} at {hI}, finite (L≈{Lf:,.0f} km): max {L.shindo_class(float(I_alert.max()))} "
          f"(I={float(I_alert.max()):.1f}) at {p.sta[int(np.argmax(I_alert))]}; {int((I_alert >= 4.5).sum())}/{len(p)} stations ≥ 5-")
    rep.p("  (compare with the agency's published predicted ranges; a match means the intensity stage did its job on bad inputs)")
    Mref = _ref_mag(reference)
    Mtrue = Mref if Mref is not None else Mdat
    hT = reference or h
    if Mtrue is not None:
        I_true = L.pgv_to_shindo(L.predict_pgv(Mtrue, hT.depth_km, L.gc_km(hT.lat, hT.lon, p.lat, p.lon), fault_type, site_amp, finite=True))
        rep.p(f"  using M{Mtrue:.1f} at {hT}: predicted max {L.shindo_class(float(I_true.max()))} (I={float(I_true.max()):.1f})")
    if ok.any():
        over = float(np.median(I_alert[ok] - p.obs_I[ok]))
        rep.flag("D6", "FLAG" if over > 1.5 else "PASS",
                 f"prediction from the alert's inputs minus observed intensity: median {over:+.1f} units "
                 f"(observed max {L.shindo_class(float(np.nanmax(p.obs_I)))}).")
        if Mtrue is not None:
            fit = float(np.median(I_true[ok] - p.obs_I[ok]))
            faithful = abs(fit) <= 1.5
            if over > 1.5:
                rep.flag("D6", "INFO" if faithful else "FLAG",
                         f"the same intensity model fed the correct M/location misses observed by {fit:+.1f} — "
                         + ("the intensity stage is faithful; the fault is upstream in M/location." if faithful
                            else "the intensity stage itself is also off."))
    else:
        rep.flag("D6", "N/A", "No observed_shindo column; predicted values above only.")
    if float(np.min(epi_I)) > 300:
        rep.flag("D6", "INFO", "All stations are beyond Si & Midorikawa's ~300 km calibration range; predictions are extrapolations.")

    # ---- D7 leave-one-out
    rep.h("D7 · Single-sensor control (leave-one-out)")
    M_all, _, _ = network_M(p, hM, rel, None)
    worst = []
    for i in range(len(p)):
        mask = np.ones(len(p), bool); mask[i] = False
        q = p.subset(mask)
        Mi, _, _ = network_M(q, hM, rel, None)
        hi, _, _, _ = locate(q, tt, center=(h.lat, h.lon), half_deg=3.0)
        dM = (Mi - M_all) if (Mi is not None and M_all is not None) else 0.0
        dx = float(L.gc_km(h.lat, h.lon, hi.lat, hi.lon))
        worst.append((abs(dM), dx, p.sta[i], dM))
    worst.sort(key=lambda w: (w[0], w[1]), reverse=True)
    top = worst[0]
    dx_thr = max(50.0, 0.3 * (reg_km if np.isfinite(reg_km) else 0))
    m_dom = top[0] > 0.5
    x_dom = max(w[1] for w in worst) > dx_thr
    xs = max(worst, key=lambda w: w[1])
    if m_dom:
        rep.flag("D7", "FLAG", f"{top[2]} controls the magnitude: removing it shifts M by {top[3]:+.2f}. "
                 "Inspect that sensor's data and power/telemetry health (cf. JMA 2016-08-01 single-station fault).")
    if x_dom:
        rep.flag("D7", "FLAG", f"{xs[2]} has outsized leverage on the location: removing it moves the solution {xs[1]:.0f} km "
                 f"(> {dx_thr:.0f} km). Fragile geometry — that station's timing must be right for the location to be.")
    if not (m_dom or x_dom):
        rep.flag("D7", "PASS", f"no single station moves M by >0.5 or the location by >{dx_thr:.0f} km "
                 f"(largest: {xs[2]}, {xs[1]:.0f} km).")
    rep.data["D7"] = [{"station": s, "dM": dm, "dx_km": dx} for _, dx, s, dm in worst[:5]]

    # ---- D8 outliers
    rep.h("D8 · Outlier picks")
    r, _ = residuals(p, tt, h, fixed_t0=True)
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    thr = max(3 * mad, 2.0)
    bad = [(p.sta[i], float(r[i])) for i in np.nonzero(np.abs(r) > thr)[0]]
    rep.flag("D8", "FLAG" if bad else "PASS",
             (f"{len(bad)} pick(s) beyond ±{thr:.1f} s: " + ", ".join(f"{s} {x:+.1f}s" for s, x in bad[:8]) +
              " — candidate second event or timestamp fault (cf. 2006-09-26, 2013-08-08).") if bad
             else f"all picks within ±{thr:.1f} s of the independent solution.")

    # ---- summary
    rep.h("Summary — failure classes supported by the data")
    codes = {c["code"] for c in rep.data.get("checks", []) if c["status"] == "FLAG"}
    lines = []
    if "D4" in codes: lines.append("Out-of-network event: location under-determined along the back-azimuth (relation-independent).")
    if d2_flag: lines.append(f"Magnitude used stations beyond the {cutoff:.0f} km safeguard (relation-independent).")
    if "D3" in codes: lines.append("Magnitude and intensity stages used different hypocentres (relation-independent).")
    if "D6" in codes and faithful: lines.append("Intensity over-prediction follows from the M/location inputs; the intensity model itself behaves.")
    elif "D6" in codes: lines.append("Intensity over-predicted, and the intensity model also misfits with correct inputs.")
    if "D7" in codes: lines.append("One station dominates M or location — inspect its data, clock and health.")
    if "D8" in codes: lines.append("Outlier picks — check for a second event or clock/timestamp faults.")
    flagged = bool(lines)
    if not lines: lines.append("No failure class flagged.")
    for s in lines: rep.p(f"  • {s}")
    if flagged:
        rep.p("")
        rep.p("  Not determinable from station data alone: whether a safeguard failed through a code regression,")
        rep.p("  a configuration change, or an unauthorised modification. That needs the agency's change history.")
    rep.data["summary"] = lines


# ============================================================== waveforms
def picks_from_waveforms(data_dir, coords_csv, inventory, t0, window, amp_window, out_csv):
    from obspy import read, read_inventory, UTCDateTime
    from obspy.signal.trigger import recursive_sta_lta, trigger_onset

    st = read(str(Path(data_dir) / "*"))
    st.merge(fill_value=0)
    inv = read_inventory(inventory) if inventory else None
    coords = {}
    if coords_csv:
        with open(coords_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                coords[r["station"]] = (float(r["lat"]), float(r["lon"]), float(r.get("counts_per_um") or 1.0))
    T0 = UTCDateTime(t0.isoformat()) if t0 else min(tr.stats.starttime for tr in st)
    rows = []
    for sta in sorted({tr.stats.station for tr in st}):
        s = st.select(station=sta).copy()
        s.trim(T0, T0 + window)
        z = s.select(component="Z") or s
        tr = z[0].copy(); tr.detrend("demean"); tr.taper(0.02)
        tr.filter("bandpass", freqmin=0.5, freqmax=8.0)
        sr = tr.stats.sampling_rate
        cft = recursive_sta_lta(tr.data.astype(float), int(1 * sr), int(20 * sr))
        on = trigger_onset(cft, 3.5, 1.0)
        on = [o for o in on if o[0] > int(1.5 * 20 * sr)]       # ignore the LTA warm-up transient
        if len(on) == 0:
            print(f"[waveforms] {sta}: no trigger", file=sys.stderr)
            continue
        i0 = int(on[0][0])
        a, b = max(i0 - int(5 * sr), 1), min(i0 + int(3 * sr), len(tr.data) - 1)   # AIC refinement
        x = tr.data[a:b].astype(float)
        k = np.arange(1, len(x) - 1)
        aic = np.array([kk * np.log(np.var(x[:kk]) + 1e-20) + (len(x) - kk - 1) * np.log(np.var(x[kk + 1:]) + 1e-20) for kk in k])
        ip = a + int(k[np.argmin(aic)])
        tP = tr.stats.starttime + ip / sr
        # amplitude: max horizontal (or Z) displacement in µm within amp_window after P
        disp = s.copy()
        if inv is not None:
            disp.remove_response(inventory=inv, output="DISP", pre_filt=(0.05, 0.1, 10, 15))
            scale = 1e6
        else:
            # No responses: data are taken to be displacement, scaled by counts_per_um from --coords.
            disp.detrend("demean")
            scale = 1.0 / coords.get(sta, (0, 0, 1.0))[2]
        hz = disp.select(component="[NE12]") or disp
        amp = max(float(np.max(np.abs(t.slice(tP, tP + amp_window).data))) for t in hz) * scale
        if inv is not None:
            c = inv.get_coordinates(z[0].id, tP)
            lat, lon = c["latitude"], c["longitude"]
        else:
            lat, lon = coords[sta][:2]
        rows.append({"station": sta, "lat": lat, "lon": lon, "p_time": tP.datetime.replace(tzinfo=timezone.utc).isoformat(),
                     "amp_um": f"{amp:.4g}"})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["station", "lat", "lon", "p_time", "amp_um"])
        w.writeheader(); w.writerows(rows)
    print(f"[waveforms] {len(rows)} stations picked → {out_csv}", file=sys.stderr)
    return out_csv


def fetch(provider, ids, t0, before, after, out):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    c = Client(provider)
    T = UTCDateTime(t0)
    Path(out).mkdir(parents=True, exist_ok=True)
    bulk = []
    for sid in ids:
        net, sta = sid.split(".")[:2]
        bulk.append((net, sta, "*", "BH?,HH?", T - before, T + after))
    st = c.get_waveforms_bulk(bulk)
    inv = c.get_stations_bulk(bulk, level="response")
    st.write(str(Path(out) / "waveforms.mseed"), format="MSEED")
    inv.write(str(Path(out) / "stations.xml"), format="STATIONXML")
    print(f"[fetch] {len(st)} traces, {len(inv.get_contents()['stations'])} stations → {out}", file=sys.stderr)


# ===================================================================== main
def _common(ap):
    ap.add_argument("--alert", required=True, help="alert JSON or JMA XML")
    ap.add_argument("--reference", help="USGS GeoJSON or hypo JSON {lat,lon,depth_km,time,mag}")
    ap.add_argument("--relation", default="eew_illustrative", choices=sorted(L.RELATIONS))
    ap.add_argument("--relation-coeffs", help="a,b,c,d for log10A = aM + b·log10R + c·R + d (A µm, R km)")
    ap.add_argument("--cutoff-km", type=float, default=700.0)
    ap.add_argument("--fault-type", default="intraplate", choices=sorted(L.SI_MIDORIKAWA_D))
    ap.add_argument("--site-amp", type=float, default=1.4, help="surface/engineering-bedrock PGV amplification")
    ap.add_argument("--json", help="write machine-readable report")


def load_reference(path):
    if not path:
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "features" in d:
        f = max(d["features"], key=lambda x: x["properties"].get("mag") or 0)
        lon, lat, dep = f["geometry"]["coordinates"][:3]
        t = datetime.fromtimestamp(f["properties"]["time"] / 1000, tz=timezone.utc)
        return L.Hypo(lat, lon, dep, t, f"catalog M{f['properties']['mag']}")
    h = L.Hypo.from_dict(d, "reference")
    if h and d.get("mag") is not None:
        h.label = f"reference M{d['mag']}"
    return h


def run_analysis(picks_csv, a):
    p = Picks.from_csv(picks_csv)
    alert = L.load_alert(a.alert)
    rel = L.relation_from_args(a.relation, a.relation_coeffs)
    tt = L.TravelTimes()
    rep = L.Report()
    analyse(p, alert, rel, a.cutoff_km, tt, load_reference(a.reference), a.fault_type, a.site_amp, rep)
    sys.stdout.write(rep.text())
    if a.json:
        L.write_json(a.json, rep.data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("picks", help="analyse a picks CSV"); _common(s)
    s.add_argument("--picks", required=True)

    s = sub.add_parser("waveforms", help="pick + measure from waveforms, then analyse"); _common(s)
    s.add_argument("--data", required=True, help="directory of miniSEED/SAC")
    s.add_argument("--inventory", help="StationXML with responses (preferred)")
    s.add_argument("--coords", help="CSV station,lat,lon[,counts_per_um] if no StationXML")
    s.add_argument("--start", help="analysis window start (ISO)")
    s.add_argument("--window", type=float, default=600)
    s.add_argument("--amp-window", type=float, default=30, help="seconds after P for peak displacement")
    s.add_argument("--picks-out", default="picks_from_waveforms.csv")

    s = sub.add_parser("fetch", help="download from an FDSN data centre")
    s.add_argument("--provider", default="IRIS")
    s.add_argument("--stations", required=True, help="comma list NET.STA, e.g. II.ERM,IU.MAJO,IU.PET,IU.YSS,G.INU")
    s.add_argument("--origin", required=True, help="origin time (ISO, UTC)")
    s.add_argument("--before", type=float, default=60); s.add_argument("--after", type=float, default=600)
    s.add_argument("--out", default="fdsn_data")

    s = sub.add_parser("jma-history", help="parse a saved JMA EEW history page")
    s.add_argument("page", help="saved HTML/text of the 緊急地震速報の内容 page")
    s.add_argument("--serial", type=int, help="write this serial as an alert JSON")
    s.add_argument("--out", help="alert JSON path (with --serial)")

    a = ap.parse_args(argv)
    if a.cmd == "picks":
        run_analysis(a.picks, a)
    elif a.cmd == "waveforms":
        out = picks_from_waveforms(a.data, a.coords, a.inventory, L.parse_time(a.start), a.window, a.amp_window, a.picks_out)
        run_analysis(out, a)
    elif a.cmd == "fetch":
        fetch(a.provider, a.stations.split(","), a.origin, a.before, a.after, a.out)
    elif a.cmd == "jma-history":
        h = L.parse_jma_history(Path(a.page).read_text(encoding="utf-8", errors="replace"))
        print(f"wave detected: {L.fmt_time(h['detected_at'])}")
        print(f"{'serial':>6} {'issued (JST)':>24} {'+s':>7} {'lat':>6} {'lon':>7} {'depth':>6} {'M':>4}")
        prev = None
        for r in h["serials"]:
            jump = "" if prev is None else f"  ΔM {r['magnitude'] - prev['magnitude']:+.1f}, moved {float(L.gc_km(prev['lat'], prev['lon'], r['lat'], r['lon'])):.0f} km"
            print(f"{r['serial']:6d} {L.fmt_time(r['issued_at']):>24} {r['elapsed_s']:7.1f} {r['lat']:6.1f} {r['lon']:7.1f} {r['depth_km']:5.0f}k {r['magnitude']:4.1f}{jump}")
            prev = r
        if a.serial is not None:
            r = next((x for x in h["serials"] if x["serial"] == a.serial), None)
            if r is None:
                raise SystemExit(f"serial {a.serial} not found")
            al = {"source": "JMA", "serial": r["serial"], "issued_at": r["issued_at"].isoformat(),
                  "detected_at": h["detected_at"].isoformat() if h["detected_at"] else None,
                  "magnitude": r["magnitude"], "hypocenter": {"lat": r["lat"], "lon": r["lon"], "depth_km": r["depth_km"]}}
            Path(a.out or f"alert_serial{a.serial}.json").write_text(json.dumps(al, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"→ {a.out or f'alert_serial{a.serial}.json'}")


if __name__ == "__main__":
    main()
