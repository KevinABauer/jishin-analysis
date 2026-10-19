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
def _log_amp(v, station):
    """log10 of a positive amplitude; zero, negative or unparseable → NaN (excluded from magnitudes)."""
    if v in (None, ""):
        return np.nan
    try:
        a = float(v)
    except ValueError:
        a = float("nan")
    if not (math.isfinite(a) and a > 0):
        print(f"[picks] {station}: amplitude {v!r} is not a positive number — ignored for magnitude", file=sys.stderr)
        return np.nan
    return math.log10(a)


class Picks:
    def __init__(self, rows):
        self.sta = [r["station"] for r in rows]
        self.lat = np.array([float(r["lat"]) for r in rows])
        self.lon = np.array([float(r["lon"]) for r in rows])
        t = [L.parse_time(r["p_time"]) for r in rows]
        self.t_ref = min(t)
        self.t = np.array([(x - self.t_ref).total_seconds() for x in t])
        self.logA = np.array([_log_amp(r.get("amp_um"), r["station"]) for r in rows])
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
        with open(path, newline="", encoding="utf-8-sig") as f:
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
D8_CAP_S = 10.0          # outlier threshold never grows beyond this, however scattered the picks are


def analyse(p: Picks, alert: L.Alert, rel: L.AmpRelation, cutoff: float, tt: L.TravelTimes,
            reference: L.Hypo | None, fault_type: str, site_amp: float, rep: L.Report):
    hM, hI, Ma = alert.h_mag, alert.h_int, alert.magnitude
    rep.p(f"Stations: {len(p)}   P-time reference: {L.fmt_time(p.t_ref)}")
    rep.p(f"Relation: {rel.describe()}")
    if rel.note:
        rep.p(f"          {rel.note}")
    rep.p(f"Travel times: {'IASP91 (TauP)' if tt.exact else 'CRUDE FALLBACK — install obspy'}")
    rep.p(f"Alert: M{Ma if Ma is not None else '? (not given)'}  intensity hypocentre {hI or '— (not given)'}  "
          f"magnitude hypocentre {hM or '—'}" + ("  (same)" if hI and alert.hypocenter_magnitude is None else ""))
    for n in alert.notes:
        rep.p(f"  note: {n}")
    classes = []

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

    Mdat, _, _ = network_M(p, h, rel, None)
    if Mdat is not None and dmin <= rel.valid_max_km:
        rep.p(f"  magnitude at this location under {rel.name}: M{Mdat:.1f}")
    else:
        Mdat = None
        rep.p(f"  magnitude: not computed — nearest station {dmin:.0f} km is beyond {rel.name}'s calibrated "
              f"{rel.valid_max_km:.0f} km, or no usable amplitudes. Use the catalogue/teleseismic Mw for this event.")

    # ---- D1 mislocation
    rep.h("D1 · Location — do the picks support the alert's hypocentres?")
    rows = [("independent", h)] + ([("reference", reference)] if reference else []) + \
           ([("alert (intensity)", hI)] if hI else []) + ([("alert (magnitude)", hM)] if alert.hypocenter_magnitude else [])
    base = None
    for name, hh in rows:
        r_free, _ = residuals(p, tt, hh, fixed_t0=False)
        mf = float(np.mean(np.abs(r_free)))
        base = base or mf
        line = f"  {name:20s} {str(hh):30s} mean|res| (free t0) {mf:6.2f} s"
        if hh.time is not None:
            r_fix, _ = residuals(p, tt, hh, fixed_t0=True)
            line += f"   (stated t0) {float(np.mean(np.abs(r_fix))):7.2f} s"
        rep.p(line)
    if not hI:
        rep.flag("D1", "N/A", "Alert carries no hypocentre — nothing to test against the picks.")
    for name, hh in rows[1:]:
        if name == "reference":
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
        he, me, rk, _ = locate(q, tt, center=(h.lat, h.lon), half_deg=20.0)
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
        classes.append("EARLY_SOLUTION_AMBIGUOUS")
        rep.flag("D1", "FLAG", f"With the first {nmax} pick(s) the alert's hypocentre is as compatible with the data as the "
                 "true one: an early-stage EEW solution could legitimately land there. Mislocation is explained by "
                 "geometry; the failure is that nothing downstream caught it.")

    # ---- D4 geometry
    rep.h("D4 · Geometry — was the event inside the network?")
    out = gap > 270 or dmin > 300
    edge = (not out) and gap > 180
    if out: classes.append("OUT_OF_NETWORK")
    rep.flag("D4", "FLAG" if out else ("INFO" if edge else "PASS"),
             f"azimuthal gap {gap:.0f}°, nearest station {dmin:.0f} km, stations lie toward bearing {baz:.0f}°. "
             + ("OUT-OF-NETWORK: distance along the back-azimuth trades off against origin time; "
                "relations calibrated on in-network events are being extrapolated." if out else
                ("Edge of network (typical offshore event): depth/offshore distance less constrained." if edge
                 else "In-network geometry.")))

    # ---- D2 distance cutoff
    rep.h(f"D2 · Distance cutoff ({cutoff:.0f} km from the hypocentre) — relation-independent part")
    cand = [("independent", h)] + ([("reference", reference)] if reference else []) + \
           ([("alert (magnitude)", hM)] if hM else []) + ([("alert (intensity)", hI)] if alert.hypocenter_magnitude else [])
    tab = []
    for name, hh in cand:
        Mc, nc, rmin = network_M(p, hh, rel, cutoff)
        Mn, _, _ = network_M(p, hh, rel, None)
        tab.append({"hypo": name, "stations_within_cutoff": nc, "nearest_R_km": rmin, "M_with_cutoff": Mc, "M_no_cutoff": Mn})
        rep.p(f"  {name:20s} stations ≤{cutoff:.0f} km: {nc:3d}   nearest {rmin:6.0f} km   "
              f"M(cutoff) {'—  no estimate' if Mc is None else f'{Mc:4.1f}'}   M(no cutoff)* {('—' if Mn is None else f'{Mn:4.1f}')}")
    rep.p("  * relation-dependent; shown to illustrate extrapolation, not as an estimate of the event")
    rep.data["D2"] = tab
    mag_row = next((t for t in tab if t["hypo"] == "alert (magnitude)"), None)
    if mag_row is None:
        rep.flag("D2", "N/A", "Alert carries no hypocentre, so which stations its magnitude could have used is unknown.")
    elif Ma is None:
        rep.flag("D2", "N/A", "Alert carries no magnitude to account for.")
    elif mag_row["stations_within_cutoff"] == 0:
        classes.append("CUTOFF_NOT_APPLIED")
        rep.flag("D2", "FLAG", f"No station lies within {cutoff:.0f} km of the alert's magnitude hypocentre "
                 f"(nearest {mag_row['nearest_R_km']:.0f} km). With the cutoff working, NO magnitude could have been "
                 "computed and no warning issued. The published M can only have come from stations the cutoff "
                 "should have excluded — the safeguard was not applied.")
    elif mag_row["M_with_cutoff"] is not None and mag_row["M_no_cutoff"] is not None and \
            abs(mag_row["M_no_cutoff"] - Ma) < abs(mag_row["M_with_cutoff"] - Ma) - 0.5:
        classes.append("CUTOFF_NOT_APPLIED")
        rep.flag("D2", "FLAG", "Alert M is reproduced only when far stations are included — cutoff likely not applied.")
    else:
        rep.flag("D2", "PASS", "Stations exist inside the cutoff; far-station contamination not required to explain M.")
    if hM:
        dnear = float(L.gc_km(hM.lat, hM.lon, p.lat, p.lon).min())
        g = float(rel.dM_dR(dnear, hM.depth_km))
        rep.flag("D2", "INFO", f"[relation-dependent] at {dnear:.0f} km, {rel.name} gives ∂M/∂R = {g*100:.2f} per 100 km "
                 f"of distance error. Every 100 km the hypocentre is pushed away adds that much to M; at 700 km "
                 f"the same relation gives {float(rel.dM_dR(700, hM.depth_km))*100:.2f}/100 km.")
        if Ma is not None and mag_row["M_no_cutoff"] is not None:
            rep.flag("D2", "INFO", f"[relation-dependent] {rel.name} at the alert's magnitude hypocentre with no cutoff → "
                     f"M{mag_row['M_no_cutoff']:.1f} (alert said M{Ma}). Agreement or disagreement here says "
                     "more about the relation's coefficients than about the event — supply the operational relation to test it.")

    # ---- D3 stage consistency
    rep.h("D3 · Stage consistency — same hypocentre for magnitude and intensity?")
    if not hI or alert.hypocenter_magnitude is None:
        rep.flag("D3", "N/A", "Alert carries at most one hypocentre; if the agency reports that the magnitude stage used "
                 "another, add it as hypocenter_magnitude.")
    else:
        sdist = float(L.gc_km(hM.lat, hM.lon, hI.lat, hI.lon))
        if sdist > 30: classes.append("STAGE_HYPOCENTRE_MISMATCH")
        rep.flag("D3", "FLAG" if sdist > 30 else "PASS",
                 f"magnitude computed at {hM}, intensity predicted from {hI}: {sdist:.0f} km apart. "
                 + ("A magnitude inflated by a far hypocentre was then applied to a near one — the errors compound."
                    if sdist > 30 else ""))

    # ---- D5 magnitude plausibility
    rep.h("D5 · Magnitude plausibility")
    Mr = _ref_mag(reference)
    if Ma is None:
        rep.flag("D5", "N/A", "Alert carries no magnitude.")
    else:
        if Ma >= 9.6:
            classes.append("IMPLAUSIBLE_MAGNITUDE")
            rep.flag("D5", "FLAG", f"M{Ma} > Mw 9.5 (Chile 1960, largest instrumentally recorded). "
                     "JMA introduced a seismologically plausible M ceiling after 2016-08-01; ask what its value is "
                     "and whether it applies on this code path.")
        if Mr is not None and hI:
            # Differential: the same relation for the alert's event and the real one, at the same stations.
            pa = rel.predict_logA(Ma, L.gc_km(hI.lat, hI.lon, p.lat, p.lon), hI.depth_km)
            pr = rel.predict_logA(Mr, L.gc_km(reference.lat, reference.lon, p.lat, p.lon), reference.depth_km)
            ex = float(np.median(pa - pr))
            if ex > 1: classes.append("MAGNITUDE_MISMATCH")
            rep.flag("D5", "FLAG" if ex > 1 else "PASS",
                     f"if the alert were right, the network would have recorded ≈10^{ex:.1f} = {10**ex:,.0f}× the ground "
                     f"motion the real M{Mr} event produces at the same stations (differential, {rel.name}).")
        elif Mr is not None:
            dM = Ma - Mr
            if abs(dM) >= 1: classes.append("MAGNITUDE_MISMATCH")
            rep.flag("D5", "FLAG" if abs(dM) >= 1 else "PASS",
                     f"alert M{Ma} vs reference M{Mr}: ΔM {dM:+.1f} (≈{10 ** (1.5 * dM):,.0f}× the energy).")
        else:
            rep.flag("D5", "N/A", "No reference magnitude; amplitude check against raw recordings would depend "
                     "entirely on the relation's coefficients at these distances, so it is not attempted.")

    # ---- D6 intensity
    rep.h("D6 · Intensity — predicted vs observed shindo (Si & Midorikawa 1999)")
    faithful = False
    ok = np.isfinite(p.obs_I)
    if Ma is None or not hI:
        rep.flag("D6", "N/A", "Needs both an alert magnitude and an alert hypocentre.")
    else:
        epi_I = L.gc_km(hI.lat, hI.lon, p.lat, p.lon)
        I_pt = L.pgv_to_shindo(L.predict_pgv(Ma, hI.depth_km, epi_I, fault_type, site_amp, finite=False))
        I_alert = L.pgv_to_shindo(L.predict_pgv(Ma, hI.depth_km, epi_I, fault_type, site_amp, finite=True))
        Lf = L.fault_length_km(Ma)
        rep.p(f"  M{Ma} at {hI}, point source:  max {L.shindo_class(float(I_pt.max()))} (I={float(I_pt.max()):.1f}); "
              f"{int((I_pt >= 4.5).sum())}/{len(p)} stations ≥ 5-")
        rep.p(f"  M{Ma} at {hI}, finite (L≈{Lf:,.0f} km): max {L.shindo_class(float(I_alert.max()))} "
              f"(I={float(I_alert.max()):.1f}) at {p.sta[int(np.argmax(I_alert))]}; {int((I_alert >= 4.5).sum())}/{len(p)} stations ≥ 5-")
        rep.p("  (compare with the agency's published predicted ranges; a match means the intensity stage did its job on bad inputs)")
        Mtrue = Mr if Mr is not None else Mdat
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
                    classes.append("INTENSITY_FROM_BAD_INPUTS" if faithful else "INTENSITY_MODEL_MISFIT")
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
    hMx = hM or h
    M_all, _, _ = network_M(p, hMx, rel, None)
    worst = []
    for i in range(len(p)):
        mask = np.ones(len(p), bool); mask[i] = False
        q = p.subset(mask)
        Mi, _, _ = network_M(q, hMx, rel, None)
        hi, _, _, _ = locate(q, tt, center=(h.lat, h.lon), half_deg=3.0)
        dM = (Mi - M_all) if (Mi is not None and M_all is not None) else 0.0
        worst.append((abs(dM), float(L.gc_km(h.lat, h.lon, hi.lat, hi.lon)), p.sta[i], dM))
    worst.sort(key=lambda w: (w[0], w[1]), reverse=True)
    top = worst[0]
    dx_thr = max(50.0, 0.3 * (reg_km if np.isfinite(reg_km) else 0))
    xs = max(worst, key=lambda w: w[1])
    m_dom, x_dom = top[0] > 0.5, xs[1] > dx_thr
    if m_dom or x_dom: classes.append("SINGLE_STATION_CONTROL")
    if m_dom:
        rep.flag("D7", "FLAG", f"{top[2]} controls the magnitude: removing it shifts M by {top[3]:+.2f}. "
                 "Inspect that sensor's data and power/telemetry health (cf. JMA 2016-08-01 single-station fault).")
    if x_dom:
        rep.flag("D7", "FLAG", f"{xs[2]} has outsized leverage on the location: removing it moves the solution {xs[1]:.0f} km "
                 f"(> {dx_thr:.0f} km). Fragile geometry — that station's timing must be right for the location to be.")
    if not (m_dom or x_dom):
        rep.flag("D7", "PASS", f"no single station moves M by >0.5 or the location by >{dx_thr:.0f} km "
                 f"(largest: {xs[2]}, {xs[1]:.0f} km).")
    rep.data["D7"] = [{"station": s_, "dM": dm, "dx_km": dx} for _, dx, s_, dm in worst[:5]]

    # ---- D8 outliers
    rep.h("D8 · Outlier picks")
    r, _ = residuals(p, tt, h, fixed_t0=True)
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    thr = min(max(3 * mad, 2.0), D8_CAP_S)
    bad = [(p.sta[i], float(r[i])) for i in np.nonzero(np.abs(r) > thr)[0]]
    rep.data["D8"] = {"threshold_s": thr, "outliers_vs_solution": bad}
    rep.flag("D8", "FLAG" if bad else "PASS",
             (f"{len(bad)} pick(s) beyond ±{thr:.1f} s of the independent solution: " + ", ".join(f"{s_} {x:+.1f}s" for s_, x in bad[:8]) +
              " — candidate second event, mis-pick, or timestamp fault (cf. 2006-09-26, 2013-08-08).") if bad
             else f"all picks within ±{thr:.1f} s of the independent solution.")
    # Bad picks can drag the independent solution with them; a catalogue origin is an outside yardstick.
    if reference is not None and reference.time is not None:
        rr, _ = residuals(p, tt, reference, fixed_t0=True)
        bad_ref = [(p.sta[i], float(rr[i])) for i in np.nonzero(np.abs(rr) > D8_CAP_S)[0]]
        rep.data["D8"]["outliers_vs_reference"] = bad_ref
        rep.flag("D8", "FLAG" if bad_ref else "PASS",
                 (f"{len(bad_ref)}/{len(p)} pick(s) more than ±{D8_CAP_S:.0f} s from the P time predicted by the reference "
                  "origin: " + ", ".join(f"{s_} {x:+.1f}s" for s_, x in bad_ref[:8]) +
                  ". Re-pick these before trusting D1/D4/D7.") if bad_ref
                 else f"all picks within ±{D8_CAP_S:.0f} s of the P times predicted by the reference origin.")
        bad = bad + bad_ref
    if bad: classes.append("OUTLIER_PICKS")

    # ---- summary
    rep.h("Summary — failure classes supported by the data")
    text = {
        "OUT_OF_NETWORK": "Out-of-network event: location under-determined along the back-azimuth (relation-independent).",
        "EARLY_SOLUTION_AMBIGUOUS": "With only the first few picks, the alert's hypocentre fits as well as the true one (relation-independent).",
        "CUTOFF_NOT_APPLIED": f"Magnitude used stations beyond the {cutoff:.0f} km safeguard (relation-independent).",
        "STAGE_HYPOCENTRE_MISMATCH": "Magnitude and intensity stages used different hypocentres (relation-independent).",
        "IMPLAUSIBLE_MAGNITUDE": "Magnitude above any instrumentally recorded event — a plausibility ceiling should have stopped it.",
        "MAGNITUDE_MISMATCH": "Magnitude far from the reference event's (≥1 unit).",
        "INTENSITY_FROM_BAD_INPUTS": "Intensity over-prediction follows from the M/location inputs; the intensity model itself behaves.",
        "INTENSITY_MODEL_MISFIT": "Intensity over-predicted, and the intensity model also misfits with correct inputs.",
        "SINGLE_STATION_CONTROL": "One station dominates M or location — inspect its data, clock and health.",
        "OUTLIER_PICKS": "Outlier picks — re-pick, or check for a second event or clock/timestamp faults.",
    }
    shown = [c for c in text if c in classes]
    for c in shown: rep.p(f"  • {text[c]}")
    if not shown:
        rep.p("  • No failure class flagged.")
    else:
        rep.p("")
        rep.p("  Not determinable from station data alone: whether a safeguard failed through a code regression,")
        rep.p("  a configuration change, or an unauthorised modification. That needs the agency's change history.")
    rep.data["failure_classes"] = shown
    rep.data["summary"] = [text[c] for c in shown]


# ============================================================== waveforms
WAVEFORM_SUFFIXES = {".mseed", ".miniseed", ".msd", ".ms", ".sac"}
LOC_PRIORITY = ["00", "", "10", "01", "20", "02"]
BAND_PRIORITY = ["BH", "HH", "EH", "SH", "BN", "HN", "LH"]


def waveform_files(data):
    """Waveform files only — never StationXML, CSV or anything else that happens to share the folder."""
    p = Path(data)
    if p.is_file():
        return [p], []
    files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in WAVEFORM_SUFFIXES)
    skipped = sorted(f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() not in WAVEFORM_SUFFIXES)
    return files, skipped


def choose_channels(traces):
    """
    One sensor per station: the best location code, then the best band, that has a vertical component.
    Returns (vertical trace, every trace of that location+band) or (None, None).
    """
    combos = {}
    for tr in traces:
        combos.setdefault((tr.stats.location, tr.stats.channel[:2]), []).append(tr)
    def rank(k):
        loc, band = k
        return (LOC_PRIORITY.index(loc) if loc in LOC_PRIORITY else len(LOC_PRIORITY),
                BAND_PRIORITY.index(band) if band in BAND_PRIORITY else len(BAND_PRIORITY))
    for key in sorted(combos, key=rank):
        z = [tr for tr in combos[key] if tr.stats.channel[-1] in "Z3"]
        if z:
            return z[0], combos[key]
    return None, None


def pick_p(tr, t_exp=None, tol=30.0, sta_s=1.0, lta_s=20.0):
    """
    P onset on one vertical trace. STA/LTA triggers inside the LTA warm-up are ignored. With an expected
    arrival time, only triggers within ±tol s of it are considered (the earliest wins); if none fire, an
    AIC pick inside the window is accepted when the onset is clearly above the noise. Returns
    (UTCDateTime or None, method, note).
    """
    from obspy.signal.trigger import recursive_sta_lta, trigger_onset
    tr = tr.copy(); tr.detrend("demean"); tr.taper(0.02)
    sr = tr.stats.sampling_rate
    tr.filter("bandpass", freqmin=0.5, freqmax=min(8.0, 0.4 * sr))
    x = tr.data.astype(float)
    n = len(x)
    warm = int(1.5 * lta_s * sr)
    cft = recursive_sta_lta(x, int(sta_s * sr), int(lta_s * sr))
    onsets = [int(o[0]) for o in trigger_onset(cft, 3.5, 1.0) if o[0] > warm]
    lo_i, hi_i = warm, n - 1
    if t_exp is not None:
        i_exp = int(round((t_exp - tr.stats.starttime) * sr))
        lo_i, hi_i = max(i_exp - int(tol * sr), warm), min(i_exp + int(tol * sr), n - 1)
        if hi_i - lo_i < int(4 * sr):
            return None, "", "expected P falls outside the record (or inside the STA/LTA warm-up)"
        onsets = [i for i in onsets if lo_i <= i <= hi_i]
    method = "sta/lta"
    if onsets:
        i0 = onsets[0]
        a, b = max(i0 - int(5 * sr), lo_i, 1), min(i0 + int(3 * sr), hi_i)
    elif t_exp is not None:
        method, a, b = "aic-window", lo_i, hi_i
    else:
        return None, "", "no trigger after warm-up"
    seg = x[a:b]
    k = np.arange(2, len(seg) - 2)
    if k.size < 4:
        return None, "", "window too short"
    aic = np.array([kk * np.log(np.var(seg[:kk]) + 1e-20) + (len(seg) - kk - 1) * np.log(np.var(seg[kk + 1:]) + 1e-20) for kk in k])
    ip = a + int(k[np.argmin(aic)])
    w = int(5 * sr)
    snr = float(np.std(x[ip:ip + w]) / (np.std(x[max(ip - w, 0):ip]) + 1e-20))
    if snr < 2.5:
        return None, "", f"no clear onset (SNR {snr:.1f})"
    return tr.stats.starttime + ip / sr, method, f"SNR {snr:.1f}"


def picks_from_waveforms(data_dir, coords_csv, inventory, expect, tol, amp_window, out_csv):
    """
    expect: L.Hypo with an origin time (normally the catalogue event) used to predict each station's
    P time; picks are restricted to ±tol s of it. None → unconstrained (not recommended for real data).
    """
    from obspy import read, read_inventory, Stream, UTCDateTime

    files, skipped = waveform_files(data_dir)
    if skipped:
        print(f"[waveforms] not waveform data, skipped: {', '.join(skipped)}", file=sys.stderr)
    if not files:
        raise SystemExit(f"no waveform files (*.mseed, *.sac, …) in {data_dir}")
    st = Stream()
    for f in files:
        st += read(str(f))
    st.merge(fill_value=0)
    if inventory is None and Path(data_dir).is_dir() and (Path(data_dir) / "stations.xml").exists():
        inventory = str(Path(data_dir) / "stations.xml")
        print(f"[waveforms] using {inventory} for coordinates and responses", file=sys.stderr)
    inv = read_inventory(inventory) if inventory else None
    coords = {}
    if coords_csv:
        with open(coords_csv, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                coords[r["station"]] = (float(r["lat"]), float(r["lon"]), float(r.get("counts_per_um") or 1.0))
    tt = L.TravelTimes(quiet=True)
    if expect is None:
        print("[waveforms] WARNING: no reference origin — picks are unconstrained. Pass --reference "
              "(e.g. the USGS event) so each pick is restricted to the expected P window.", file=sys.stderr)

    rows = []
    for net, sta in sorted({(tr.stats.network, tr.stats.station) for tr in st}):
        z, group = choose_channels([tr for tr in st if tr.stats.network == net and tr.stats.station == sta])
        name = f"{net}.{sta}" if net else sta
        if z is None:
            print(f"[waveforms] {name}: no vertical component — skipped", file=sys.stderr)
            continue
        if inv is not None:
            try:
                c = inv.get_coordinates(z.id, z.stats.starttime)
                lat, lon = c["latitude"], c["longitude"]
            except Exception:  # noqa: BLE001
                print(f"[waveforms] {name}: {z.id} not in the inventory — skipped", file=sys.stderr)
                continue
        elif name in coords or sta in coords:
            lat, lon = coords.get(name, coords.get(sta))[:2]
        else:
            print(f"[waveforms] {name}: no coordinates — skipped", file=sys.stderr)
            continue
        t_exp = None
        if expect is not None:
            d = float(L.gc_km(expect.lat, expect.lon, lat, lon))
            t_exp = UTCDateTime(expect.time.isoformat()) + float(tt.p(d, expect.depth_km))
        tP, method, note = pick_p(z, t_exp, tol)
        if tP is None:
            print(f"[waveforms] {name} {z.stats.channel}: no pick — {note}", file=sys.stderr)
            continue
        disp = Stream([tr.copy() for tr in group])
        if inv is not None:
            disp.remove_response(inventory=inv, output="DISP", pre_filt=(0.05, 0.1, 0.4 * z.stats.sampling_rate * 0.9,
                                                                          0.4 * z.stats.sampling_rate))
            scale = 1e6
        else:
            # No responses: data are taken to be displacement, scaled by counts_per_um from --coords.
            disp.detrend("demean")
            scale = 1.0 / coords.get(name, coords.get(sta))[2]
        hz = [tr for tr in disp if tr.stats.channel[-1] in "NE12"] or [tr for tr in disp if tr.stats.channel[-1] in "Z3"]
        amp = max(float(np.max(np.abs(tr.slice(tP, tP + amp_window).data))) for tr in hz) * scale
        off = (tP - t_exp) if t_exp is not None else None
        rows.append({"station": name, "lat": lat, "lon": lon,
                     "p_time": tP.datetime.replace(tzinfo=timezone.utc).isoformat(), "amp_um": f"{amp:.4g}",
                     "channel": z.id, "pick_method": method,
                     "offset_from_expected_s": "" if off is None else f"{off:+.2f}"})
        print(f"[waveforms] {name} {z.id}: P {tP.isoformat()} ({method}, {note}"
              + (f", {off:+.1f} s from expected)" if off is not None else ")"), file=sys.stderr)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["station", "lat", "lon", "p_time", "amp_um", "channel", "pick_method",
                                          "offset_from_expected_s"])
        w.writeheader(); w.writerows(rows)
    print(f"[waveforms] {len(rows)} stations picked → {out_csv}", file=sys.stderr)
    if len(rows) < 3:
        raise SystemExit("fewer than 3 usable picks — cannot locate")
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
    d = json.loads(L.read_text(path))
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
    return rep.data


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("picks", help="analyse a picks CSV"); _common(s)
    s.add_argument("--picks", required=True)

    s = sub.add_parser("waveforms", help="pick + measure from waveforms, then analyse"); _common(s)
    s.add_argument("--data", required=True, help="directory of miniSEED/SAC (e.g. the output of fetch), or one file")
    s.add_argument("--inventory", help="StationXML with responses (default: stations.xml in --data, if present)")
    s.add_argument("--coords", help="CSV station,lat,lon[,counts_per_um] if no StationXML")
    s.add_argument("--pick-tol", type=float, default=30.0,
                   help="accept P picks only within ±this many seconds of the P time predicted from --reference")
    s.add_argument("--amp-window", type=float, default=30, help="seconds after P for peak displacement")
    s.add_argument("--picks-out", default="picks_from_waveforms.csv")

    s = sub.add_parser("fetch", help="download from an FDSN data centre")
    s.add_argument("--provider", default="EARTHSCOPE", help="FDSN data centre (default EARTHSCOPE)")
    s.add_argument("--stations", required=True, help="comma list NET.STA, e.g. II.ERM,IU.MAJO,IU.PET,IU.YSS,G.INU")
    s.add_argument("--origin", required=True, help="origin time (ISO, UTC)")
    s.add_argument("--before", type=float, default=60); s.add_argument("--after", type=float, default=600)
    s.add_argument("--out", default="fdsn_data")

    s = sub.add_parser("jma-history", help="parse a saved JMA EEW history page")
    s.add_argument("page", help="saved HTML/text of the 緊急地震速報の内容 page")
    s.add_argument("--serial", type=int, help="write this serial as an alert JSON")
    s.add_argument("--out", help="alert JSON path (with --serial)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "picks":
        return run_analysis(a.picks, a)
    if a.cmd == "waveforms":
        ref = load_reference(a.reference)
        alert = L.load_alert(a.alert)
        expect = ref if (ref is not None and ref.time is not None) else None
        if expect is None and alert.h_int is not None and alert.origin_time is not None:
            expect = L.Hypo(alert.h_int.lat, alert.h_int.lon, alert.h_int.depth_km, alert.origin_time, "alert")
            print("[waveforms] expected P times from the ALERT's hypocentre — if the alert is wrong, so is the "
                  "window; prefer --reference", file=sys.stderr)
        out = picks_from_waveforms(a.data, a.coords, a.inventory, expect, a.pick_tol, a.amp_window, a.picks_out)
        return run_analysis(out, a)
    if a.cmd == "fetch":
        fetch(a.provider, a.stations.split(","), a.origin, a.before, a.after, a.out)
        return None
    if a.cmd == "jma-history":
        h = L.parse_jma_history(L.read_text(a.page))
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
            out = a.out or f"alert_serial{a.serial}.json"
            Path(out).write_text(json.dumps(al, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"→ {out}")
        return h


if __name__ == "__main__":
    main()
