"""
eewlib — shared pieces for alert_triangulate.py and eew_diagnose.py

Geodesy, P travel times (IASP91 via ObsPy TauP, cached grid), amplitude–magnitude
relations, a JMA-intensity predictor, and parsers for JMA alert formats.

Everything here is deliberately transparent and parameterised. Where a relation is
*illustrative* rather than an agency's operational formula, it says so, and the
tools report which relation produced every number.
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

EARTH_R_KM = 6371.0
JST = timezone(timedelta(hours=9))


# --------------------------------------------------------------------------- time
def parse_time(s: str | None) -> datetime | None:
    """ISO-8601 → aware UTC datetime. Naive strings are treated as UTC."""
    if s is None or s == "":
        return None
    if isinstance(s, datetime):
        dt = s
    else:
        s = str(s).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fmt_time(dt: datetime | None, tz=JST) -> str:
    if dt is None:
        return "—"
    return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S.%f")[:-5] + (" JST" if tz is JST else " UTC")


# ------------------------------------------------------------------------ geodesy
def gc_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (haversine). Broadcasts over numpy arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def destination(lat, lon, bearing, dist_km):
    """Point reached travelling dist_km from (lat, lon) on an initial bearing."""
    d = dist_km / EARTH_R_KM
    b = math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1),
                         math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def hypo_km(epi_km, depth_km):
    return np.sqrt(np.asarray(epi_km) ** 2 + np.asarray(depth_km) ** 2)


def azimuthal_gap(ev_lat, ev_lon, st_lat, st_lon) -> float:
    az = np.sort(np.atleast_1d(bearing_deg(ev_lat, ev_lon, st_lat, st_lon)))
    if az.size < 2:
        return 360.0
    gaps = np.diff(np.concatenate([az, [az[0] + 360]]))
    return float(gaps.max())


def to_enu_km(lat, lon, lat0, lon0):
    """Local equirectangular projection (fine below ~200 km)."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R_KM * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R_KM
    return x, y


def from_enu_km(x, y, lat0, lon0):
    lat = lat0 + np.degrees(np.asarray(y) / EARTH_R_KM)
    lon = lon0 + np.degrees(np.asarray(x) / (EARTH_R_KM * math.cos(math.radians(lat0))))
    return lat, lon


# ----------------------------------------------------------------------- hypocentre
@dataclass
class Hypo:
    lat: float
    lon: float
    depth_km: float
    time: datetime | None = None   # origin time (UTC), optional
    label: str = ""

    @classmethod
    def from_dict(cls, d: dict | None, label: str = "") -> "Hypo | None":
        if not d or d.get("lat") is None or d.get("lon") is None:
            return None
        return cls(float(d["lat"]), float(d["lon"]), float(d.get("depth_km") or 0.0),
                   parse_time(d.get("time")), label or d.get("label", ""))

    def to_dict(self) -> dict:
        out = {"lat": round(self.lat, 4), "lon": round(self.lon, 4),
               "depth_km": round(self.depth_km, 1)}
        if self.time:
            out["time"] = self.time.isoformat()
        return out

    def __str__(self) -> str:
        return f"{self.lat:.3f}°N {self.lon:.3f}°E {self.depth_km:.0f} km"


# ------------------------------------------------------------------- travel times
class TravelTimes:
    """
    First-arriving P time (s) as a function of epicentral distance and depth.

    Uses a grid computed once with ObsPy TauP (IASP91) and cached to disk, then
    bilinear interpolation, so a grid-search locator can evaluate millions of
    times per second. Without ObsPy it falls back to a crude two-velocity model
    and says so loudly — fine for the synthetic demo, NOT for real data.
    """
    DIST_DEG = np.arange(0.0, 40.01, 0.2)
    DEPTH_KM = np.arange(0.0, 700.1, 20.0)

    def __init__(self, model: str = "iasp91", cache_dir: str | None = None, quiet=False):
        self.model = model
        self.exact = False
        cache_dir = Path(cache_dir or os.environ.get("EEW_CACHE", Path.home() / ".cache" / "eew_forensics"))
        cache = cache_dir / f"ptt_{model}_v1.npz"
        try:
            from scipy.interpolate import RegularGridInterpolator
        except ImportError:  # pragma: no cover
            RegularGridInterpolator = None
        table = None
        if cache.exists():
            table = np.load(cache)["t"]
            self.exact = True
        else:
            try:
                from obspy.taup import TauPyModel
                m = TauPyModel(model)
                if not quiet:
                    print(f"[eewlib] building P travel-time table ({model}); cached after first run…", file=sys.stderr)
                table = np.full((self.DIST_DEG.size, self.DEPTH_KM.size), np.nan)
                for j, z in enumerate(self.DEPTH_KM):
                    for i, d in enumerate(self.DIST_DEG):
                        arr = m.get_travel_times(source_depth_in_km=float(z), distance_in_degree=float(max(d, 1e-3)),
                                                 phase_list=["p", "P", "Pn", "Pdiff"])
                        if arr:
                            table[i, j] = min(a.time for a in arr)
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, t=table)
                self.exact = True
            except ImportError:
                if not quiet:
                    print("[eewlib] WARNING: ObsPy not installed — using a crude 6.0/8.0 km/s model. "
                          "Install obspy before analysing real data.", file=sys.stderr)
        self._table = table
        self._interp = None
        if table is not None and RegularGridInterpolator is not None:
            self._interp = RegularGridInterpolator((self.DIST_DEG, self.DEPTH_KM), table,
                                                   bounds_error=False, fill_value=None)

    def p(self, dist_km, depth_km):
        dist_km = np.asarray(dist_km, dtype=float)
        depth = np.broadcast_to(np.asarray(depth_km, dtype=float), dist_km.shape)
        if self._interp is not None:
            pts = np.stack([np.clip(dist_km / 111.195, 0, self.DIST_DEG[-1]),
                            np.clip(depth, 0, self.DEPTH_KM[-1])], axis=-1)
            return np.asarray(self._interp(pts)).reshape(dist_km.shape)
        r = hypo_km(dist_km, depth)                           # crude fallback
        return np.where(r < 150, r / 6.0, 150 / 6.0 + (r - 150) / 8.0)


# ------------------------------------------------------- amplitude ⇄ magnitude
@dataclass
class AmpRelation:
    """
    log10 A = a·M + b·log10 R + c·R + d       (A in µm, R in km)

    R is epicentral ("epi") or hypocentral ("hypo") distance.
    Inversion: M = (log10 A − b·log10 R − c·R − d) / a

    The mislocation→magnitude gain is ∂M/∂R = −(b/(R·ln10) + c)/a. With an
    anelastic term (c < 0) and a < 1 this grows without limit as R increases —
    which is why relations calibrated at short range blow up when extrapolated.
    """
    name: str
    a: float
    b: float
    c: float
    d: float
    dist: str = "hypo"
    valid_max_km: float = 700.0
    note: str = ""

    def _r(self, epi_km, depth_km):
        return np.asarray(epi_km, float) if self.dist == "epi" else hypo_km(epi_km, depth_km)

    def predict_logA(self, M, epi_km, depth_km):
        R = np.maximum(self._r(epi_km, depth_km), 1.0)
        return self.a * M + self.b * np.log10(R) + self.c * R + self.d

    def invert_M(self, logA, epi_km, depth_km):
        R = np.maximum(self._r(epi_km, depth_km), 1.0)
        return (np.asarray(logA) - self.b * np.log10(R) - self.c * R - self.d) / self.a

    def dM_dR(self, epi_km, depth_km):
        R = np.maximum(self._r(epi_km, depth_km), 1.0)
        return -(self.b / (R * math.log(10)) + self.c) / self.a

    def describe(self) -> str:
        return (f"{self.name}: log10A = {self.a:g}·M {self.b:+g}·log10R {self.c:+g}·R {self.d:+g} "
                f"[R={self.dist}, A µm, calibrated to {self.valid_max_km:.0f} km]")


RELATIONS: dict[str, AmpRelation] = {
    # Tsuboi (1954) JMA displacement magnitude: M = log10 A + 1.73 log10 Δ − 0.83
    # (A = max horizontal ground displacement in µm, Δ epicentral km, intended Δ ≲ 500 km)
    "tsuboi": AmpRelation("tsuboi", 1.0, -1.73, 0.0, 0.83, "epi", 500.0,
                          "Tsuboi (1954) JMA magnitude. Published, but for shallow events at Δ ≲ 500 km."),
    # Generic EEW-style relation WITH an anelastic term. Coefficients are ILLUSTRATIVE:
    # the shape (a<1, geometric + anelastic decay) is typical of displacement-based EEW
    # relations; the numbers are NOT JMA's operational values. d is set so it agrees with
    # Tsuboi at M6, 200 km. Use --relation-coeffs to supply the relation you want to test.
    "eew_illustrative": AmpRelation("eew_illustrative", 0.72, -1.2, -0.002, 1.69, "hypo", 700.0,
                                    "ILLUSTRATIVE coefficients — shape only, not an agency's formula."),
}


def relation_from_args(name: str, coeffs: str | None) -> AmpRelation:
    if coeffs:
        a, b, c, d = (float(x) for x in coeffs.split(","))
        return AmpRelation("custom", a, b, c, d, "hypo", 700.0, "user-supplied coefficients")
    if name not in RELATIONS:
        raise SystemExit(f"unknown relation {name!r}; choose from {', '.join(RELATIONS)} or pass --relation-coeffs a,b,c,d")
    return RELATIONS[name]


# ------------------------------------------------------------ intensity (shindo)
SI_MIDORIKAWA_D = {"crustal": 0.0, "interplate": -0.02, "intraplate": 0.12}


def fault_length_km(M):
    """Utsu-type empirical rupture length, log10 L = 0.5·M − 1.85 (approximate)."""
    return 10 ** (0.5 * M - 1.85)


def predict_pgv(Mw, depth_km, epi_km, fault_type="intraplate", site_amp=1.4, finite=False):
    """
    Si & Midorikawa (1999) PGV attenuation (cm/s). Calibrated to ~300 km; beyond that
    it is an extrapolation and is reported as such.
    log PGV = 0.58·Mw + 0.0038·D + d − 1.29 − log(X + 0.0028·10^(0.50·Mw)) − 0.002·X
    X is fault distance. finite=False: X = hypocentral distance (point source).
    finite=True: crude extended source — rupture of length L(M) assumed to extend half
    its length toward the site, X = max(R − L/2, D). For great earthquakes this is what
    turns a distant hypocentre into strong predicted shaking (an M9.9 rupture is >1000 km long).
    """
    X = hypo_km(epi_km, depth_km)
    if finite:
        X = np.maximum(X - fault_length_km(Mw) / 2, depth_km)
    D = min(float(depth_km), 300.0)
    dterm = SI_MIDORIKAWA_D.get(fault_type, 0.0)
    logv = 0.58 * Mw + 0.0038 * D + dterm - 1.29 - np.log10(X + 0.0028 * 10 ** (0.5 * Mw)) - 0.002 * X
    return site_amp * 10 ** logv


def pgv_to_shindo(pgv):
    """Midorikawa et al. (1999): I = 2.30 + 2.01·log10 PGV (0 ≤ I ≤ 7)."""
    return np.clip(2.30 + 2.01 * np.log10(np.maximum(pgv, 1e-6)), -1, 7.4)


def shindo_class(I: float) -> str:
    """Instrumental intensity → JMA class (rounding per JMA thresholds)."""
    if I < 0.5: return "0"
    if I < 1.5: return "1"
    if I < 2.5: return "2"
    if I < 3.5: return "3"
    if I < 4.5: return "4"
    if I < 5.0: return "5-"
    if I < 5.5: return "5+"
    if I < 6.0: return "6-"
    if I < 6.5: return "6+"
    return "7"


SHINDO_NUM = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5-": 4.75, "5弱": 4.75, "5+": 5.25, "5強": 5.25,
              "6-": 5.75, "6弱": 5.75, "6+": 6.25, "6強": 6.25, "7": 6.75}


# -------------------------------------------------------------------- the alert
@dataclass
class Alert:
    source: str = ""
    event_id: str = ""
    serial: int | None = None
    info_type: str = ""              # 発表 / 訂正 / 取消
    issued_at: datetime | None = None
    detected_at: datetime | None = None
    origin_time: datetime | None = None
    magnitude: float | None = None
    hypocenter: Hypo | None = None             # as published (used for intensity)
    hypocenter_magnitude: Hypo | None = None   # used for magnitude, if known to differ
    region: str = ""
    message_digest: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def h_int(self) -> Hypo | None:
        return self.hypocenter

    @property
    def h_mag(self) -> Hypo | None:
        return self.hypocenter_magnitude or self.hypocenter

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("hypocenter", "hypocenter_magnitude")}
        for k in ("issued_at", "detected_at", "origin_time"):
            d[k] = getattr(self, k).isoformat() if getattr(self, k) else None
        d["hypocenter"] = self.hypocenter.to_dict() if self.hypocenter else None
        d["hypocenter_magnitude"] = self.hypocenter_magnitude.to_dict() if self.hypocenter_magnitude else None
        return d


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_all(el, name):
    return [e for e in el.iter() if _local(e.tag) == name]


def _find(el, path: str):
    """Namespace-agnostic path lookup: 'Body/Earthquake/OriginTime'."""
    cur = [el]
    for part in path.split("/"):
        nxt = []
        for c in cur:
            nxt.extend(ch for ch in c if _local(ch.tag) == part)
        cur = nxt
        if not cur:
            return None
    return cur[0]


_COORD = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/?")


def parse_jma_coordinate(s: str) -> tuple[float, float, float] | None:
    """JMA XML jmx_eb:Coordinate, e.g. '+48.2+153.4-140000/' (depth in metres, negative down)."""
    if not s or s.strip() in ("", "/"):
        return None
    m = _COORD.match(s.strip())
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    depth = -float(m.group(3)) / 1000.0 if m.group(3) else 0.0
    return lat, lon, depth


def _num_or_none(v):
    """A finite float, or None for missing / NaN / unparseable values."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read_text(path) -> str:
    """Read UTF-8 text, tolerating a byte-order mark."""
    return Path(path).read_text(encoding="utf-8-sig")


def load_alert(path: str | Path) -> Alert:
    """Load an alert from normalised JSON or from JMA 防災情報XML (VXSE43/44/45)."""
    text = read_text(path)
    if text.lstrip().startswith("{"):
        d = json.loads(text)
        return Alert(
            source=d.get("source", ""), event_id=str(d.get("event_id", "") or ""), serial=d.get("serial"),
            info_type=d.get("info_type", ""), issued_at=parse_time(d.get("issued_at")),
            detected_at=parse_time(d.get("detected_at")), origin_time=parse_time(d.get("origin_time")),
            magnitude=_num_or_none(d.get("magnitude")), hypocenter=Hypo.from_dict(d.get("hypocenter"), "alert"),
            hypocenter_magnitude=Hypo.from_dict(d.get("hypocenter_magnitude"), "alert-M"),
            region=d.get("region", ""), message_digest=d.get("message_digest", ""), notes=d.get("notes", []))
    root = ET.fromstring(text)
    a = Alert(source="JMA-XML")
    t = _find(root, "Head/ReportDateTime"); a.issued_at = parse_time(t.text) if t is not None else None
    t = _find(root, "Head/EventID"); a.event_id = t.text.strip() if t is not None and t.text else ""
    t = _find(root, "Head/Serial"); a.serial = int(t.text) if t is not None and t.text and t.text.isdigit() else None
    t = _find(root, "Head/InfoType"); a.info_type = t.text.strip() if t is not None and t.text else ""
    t = _find(root, "Body/Earthquake/OriginTime"); a.origin_time = parse_time(t.text) if t is not None else None
    t = _find(root, "Body/Earthquake/ArrivalTime"); a.detected_at = parse_time(t.text) if t is not None else None
    hyp = _find(root, "Body/Earthquake/Hypocenter/Area")
    if hyp is not None:
        nm = _find(hyp, "Name"); a.region = nm.text.strip() if nm is not None and nm.text else ""
        for c in _find_all(hyp, "Coordinate"):
            xy = parse_jma_coordinate(c.text or "")
            if xy and "日本測地系" not in c.get("type", ""):
                a.hypocenter = Hypo(*xy, time=a.origin_time, label="alert")
                break
    mags = _find_all(root, "Magnitude")
    if mags:
        m = mags[0]
        # JMA writes an undetermined magnitude as condition="不明" with the text "NaN".
        cond = (m.get("condition") or "").strip()
        v = _num_or_none(m.text)
        if cond or v is None:
            a.magnitude = None
            a.notes.append(f"magnitude not determined (condition={cond or '—'}, description={m.get('description', '')})")
        else:
            a.magnitude = v
    return a


# JMA EEW history page (気象庁 緊急地震速報の内容): per-serial rows like
#   "17 12時40分02.6秒 112.5 48.2 153.4 140km 9.9"
_ROW = re.compile(r"(?<!\d)(\d{1,3})\s+(\d{1,2})時(\d{1,2})分(\d{1,2}(?:\.\d)?)秒\s+([\d.]+)\s+"
                  r"(\d{1,2}\.\d)\s+(\d{2,3}\.\d)\s+(\d{1,3})\s*km\s+(\d\.\d)")
_DET = re.compile(r"地震波\s*検知時刻\s*(\d{1,2})時(\d{1,2})分(\d{1,2}(?:\.\d)?)秒")
_DATE = re.compile(r"令和\s*(\d+)年\s*(\d{1,2})月(\d{1,2})日")


def parse_jma_history(text: str) -> dict:
    """Extract detection time and the per-serial hypocentre/M sequence from a saved page."""
    t = html.unescape(re.sub(r"<[^>]+>", " ", text))
    t = re.sub(r"[|｜]", " ", t)
    t = re.sub(r"\s+", " ", t)
    date = _DATE.search(t)
    y = 2018 + int(date.group(1)) if date else None
    mo, dd = (int(date.group(2)), int(date.group(3))) if date else (None, None)

    def mk(h, mi, s):
        if y is None:
            return None
        sec = float(s)
        return datetime(y, mo, dd, int(h), int(mi), int(sec), int(round((sec % 1) * 1e6)), tzinfo=JST).astimezone(timezone.utc)

    det = _DET.search(t)
    rows = []
    for m in _ROW.finditer(t):
        rows.append({"serial": int(m.group(1)), "issued_at": mk(m.group(2), m.group(3), m.group(4)),
                     "elapsed_s": float(m.group(5)), "lat": float(m.group(6)), "lon": float(m.group(7)),
                     "depth_km": float(m.group(8)), "magnitude": float(m.group(9))})
    seen, uniq = set(), []
    for r in rows:
        if r["serial"] not in seen:
            seen.add(r["serial"]); uniq.append(r)
    return {"detected_at": mk(*det.groups()) if det else None, "serials": uniq}


# ---------------------------------------------------------- reference geography
# Approximate coastal/peripheral points of Japan's onshore network, used ONLY to
# estimate when a distant event's P wave would first reach Japan. Supply a real
# station list (--stations) to replace these with actual sensor coordinates.
JAPAN_REFERENCE_POINTS = {
    "Shiretoko/Rausu": (44.02, 145.19), "Nemuro": (43.33, 145.58), "Erimo": (41.93, 143.25),
    "Wakkanai": (45.42, 141.67), "Okushiri": (42.15, 139.47), "Shimokita": (41.43, 141.45),
    "Miyako": (39.64, 141.95), "Choshi": (35.73, 140.83), "Hachijo-jima": (33.11, 139.79),
    "Chichi-jima": (27.09, 142.19), "Haha-jima": (26.66, 142.16), "Omaezaki": (34.60, 138.22),
    "Shiono-misaki": (33.43, 135.76), "Muroto": (33.25, 134.18), "Sado": (38.03, 138.37),
    "Noto": (37.50, 137.25), "Oki": (36.20, 133.32), "Tsushima": (34.40, 129.30),
    "Goto": (32.70, 128.84), "Sata-misaki": (31.00, 130.66), "Amami": (28.38, 129.49),
    "Naha": (26.21, 127.68), "Ishigaki": (24.34, 124.16), "Yonaguni": (24.47, 123.00),
}


def load_points_csv(path: str | Path) -> dict[str, tuple[float, float]]:
    import csv
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r.get("station") or r.get("name")] = (float(r["lat"]), float(r["lon"]))
    return out


def first_arrival_in_network(h: Hypo, tt: TravelTimes, points: dict[str, tuple[float, float]]):
    names = list(points)
    la = np.array([points[n][0] for n in names]); lo = np.array([points[n][1] for n in names])
    d = gc_km(h.lat, h.lon, la, lo)
    t = tt.p(d, h.depth_km)
    i = int(np.argmin(t))
    return names[i], float(d[i]), float(t[i])


# ------------------------------------------------------------------ formatting
class Report:
    """Collects lines for the console and a machine-readable dict for --json."""
    def __init__(self):
        self.lines: list[str] = []
        self.data: dict = {}

    def h(self, s):  self.lines += ["", s, "─" * len(s)]
    def p(self, s=""): self.lines.append(s)

    def flag(self, code, status, msg):
        mark = {"FLAG": "⚑ FLAG", "PASS": "✓ pass", "INFO": "· info", "N/A": "– n/a"}[status]
        self.lines.append(f"  [{code}] {mark:7s} {msg}")
        self.data.setdefault("checks", []).append({"code": code, "status": status, "message": msg})

    def text(self) -> str:
        return "\n".join(self.lines).strip() + "\n"


def write_json(path, obj):
    def conv(o):
        if isinstance(o, datetime): return o.isoformat()
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Hypo): return o.to_dict()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding="utf-8")
