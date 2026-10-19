"""
eewlib — alert_triangulate.py と eew_diagnose.py の共通部品

測地計算、P波走時（ObsPy TauP の IASP91 をグリッド化してキャッシュ）、振幅とマグニチュードの
関係式、計測震度の予測式、気象庁の速報形式のパーサ、日本語環境向けの入出力をまとめています。

どの数値も追跡できるよう、式の係数はすべて外から与えられる形にしています。機関の運用式ではなく
「説明用」の式を使う場合はそのことを明記し、各ツールは数値ごとにどの式で得たかを併記します。

日本語環境への対応
  ・CSV は UTF-8（BOM 有無とも）と Shift_JIS（cp932、Excel の既定）を自動判別して読み込みます。
  ・CSV の列名は英語・日本語のどちらでも構いません（例: lat / 緯度、observed_shindo / 観測震度）。
  ・全角数字・全角記号は NFKC 正規化してから解釈します（「５弱」「震度５弱」も可）。
  ・タイムゾーン指定のない時刻は日本時間（JST）とみなします。「2026年10月16日12時40分02.6秒」形式も可。
  ・画面出力は Shift_JIS で表せる文字だけを使い、全角文字の桁ずれも補正します。
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

EARTH_R_KM = 6371.0
JST = timezone(timedelta(hours=9))


# ===================================================================== 日本語環境
def setup_console() -> None:
    """標準出力・標準エラーで表せない文字があっても止まらないようにする（Windows の cp932 対策）。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


_ARGPARSE_JA = {
    "usage: ": "使い方: ",
    "options": "オプション",
    "positional arguments": "位置引数",
    "show this help message and exit": "このヘルプを表示して終了します",
    "%(prog)s: error: %(message)s\n": "%(prog)s: エラー: %(message)s\n",
    "the following arguments are required: %s": "次の引数は必須です: %s",
    "unrecognized arguments: %s": "認識できない引数です: %s",
    "invalid choice: %(value)r (choose from %(choices)s)": "選択肢にない値です: %(value)r（%(choices)s から選んでください）",
    "expected one argument": "値を1つ指定してください",
    "expected at least one argument": "値を1つ以上指定してください",
    "expected at most one argument": "値は1つまでです",
    "argument %(argument_name)s: %(message)s": "引数 %(argument_name)s: %(message)s",
    "invalid %(type)s value: %(value)r": "%(type)s として解釈できない値です: %(value)r",
    "ambiguous option: %(option)s could match %(matches)s": "あいまいなオプションです: %(option)s は %(matches)s のいずれにも該当します",
    "one of the arguments %s is required": "%s のいずれかが必須です",
    "not allowed with argument %s": "引数 %s とは同時に指定できません",
    " (default: %(default)s)": "（既定値: %(default)s）",
}


def localize_argparse() -> None:
    """argparse の定型文（usage:、エラー文など）を日本語にする。"""
    argparse._ = lambda s: _ARGPARSE_JA.get(s, s)   # argparse はモジュール内の _() で翻訳を引く


def nfkc(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip() if s is not None else ""


def display_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def ljust_w(s, width: int) -> str:
    """全角文字を2桁として左寄せ。"""
    s = str(s)
    return s + " " * max(width - display_width(s), 0)


def rjust_w(s, width: int) -> str:
    s = str(s)
    return " " * max(width - display_width(s), 0) + s


def read_text(path) -> str:
    """UTF-8（BOM 可）→ Shift_JIS（cp932）の順に試して読む。"""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# 列名の別名（英語・日本語）→ 内部名
COLUMN_ALIASES = {
    "station": ["station", "name", "sta", "観測点", "観測点名", "地点", "地点名"],
    "lat": ["lat", "latitude", "緯度", "北緯"],
    "lon": ["lon", "lng", "longitude", "経度", "東経"],
    "p_time": ["p_time", "p到達時刻", "p波到達時刻", "到達時刻", "p波"],
    "amp_um": ["amp_um", "振幅_um", "振幅(um)", "振幅(μm)", "最大変位_um", "最大変位(μm)", "最大変位振幅"],
    "observed_shindo": ["observed_shindo", "観測震度", "震度"],
    "received_at": ["received_at", "受信時刻", "受信日時"],
    "carrier": ["carrier", "事業者", "キャリア", "携帯事業者"],
    "channel": ["channel", "経路", "伝達手段", "受信経路"],
    "rssi_dbm": ["rssi_dbm", "rssi", "受信強度", "受信強度_dbm", "受信強度(dbm)"],
    "cell_id": ["cell_id", "セルid", "基地局id"],
    "message_digest": ["message_digest", "本文ハッシュ", "ハッシュ"],
    "counts_per_um": ["counts_per_um", "感度", "感度_counts_per_um"],
}
_ALIAS = {nfkc(a).lower(): k for k, v in COLUMN_ALIASES.items() for a in v}


def read_csv_rows(path) -> list[dict]:
    """文字コードを自動判別し、列名を内部名にそろえて返す。"""
    rows = []
    for r in csv.DictReader(io.StringIO(read_text(path))):
        rows.append({_ALIAS.get(nfkc(k).lower(), nfkc(k)): (nfkc(v) if v is not None else "") for k, v in r.items() if k})
    return rows


def write_csv_rows(path, fieldnames, rows) -> None:
    """Excel でそのまま開けるよう BOM 付き UTF-8 で書く。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


CARRIER_ALIASES = {"ドコモ": "docomo", "nttドコモ": "docomo", "docomo": "docomo", "au": "au", "kddi": "au",
                   "ソフトバンク": "softbank", "softbank": "softbank", "楽天": "rakuten", "楽天モバイル": "rakuten",
                   "rakuten": "rakuten", "ワイモバイル": "softbank", "uqモバイル": "au", "ahamo": "docomo",
                   "povo": "au", "linemo": "softbank"}
CHANNEL_ALIASES = {"cbs": "cbs", "セルブロードキャスト": "cbs", "エリアメール": "cbs", "緊急速報メール": "cbs",
                   "etws": "cbs", "tv": "tv", "テレビ": "tv", "radio": "radio", "ラジオ": "radio",
                   "jalert": "jalert", "jアラート": "jalert", "防災無線": "jalert", "防災行政無線": "jalert",
                   "app": "app", "アプリ": "app", "other": "other", "その他": "other"}


def norm_carrier(s: str) -> str:
    k = nfkc(s).lower()
    return CARRIER_ALIASES.get(k, k)


def norm_channel(s: str) -> str:
    k = nfkc(s).lower()
    return CHANNEL_ALIASES.get(k, k or "cbs")


# ======================================================================== 時刻
_JDATE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})時(\d{1,2})分(\d{1,2}(?:\.\d+)?)秒")


def parse_time(s) -> datetime | None:
    """
    時刻を UTC の datetime にする。ISO 8601、「2026/10/16 12:40:02.6」、
    「2026年10月16日12時40分02.6秒」を受け付ける。タイムゾーン指定がなければ JST とみなす。
    """
    if s is None or s == "":
        return None
    if isinstance(s, datetime):
        dt = s
    else:
        t = nfkc(s)
        m = _JDATE.search(t)
        if m:
            y, mo, d, h, mi, sec = m.groups()
            secf = float(sec)
            dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(secf), int(round((secf % 1) * 1e6)))
        else:
            t = t.replace("Z", "+00:00").replace("/", "-")
            if re.search(r"\d \d", t):
                t = t.replace(" ", "T", 1)
            t = re.sub(r"\s*JST$", "+09:00", t)
            dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=JST)
    return dt.astimezone(timezone.utc)


def fmt_time(dt: datetime | None, tz=JST) -> str:
    if dt is None:
        return "－"
    return dt.astimezone(tz).strftime("%Y/%m/%d %H:%M:%S.%f")[:-5] + (" JST" if tz is JST else " UTC")


# ======================================================================== 測地
def gc_km(lat1, lon1, lat2, lon2):
    """大円距離（km、ハバーサイン公式）。numpy 配列でブロードキャスト可。"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """方位角（度、北から時計回り）。"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def destination(lat, lon, bearing, dist_km):
    """(lat, lon) から方位 bearing に dist_km 進んだ地点。"""
    d = dist_km / EARTH_R_KM
    b = math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def hypo_km(epi_km, depth_km):
    """震央距離と深さから震源距離。"""
    return np.sqrt(np.asarray(epi_km) ** 2 + np.asarray(depth_km) ** 2)


def azimuthal_gap(ev_lat, ev_lon, st_lat, st_lon) -> float:
    """震央から見た観測点の方位の最大空白（度）。"""
    az = np.sort(np.atleast_1d(bearing_deg(ev_lat, ev_lon, st_lat, st_lon)))
    if az.size < 2:
        return 360.0
    return float(np.diff(np.concatenate([az, [az[0] + 360]])).max())


def to_enu_km(lat, lon, lat0, lon0):
    """局所平面への投影（正距円筒、200 km 程度まで）。"""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R_KM * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R_KM
    return x, y


def from_enu_km(x, y, lat0, lon0):
    lat = lat0 + np.degrees(np.asarray(y) / EARTH_R_KM)
    lon = lon0 + np.degrees(np.asarray(x) / (EARTH_R_KM * math.cos(math.radians(lat0))))
    return lat, lon


def latlon_ja(lat: float, lon: float, nd: int = 3) -> str:
    ns = "北緯" if lat >= 0 else "南緯"
    ew = "東経" if lon >= 0 else "西経"
    return f"{ns}{abs(lat):.{nd}f}° {ew}{abs(lon):.{nd}f}°"


# ======================================================================== 震源
@dataclass
class Hypo:
    lat: float
    lon: float
    depth_km: float
    time: datetime | None = None   # 発生時刻（UTC）。不明なら None
    label: str = ""

    @classmethod
    def from_dict(cls, d: dict | None, label: str = "") -> "Hypo | None":
        if not d or d.get("lat") is None or d.get("lon") is None:
            return None
        return cls(float(d["lat"]), float(d["lon"]), float(d.get("depth_km") or 0.0),
                   parse_time(d.get("time")), label or d.get("label", ""))

    def to_dict(self) -> dict:
        out = {"lat": round(self.lat, 4), "lon": round(self.lon, 4), "depth_km": round(self.depth_km, 1)}
        if self.time:
            out["time"] = self.time.isoformat()
        return out

    def __str__(self) -> str:
        return f"{latlon_ja(self.lat, self.lon)} 深さ{self.depth_km:.0f}km"


# ====================================================================== P波走時
class TravelTimes:
    """
    震央距離と深さに対する最初に到達する P 波の走時（秒）。

    ObsPy TauP（IASP91）で一度だけ表を作ってディスクにキャッシュし、以後は双線形補間で引く
    ので、グリッドサーチの震源決定でも高速に評価できます。ObsPy がない場合は 6.0/8.0 km/s の
    簡易モデルで代用し、その旨をはっきり警告します（合成データの動作確認用で、実データには不可）。
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
                    print(f"[eewlib] P波走時表（{model}）を作成しています。初回のみ時間がかかり、以後はキャッシュを使います…",
                          file=sys.stderr)
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
                    print("[eewlib] 警告: ObsPy が見つかりません。6.0/8.0 km/s の簡易モデルで代用します。"
                          "実データを解析する前に obspy をインストールしてください。", file=sys.stderr)
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
            return self._interp(pts)
        r = hypo_km(dist_km, depth)                                   # 簡易モデル
        return np.where(r < 150, r / 6.0, 150 / 6.0 + (r - 150) / 8.0)


# ============================================================ 振幅 ⇔ マグニチュード
@dataclass
class AmpRelation:
    """
    log10 A = a・M + b・log10 R + c・R + d　（A は μm、R は km）

    R は震央距離（"epi"）または震源距離（"hypo"）。
    逆算: M = (log10 A − b・log10 R − c・R − d) / a

    震源位置の誤差がマグニチュードに効く度合いは ∂M/∂R = −(b/(R・ln10) + c)/a。
    内部減衰の項（c < 0）があり a < 1 のとき、この値は R とともに際限なく大きくなります。
    近距離で較正した式を遠距離に外挿すると M が跳ね上がるのはこのためです。
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
        r = "震央距離" if self.dist == "epi" else "震源距離"
        return (f"{self.name}: log10A = {self.a:g}・M {self.b:+g}・log10R {self.c:+g}・R {self.d:+g}"
                f"（R＝{r}、A は μm、較正範囲 {self.valid_max_km:.0f}km まで）")


RELATIONS: dict[str, AmpRelation] = {
    # 坪井（1954）の気象庁変位マグニチュード: M = log10 A + 1.73 log10 Δ − 0.83
    # （A＝水平動最大変位振幅 μm、Δ＝震央距離 km。おおむね Δ ≦ 500 km を想定）
    "tsuboi": AmpRelation("tsuboi", 1.0, -1.73, 0.0, 0.83, "epi", 500.0,
                          "坪井（1954）の気象庁マグニチュード。公表された式だが、浅い地震・震央距離おおむね500km以内が前提。"),
    # 内部減衰項を含む緊急地震速報型の汎用式。係数は【説明用】:
    # a<1、幾何減衰＋内部減衰という形は変位型の速報用関係式に典型的だが、数値は気象庁の運用値ではない。
    # d は M6・200km で坪井式と一致するよう決めた。検証したい式は --relation-coeffs で与えること。
    "eew_illustrative": AmpRelation("eew_illustrative", 0.72, -1.2, -0.002, 1.69, "hypo", 700.0,
                                    "【説明用の係数】式の形を示すためのもので、機関の運用式ではありません。"),
}


def relation_from_args(name: str, coeffs: str | None) -> AmpRelation:
    if coeffs:
        a, b, c, d = (float(x) for x in nfkc(coeffs).split(","))
        return AmpRelation("custom", a, b, c, d, "hypo", 700.0, "利用者が指定した係数")
    if name not in RELATIONS:
        raise SystemExit(f"関係式 {name!r} はありません。{', '.join(RELATIONS)} から選ぶか、"
                         "--relation-coeffs a,b,c,d で係数を指定してください。")
    return RELATIONS[name]


# ======================================================================== 震度
SI_MIDORIKAWA_D = {"crustal": 0.0, "interplate": -0.02, "intraplate": 0.12}
FAULT_TYPE_JA = {"crustal": "地殻内", "interplate": "プレート間", "intraplate": "プレート内"}


def fault_length_km(M):
    """宇津の経験式に準じた断層長さ log10 L = 0.5・M − 1.85（概算）。"""
    return 10 ** (0.5 * M - 1.85)


def predict_pgv(Mw, depth_km, epi_km, fault_type="intraplate", site_amp=1.4, finite=False):
    """
    司・翠川（1999）の最大速度（PGV、cm/s）の距離減衰式。較正はおおむね 300 km まで、それ以遠は外挿。
    log PGV = 0.58・Mw + 0.0038・D + d − 1.29 − log(X + 0.0028・10^(0.50・Mw)) − 0.002・X
    X は断層最短距離。finite=False: X＝震源距離（点震源）。
    finite=True: 断層長 L(M) の半分が地点側へ延びると仮定した粗い面震源近似 X = max(R − L/2, D)。
    巨大地震では、遠い震源から強い揺れが予測されるのはこの効果による（M9.9 の断層は 1000 km を超える）。
    """
    X = hypo_km(epi_km, depth_km)
    if finite:
        X = np.maximum(X - fault_length_km(Mw) / 2, depth_km)
    D = min(float(depth_km), 300.0)
    dterm = SI_MIDORIKAWA_D.get(fault_type, 0.0)
    logv = 0.58 * Mw + 0.0038 * D + dterm - 1.29 - np.log10(X + 0.0028 * 10 ** (0.5 * Mw)) - 0.002 * X
    return site_amp * 10 ** logv


def pgv_to_shindo(pgv):
    """翠川・藤本・村松（1999）: I = 2.30 + 2.01・log10 PGV（0 ≦ I ≦ 7）。"""
    return np.clip(2.30 + 2.01 * np.log10(np.maximum(pgv, 1e-6)), -1, 7.4)


def shindo_class(I: float) -> str:
    """計測震度 → 震度階級（気象庁の区分）。"""
    if I < 0.5: return "0"
    if I < 1.5: return "1"
    if I < 2.5: return "2"
    if I < 3.5: return "3"
    if I < 4.5: return "4"
    if I < 5.0: return "5弱"
    if I < 5.5: return "5強"
    if I < 6.0: return "6弱"
    if I < 6.5: return "6強"
    return "7"


_SHINDO_NUM = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5弱": 4.75, "5-": 4.75, "5強": 5.25, "5+": 5.25,
               "6弱": 5.75, "6-": 5.75, "6強": 6.25, "6+": 6.25, "7": 6.75}


def parse_shindo(s) -> float:
    """「5弱」「５弱」「震度5弱」「5-」などを数値に。空欄は NaN。"""
    t = nfkc(s).replace("震度", "").replace(" ", "")
    return _SHINDO_NUM.get(t, float("nan"))


# ======================================================================== 速報
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
    hypocenter: Hypo | None = None             # 発表された震源（震度予測に使われたもの）
    hypocenter_magnitude: Hypo | None = None   # マグニチュード計算に使われた震源（異なる場合）
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
    """名前空間を無視したパス検索（例 'Body/Earthquake/OriginTime'）。"""
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
    """気象庁防災情報XML の jmx_eb:Coordinate（例 '+48.2+153.4-140000/'、深さはメートル・下向き負）。"""
    if not s or s.strip() in ("", "/"):
        return None
    m = _COORD.match(s.strip())
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    depth = -float(m.group(3)) / 1000.0 if m.group(3) else 0.0
    return lat, lon, depth


def load_alert(path) -> Alert:
    """正規化済み JSON、または気象庁防災情報XML（VXSE43/44/45）から速報を読む。"""
    text = read_text(path)
    if text.lstrip().startswith("{"):
        d = json.loads(text)
        return Alert(
            source=d.get("source", ""), event_id=str(d.get("event_id", "") or ""), serial=d.get("serial"),
            info_type=d.get("info_type", ""), issued_at=parse_time(d.get("issued_at")),
            detected_at=parse_time(d.get("detected_at")), origin_time=parse_time(d.get("origin_time")),
            magnitude=d.get("magnitude"), hypocenter=Hypo.from_dict(d.get("hypocenter"), "速報"),
            hypocenter_magnitude=Hypo.from_dict(d.get("hypocenter_magnitude"), "速報M"),
            region=d.get("region", ""), message_digest=d.get("message_digest", ""), notes=d.get("notes", []))
    root = ET.fromstring(text)
    a = Alert(source="気象庁XML")
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
                a.hypocenter = Hypo(*xy, time=a.origin_time, label="速報")
                break
    for m in _find_all(root, "Magnitude"):
        try:
            a.magnitude = float(m.text)
            break
        except (TypeError, ValueError):
            continue
    return a


# 気象庁「緊急地震速報の内容」ページの各報の行。例:
#   "17 12時40分02.6秒 112.5 48.2 153.4 140km 9.9"
_ROW = re.compile(r"(?<!\d)(\d{1,3})\s+(\d{1,2})時(\d{1,2})分(\d{1,2}(?:\.\d)?)秒\s+([\d.]+)\s+"
                  r"(\d{1,2}\.\d)\s+(\d{2,3}\.\d)\s+(\d{1,3})\s*km\s+(\d\.\d)")
_DET = re.compile(r"地震波\s*検知時刻\s*(\d{1,2})時(\d{1,2})分(\d{1,2}(?:\.\d)?)秒")
_DATE = re.compile(r"令和\s*(\d+)年\s*(\d{1,2})月(\d{1,2})日")


def parse_jma_history(text: str) -> dict:
    """保存したページから地震波検知時刻と、各報の震源・M の推移を取り出す。"""
    t = nfkc(html.unescape(re.sub(r"<[^>]+>", " ", text)))
    t = re.sub(r"[|]", " ", t)
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
    rows, seen = [], set()
    for m in _ROW.finditer(t):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        rows.append({"serial": n, "issued_at": mk(m.group(2), m.group(3), m.group(4)),
                     "elapsed_s": float(m.group(5)), "lat": float(m.group(6)), "lon": float(m.group(7)),
                     "depth_km": float(m.group(8)), "magnitude": float(m.group(9))})
    return {"detected_at": mk(*det.groups()) if det else None, "serials": rows}


# ===================================================================== 参照地点
# 日本の陸上観測網の外縁にあたるおおよその地点。遠地地震の P 波が「最初に日本へ届く時刻」を
# 見積もるためだけに使います。実際の観測点座標は --stations で与えてください。
JAPAN_REFERENCE_POINTS = {
    "知床・羅臼": (44.02, 145.19), "根室": (43.33, 145.58), "えりも": (41.93, 143.25),
    "稚内": (45.42, 141.67), "奥尻島": (42.15, 139.47), "下北": (41.43, 141.45),
    "宮古": (39.64, 141.95), "銚子": (35.73, 140.83), "八丈島": (33.11, 139.79),
    "父島": (27.09, 142.19), "母島": (26.66, 142.16), "御前崎": (34.60, 138.22),
    "潮岬": (33.43, 135.76), "室戸": (33.25, 134.18), "佐渡": (38.03, 138.37),
    "能登": (37.50, 137.25), "隠岐": (36.20, 133.32), "対馬": (34.40, 129.30),
    "五島": (32.70, 128.84), "佐多岬": (31.00, 130.66), "奄美": (28.38, 129.49),
    "那覇": (26.21, 127.68), "石垣": (24.34, 124.16), "与那国": (24.47, 123.00),
}


def load_points_csv(path) -> dict[str, tuple[float, float]]:
    return {r["station"]: (float(r["lat"]), float(r["lon"])) for r in read_csv_rows(path)}


def first_arrival_in_network(h: Hypo, tt: TravelTimes, points: dict[str, tuple[float, float]]):
    names = list(points)
    la = np.array([points[n][0] for n in names]); lo = np.array([points[n][1] for n in names])
    d = gc_km(h.lat, h.lon, la, lo)
    t = tt.p(d, h.depth_km)
    i = int(np.argmin(t))
    return names[i], float(d[i]), float(t[i])


# ======================================================================== 出力
class Report:
    """画面用の行と、--json 用の機械可読データを集める。判定コード（FLAG/PASS など）は英語のまま。"""
    MARK = {"FLAG": "▲ 要注意", "PASS": "○ 問題なし", "INFO": "・ 参考", "N/A": "－ 対象外"}

    def __init__(self):
        self.lines: list[str] = []
        self.data: dict = {}

    def h(self, s):
        self.lines += ["", s, "─" * max(display_width(s) // 2, 10)]

    def p(self, s=""):
        self.lines.append(s)

    def flag(self, code, status, msg):
        self.lines.append(f"  [{code}] {ljust_w(self.MARK[status], 11)}{msg}")
        self.data.setdefault("checks", []).append({"code": code, "status": status, "message": msg})

    def text(self) -> str:
        return "\n".join(self.lines).strip() + "\n"


def write_json(path, obj):
    def conv(o):
        if isinstance(o, datetime): return o.isoformat()
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Hypo): return o.to_dict()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding="utf-8")
