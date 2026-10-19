#!/usr/bin/env python3
"""
eew_diagnose.py ― 観測データから、緊急地震速報の推定がどこで誤ったかを調べる

観測点のデータから震源とマグニチュードを独立に求め直し、気象庁の緊急地震速報で実際に
起きたことのある不具合の類型と照らし合わせます。

  D1  震源の誤決定          速報の震源を P 波の読み取り値が支持するか（最初の数観測点だけの解も確認）
  D2  距離による除外        マグニチュードの計算に、震源から N km 超の観測点が必要だったか
                            （気象庁は 2020年9月7日以降、700km 超の観測点を除外する運用）
  D3  処理段階間の不整合    マグニチュードと震度予測で異なる震源を使っていないか
  D4  観測網の外の地震      方位角ギャップと最寄り観測点までの距離。網の外では、震源は
                            観測網から見た方向に沿って決まりにくくなる
  D5  マグニチュードの妥当性 物理的な上限、および実際の地震との振幅比
  D6  震度の過大予測        予測震度と観測震度の比較。震度予測の段階が入力に忠実だったか
  D7  単一観測点への依存    1点ずつ抜いて、M や震源が1観測点に左右されていないか
  D8  外れ値の読み取り      複数の地震の重なり、時刻付与の異常

サブコマンド
  picks        読み取り値の CSV を解析（観測点,緯度,経度,P波到達時刻[,最大変位(μm),観測震度]）
  waveforms    miniSEED／SAC から P 波を読み取り振幅を測ってから解析
  fetch        FDSN データセンターから波形と StationXML を取得（要ネットワーク）
  jma-history  保存した気象庁「緊急地震速報の内容」ページから各報の推移を取り出す

マグニチュードの関係式はパラメータです（--relation／--relation-coeffs）。気象庁の運用係数は
公開されていないため、係数に依存しない結果（D2 の観測点数、D3、D4、D6 の忠実性、時刻の照合）には
その旨を明記します。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import eewlib as L

N_IND, N_REF, N_INT, N_MAG = "独立解", "参照カタログ", "速報（震度予測）", "速報（M計算）"


# ===================================================================== データ
def _log_amp(v, station):
    """正の振幅の常用対数。0・負・解釈不能は NaN（マグニチュード計算から除外）。"""
    if v in (None, ""):
        return np.nan
    try:
        a = float(L.nfkc(v))
    except ValueError:
        a = float("nan")
    if not (math.isfinite(a) and a > 0):
        print(f"[読み取り値] {station}: 振幅 {v!r} は正の数ではないため、マグニチュード計算から除外します", file=sys.stderr)
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
        self.obs_I = np.array([L.parse_shindo(r.get("observed_shindo", "")) for r in rows])

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
        return cls(L.read_csv_rows(path))


# ===================================================================== 震源決定
def misfit_grid(p: Picks, tt: L.TravelTimes, lats, lons, depths, fixed_t0=None):
    """緯度×経度×深さのグリッドで L1 残差を計算。発生時刻は各点で中央値として解く（固定も可）。"""
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    D = L.gc_km(LA[..., None], LO[..., None], p.lat, p.lon)
    best = (np.inf, None)
    cube = np.empty((lats.size, lons.size, depths.size))
    for k, z in enumerate(depths):
        T = tt.p(D, z)
        t0 = np.median(p.t - T, axis=-1) if fixed_t0 is None else np.full(T.shape[:-1], fixed_t0)
        m = np.mean(np.abs(p.t - T - t0[..., None]), axis=-1)
        cube[..., k] = m
        i, j = np.unravel_index(np.argmin(m), m.shape)
        if m[i, j] < best[0]:
            best = (m[i, j], (i, j, k, t0[i, j]))
    return cube, best


def locate(p: Picks, tt: L.TravelTimes, center=None, half_deg=20.0, fixed_t0=None):
    """粗→細の3段グリッドサーチ。最良解、平均残差、候補域の広がり（km）とその向きを返す。"""
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
    h = L.Hypo(clat, clon, cz, p.t_ref + timedelta(seconds=float(t0)), "独立解")
    cube, lats, lons, depths = coarse                     # 不確かさ: 残差が最良＋max(0.5秒, 50%) 以下の範囲
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
    t0 = (h.time - p.t_ref).total_seconds() if (fixed_t0 and h.time is not None) else float(np.median(p.t - T))
    return p.t - T - t0, t0


# ================================================================ マグニチュード
def station_M(p: Picks, h: L.Hypo, rel: L.AmpRelation):
    d = L.gc_km(h.lat, h.lon, p.lat, p.lon)
    return rel.invert_M(p.logA, d, h.depth_km), d, L.hypo_km(d, h.depth_km)


def network_M(p: Picks, h: L.Hypo, rel: L.AmpRelation, cutoff_km: float | None):
    """観測点ごとの M の中央値。cutoff_km 指定時は震源距離がそれ以下の観測点のみ。"""
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


# ======================================================================== 解析
D8_CAP_S = 10.0          # 外れ値のしきい値は、読み取り値がどれほどばらついてもこれ以上は広げない


def analyse(p: Picks, alert: L.Alert, rel: L.AmpRelation, cutoff: float, tt: L.TravelTimes,
            reference: L.Hypo | None, fault_type: str, site_amp: float, rep: L.Report):
    W = 18
    hM, hI, Ma = alert.h_mag, alert.h_int, alert.magnitude
    rep.p(f"観測点数: {len(p)}　P波時刻の基準: {L.fmt_time(p.t_ref)}")
    rep.p(f"関係式: {rel.describe()}")
    if rel.note:
        rep.p(f"　　　　{rel.note}")
    rep.p(f"走時: {'IASP91（TauP）' if tt.exact else '簡易モデル ― obspy を入れてください'}")
    rep.p(f"速報: M{Ma if Ma is not None else '不明'}　震度予測の震源 {hI or '－（なし）'}　M計算の震源 {hM or '－'}"
          + ("（同じ）" if hI and alert.hypocenter_magnitude is None else ""))
    for n in alert.notes:
        rep.p(f"  注記: {n}")
    classes = []

    # ---- 独立解
    rep.h("P波の読み取り値による独立した震源決定")
    h, mis, reg_km, reg_az = locate(p, tt)
    rep.p(f"  {h}　発生 {L.fmt_time(h.time)}　平均残差 {mis:.2f} 秒")
    rep.p(f"  候補域（残差が最良解に近い範囲）の広がり {reg_km:.0f}km"
          + (f"、方位 {reg_az:.0f}°／{reg_az+180:.0f}° 方向に細長い" if reg_az is not None else ""))
    gap = L.azimuthal_gap(h.lat, h.lon, p.lat, p.lon)
    dmin = float(L.gc_km(h.lat, h.lon, p.lat, p.lon).min())
    baz = float(np.median(L.bearing_deg(h.lat, h.lon, p.lat, p.lon)))
    rep.data["solution"] = {"hypo": h.to_dict(), "mean_abs_res_s": mis, "region_km": reg_km,
                            "region_bearing": reg_az, "azimuthal_gap": gap, "nearest_station_km": dmin}
    if reference:
        sep = float(L.gc_km(h.lat, h.lon, reference.lat, reference.lon))
        rep.p(f"  参照カタログ: {reference} → 独立解との差 {sep:.0f}km")
        rep.data["solution"]["sep_from_reference_km"] = sep

    Mdat, _, _ = network_M(p, h, rel, None)
    if Mdat is not None and dmin <= rel.valid_max_km:
        rep.p(f"  この震源での {rel.name} によるマグニチュード: M{Mdat:.1f}")
    else:
        Mdat = None
        rep.p(f"  マグニチュード: 計算しません。最寄り観測点 {dmin:.0f}km が {rel.name} の較正範囲 "
              f"{rel.valid_max_km:.0f}km を超えているか、使える振幅がありません。この地震にはカタログ（遠地）の Mw を使ってください。")

    # ---- D1 震源の誤決定
    rep.h("D1 ・ 震源 ― 読み取り値は速報の震源を支持するか")
    rows = [(N_IND, h)] + ([(N_REF, reference)] if reference else []) + \
           ([(N_INT, hI)] if hI else []) + ([(N_MAG, hM)] if alert.hypocenter_magnitude else [])
    base = None
    for name, hh in rows:
        r_free, _ = residuals(p, tt, hh, fixed_t0=False)
        mf = float(np.mean(np.abs(r_free)))
        base = base or mf
        line = f"  {L.ljust_w(name, W)}{L.ljust_w(str(hh), 40)}平均残差（発生時刻自由）{mf:6.2f} 秒"
        if hh.time is not None:
            r_fix, _ = residuals(p, tt, hh, fixed_t0=True)
            line += f"　（発生時刻固定）{float(np.mean(np.abs(r_fix))):7.2f} 秒"
        rep.p(line)
    if not hI:
        rep.flag("D1", "N/A", "速報に震源がないため、読み取り値と照合できません。")
    for name, hh in rows[1:]:
        if name == N_REF:
            continue
        r_free, _ = residuals(p, tt, hh, fixed_t0=False)
        mf = float(np.mean(np.abs(r_free)))
        sep = float(L.gc_km(h.lat, h.lon, hh.lat, hh.lon))
        if sep <= 30:
            rep.flag("D1", "PASS", f"{name}は独立解から {sep:.0f}km 以内です。")
        elif mf <= max(1.5 * base, base + 0.5):
            rep.flag("D1", "FLAG", f"{name}は独立解から {sep:.0f}km 離れているのに、読み取り値への当てはまりはほぼ同じです"
                     f"（{mf:.2f} 秒 対 {base:.2f} 秒）。読み取り値では両者を区別できず、震源を決めているのはデータではなく"
                     "観測点配置です。速報の震源が観測網から見た方向に沿って引きずられるのは、こうした場合です。")
        else:
            rep.flag("D1", "FLAG", f"{name}は {sep:.0f}km 離れており、全観測点の読み取り値はこれを支持しません"
                     f"（{mf:.2f} 秒 対 {base:.2f} 秒）。ただし最初の数観測点だけで決めた解なら、そこに落ちることはあり得ます（下記）。")

    order = np.argsort(p.t)
    alert_h = [(n, x) for n, x in ((N_INT, hI), (N_MAG, hM if alert.hypocenter_magnitude else None))
               if x and float(L.gc_km(h.lat, h.lon, x.lat, x.lon)) > 30]
    rep.p("")
    rep.p("  早い段階の解（最初の N 観測点のみ ＝ 速報システムが判断する時点で持っている情報）:")
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
        rep.p(f"    N={n}: 最良解 {he}（全観測点の解から {float(L.gc_km(h.lat, h.lon, he.lat, he.lon)):.0f}km）、"
              f"候補域 {rk:.0f}km" + (f"。{'・'.join(permits)}も同程度に当てはまる" if permits else ""))
    rep.data["early_solutions"] = early
    if any(e["permits"] for e in early):
        nmax = max(e["n"] for e in early if e["permits"])
        classes.append("EARLY_SOLUTION_AMBIGUOUS")
        rep.flag("D1", "FLAG", f"最初の {nmax} 観測点までの情報では、速報の震源は真の震源と同じくらいデータに合います。"
                 "早い段階の速報の解がそこに落ちることは、処理として起こり得ます。誤決定自体は観測点配置で説明でき、"
                 "問題はそれが後段の処理で止められなかったことです。")

    # ---- D4 観測網との位置関係
    rep.h("D4 ・ 位置関係 ― 地震は観測網の中で起きたか")
    out = gap > 270 or dmin > 300
    edge = (not out) and gap > 180
    if out: classes.append("OUT_OF_NETWORK")
    rep.flag("D4", "FLAG" if out else ("INFO" if edge else "PASS"),
             f"方位角ギャップ {gap:.0f}°、最寄り観測点 {dmin:.0f}km、観測点は方位 {baz:.0f}° の側に偏在。"
             + ("観測網の外の地震です。観測網から見た方向の距離と発生時刻とがトレードオフの関係になり、網内の地震で較正した"
                "関係式を外挿して使うことになります。" if out else
                ("観測網の縁（海域の地震に典型的）です。深さと沖合方向の距離の拘束が弱くなります。" if edge
                 else "観測網の内側です。")))

    # ---- D2 距離による除外
    rep.h(f"D2 ・ 距離による除外（震源から {cutoff:.0f}km）― 関係式に依存しない部分")
    cand = [(N_IND, h)] + ([(N_REF, reference)] if reference else []) + \
           ([(N_MAG, hM)] if hM else []) + ([(N_INT, hI)] if alert.hypocenter_magnitude else [])
    tab = []
    for name, hh in cand:
        Mc, nc, rmin = network_M(p, hh, rel, cutoff)
        Mn, _, _ = network_M(p, hh, rel, None)
        tab.append({"hypo": name, "stations_within_cutoff": nc, "nearest_R_km": rmin, "M_with_cutoff": Mc, "M_no_cutoff": Mn})
        rep.p(f"  {L.ljust_w(name, W)}{cutoff:.0f}km 以内の観測点 {nc:3d}　最寄り {rmin:6.0f}km　"
              f"M（除外あり）{'  － 推定不能' if Mc is None else f'{Mc:5.1f}'}　M（除外なし）※ {('－' if Mn is None else f'{Mn:4.1f}')}")
    rep.p("  ※ 関係式に依存します。外挿の影響を示すための値で、地震の規模の推定値ではありません。")
    rep.data["D2"] = tab
    mag_row = next((t for t in tab if t["hypo"] == N_MAG), None)
    if mag_row is None:
        rep.flag("D2", "N/A", "速報に震源がないため、マグニチュードの計算にどの観測点が使われ得たかが分かりません。")
    elif Ma is None:
        rep.flag("D2", "N/A", "速報にマグニチュードがないため、説明すべき値がありません。")
    elif mag_row["stations_within_cutoff"] == 0:
        classes.append("CUTOFF_NOT_APPLIED")
        rep.flag("D2", "FLAG", f"速報のM計算の震源から {cutoff:.0f}km 以内に観測点が1つもありません"
                 f"（最寄り {mag_row['nearest_R_km']:.0f}km）。除外機能が働いていれば、マグニチュードは計算できず、"
                 "警報も出なかったはずです。発表された M は、除外されるべき観測点からしか得られません。"
                 "つまり、この除外機能が働いていませんでした。")
    elif mag_row["M_with_cutoff"] is not None and mag_row["M_no_cutoff"] is not None and \
            abs(mag_row["M_no_cutoff"] - Ma) < abs(mag_row["M_with_cutoff"] - Ma) - 0.5:
        classes.append("CUTOFF_NOT_APPLIED")
        rep.flag("D2", "FLAG", "速報の M は、遠方の観測点を含めたときにだけ再現されます。除外機能が適用されていなかった可能性があります。")
    else:
        rep.flag("D2", "PASS", "除外範囲の内側に観測点があり、M の説明に遠方観測点の混入を持ち出す必要はありません。")
    if hM:
        dnear = float(L.gc_km(hM.lat, hM.lon, p.lat, p.lon).min())
        g = float(rel.dM_dR(dnear, hM.depth_km))
        rep.flag("D2", "INFO", f"［関係式に依存］{dnear:.0f}km では、{rel.name} の ∂M/∂R は距離の誤り 100km あたり {g*100:.2f}。"
                 f"震源が 100km 遠くに置かれるごとに M がこれだけ増えます。同じ式で 700km では 100km あたり "
                 f"{float(rel.dM_dR(700, hM.depth_km))*100:.2f}。")
        if Ma is not None and mag_row["M_no_cutoff"] is not None:
            rep.flag("D2", "INFO", f"［関係式に依存］速報のM計算の震源で除外なしに {rel.name} を使うと M{mag_row['M_no_cutoff']:.1f}"
                     f"（速報は M{Ma}）。ここで一致するかどうかは、地震そのものより係数の選び方で決まります。"
                     "検証するには運用式の係数を与えてください。")

    # ---- D3 処理段階間の整合
    rep.h("D3 ・ 段階間の整合 ― マグニチュードと震度予測で同じ震源を使ったか")
    if not hI or alert.hypocenter_magnitude is None:
        rep.flag("D3", "N/A", "速報の震源は1つ以下です。M計算で別の震源を使ったと発表機関が説明している場合は、"
                 "hypocenter_magnitude として追加してください。")
    else:
        sdist = float(L.gc_km(hM.lat, hM.lon, hI.lat, hI.lon))
        if sdist > 30: classes.append("STAGE_HYPOCENTRE_MISMATCH")
        rep.flag("D3", "FLAG" if sdist > 30 else "PASS",
                 f"マグニチュードは {hM}、震度予測は {hI} で計算: 両者は {sdist:.0f}km 離れています。"
                 + ("遠い震源で膨らんだ M を、近い震源に当てはめたことになり、誤差が重なります。" if sdist > 30 else ""))

    # ---- D5 マグニチュードの妥当性
    rep.h("D5 ・ マグニチュードの妥当性")
    Mr = _ref_mag(reference)
    if Ma is None:
        rep.flag("D5", "N/A", "速報にマグニチュードがありません。")
    else:
        if Ma >= 9.6:
            classes.append("IMPLAUSIBLE_MAGNITUDE")
            rep.flag("D5", "FLAG", f"M{Ma} は観測史上最大の 1960年チリ地震（Mw 9.5）を上回ります。気象庁は"
                     "2016年8月1日の事案の後、地震学的に考えられるマグニチュードの上限値を設けるとしています。"
                     "その値と、今回の処理経路でそれが適用されるのかを確認すべきです。")
        if Mr is not None and hI:
            # 差分: 同じ式を速報の地震と実際の地震に、同じ観測点で当てはめる。距離項の大部分が相殺される。
            pa = rel.predict_logA(Ma, L.gc_km(hI.lat, hI.lon, p.lat, p.lon), hI.depth_km)
            pr = rel.predict_logA(Mr, L.gc_km(reference.lat, reference.lon, p.lat, p.lon), reference.depth_km)
            ex = float(np.median(pa - pr))
            if ex > 1: classes.append("MAGNITUDE_MISMATCH")
            rep.flag("D5", "FLAG" if ex > 1 else "PASS",
                     f"速報が正しければ、各観測点は実際の M{Mr} の地震の 約10^{ex:.1f} ＝ {10**ex:,.0f} 倍の揺れを"
                     f"記録していたはずです（差分による比較、{rel.name}）。")
        elif Mr is not None:
            dM = Ma - Mr
            if abs(dM) >= 1: classes.append("MAGNITUDE_MISMATCH")
            rep.flag("D5", "FLAG" if abs(dM) >= 1 else "PASS",
                     f"速報の M{Ma} と参照の M{Mr} の差は {dM:+.1f}（エネルギーにして約 {10 ** (1.5 * dM):,.0f} 倍）。")
        else:
            rep.flag("D5", "N/A", "参照マグニチュードがありません。記録そのものとの振幅比較はこの距離では係数次第に"
                     "なるため、行いません。")

    # ---- D6 震度
    rep.h("D6 ・ 震度 ― 予測震度と観測震度（司・翠川 1999）")
    faithful = False
    ok = np.isfinite(p.obs_I)
    if Ma is None or not hI:
        rep.flag("D6", "N/A", "速報のマグニチュードと震源の両方が必要です。")
    else:
        epi_I = L.gc_km(hI.lat, hI.lon, p.lat, p.lon)
        I_pt = L.pgv_to_shindo(L.predict_pgv(Ma, hI.depth_km, epi_I, fault_type, site_amp, finite=False))
        I_alert = L.pgv_to_shindo(L.predict_pgv(Ma, hI.depth_km, epi_I, fault_type, site_amp, finite=True))
        Lf = L.fault_length_km(Ma)
        rep.p(f"  M{Ma}・{hI}、点震源: 最大 震度{L.shindo_class(float(I_pt.max()))}"
              f"（計測震度 {float(I_pt.max()):.1f}）、震度5弱以上 {int((I_pt >= 4.5).sum())}/{len(p)} 観測点")
        rep.p(f"  M{Ma}・{hI}、面震源（断層長 約{Lf:,.0f}km）: 最大 震度{L.shindo_class(float(I_alert.max()))}"
              f"（計測震度 {float(I_alert.max()):.1f}、{p.sta[int(np.argmax(I_alert))]}）、"
              f"震度5弱以上 {int((I_alert >= 4.5).sum())}/{len(p)} 観測点")
        rep.p("  （発表機関が公表した予測震度の幅と比べてください。一致すれば、震度予測の段階は誤った入力に対して"
              "正しく動いたことになります）")
        Mtrue = Mr if Mr is not None else Mdat
        hT = reference or h
        if Mtrue is not None:
            I_true = L.pgv_to_shindo(L.predict_pgv(Mtrue, hT.depth_km, L.gc_km(hT.lat, hT.lon, p.lat, p.lon), fault_type, site_amp, finite=True))
            rep.p(f"  M{Mtrue:.1f}・{hT}: 最大 震度{L.shindo_class(float(I_true.max()))}（計測震度 {float(I_true.max()):.1f}）")
        if ok.any():
            over = float(np.median(I_alert[ok] - p.obs_I[ok]))
            rep.flag("D6", "FLAG" if over > 1.5 else "PASS",
                     f"速報の入力による予測震度と観測震度の差: 中央値 {over:+.1f}（観測の最大 震度{L.shindo_class(float(np.nanmax(p.obs_I)))}）。")
            if Mtrue is not None:
                fit = float(np.median(I_true[ok] - p.obs_I[ok]))
                faithful = abs(fit) <= 1.5
                if over > 1.5:
                    classes.append("INTENSITY_FROM_BAD_INPUTS" if faithful else "INTENSITY_MODEL_MISFIT")
                    rep.flag("D6", "INFO" if faithful else "FLAG",
                             f"同じ予測式に正しい M と震源を与えると、観測との差は {fit:+.1f}。"
                             + ("震度予測の段階は入力に忠実で、不具合はその上流（M・震源）にあります。" if faithful
                                else "震度予測の段階そのものもずれています。"))
        else:
            rep.flag("D6", "N/A", "観測震度の列がありません。上の予測値のみです。")
        if float(np.min(epi_I)) > 300:
            rep.flag("D6", "INFO", "すべての観測点が司・翠川式の較正範囲（約300km）の外にあり、予測値は外挿です。")

    # ---- D7 1点抜き
    rep.h("D7 ・ 単一観測点への依存（1点ずつ除いて再計算）")
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
        rep.flag("D7", "FLAG", f"{top[2]} がマグニチュードを左右しています（除くと M が {top[3]:+.2f} 変化）。"
                 "その観測点のデータと、電源・伝送の状態を確認してください（2016年8月1日 富津観測点の事案を参照）。")
    if x_dom:
        rep.flag("D7", "FLAG", f"{xs[2]} が震源位置に大きく効いています（除くと解が {xs[1]:.0f}km 動き、目安の {dx_thr:.0f}km を超える）。"
                 "観測点配置がもろく、この観測点の時刻が正しいことが震源の正しさの前提になっています。")
    if not (m_dom or x_dom):
        rep.flag("D7", "PASS", f"M を 0.5 超、震源を {dx_thr:.0f}km 超動かす観測点はありません（最大は {xs[2]}、{xs[1]:.0f}km）。")
    rep.data["D7"] = [{"station": s_, "dM": dm, "dx_km": dx} for _, dx, s_, dm in worst[:5]]

    # ---- D8 外れ値
    rep.h("D8 ・ 外れ値の読み取り")
    r, _ = residuals(p, tt, h, fixed_t0=True)
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    thr = min(max(3 * mad, 2.0), D8_CAP_S)
    bad = [(p.sta[i], float(r[i])) for i in np.nonzero(np.abs(r) > thr)[0]]
    rep.data["D8"] = {"threshold_s": thr, "outliers_vs_solution": bad}
    rep.flag("D8", "FLAG" if bad else "PASS",
             (f"独立解から ±{thr:.1f} 秒を超える読み取りが {len(bad)} 件: " + "、".join(f"{s_} {x:+.1f}秒" for s_, x in bad[:8]) +
              "。誤読み取り、別の地震の重なり、または時刻付与の異常の可能性（2006年9月26日、2013年8月8日の事案を参照）。") if bad
             else f"すべての読み取りが独立解から ±{thr:.1f} 秒以内です。")
    # 誤った読み取りは独立解そのものを引きずるので、カタログの発生時刻・震源という外部の物差しでも確かめる
    if reference is not None and reference.time is not None:
        rr, _ = residuals(p, tt, reference, fixed_t0=True)
        bad_ref = [(p.sta[i], float(rr[i])) for i in np.nonzero(np.abs(rr) > D8_CAP_S)[0]]
        rep.data["D8"]["outliers_vs_reference"] = bad_ref
        rep.flag("D8", "FLAG" if bad_ref else "PASS",
                 (f"参照の震源・発生時刻から予測される P 波到達時刻と ±{D8_CAP_S:.0f} 秒以上ずれた読み取りが "
                  f"{len(bad_ref)}/{len(p)} 件: " + "、".join(f"{s_} {x:+.1f}秒" for s_, x in bad_ref[:8]) +
                  "。D1・D4・D7 を信頼する前に読み取り直してください。") if bad_ref
                 else f"すべての読み取りが、参照の震源から予測される P 波到達時刻の ±{D8_CAP_S:.0f} 秒以内です。")
        bad = bad + bad_ref
    if bad: classes.append("OUTLIER_PICKS")

    # ---- まとめ
    rep.h("まとめ ― データが支持する不具合の類型")
    text = {
        "OUT_OF_NETWORK": "観測網の外の地震: 観測網から見た方向に沿って震源が決まりにくい（関係式に依存しない）。",
        "EARLY_SOLUTION_AMBIGUOUS": "最初の数観測点だけでは、速報の震源が真の震源と同程度にデータに合う（関係式に依存しない）。",
        "CUTOFF_NOT_APPLIED": f"マグニチュードの計算に、{cutoff:.0f}km の除外範囲を超える観測点が使われた（関係式に依存しない）。",
        "STAGE_HYPOCENTRE_MISMATCH": "マグニチュードと震度予測で異なる震源が使われた（関係式に依存しない）。",
        "IMPLAUSIBLE_MAGNITUDE": "観測史上どの地震よりも大きいマグニチュード。妥当性の上限で止まるべきだった。",
        "MAGNITUDE_MISMATCH": "マグニチュードが参照の地震と 1 以上異なる。",
        "INTENSITY_FROM_BAD_INPUTS": "震度の過大予測は M・震源の入力に起因し、震度予測式そのものは正常。",
        "INTENSITY_MODEL_MISFIT": "震度が過大に予測され、正しい入力でも予測式が観測と合わない。",
        "SINGLE_STATION_CONTROL": "1観測点が M または震源を左右している。その観測点のデータ・時計・状態を確認。",
        "OUTLIER_PICKS": "外れ値の読み取りがある。読み取り直すか、別の地震の重なりや時刻付与の異常を確認。",
    }
    shown = [c for c in text if c in classes]
    for c in shown: rep.p(f"  ・{text[c]}")
    if not shown:
        rep.p("  ・該当する不具合の類型はありません。")
    else:
        rep.p("")
        rep.p("  観測データだけでは判断できないこと: 除外機能などの歯止めが働かなかった原因が、改修時のデグレードか、設定変更か、")
        rep.p("  無許可の改変か。これには発表機関の変更履歴が必要です。")
    rep.data["failure_classes"] = shown
    rep.data["summary"] = [text[c] for c in shown]


# ======================================================================== 波形
WAVEFORM_SUFFIXES = {".mseed", ".miniseed", ".msd", ".ms", ".sac"}
LOC_PRIORITY = ["00", "", "10", "01", "20", "02"]
BAND_PRIORITY = ["BH", "HH", "EH", "SH", "BN", "HN", "LH"]


def waveform_files(data):
    """波形ファイルだけを返す。同じフォルダの StationXML や CSV は読まない。"""
    p = Path(data)
    if p.is_file():
        return [p], []
    files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in WAVEFORM_SUFFIXES)
    skipped = sorted(f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() not in WAVEFORM_SUFFIXES)
    return files, skipped


def choose_channels(traces):
    """
    観測点ごとに1つの地震計を選ぶ。ロケーションコード（00 → "" → 10 …）、帯域（BH → HH → EH …）の順に、
    上下動成分を持つ最初の組み合わせ。戻り値は（上下動トレース、その組み合わせの全トレース）または (None, None)。
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
    上下動トレース1本から P 波の立ち上がりを読む。LTA の立ち上がり区間のトリガーは無視する。予測到達時刻が
    あれば、その ±tol 秒以内のトリガーだけを候補にし（最も早いものを採用）、1つもなければ窓内の AIC 読み取りを
    雑音より明瞭な場合に限って採用する。戻り値は（UTCDateTime または None、方法、メモ）。
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
            return None, "", "予測到達時刻が記録の外（または STA/LTA の立ち上がり区間）にある"
        onsets = [i for i in onsets if lo_i <= i <= hi_i]
    method = "sta/lta"
    if onsets:
        i0 = onsets[0]
        a, b = max(i0 - int(5 * sr), lo_i, 1), min(i0 + int(3 * sr), hi_i)
    elif t_exp is not None:
        method, a, b = "aic-window", lo_i, hi_i
    else:
        return None, "", "立ち上がり区間の後にトリガーなし"
    seg = x[a:b]
    k = np.arange(2, len(seg) - 2)
    if k.size < 4:
        return None, "", "窓が短すぎる"
    aic = np.array([kk * np.log(np.var(seg[:kk]) + 1e-20) + (len(seg) - kk - 1) * np.log(np.var(seg[kk + 1:]) + 1e-20) for kk in k])
    ip = a + int(k[np.argmin(aic)])
    w = int(5 * sr)
    snr = float(np.std(x[ip:ip + w]) / (np.std(x[max(ip - w, 0):ip]) + 1e-20))
    if snr < 2.5:
        return None, "", f"明瞭な立ち上がりなし（SN比 {snr:.1f}）"
    return tr.stats.starttime + ip / sr, method, f"SN比 {snr:.1f}"


def picks_from_waveforms(data_dir, coords_csv, inventory, expect, tol, amp_window, out_csv):
    """
    expect: 発生時刻つきの L.Hypo（通常はカタログの地震）。各観測点の P 波到達時刻の予測に使い、
    読み取りをその ±tol 秒以内に限る。None なら制約なし（実データには非推奨）。
    """
    from obspy import read, read_inventory, Stream, UTCDateTime

    files, skipped = waveform_files(data_dir)
    if skipped:
        print(f"[波形] 波形データではないため読み飛ばし: {'、'.join(skipped)}", file=sys.stderr)
    if not files:
        raise SystemExit(f"{data_dir} に波形ファイル（*.mseed、*.sac など）がありません。")
    st = Stream()
    for f in files:
        st += read(str(f))
    st.merge(fill_value=0)
    if inventory is None and Path(data_dir).is_dir() and (Path(data_dir) / "stations.xml").exists():
        inventory = str(Path(data_dir) / "stations.xml")
        print(f"[波形] 座標と計器応答に {inventory} を使います", file=sys.stderr)
    inv = read_inventory(inventory) if inventory else None
    coords = {}
    if coords_csv:
        for r in L.read_csv_rows(coords_csv):
            coords[r["station"]] = (float(r["lat"]), float(r["lon"]), float(r.get("counts_per_um") or 1.0))
    tt = L.TravelTimes(quiet=True)
    if expect is None:
        print("[波形] 警告: 参照の震源・発生時刻がないため、読み取りに制約がかかりません。--reference（USGS の地震など）を"
              "与えると、各観測点の読み取りを予測到達時刻の前後に限定できます。", file=sys.stderr)

    rows = []
    for net, sta in sorted({(tr.stats.network, tr.stats.station) for tr in st}):
        z, group = choose_channels([tr for tr in st if tr.stats.network == net and tr.stats.station == sta])
        name = f"{net}.{sta}" if net else sta
        if z is None:
            print(f"[波形] {name}: 上下動成分がないため除外", file=sys.stderr)
            continue
        if inv is not None:
            try:
                c = inv.get_coordinates(z.id, z.stats.starttime)
                lat, lon = c["latitude"], c["longitude"]
            except Exception:  # noqa: BLE001
                print(f"[波形] {name}: {z.id} が StationXML にないため除外", file=sys.stderr)
                continue
        elif name in coords or sta in coords:
            lat, lon = coords.get(name, coords.get(sta))[:2]
        else:
            print(f"[波形] {name}: 座標がないため除外", file=sys.stderr)
            continue
        t_exp = None
        if expect is not None:
            d = float(L.gc_km(expect.lat, expect.lon, lat, lon))
            t_exp = UTCDateTime(expect.time.isoformat()) + float(tt.p(d, expect.depth_km))
        tP, method, note = pick_p(z, t_exp, tol)
        if tP is None:
            print(f"[波形] {name} {z.stats.channel}: 読み取りなし ― {note}", file=sys.stderr)
            continue
        disp = Stream([tr.copy() for tr in group])
        if inv is not None:
            sr = z.stats.sampling_rate
            disp.remove_response(inventory=inv, output="DISP", pre_filt=(0.05, 0.1, 0.36 * sr, 0.4 * sr))
            scale = 1e6
        else:
            disp.detrend("demean")                     # 応答なし: 変位データとみなし、感度（counts/μm）で換算
            scale = 1.0 / coords.get(name, coords.get(sta))[2]
        hz = [tr for tr in disp if tr.stats.channel[-1] in "NE12"] or [tr for tr in disp if tr.stats.channel[-1] in "Z3"]
        amp = max(float(np.max(np.abs(tr.slice(tP, tP + amp_window).data))) for tr in hz) * scale
        off = (tP - t_exp) if t_exp is not None else None
        rows.append({"観測点": name, "緯度": lat, "経度": lon,
                     "P波到達時刻": tP.datetime.replace(tzinfo=timezone.utc).astimezone(L.JST).isoformat(timespec="milliseconds"),
                     "最大変位(μm)": f"{amp:.4g}", "チャンネル": z.id, "読み取り方法": method,
                     "予測時刻との差(秒)": "" if off is None else f"{off:+.2f}"})
        print(f"[波形] {name} {z.id}: P {L.fmt_time(tP.datetime.replace(tzinfo=timezone.utc))}（{method}、{note}"
              + (f"、予測との差 {off:+.1f} 秒）" if off is not None else "）"), file=sys.stderr)
    L.write_csv_rows(out_csv, ["観測点", "緯度", "経度", "P波到達時刻", "最大変位(μm)", "チャンネル", "読み取り方法",
                               "予測時刻との差(秒)"], rows)
    print(f"[波形] {len(rows)} 観測点を読み取り → {out_csv}", file=sys.stderr)
    if len(rows) < 3:
        raise SystemExit("使える読み取りが3観測点未満のため、震源を決められません。")
    return out_csv


def fetch(provider, ids, t0, before, after, out):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    c = Client(provider)
    T = UTCDateTime(L.parse_time(t0).isoformat())
    Path(out).mkdir(parents=True, exist_ok=True)
    bulk = [(sid.split(".")[0], sid.split(".")[1], "*", "BH?,HH?", T - before, T + after) for sid in ids]
    st = c.get_waveforms_bulk(bulk)
    inv = c.get_stations_bulk(bulk, level="response")
    st.write(str(Path(out) / "waveforms.mseed"), format="MSEED")
    inv.write(str(Path(out) / "stations.xml"), format="STATIONXML")
    print(f"[取得] {len(st)} トレース、{len(inv.get_contents()['stations'])} 観測点 → {out}", file=sys.stderr)


# ======================================================================== main
def _common(ap):
    ap.add_argument("--alert", required=True, help="速報（正規化 JSON または気象庁XML）")
    ap.add_argument("--reference", help="参照する震源（USGS GeoJSON、または {lat,lon,depth_km,time,mag} の JSON）")
    ap.add_argument("--relation", default="eew_illustrative", choices=sorted(L.RELATIONS), help="マグニチュードの関係式")
    ap.add_argument("--relation-coeffs", help="log10A = aM + b・log10R + c・R + d の係数 a,b,c,d（A は μm、R は km）")
    ap.add_argument("--cutoff-km", type=float, default=700.0, help="M 計算から除外する震源距離（km）")
    ap.add_argument("--fault-type", default="intraplate", choices=sorted(L.SI_MIDORIKAWA_D),
                    help="司・翠川式の地震タイプ（crustal＝地殻内、interplate＝プレート間、intraplate＝プレート内）")
    ap.add_argument("--site-amp", type=float, default=1.4, help="工学的基盤から地表への PGV 増幅率")
    ap.add_argument("--json", help="機械可読の結果をこのファイルに書き出す")


def load_reference(path):
    if not path:
        return None
    d = json.loads(L.read_text(path))
    if "features" in d:
        f = max(d["features"], key=lambda x: x["properties"].get("mag") or 0)
        lon, lat, dep = f["geometry"]["coordinates"][:3]
        t = datetime.fromtimestamp(f["properties"]["time"] / 1000, tz=timezone.utc)
        return L.Hypo(lat, lon, dep, t, f"カタログ M{f['properties']['mag']}")
    h = L.Hypo.from_dict(d, "参照")
    if h and d.get("mag") is not None:
        h.label = f"参照 M{d['mag']}"
    return h


def run_analysis(picks_csv, a):
    p = Picks.from_csv(picks_csv)
    alert = L.load_alert(a.alert)
    rel = L.relation_from_args(a.relation, a.relation_coeffs)
    rep = L.Report()
    analyse(p, alert, rel, a.cutoff_km, L.TravelTimes(), load_reference(a.reference), a.fault_type, a.site_amp, rep)
    sys.stdout.write(rep.text())
    if a.json:
        L.write_json(a.json, rep.data)
    return rep.data


def build_parser() -> argparse.ArgumentParser:
    L.localize_argparse()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, title="サブコマンド")

    s = sub.add_parser("picks", help="読み取り値の CSV を解析する"); _common(s)
    s.add_argument("--picks", required=True, help="読み取り値の CSV")

    s = sub.add_parser("waveforms", help="波形から読み取ってから解析する"); _common(s)
    s.add_argument("--data", required=True, help="miniSEED／SAC のディレクトリ（fetch の出力先など）またはファイル1つ")
    s.add_argument("--inventory", help="計器応答を含む StationXML（省略時、--data に stations.xml があればそれを使う）")
    s.add_argument("--coords", help="StationXML がない場合の観測点 CSV（観測点,緯度,経度[,感度]）")
    s.add_argument("--pick-tol", type=float, default=30.0,
                   help="--reference から予測した P 波到達時刻の前後この秒数以内の読み取りだけを採用する")
    s.add_argument("--amp-window", type=float, default=30, help="最大変位を測る P 波後の秒数")
    s.add_argument("--picks-out", default="picks_from_waveforms.csv", help="読み取り結果の出力先")

    s = sub.add_parser("fetch", help="FDSN データセンターから取得する")
    s.add_argument("--provider", default="EARTHSCOPE", help="FDSN データセンター名（既定 EARTHSCOPE）")
    s.add_argument("--stations", required=True, help="NET.STA のカンマ区切り（例 II.ERM,IU.MAJO,IU.PET,IU.YSS,G.INU）")
    s.add_argument("--origin", required=True, help="発生時刻（タイムゾーン省略時は JST）")
    s.add_argument("--before", type=float, default=60, help="発生時刻の何秒前から")
    s.add_argument("--after", type=float, default=600, help="発生時刻の何秒後まで")
    s.add_argument("--out", default="fdsn_data", help="保存先ディレクトリ")

    s = sub.add_parser("jma-history", help="保存した気象庁の緊急地震速報ページを解析する")
    s.add_argument("page", help="保存した「緊急地震速報の内容」ページ（HTML またはテキスト）")
    s.add_argument("--serial", type=int, help="この報を速報 JSON として書き出す")
    s.add_argument("--out", help="速報 JSON の出力先（--serial と併用）")
    return ap


def main(argv=None):
    L.setup_console()
    a = build_parser().parse_args(argv)
    if a.cmd == "picks":
        return run_analysis(a.picks, a)
    if a.cmd == "waveforms":
        ref = load_reference(a.reference)
        alert = L.load_alert(a.alert)
        expect = ref if (ref is not None and ref.time is not None) else None
        if expect is None and alert.h_int is not None and alert.origin_time is not None:
            expect = L.Hypo(alert.h_int.lat, alert.h_int.lon, alert.h_int.depth_km, alert.origin_time, "速報")
            print("[波形] 予測到達時刻を【速報の震源】から計算します。速報が誤っていれば窓も誤るので、"
                  "--reference の指定を勧めます。", file=sys.stderr)
        out = picks_from_waveforms(a.data, a.coords, a.inventory, expect, a.pick_tol, a.amp_window, a.picks_out)
        return run_analysis(out, a)
    if a.cmd == "fetch":
        fetch(a.provider, a.stations.split(","), a.origin, a.before, a.after, a.out)
        return None
    if a.cmd == "jma-history":
        h = L.parse_jma_history(L.read_text(a.page))
        print(f"地震波検知: {L.fmt_time(h['detected_at'])}")
        print(f"{L.rjust_w('報', 4)}  {L.ljust_w('発表時刻', 25)}{L.rjust_w('経過秒', 8)}{L.rjust_w('北緯', 7)}"
              f"{L.rjust_w('東経', 8)}{L.rjust_w('深さ', 7)}{L.rjust_w('M', 5)}")
        prev = None
        for r in h["serials"]:
            jump = "" if prev is None else (f"　M {r['magnitude'] - prev['magnitude']:+.1f}、震源の移動 "
                                            f"{float(L.gc_km(prev['lat'], prev['lon'], r['lat'], r['lon'])):.0f}km")
            print(f"{r['serial']:4d}  {L.ljust_w(L.fmt_time(r['issued_at']), 25)}{r['elapsed_s']:8.1f}{r['lat']:7.1f}"
                  f"{r['lon']:8.1f}{r['depth_km']:5.0f}km{r['magnitude']:5.1f}{jump}")
            prev = r
        if a.serial is not None:
            r = next((x for x in h["serials"] if x["serial"] == a.serial), None)
            if r is None:
                raise SystemExit(f"第{a.serial}報が見つかりません。")
            al = {"source": "気象庁", "serial": r["serial"], "issued_at": r["issued_at"].isoformat(),
                  "detected_at": h["detected_at"].isoformat() if h["detected_at"] else None,
                  "magnitude": r["magnitude"], "hypocenter": {"lat": r["lat"], "lon": r["lon"], "depth_km": r["depth_km"]}}
            out = a.out or f"alert_serial{a.serial}.json"
            Path(out).write_text(json.dumps(al, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"→ {out}")
        return h


if __name__ == "__main__":
    main()
