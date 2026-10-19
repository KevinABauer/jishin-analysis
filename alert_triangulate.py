#!/usr/bin/env python3
"""
alert_triangulate.py ― その速報は、どこから来たのか

地震の速報を、互いに独立した3つの観点から「三角測量」します。どの観点でも、発表機関側の
処理誤りと、偽造・注入された速報とを切り分けられます。

  A. 発表元    受信した速報は、発表機関自身の記録と一致するか
               （気象庁防災情報XML、気象庁の緊急地震速報の発表履歴ページ、またはその JSON 書き起こし）
  B. 受信範囲  誰が、どこで、どの事業者・どの経路で受け取ったか
               偽基地局がセルブロードキャスト（ETWS）を偽装しても、届くのは成りすました事業者の
               数km圏内だけ。発表機関から出た速報は、警報対象の都道府県全域で全事業者・テレビ・
               ラジオ・Jアラートに同時に届く。範囲が局所的なら、受信強度から送信源の位置を推定する。
  C. 物理      その速報の裏に実際の地震があるか
               独立した地震カタログ（既定は USGS）と照合し、さらに最も強力な単独の検証として、
               発表機関の「地震波検知時刻」が、実際の地震の P 波が観測網に最初に届いた時刻と
               一致するかを確かめる。

使い方
  python alert_triangulate.py --alert 速報.json [--authority 気象庁.xml ...]
         [--receipts 受信記録.csv] [--catalog usgs.geojson | --fetch-usgs]
         [--stations 観測点.csv] [--json 結果.json]

受信記録 CSV の列（英語・日本語どちらの列名でも可、UTF-8 / Shift_JIS 自動判別）
  受信時刻, 緯度, 経度, 事業者, 経路 [, 受信強度_dbm, セルID, 本文ハッシュ]
  経路: エリアメール（緊急速報メール）/ テレビ / ラジオ / Jアラート / アプリ / その他

いずれの判定も確定ではありません。各結論には「何が分かれば覆るか」を併記します。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

import eewlib as L

CH_JA = {"cbs": "エリアメール等", "tv": "テレビ", "radio": "ラジオ", "jalert": "Jアラート", "app": "アプリ", "other": "その他"}
CA_JA = {"docomo": "ドコモ", "au": "au", "softbank": "ソフトバンク", "rakuten": "楽天モバイル"}


# ------------------------------------------------------------------ A. 発表元
def check_authority(alert: L.Alert, auth_paths: list[str], rep: L.Report) -> str:
    rep.h("A ・ 発表元 ― 発表機関自身の記録にこの速報はあるか")
    if not auth_paths:
        rep.flag("A0", "N/A", "発表機関の記録が指定されていません（--authority）。気象庁は緊急地震速報（警報）の内容を"
                 "「緊急地震速報（警報）発表状況」ページと防災情報XML（VXSE43）で公開しています。")
        return "NOT_CHECKED"
    best, best_score = None, -1e9
    for p in auth_paths:
        try:
            a = L.load_alert(p)
        except Exception as e:  # noqa: BLE001
            rep.flag("A1", "INFO", f"{p} を読み込めませんでした: {e}")
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
        rep.flag("A1", "FLAG", "読み込める発表機関の記録がありません。")
        return "ABSENT"

    diffs = []
    if alert.magnitude is not None and best.magnitude is not None and abs(alert.magnitude - best.magnitude) > 0.05:
        diffs.append(f"M{alert.magnitude}（機関の記録は M{best.magnitude}）")
    if alert.h_int and best.h_int:
        sep = float(L.gc_km(alert.h_int.lat, alert.h_int.lon, best.h_int.lat, best.h_int.lon))
        ddep = abs(alert.h_int.depth_km - best.h_int.depth_km)
        if sep > 15 or ddep > 15:
            diffs.append(f"震源が {sep:.0f}km、深さが {ddep:.0f}km 異なる")
    if alert.issued_at and best.issued_at:
        dt = (alert.issued_at - best.issued_at).total_seconds()
        if abs(dt) > 60:
            diffs.append(f"発表時刻が {dt:+.0f} 秒異なる")
    rep.data["authority_record"] = best.to_dict()
    if diffs:
        rep.flag("A2", "FLAG", "受信した速報は発表機関の記録と一致しません: " + "、".join(diffs))
        return "MISMATCH"
    rep.flag("A2", "PASS", f"発表機関の記録と一致します（M{best.magnitude}、{best.h_int}、"
             f"発表 {L.fmt_time(best.issued_at)}）。この内容は発表機関自身が出したものです。")
    return "MATCH"


# ---------------------------------------------------------------- B. 受信範囲
def load_receipts(path):
    rows = []
    for r in L.read_csv_rows(path):
        rows.append({
            "t": L.parse_time(r.get("received_at")), "lat": float(r["lat"]), "lon": float(r["lon"]),
            "carrier": L.norm_carrier(r.get("carrier", "")), "channel": L.norm_channel(r.get("channel", "")),
            "rssi": float(r["rssi_dbm"]) if r.get("rssi_dbm") not in (None, "") else None,
            "cell": r.get("cell_id", ""), "digest": r.get("message_digest", ""),
        })
    return rows


def multilaterate_rssi(rows, rep: L.Report):
    """
    距離減衰モデル: RSSI = P0 − 10・n・log10(d / 1 m)。送信源の候補位置ごとに (P0, n) を
    線形最小二乗で解き（n は 1.6～4.5 に制限）、残差最小の位置を最良推定とする。95% 範囲は
    最良解から推定した分散を用い、Δχ2 ≦ 5.99（自由度2）で求める。
    スマートフォンの受信強度は誤差が大きいので、結果は「位置の確定」ではなく
    方向探知で絞り込むための探索範囲として扱うこと。
    """
    pts = [r for r in rows if r["rssi"] is not None]
    if len(pts) < 5:
        rep.flag("B5", "N/A", f"受信強度のある受信記録が {len(pts)} 件のみです。位置推定には5件以上必要なため、"
                 "重心のみを示します。")
        return None
    lat0 = float(np.mean([r["lat"] for r in pts])); lon0 = float(np.mean([r["lon"] for r in pts]))
    x, y = L.to_enu_km([r["lat"] for r in pts], [r["lon"] for r in pts], lat0, lon0)
    rssi = np.array([r["rssi"] for r in pts])
    span = max(np.ptp(x), np.ptp(y), 1.0)
    g = np.linspace(-span, span, 241)
    GX, GY = np.meshgrid(g, g)
    d_m = np.sqrt((GX[..., None] - x) ** 2 + (GY[..., None] - y) ** 2) * 1000 + 10.0
    X = -10 * np.log10(d_m)
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
    rep.flag("B5", "INFO", f"送信源の推定位置 {L.latlon_ja(float(blat), float(blon), 5)}、95% 範囲の半径 約{radius:.2f}km"
             f"（距離減衰係数 {n[k]:.2f}、残差 RMS {math.sqrt(sigma2):.1f}dB、受信記録 {len(pts)} 件）。")
    return res


def check_footprint(rows, rep: L.Report, min_local: int = 10) -> str:
    rep.h("B ・ 受信範囲 ― 届き方から見た発信元")
    if not rows:
        rep.flag("B0", "N/A", "受信記録が指定されていません（--receipts）。")
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
    ca = "、".join(CA_JA.get(c, c) for c in carriers) or "－"
    ch = "、".join(CH_JA.get(c, c) for c in channels)
    rep.flag("B1", "INFO", f"受信記録 {len(rows)} 件、広がり {extent:.0f}km。事業者: {ca}。経路: {ch}。"
             f"受信時刻の 90% が {spread:.1f} 秒の幅に収まる。")
    if len(digests) > 1:
        rep.flag("B2", "FLAG", f"本文が {len(digests)} 種類出回っています。配信のどこかで内容が分かれています。"
                 "それぞれを発表機関の文面と照合してください。")
    # 判定はあえて非対称にしている。「広い」ことは積極的な証拠で、少数の受信記録でも示せる。
    # 「狭い」ことは広がりがないというだけで、標本が少なければ必ず狭く見える。
    # そのため局所型の判定には一定数以上の受信記録を求める。
    non_cbs = [c for c in channels if c not in ("cbs", "")]
    if extent >= 150 or (len(carriers) >= 2 and extent >= 50) or non_cbs:
        why = []
        if extent >= 150: why.append(f"広がり {extent:.0f}km")
        if len(carriers) >= 2: why.append(f"独立した {len(carriers)} 事業者")
        if non_cbs: why.append("携帯以外の経路（" + "、".join(CH_JA.get(c, c) for c in non_cbs) + "）")
        rep.flag("B3", "PASS", "全国型の届き方です（" + "、".join(why) + "）。無線区間での偽装ではこうはなりません。"
                 "数百km にわたり全事業者に偽基地局を同時に置き、さらにテレビや Jアラートまで押さえる必要があるため"
                 "です。共通の発信点は各事業者より上流にあります。")
        return "NATIONAL"
    if extent <= 30 and len(carriers) <= 1 and set(channels) <= {"cbs"}:
        if len(rows) < min_local:
            rep.flag("B3", "INFO", f"受信記録が {len(rows)} 件しかなく、互いに近接しています。局所型と判定するには "
                     f"{min_local} 件以上が必要です（--min-local-receipts）。標本が少なければ、どんな届き方も局所的に"
                     "見えます。近くにいた他事業者の利用者で「受信しなかった」人の記録も含め、追加で集めてください。")
            return "INDETERMINATE"
        rep.flag("B3", "FLAG", f"局所型の届き方です（受信記録 {len(rows)} 件が {extent:.1f}km 以内、単一事業者、"
                 "エリアメール等のみ）。偽基地局（偽の eNB／gNB）による ETWS の送信と矛盾しません。")
        cells = sorted({r["cell"] for r in rows if r["cell"]})
        if cells:
            rep.flag("B4", "INFO", f"報告されたセルID: {cells[:12]}{' …' if len(cells) > 12 else ''}。"
                     "事業者自身の基地局リストにない ID は、偽基地局を強く示唆します。")
        tx = multilaterate_rssi(rows, rep)
        if tx is None:
            rep.flag("B5", "INFO", f"受信記録の重心: {L.latlon_ja(float(lat.mean()), float(lon.mean()), 5)}"
                     "（参考程度。送信源は受信記録の分布の外にあることもあります）。")
        rep.data["transmitter_estimate"] = tx
        return "LOCAL"
    rep.flag("B3", "INFO", "全国型とも局所型とも言い切れません。結論を出す前に、他の事業者・テレビ・Jアラートの"
             "記録などを追加で集めてください。")
    return "INDETERMINATE"


# -------------------------------------------------------------------- C. 物理
def fetch_usgs(alert: L.Alert, minmag=4.0) -> dict:
    t0 = alert.origin_time or alert.detected_at or alert.issued_at
    if t0 is None:
        raise SystemExit("--fetch-usgs には、速報に発生時刻・検知時刻・発表時刻のいずれかが必要です。")
    q = {"format": "geojson", "minmagnitude": minmag,
         "starttime": (t0 - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S"),
         "endtime": (t0 + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")}
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def catalog_events(geo: dict) -> list[dict]:
    out = []
    for ft in geo.get("features", []):
        pr, (lon, lat, dep) = ft["properties"], ft["geometry"]["coordinates"][:3]
        t = datetime.fromtimestamp(pr["time"] / 1000, tz=timezone.utc)
        out.append({"id": ft.get("id"), "mag": pr.get("mag"), "magType": pr.get("magType"),
                    "place": pr.get("place"), "time": t, "hypo": L.Hypo(lat, lon, dep, t, "カタログ")})
    return out


def check_physics(alert: L.Alert, geo: dict | None, points, tt: L.TravelTimes, rep: L.Report) -> tuple[str, str]:
    rep.h("C ・ 物理 ― 速報の裏に実際の地震はあるか")
    if geo is None:
        rep.flag("C0", "N/A", "独立した地震カタログが指定されていません（--catalog または --fetch-usgs）。")
        return "NOT_CHECKED", "NOT_CHECKED"
    evs = catalog_events(geo)
    ref_t = alert.origin_time or alert.detected_at or alert.issued_at
    cands = [e for e in evs if ref_t and timedelta(minutes=-20) <= (e["time"] - ref_t) <= timedelta(minutes=10)]
    if not cands:
        rep.flag("C1", "FLAG", "前20分～後10分に M4 以上の地震が独立カタログにありません。何も起きていない"
                 "（センサーや伝送の不具合、または捏造）か、カタログの掲載が遅れているかです。時間をおいて再照会してください。")
        return "NO_PHYSICAL_EVENT", "NOT_CHECKED"

    def p_arrival(e):
        _, _, t = L.first_arrival_in_network(e["hypo"], tt, points)
        return e["time"] + timedelta(seconds=t)
    if alert.detected_at:
        ev = min(cands, key=lambda e: abs((p_arrival(e) - alert.detected_at).total_seconds()))
    else:
        ev = max(cands, key=lambda e: e["mag"] or 0)
    rep.data["catalog_event"] = {k: v for k, v in ev.items() if k != "hypo"} | {"hypo": ev["hypo"].to_dict()}
    rep.flag("C1", "INFO", f"独立カタログ: M{ev['mag']}（{ev['magType'] or '－'}）{ev['place']}、"
             f"{ev['hypo']}、発生 {L.fmt_time(ev['time'])}（{ev['id']}）。")

    # 速報が実際に持っている要素だけを比べる。何も比べられなければ NOT_CHECKED で、「整合」とはしない。
    physics, compared, disagree, disc = "NOT_CHECKED", [], [], {}
    if alert.magnitude is not None and ev["mag"] is not None:
        dM = alert.magnitude - ev["mag"]
        disc |= {"dM": dM, "energy_ratio": 10 ** (1.5 * dM)}
        compared.append("magnitude")
        if abs(dM) >= 1.0:
            disagree.append(f"マグニチュードの差は {dM:+.1f}（エネルギーにして約 {10 ** (1.5 * dM):,.0f} 倍）")
    if alert.h_int:
        sep = float(L.gc_km(alert.h_int.lat, alert.h_int.lon, ev["hypo"].lat, ev["hypo"].lon))
        disc["epicentre_sep_km"] = sep
        compared.append("location")
        if sep > 100:
            disagree.append(f"震央は {sep:.0f}km 離れています")
    rep.data["discrepancy"] = disc | {"compared": compared}
    names = {"magnitude": "マグニチュード", "location": "震源"}
    if disagree:
        physics = "REAL_EVENT_WRONG_PARAMETERS"
        rep.flag("C2", "FLAG", "速報の震源要素が実際の地震と食い違います。" + "、".join(disagree) + "。"
                 "地震そのものは本物ですが、速報はその地震を正しく表していません。")
    elif compared:
        physics = "CONSISTENT"
        missing = [names[k] for k in ("magnitude", "location") if k not in compared]
        rep.flag("C2", "PASS", f"{'と'.join(names[k] for k in compared)}はカタログと一致します"
                 + (f"（速報に{missing[0]}がないため、そちらは未確認）" if missing else "") + "。")
    else:
        rep.flag("C2", "N/A", "速報にマグニチュードも震源もなく、比べられるものがありません。")
    if alert.magnitude is not None and alert.magnitude >= 9.6:
        rep.flag("C3", "FLAG", f"M{alert.magnitude} は観測史上最大の地震（1960年チリ地震、Mw 9.5）を上回ります。"
                 "原理的にあり得ないとは言えませんが、事前の見込みとしては、推定処理が想定外の領域で動いていたことを示します。")

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
                 f"機関の地震波検知 {L.fmt_time(alert.detected_at)} に対し、実際の地震の P 波が観測網に最初に届く時刻は"
                 f"{name}（{dkm:.0f}km、走時 {tP:.1f} 秒）で {L.fmt_time(pred)}。差は {lag:+.1f} 秒。"
                 + ("検知のきっかけは、まさにこの地震の P 波です。" if ok else
                    "この地震の P 波到達とは合いません。別の原因を探してください。"))
        if not tt.exact:
            rep.flag("C4", "INFO", "走時は簡易モデルによるものです。C4 を根拠にするには obspy を入れてください。")
    return physics, timing


# -------------------------------------------------------------------- 総合判定
VERDICTS = {
    "LOCAL_INJECTION": "無線区間での局所的な注入（偽基地局）の疑い",
    "DISTRIBUTION_ALTERED": "配信途中で内容が改変された疑い（発表機関から受信者までの経路を調査）",
    "PROCESSING_FAULT": "発表機関から出た本物の速報で、内容が誤り ― 処理不具合の類型",
    "UPSTREAM_FALSE_TRIGGER": "上流での誤検知（実際の地震なし）― センサー・伝送の不具合、または上流での侵害",
    "CONSISTENT": "整合した速報 ― 確認した範囲では異常は見つかりません",
    "INDETERMINATE": "判定保留 ― 与えられた材料では結論を出せません",
}
AXIS_JA = {"authority": "発表元", "footprint": "受信範囲", "physics": "物理", "timing": "時刻"}


def verdict(auth, foot, phys, timing, rep: L.Report, compared=()):
    rep.h("総合判定")
    code, why, overturn = "INDETERMINATE", [], []
    if foot == "LOCAL" and auth in ("ABSENT", "MISMATCH", "NOT_CHECKED"):
        code = "LOCAL_INJECTION"
        why = ["十分な数の受信記録が、狭い範囲・単一事業者・エリアメール等のみに限られている"]
        overturn = ["発表機関の記録に同じ速報がある", "他の事業者・経路・地域での受信記録が見つかる"]
    elif auth == "MISMATCH" and foot in ("NATIONAL", "NOT_CHECKED", "INDETERMINATE"):
        code = "DISTRIBUTION_ALTERED"
        why = ["受信した内容が、発表機関の公開内容と異なる"]
        overturn = ["差異が受信側の書き写しの誤りにすぎない"]
    elif auth in ("MATCH", "NOT_CHECKED") and phys == "REAL_EVENT_WRONG_PARAMETERS" and timing in ("CONSISTENT", "NOT_CHECKED"):
        code = "PROCESSING_FAULT"
        why = ["実際に地震が起きている", "速報はその地震を誤って表している"]
        if timing == "CONSISTENT": why.append("機関の検知時刻が、その地震の P 波到達時刻と一致する")
        if auth == "MATCH": why.append("発表機関自身の記録に同じ内容がある")
        if foot == "NATIONAL": why.append("届き方が全国型で、無線区間での偽装ではない")
        overturn = ["推定処理への入力（観測点データ・伝送）が改ざんされた証拠 ― 確認には観測点単位のデータが必要。"
                    "eew_diagnose.py を実行してください",
                    "地震前に緊急地震速報のソフトウェア・設定に説明のつかない変更があった（変更管理記録）"]
    elif phys == "NO_PHYSICAL_EVENT" and auth in ("MATCH", "NOT_CHECKED"):
        code = "UPSTREAM_FALSE_TRIGGER"
        why = ["発表機関が発表している", "独立カタログに対応する地震がない"]
        overturn = ["遅れてカタログに掲載される",
                    "観測点単位のデータで単一センサーの故障が確認される"
                    "（例: 2016年8月1日 富津観測点の電源部故障、2013年8月8日 海底地震計の時刻付与異常）"]
    elif phys == "CONSISTENT" and timing != "INCONSISTENT" and auth != "MISMATCH":
        code = "CONSISTENT"
        names = {"magnitude": "マグニチュード", "location": "震源"}
        why = [f"{'と'.join(names[k] for k in compared) or '－'}が独立カタログと一致する"]
        if "location" not in compared: overturn.append("速報の震源が分かり、それがカタログと食い違う")
    else:
        why = ["、".join(f"{AXIS_JA[k]}={v}" for k, v in
                        {"authority": auth, "footprint": foot, "physics": phys, "timing": timing}.items())]
        if timing == "INCONSISTENT":
            why.append("機関の検知時刻が、カタログの地震の P 波到達時刻と合わない")
        overturn = ["NOT_CHECKED／INDETERMINATE となっている観点について、材料を追加する"]
    rep.p(f"  {VERDICTS[code]}")
    for w in why: rep.p(f"    根拠: {w}")
    for o in overturn: rep.p(f"    覆る条件: {o}")
    rep.data["verdict"] = {"code": code, "verdict": VERDICTS[code], "because": why, "would_change_if": overturn,
                           "axes": {"authority": auth, "footprint": foot, "physics": phys, "timing": timing}}


def build_parser() -> argparse.ArgumentParser:
    L.localize_argparse()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alert", required=True, help="受信した速報（正規化 JSON または気象庁XML）")
    ap.add_argument("--authority", nargs="*", default=[], help="発表機関の記録（気象庁XML または JSON、複数可）")
    ap.add_argument("--receipts", help="受信記録の CSV")
    ap.add_argument("--min-local-receipts", type=int, default=10,
                    help="局所型と判定するのに必要な受信記録の件数（既定 10）")
    ap.add_argument("--catalog", help="独立した地震カタログ（USGS GeoJSON）")
    ap.add_argument("--fetch-usgs", action="store_true", help="USGS の FDSN イベントサービスに照会する")
    ap.add_argument("--stations", help="観測点 CSV（観測点,緯度,経度）。省略時は日本の参照地点を使う")
    ap.add_argument("--json", help="機械可読の結果をこのファイルに書き出す")
    return ap


def main(argv=None):
    L.setup_console()
    a = build_parser().parse_args(argv)
    alert = L.load_alert(a.alert)
    rep = L.Report()
    rep.p(f"速報: {alert.source} {alert.region} M{alert.magnitude if alert.magnitude is not None else '不明'}、"
          f"{alert.h_int or '震源なし'}"
          f"（{f'第{alert.serial}報、' if alert.serial is not None else ''}発表 {L.fmt_time(alert.issued_at)}、"
          f"地震波検知 {L.fmt_time(alert.detected_at)}）")
    for n in alert.notes:
        rep.p(f"  注記: {n}")
    rep.data["alert"] = alert.to_dict()

    auth = check_authority(alert, a.authority, rep)
    foot = check_footprint(load_receipts(a.receipts) if a.receipts else [], rep, a.min_local_receipts)
    geo = json.loads(L.read_text(a.catalog)) if a.catalog else (fetch_usgs(alert) if a.fetch_usgs else None)
    points = L.load_points_csv(a.stations) if a.stations else L.JAPAN_REFERENCE_POINTS
    tt = L.TravelTimes()
    phys, timing = check_physics(alert, geo, points, tt, rep)
    verdict(auth, foot, phys, timing, rep, rep.data.get("discrepancy", {}).get("compared", ()))

    sys.stdout.write(rep.text())
    if a.json:
        L.write_json(a.json, rep.data)
    return rep.data


if __name__ == "__main__":
    main()
