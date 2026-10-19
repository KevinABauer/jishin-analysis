"""
Build a folder shaped like `eew_diagnose.py fetch` output — waveforms.mseed + stations.xml — with the
things that broke real data: several location codes and bands per station, a long-period channel,
and large glitches well before the P wave.
"""
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Site, Station
from obspy.core.inventory.response import Response

import eewlib as L

GAIN = 1.0e9          # counts per metre of displacement (flat)


def _response():
    return Response.from_paz(zeros=[], poles=[], stage_gain=GAIN, input_units="M", output_units="COUNTS")


def make_fetch_like_dir(out: Path, hypo: L.Hypo, mag: float, stations: dict, seed=3):
    """stations: {SEED station code (≤5 ASCII chars): (lat, lon)}"""
    rng = np.random.default_rng(seed)
    tt = L.TravelTimes(quiet=True)
    rel = L.RELATIONS["tsuboi"]
    t0 = UTCDateTime(hypo.time.isoformat()) - 60
    st, net_stations, truth = Stream(), [], {}
    for code, (lat, lon) in stations.items():
        d = float(L.gc_km(hypo.lat, hypo.lon, lat, lon))
        tp = UTCDateTime(hypo.time.isoformat()) + float(tt.p(d, hypo.depth_km))
        amp_m = 10 ** float(rel.predict_logA(mag, d, hypo.depth_km)) * 1e-6
        truth[f"SY.{code}"] = (tp, amp_m * 1e6)
        chans = []
        for loc, band, sr in (("00", "BH", 20.0), ("10", "HH", 40.0), ("00", "LH", 1.0)):
            n = int(660 * sr)
            t = np.arange(n) / sr
            for comp in "ZNE":
                x = rng.normal(0, amp_m * 0.01, n)
                k = t >= (tp - t0)
                tau = t[k] - (tp - t0)
                sig = (1 - np.exp(-tau / 2.0)) * np.exp(-tau / 40.0) * np.sin(2 * np.pi * 0.8 * tau + rng.uniform(0, 6))
                x[k] += amp_m * (0.5 if comp == "Z" else 1.0) * sig / (np.abs(sig).max() + 1e-30)
                if loc == "10" or (loc == "00" and band == "BH" and comp == "Z"):
                    # A strong glitch ~150 s before P, after STA/LTA warm-up — the old picker took these.
                    g0 = int(max((tp - t0) - 150, 45) * sr)
                    x[g0:g0 + int(2 * sr)] += amp_m * 3 * rng.normal(0, 1, int(2 * sr))
                st += Trace((x * GAIN).astype("float64"), header={
                    "network": "SY", "station": code, "location": loc, "channel": band + comp,
                    "sampling_rate": sr, "starttime": t0})
                chans.append(Channel(code=band + comp, location_code=loc, latitude=lat, longitude=lon,
                                     elevation=0, depth=0, sample_rate=sr, response=_response()))
        net_stations.append(Station(code=code, latitude=lat, longitude=lon, elevation=0, channels=chans,
                                    site=Site(name=code)))
    out.mkdir(parents=True, exist_ok=True)
    st.write(str(out / "waveforms.mseed"), format="MSEED", encoding="FLOAT64")
    Inventory(networks=[Network(code="SY", stations=net_stations)], source="synthetic").write(
        str(out / "stations.xml"), format="STATIONXML")
    return truth
