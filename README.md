# eew-forensics

Two tools for interrogating an earthquake early-warning (EEW) alert, built around the
JMA alert of 16 Oct 2026 (M9.9 "Kuril Islands", withdrawn after 8 minutes; the real
event was a Mww 6.9 at 135 km depth near Kamchatka).

| Tool | Question it answers |
|---|---|
| `alert_triangulate.py` | Where did the alert come from — the agency, the distribution chain, or a radio-layer spoof? Is there a real earthquake behind it? |
| `eew_diagnose.py` | Given station data, which stage of the EEW estimate failed, and which known failure class does it match? |

```
pip install -r requirements.txt
python make_synthetic.py            # rebuild the test cases (already included)

python alert_triangulate.py --alert cases/2026-10-16/alert.json \
    --authority cases/2026-10-16/authority_jma.json \
    --receipts  cases/2026-10-16/receipts_synthetic.csv \
    --catalog   cases/2026-10-16/usgs_event.geojson

python eew_diagnose.py picks --picks cases/2026-10-16/picks_synthetic.csv \
    --alert cases/2026-10-16/alert.json --reference cases/2026-10-16/usgs_event.geojson
```

Saved outputs for both are in `cases/2026-10-16/*_report.txt`; the rogue-cell demo is in `cases/spoof-demo/`.

## alert_triangulate.py

Three independent axes:

- **A · Authority** — matches the received alert against the agency's own record (JMA 防災情報XML
  VXSE43/44/45, or JSON). A mismatch means content changed after the agency.
- **B · Footprint** — from a CSV of receipts (time, position, carrier, channel, optional RSSI/cell ID).
  National, multi-carrier, multi-channel → common source upstream of the carriers. Local, single
  carrier, cell broadcast only → rogue base station candidate, and the transmitter is multilaterated
  from RSSI (log-distance path loss, grid search, 95 % region by Δχ²).
- **C · Physics** — compares the alert with an independent catalogue (`--catalog` file or
  `--fetch-usgs`) and checks whether the agency's *wave-detected* time equals the moment the real
  event's P wave first reached the network (IASP91 via TauP; `--stations` for real sensor coordinates).

Each verdict lists what would overturn it.

## eew_diagnose.py

Re-locates the event from P picks and tests the alert against JMA's documented EEW failure classes:

| Check | Failure class | Depends on magnitude coefficients? |
|---|---|---|
| D1 | mislocation; including where solutions from only the **first N picks** can land | no |
| D2 | magnitude computed with stations beyond the distance cutoff (700 km since 2020-09-07) | station counts: no · M values: yes |
| D3 | magnitude and intensity stages using different hypocentres | no |
| D4 | out-of-network event (azimuthal gap, nearest-station distance) | no |
| D5 | implausible M; differential amplitude check against the real event | partly |
| D6 | intensity over-prediction; whether the intensity model was faithful to its inputs (Si & Midorikawa 1999, point and crude finite source) | no (uses published GMPE) |
| D7 | one station controlling M (2016 Futtsu class) or location | no |
| D8 | outlier picks — overlapping events, timestamp faults (2006, 2013 classes) | no |

Subcommands: `picks` (CSV `station,lat,lon,p_time[,amp_um,observed_shindo]`), `waveforms`
(miniSEED/SAC + StationXML or coords CSV → STA/LTA + AIC picks, peak displacement), `fetch`
(FDSN download), `jma-history` (parse a saved JMA 緊急地震速報 history page into the serial-by-serial
sequence and emit an alert JSON).

## Getting real data

- **The alert as issued**: JMA 緊急地震速報(警報)発表状況 → this event → 緊急地震速報の内容. Save the page
  and run `eew_diagnose.py jma-history page.html --serial 17 --out alert.json`. JMA 防災情報XML archives
  also carry VXSE43.
- **Japanese station data** (what JMA's system actually saw): NIED Hi-net and F-net (registration
  required; `HinetPy` reads Hi-net win32 into SAC). JMA's own station waveforms are not public.
- **Global broadband** via FDSN, e.g. `II.ERM` (Erimo), `IU.MAJO` (Matsushiro), `IU.YSS`
  (Yuzhno-Sakhalinsk), `IU.PET` (Petropavlovsk), `G.INU` (Inuyama):
  `eew_diagnose.py fetch --stations II.ERM,IU.MAJO,IU.YSS,IU.PET,G.INU --origin 2026-10-16T03:35:31`
- **Receipts** for footprint analysis: your own handsets/logs, staff reports, municipal J-ALERT logs,
  broadcaster logs. A suspected rogue cell belongs with the carrier and the 総合通信局 (MIC runs
  DEURAS radio direction finding).

## What's verified and what isn't

- Verified here: travel times against direct TauP calls; the triangulation verdicts on the 2026 case
  and the spoof demo (transmitter recovered to ~10 m on clean synthetic RSSI — real handset RSSI will
  be far worse); the diagnostic on the 2026 case and on a correct in-network control (no failure class
  flagged); the waveform path on synthetic waveforms (picks within +0.1–0.4 s, amplitudes within ~3 %);
  the history-page parser on an excerpt in the page's layout.
- **Not** run here: `fetch` and `--fetch-usgs` (no network route to those services from the build
  sandbox), and the waveform path on real recordings.
- **Illustrative, not operational**: `eew_illustrative` magnitude coefficients. JMA's operational
  magnitude relation is not public; pass it with `--relation-coeffs a,b,c,d` if you have it.
  Findings marked relation-independent do not rely on it.
- The 2026 case's **station picks, amplitudes, intensities and receipts are synthetic**. The alert
  values, detection time and USGS solution are real. `hypocenter_magnitude` is a reconstruction from
  JMA's verbal statement (computed ~680 km farther from Japan than the true source); JMA has not
  published those coordinates.
- Si & Midorikawa is calibrated to ~300 km; everything here is an extrapolation, and flagged as one.
