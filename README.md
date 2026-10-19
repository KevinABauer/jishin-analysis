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
pip install -r requirements-dev.txt && python -m pytest    # 25 tests, ~30 s

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
  from RSSI (log-distance path loss, grid search, 95 % region by Δχ²). A local verdict needs at least
  `--min-local-receipts` receipts (default 10): a wide footprint is positive evidence and a few receipts
  can show it, but every small sample looks local, so fewer than that is reported as indeterminate.
- **C · Physics** — compares the alert with an independent catalogue (`--catalog` file or
  `--fetch-usgs`) and checks whether the agency's *wave-detected* time equals the moment the real
  event's P wave first reached the network (IASP91 via TauP; `--stations` for real sensor coordinates).
  It compares whatever the alert carries — magnitude, location, or both. If it carries neither, the
  physics axis is `NOT_CHECKED`, never "consistent".

Each verdict lists what would overturn it. `--json` output carries a language-independent
`verdict.code` (`PROCESSING_FAULT`, `LOCAL_INJECTION`, `DISTRIBUTION_ALTERED`, `UPSTREAM_FALSE_TRIGGER`,
`CONSISTENT`, `INDETERMINATE`).

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
| D8 | outlier picks — vs the independent solution (threshold 3×MAD, floor 2 s, **capped at 10 s**) and, with `--reference`, vs the P times the catalogue origin predicts | no |

`--json` output lists the supported failure classes as codes in `failure_classes`. An alert with no
hypocentre or no magnitude is analysed as far as it can be; the checks that need the missing part
report `N/A`.

Subcommands:

- `picks` — analyse a CSV `station,lat,lon,p_time[,amp_um,observed_shindo]`. Zero or negative
  amplitudes are ignored for magnitude, with a warning.
- `waveforms` — pick and measure from waveforms, then analyse. Point `--data` at the output of `fetch`:
  - only `*.mseed`, `*.miniseed`, `*.msd`, `*.ms`, `*.sac` are read; `stations.xml` in the same folder
    is used for coordinates and responses automatically;
  - one sensor per station — location `00`, then `""`, `10`, …; band `BH`, then `HH`, `EH`, …; vertical
    component for the pick, the same location/band for amplitudes;
  - with `--reference`, each pick must fall within `--pick-tol` s (default 30) of the P time that origin
    predicts. Triggers outside the window — glitches, other events — are ignored; if none fire inside
    it, an AIC pick is accepted only above SNR 2.5. Without `--reference` it warns that picks are
    unconstrained;
  - the picks CSV records the channel, pick method and offset from the expected time.
- `fetch` — download waveforms and StationXML from an FDSN data centre (default `EARTHSCOPE`).
- `jma-history` — parse a saved JMA 緊急地震速報 history page into the serial-by-serial sequence and
  emit an alert JSON.

## Getting real data

- **The alert as issued**: JMA 緊急地震速報(警報)発表状況 → this event → 緊急地震速報の内容. Save the page
  and run `eew_diagnose.py jma-history page.html --serial 17 --out alert.json`. JMA 防災情報XML archives
  also carry VXSE43.
- **Japanese station data** (what JMA's system actually saw): NIED Hi-net and F-net (registration
  required; `HinetPy` reads Hi-net win32 into SAC). JMA's own station waveforms are not public.
- **Global broadband** via FDSN, e.g. `II.ERM` (Erimo), `IU.MAJO` (Matsushiro), `IU.YSS`
  (Yuzhno-Sakhalinsk), `IU.PET` (Petropavlovsk), `G.INU` (Inuyama):
  ```
  python eew_diagnose.py fetch --stations II.ERM,IU.MAJO,IU.YSS,IU.PET,G.INU --origin 2026-10-16T03:35:31.9
  python eew_diagnose.py waveforms --data fdsn_data --alert alert.json --reference usgs_event.geojson
  ```
- **Receipts** for footprint analysis: your own handsets/logs, staff reports, municipal J-ALERT logs,
  broadcaster logs. A suspected rogue cell belongs with the carrier and the 総合通信局 (MIC runs
  DEURAS radio direction finding).

## What's verified and what isn't

- **Test suite** (`python -m pytest`, 25 tests): BOM-prefixed JSON, JMA `condition="不明"` magnitudes,
  the history-page parser; the 2026 verdict and its P-arrival timing; magnitude-only and parameter-less
  alerts never coming back "consistent"; the 10-receipt boundary for a local verdict; transmitter
  recovery in the spoof demo; the 2026 failure classes; a correct in-network control returning no
  failure class; alerts with no hypocentre / no magnitude / zero amplitudes; the D8 cap with 4 of 5
  picks 113–425 s early; the `EARTHSCOPE` default; and the `waveforms` path on a synthetic folder shaped
  like `fetch` output (multiple location codes and bands, glitches 150 s before P) — picks within 2 s of
  P on `00.BHZ`, amplitudes within 15 % after response removal.
- **Not** run from the build environment: `fetch` and `--fetch-usgs` (no network route to those
  services), and `waveforms` on real recordings.
- **Illustrative, not operational**: `eew_illustrative` magnitude coefficients. JMA's operational
  magnitude relation is not public; pass it with `--relation-coeffs a,b,c,d` if you have it.
  Findings marked relation-independent do not rely on it.
- The 2026 case's **station picks, amplitudes, intensities and receipts are synthetic**. The alert
  values, detection time and USGS solution (03:35:31.9 UTC) are real. `hypocenter_magnitude` is a
  reconstruction from JMA's verbal statement (computed ~680 km farther from Japan than the true
  source); JMA has not published those coordinates.
- Si & Midorikawa is calibrated to ~300 km; everything here is an extrapolation, and flagged as one.
