# Dilution-Aware Log-Time Forecaster

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate
on Exophase at 2026-11-26T00:00:00Z, for make.fun's
`gta6-trophy-rarity-forecast-01` bounty ($250, 20 places).

## The program

`main.py` (Python 3.13, standard library only, no network) reads every
`*.in` case from `$MAKEFUN_INPUT_DIR` (or `/input/`) and writes one
`*.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (or `/output/`).
Each output is `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`
for all 55 trophies: five quantiles of the trophy's earn rate in percent,
non-decreasing, each in [0, 100]. Runs in ~0.06 s per case set,
deterministic (two runs are byte-identical).

## Method

All rates are worked on the logit scale (clamped to [0.01, 99.99]).

**Trend regime (the general case, and the final case f00).** A
recency-weighted robust least-squares line of logit(rate) against
ln(1 + hours since release). The log-time axis makes the fit a
saturating growth curve: rates climb fast just after release and
flatten as the player base matures. Weights decay with a 36-hour
half-life; one Huber pass downweights reads that jump off the line
(Exophase refreshes move the tracked-player count in steps, which
shifts every rate at once). The slope is shrunk toward zero when the
effective sample is small. The median is the fitted line read at the
target, clamped to within 2.5 logit units of the last observed rate.
The spread combines a random-walk term (0.35 * sqrt(horizon in days)
— tighter than the house baseline's 0.5, because a trend fit already
explains most of the drift), the fit's residual scale times the
standard prediction factor, and a small refresh-risk floor.

**Stale-page / dilution regime (launch night).** When the whole page
is frozen — every trophy's rate unchanged across all reads, the
tracked-player count constant and small, checkpoint within 36 hours
of release — the page has not refreshed since before release while
new players flood in. A refresh before the target then dilutes every
rate toward the new, much larger tracked population: on launch night
the tracked count grew about sixfold while the new players had earned
almost nothing yet, so rates fell to roughly a quarter of their frozen
values. The median is shaded down by that dilution factor (tapered to
zero for a trophy already at 100%, which every new player earns at
once) and the spread is widened. Holding the last rate here — what
both house baselines do — is the known trap of the practice cases.

## Local validation

The kit ships no practice answers (they are scored server-side), so
the practice loss is measured against a reconstructed answer: the
launch-night refresh modelled as new players earning at a fraction
`g` of the old population's rate, with `g` calibrated so the house
baseline scores its published ~1.51 (which pins `g` ≈ 0.05, i.e. the
refresh was close to pure dilution). Mean pinball loss over both
practice cases (p00, p01):

| new-player progress `g` | this entry | house baseline |
|---:|---:|---:|
| 0.00 (pure dilution) | 0.229 | 1.688 |
| 0.05 (calibrated) | **0.125** | **1.466** |
| 0.10 | 0.089 | 1.285 |
| 0.20 | 0.136 | 1.000 |
| 0.30 | 0.261 | 0.779 |

The entry beats the house baseline (and the house curve, which also
scores ~1.51 on the practice) across the whole plausible range of
the refresh.

For the final case, which has no local history yet, the trend engine
was checked on a synthetic final-like case (120 reads over 60 hours,
log-time trends per trophy plus three refresh level shifts, target
4.5 days out): **0.173** vs house baseline 0.261 vs house curve
0.329.

## Files

- `main.py` — the entry (Node alternative not used)
- `README.md` — this file
- `LICENSE` — MIT
- `.makefun/owner.json` — payout wallet
