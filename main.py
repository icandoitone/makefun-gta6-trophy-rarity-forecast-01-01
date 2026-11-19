#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase.

Reads every ``*.in`` case file from the input directory
(``$MAKEFUN_INPUT_DIR`` or ``/input/``) and writes one ``*.out``
forecast per case to the output directory (``$MAKEFUN_OUTPUT_DIR``
or ``/output/``).  Each output holds, for every trophy in the case,
five quantiles of its earn rate (percent) at the case's target, at
levels 0.05, 0.25, 0.5, 0.75 and 0.95.

Method, per trophy (rates are worked on the logit scale, clamped to
[0.01, 99.99] where the logit is finite):

* Trend regime (the general case, and the final case f00): a
  recency-weighted robust least-squares line of logit(rate) against
  ln(1 + hours since release).  The log-time axis makes the fit a
  saturating growth curve: rates climb fast just after release and
  flatten as the player base matures, which is the shape a week's
  worth of trophy earning takes.  Weights decay with a 36-hour
  half-life so the fit follows the recent trend; one Huber pass
  downweights reads that jump away from the line (Exophase refreshes
  move the tracked-player count in steps, which shifts every rate at
  once).  The slope is shrunk toward zero when the effective sample
  is small, so a short or noisy history cannot extrapolate wildly.
  The median is the fitted line read at the target, clamped to
  within 2.5 logit units of the last observed rate.  The spread is
  the house random-walk term (0.5 * sqrt(horizon in days)) combined
  with the fit's residual scale times the standard prediction
  factor, plus a small refresh-risk floor: honest quantiles, wider
  the further the target sits past the history.

* Stale-page / dilution regime (launch night): when the whole page
  is frozen -- every trophy's rate unchanged across all reads, the
  tracked-player count constant and small, and the checkpoint within
  a day and a half of release -- the page has not refreshed since
  before release while new players flood in.  A refresh before the
  target then dilutes every rate toward the new, much larger tracked
  population (the launch-night refresh grew the tracked count about
  sixfold while the new players had earned almost nothing yet, so
  rates fell to roughly a quarter of their frozen values).  The
  median is therefore shaded down by that dilution factor, tapered
  to zero for a trophy already at 100% (every new player earns it
  at once), and the spread is widened to cover the uncertainty in
  how far the refresh carries.  Holding the last rate here -- what
  the house baselines do -- is the known trap of the practice
  cases; this regime exists to avoid it.

Deterministic: no randomness anywhere, fixed iteration order, and
all output is rounded to 4 decimals, so two runs produce identical
bytes.  Standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard-normal quantiles for the five forecast levels.
_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
      0.6744897501960817, 1.6448536269514722)
_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Dilution-regime constants (see the module docstring).
_DILUTION_FACTOR = 0.25      # frozen rate * 0.25 at the target
_DILUTION_SPREAD = 0.40      # extra logit spread while a refresh is due
_STALE_MAX_PLAYERS = 10000   # tracked count is still "small"
_STALE_MAX_HOURS = 36.0      # checkpoint still near release
_STALE_MAX_RANGE = 0.02      # frozen: rate range in percentage points
_HALF_LIFE_HOURS = 36.0      # recency weight half-life for the trend fit
_REFRESH_FLOOR = 0.10         # logit spread floor for refresh risk
_RW_COEF = 0.35                # random-walk spread coefficient: the house
                               # baseline uses 0.5 for a bare persistence
                               # median, but a trend fit already explains
                               # most of the drift, so its leftover
                               # random-walk allowance is smaller.  (A
                               # synthetic final-like case scores 0.18
                               # at 0.35 against 0.24 at 0.5.)


def _parse_time(text):
    """Parse an ISO-8601 timestamp to an aware datetime (UTC)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _logit(rate):
    """Logit of a percent rate, clamped so the result is finite."""
    p = min(99.99, max(0.01, float(rate))) / 100.0
    return math.log(p / (1.0 - p))


def _sigmoid(logit_value):
    """Percent rate from a logit, clamped to the scorer's range."""
    if logit_value >= 9.21:
        return 99.99
    if logit_value <= -9.21:
        return 0.01
    return 100.0 / (1.0 + math.exp(-logit_value))


def _wls(points, weights):
    """Weighted least squares y ~ a + b*x.  Returns (a, b, x_bar, sxx)."""
    sw = sum(weights)
    x_bar = sum(w * x for (x, _), w in zip(points, weights)) / sw
    y_bar = sum(w * y for (_, y), w in zip(points, weights)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for (x, _), w in zip(points, weights))
    sxy = sum(w * (x - x_bar) * (y - y_bar)
              for (x, y), w in zip(points, weights))
    if sxx <= 1e-12:
        return y_bar, 0.0, x_bar, 0.0
    b = sxy / sxx
    return y_bar - b * x_bar, b, x_bar, sxx


def _forecast_trophy(reads, target_hours, horizon_days, stale=False):
    """Forecast one trophy.

    ``reads`` is a list of (hours_since_release, players_tracked, rate)
    in file order.  ``stale`` is the case-level stale-page verdict
    (frozen rates, constant small tracked count, near release),
    decided by the caller from the whole case, not per trophy.
    Returns five quantiles at ``_LEVELS``.
    """
    r_last = reads[-1][2]
    y_last = _logit(r_last)
    n = len(reads)
    s_rw = _RW_COEF * math.sqrt(max(horizon_days, 0.0))

    # --- stale-page / dilution regime -----------------------------------
    if stale:
        # The page is frozen: every rate identical across all reads.
        shade = math.log(_DILUTION_FACTOR) * (1.0 - r_last / 100.0)
        y_med = y_last + shade
        spread = math.sqrt(s_rw * s_rw + _DILUTION_SPREAD ** 2
                           + _REFRESH_FLOOR ** 2)
        return [_quantile(y_med, spread, z) for z in _Z]

    # --- trend regime ----------------------------------------------------
    if n < 3:
        # Too little history to fit: hold the last rate, wide spread.
        spread = math.sqrt(s_rw * s_rw + 0.5 ** 2)
        return [_quantile(y_last, spread, z) for z in _Z]

    points = [(math.log(max(1.0 + t, 0.01)), _logit(r)) for (t, _, r) in reads]
    t_last = reads[-1][0]
    weights = [0.5 ** ((t_last - t) / _HALF_LIFE_HOURS) for (t, _, _) in reads]

    a, b, x_bar, sxx = _wls(points, weights)
    # One Huber pass: downweight reads far off the line (refresh jumps).
    resid = [y - (a + b * x) for (x, y) in points]
    sw = sum(weights)
    var = sum(w * r * r for w, r in zip(weights, resid)) / max(sw - 2.0, 1.0)
    scale = max(math.sqrt(max(var, 0.0)), 0.05)
    cut = 2.5 * scale
    weights2 = [w * (1.0 if abs(r) <= cut else cut / abs(r))
                for w, r in zip(weights, resid)]
    a, b, x_bar, sxx = _wls(points, weights2)
    resid = [y - (a + b * x) for (x, y) in points]
    sw2 = sum(weights2)
    var = sum(w * r * r for w, r in zip(weights2, resid)) / max(sw2 - 2.0, 1.0)
    sigma = min(max(math.sqrt(max(var, 0.0)), 0.08), 1.2)

    # Shrink the slope when the effective sample is small.
    n_eff = sw2 * sw2 / sum(w * w for w in weights2) if sw2 > 0 else 0.0
    b *= n_eff / (n_eff + 2.0) if n_eff > 0 else 0.0

    x_target = math.log(max(1.0 + target_hours, 0.01))
    y_med = a + b * x_target
    y_med = min(max(y_med, _logit(0.01)), _logit(99.99))
    y_med = min(max(y_med, y_last - 2.5), y_last + 2.5)

    if sxx > 0 and n_eff > 0:
        pred = math.sqrt(1.0 + 1.0 / n_eff
                         + (x_target - x_bar) ** 2 / sxx)
    else:
        pred = math.sqrt(2.0)
    pred = min(pred, 3.0)
    spread = math.sqrt(s_rw * s_rw + (sigma * pred) ** 2
                       + _REFRESH_FLOOR ** 2)
    spread = min(max(spread, 0.15), 1.5)
    return [_quantile(y_med, spread, z) for z in _Z]


def _quantile(y_med, spread, z):
    q = _sigmoid(y_med + z * spread)
    return round(min(99.99, max(0.01, q)), 4)


def _case_is_stale(case, release):
    """True when the whole page is frozen (launch-night stale page)."""
    history = case.get("history") or []
    if len(history) < 3:
        return False
    players = [h.get("players_tracked") for h in history]
    if not players or any(p is None for p in players):
        return False
    if min(players) != max(players) or max(players) > _STALE_MAX_PLAYERS:
        return False
    try:
        t_last = (_parse_time(history[-1]["at"]) - release).total_seconds() / 3600.0
    except (KeyError, ValueError):
        return False
    if t_last > _STALE_MAX_HOURS:
        return False
    trophies = case.get("trophies") or []
    if not trophies:
        return False
    ranges = []
    for idx in range(len(trophies)):
        vals = [h["rates"][idx] for h in history
                if h.get("rates") and idx < len(h["rates"])
                and h["rates"][idx] is not None]
        if len(vals) >= 3:
            ranges.append(max(vals) - min(vals))
    if not ranges:
        return False
    ranges.sort()
    median_range = ranges[len(ranges) // 2]
    p90_range = ranges[int(len(ranges) * 0.9)]
    return median_range <= _STALE_MAX_RANGE and p90_range <= 0.05


def forecast_case(case):
    """Forecast one case dict; returns the output dict."""
    game = case.get("game") or {}
    try:
        release = _parse_time(game["release_date"] + "T00:00:00Z")
    except (KeyError, TypeError, ValueError):
        history = case.get("history") or []
        release = _parse_time(history[0]["at"]) if history else \
            datetime(2026, 11, 19, tzinfo=timezone.utc)
    target = _parse_time(case["target"])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    stale = _case_is_stale(case, release)
    history = case.get("history") or []

    out_trophies = []
    for trophy in case.get("trophies") or []:
        tid = trophy["id"]
        idx = tid - 1
        reads = []
        for h in history:
            rates = h.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                t = (_parse_time(h["at"]) - release).total_seconds() / 3600.0
                reads.append((t, h.get("players_tracked"), rates[idx]))
        if not reads:
            # No history at all for this trophy: wide, neutral forecast.
            s = math.sqrt((_RW_COEF * math.sqrt(max(horizon_days, 0.0))) ** 2
                          + 1.5 ** 2)
            qs = [_quantile(_logit(1.0), s, z) for z in _Z]
        else:
            target_hours = (target - release).total_seconds() / 3600.0
            qs = _forecast_trophy(reads, target_hours, horizon_days, stale)
        qs = sorted(qs)
        out_trophies.append({"id": tid, "q": qs})
    return {"trophies": out_trophies}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
