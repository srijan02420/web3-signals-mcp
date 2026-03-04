from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from shared.base_agent import BaseAgent
from shared.profile_loader import load_profile, get_assets, get_threshold
from shared.storage import Storage


class DerivativesAgent(BaseAgent):
    """
    Collects derivatives data (long/short, funding, OI) from Binance Futures.
    Everything is driven by profiles/default.yaml — no hardcoded values.

    Source: Binance Futures API (free, no key).
    """

    def __init__(self, profile_path: str | None = None) -> None:
        default = Path(__file__).resolve().parent / "profiles" / "default.yaml"
        self.profile = load_profile(Path(profile_path) if profile_path else default)
        self.assets = get_assets(self.profile)
        self.timeout = int(self.profile.get("http_timeout_sec", 15))
        self.futures_map: Dict[str, str] = self.profile.get("binance_futures_map", {})
        self.binance_cfg = self.profile.get("binance", {})
        self.base_url = self.binance_cfg.get("base_url", "https://fapi.binance.com")
        self.endpoints = self.binance_cfg.get("endpoints", {})

        super().__init__(
            agent_name="derivatives_agent",
            profile_name=self.profile.get("name", "derivatives_default"),
        )

    def empty_data(self) -> Dict[str, Any]:
        return {
            "by_asset": {sym: self._empty_asset() for sym in self.assets},
            "summary": {
                "healthy_assets": [],
                "overcrowded_longs": [],
                "bearish_dominance": [],
                "high_funding": [],
            },
        }

    @staticmethod
    def _empty_asset() -> Dict[str, Any]:
        return {
            "long_pct": None,
            "short_pct": None,
            "long_short_ratio": None,
            "funding_rate": None,
            "open_interest_usd": None,
            "ls_status": "unknown",
            "funding_status": "unknown",
            "derivatives_condition": False,
            # Lead indicators (computed from historical snapshots)
            "funding_rate_change_4h": None,
            "funding_rate_change_24h": None,
            "oi_change_pct_4h": None,
            "oi_change_pct_24h": None,
        }

    def collect(self) -> Tuple[Dict[str, Any], List[str]]:
        data = self.empty_data()
        errors: List[str] = []

        # Thresholds from YAML
        ls_min = float(get_threshold(self.profile, "thresholds", "long_short_min", default=0.55))
        ls_max = float(get_threshold(self.profile, "thresholds", "long_short_max", default=0.65))
        fr_max = float(get_threshold(self.profile, "thresholds", "funding_rate_max", default=0.0005))
        ls_period = self.binance_cfg.get("long_short_period", "1h")
        ls_limit = int(self.binance_cfg.get("long_short_limit", 1))

        for sym in self.assets:
            futures_sym = self.futures_map.get(sym)
            if not futures_sym:
                errors.append(f"{sym}: no Binance futures mapping in profile")
                continue

            asset = data["by_asset"][sym]

            # --- Long/Short ratio ---
            try:
                ep = self.endpoints.get("long_short", "/futures/data/globalLongShortAccountRatio")
                url = f"{self.base_url}{ep}?symbol={futures_sym}&period={ls_period}&limit={ls_limit}"
                rows = self._get_json(url)
                if rows:
                    row = rows[0]
                    asset["long_pct"] = round(float(row["longAccount"]), 4)
                    asset["short_pct"] = round(float(row["shortAccount"]), 4)
                    asset["long_short_ratio"] = asset["long_pct"]
            except Exception as exc:
                errors.append(f"long_short {sym}: {exc}")

            # --- Funding rate ---
            try:
                ep = self.endpoints.get("funding_rate", "/fapi/v1/premiumIndex")
                url = f"{self.base_url}{ep}?symbol={futures_sym}"
                row = self._get_json(url)
                if isinstance(row, dict):
                    asset["funding_rate"] = float(row.get("lastFundingRate", 0.0))
            except Exception as exc:
                errors.append(f"funding {sym}: {exc}")

            # --- Open Interest ---
            try:
                ep = self.endpoints.get("open_interest", "/fapi/v1/openInterest")
                url = f"{self.base_url}{ep}?symbol={futures_sym}"
                row = self._get_json(url)
                if isinstance(row, dict):
                    asset["open_interest_usd"] = float(row.get("openInterest", 0.0))
            except Exception as exc:
                errors.append(f"oi {sym}: {exc}")

            # --- Score (thresholds from YAML) ---
            ls = asset.get("long_short_ratio")
            fr = asset.get("funding_rate")

            if ls is not None:
                if ls_min <= ls <= ls_max:
                    asset["ls_status"] = "healthy"
                elif ls > ls_max:
                    asset["ls_status"] = "overcrowded"
                else:
                    asset["ls_status"] = "bearish"

            if fr is not None:
                if 0 <= fr <= fr_max:
                    asset["funding_status"] = "normal"
                elif fr > fr_max:
                    asset["funding_status"] = "high"
                else:
                    asset["funding_status"] = "negative"

            asset["derivatives_condition"] = (
                asset["ls_status"] == "healthy"
                and asset["funding_status"] in ("normal", "negative", "unknown")
            )

        # --- Lead indicators: compute deltas from historical snapshots ---
        try:
            store = Storage()
            history = store.load_history("derivatives_agent", limit=20)
            if len(history) >= 2:
                self._compute_deltas(data, history, errors)
        except Exception as exc:
            errors.append(f"lead indicators: {exc}")

        # Build summary
        healthy, overcrowded, bearish, high_fr = [], [], [], []
        for sym, asset in data["by_asset"].items():
            s = asset["ls_status"]
            if s == "healthy":
                healthy.append(sym)
            elif s == "overcrowded":
                overcrowded.append(sym)
            elif s == "bearish":
                bearish.append(sym)
            if asset["funding_status"] == "high":
                high_fr.append(sym)

        data["summary"] = {
            "healthy_assets": healthy,
            "overcrowded_longs": overcrowded,
            "bearish_dominance": bearish,
            "high_funding": high_fr,
        }

        return data, errors

    # ------------------------------------------------------------------ #
    # Lead indicator computation
    # ------------------------------------------------------------------ #

    def _compute_deltas(
        self,
        current_data: Dict[str, Any],
        history: List[Dict[str, Any]],
        errors: List[str],
    ) -> None:
        """Compute funding rate change and OI change from historical snapshots.

        History is sorted newest-first. We find snapshots closest to 4h and 24h
        ago and compute deltas against the current data.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Parse timestamps from history and index by age in hours
        timed: List[Tuple[float, Dict[str, Any]]] = []
        for snap in history:
            ts_str = snap.get("timestamp") or ""
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age_hours = (now - ts).total_seconds() / 3600
                by_asset = snap.get("data", {}).get("data", {}).get("by_asset", {})
                if not by_asset:
                    by_asset = snap.get("data", {}).get("by_asset", {})
                if by_asset:
                    timed.append((age_hours, by_asset))
            except Exception:
                continue

        if not timed:
            return

        # Find closest snapshot to each target window (with max tolerance)
        targets = {"4h": (4.0, 1.5, 12.0), "24h": (24.0, 8.0, 48.0)}
        closest: Dict[str, Optional[Dict[str, Any]]] = {}
        for label, (target_h, min_age, max_age) in targets.items():
            best = None
            best_diff = float("inf")
            for age_h, by_asset in timed:
                if age_h < min_age or age_h > max_age:
                    continue
                diff = abs(age_h - target_h)
                if diff < best_diff:
                    best = by_asset
                    best_diff = diff
            closest[label] = best

        # Compute deltas per asset
        for sym in self.assets:
            asset = current_data["by_asset"].get(sym, {})
            cur_fr = asset.get("funding_rate")
            cur_oi = asset.get("open_interest_usd")

            for label, snap_data in closest.items():
                if snap_data is None:
                    continue
                prev = snap_data.get(sym, {})
                prev_fr = prev.get("funding_rate")
                prev_oi = prev.get("open_interest_usd")

                # Funding rate change (absolute delta)
                if cur_fr is not None and prev_fr is not None:
                    delta = cur_fr - prev_fr
                    asset[f"funding_rate_change_{label}"] = round(delta, 8)

                # OI change (percentage)
                if cur_oi is not None and prev_oi is not None and prev_oi > 0:
                    pct = ((cur_oi - prev_oi) / prev_oi) * 100
                    asset[f"oi_change_pct_{label}"] = round(pct, 2)

        n_with_delta = sum(
            1 for sym in self.assets
            if current_data["by_asset"].get(sym, {}).get("funding_rate_change_4h") is not None
        )
        errors.append(f"lead indicators: {n_with_delta}/{len(self.assets)} assets have 4h deltas")

    # ------------------------------------------------------------------ #
    # HTTP helper
    # ------------------------------------------------------------------ #

    def _get_json(self, url: str, retries: int = 2) -> Any:
        import time as _time
        last_exc = None
        for attempt in range(retries + 1):
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
                with urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    _time.sleep(1 * (attempt + 1))  # backoff: 1s, 2s
        raise last_exc
