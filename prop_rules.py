# prop_rules.py
# Simulation Apex/risk + règles journalières (verbeux + testable)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketSpec:
    tick_size: float = 0.25
    tick_value: float = 0.50  # MNQ: $0.50/tick
    point_value: float = 2.0  # MNQ: $2/point (0.25 tick => $0.50)


@dataclass
class StrategySpec:
    sl_ticks: int = 15
    tp_ticks: int = 22
    max_trades_per_day: int = 2
    max_consecutive_losses: int = 2
    daily_stop_r: float = -1.0  # stop si PnL jour <= -1R
    slippage_ticks: int = 0     # stress test: 1,2 ticks
    commission_roundturn_usd: float = 0.0
    intrabar_priority: str = "worst"  # worst: si SL et TP touchés même minute => SL


@dataclass
class ApexSpec:
    starting_balance: float = 50_000.0
    trailing_dd: float = 2_500.0

    # trailing_mode:
    # - "strict": trailing = peak_equity - dd, peak inclut latent favorable
    #             (conservateur; ne "gèle" jamais)
    # - "cap_at_start_plus": trailing ne monte pas au-delà starting + safety_net
    trailing_mode: str = "strict"
    safety_net_extra: float = 100.0


@dataclass
class DailyRiskState:
    trades_taken: int = 0
    consecutive_losses: int = 0
    pnl_usd: float = 0.0

    def reset(self):
        self.trades_taken = 0
        self.consecutive_losses = 0
        self.pnl_usd = 0.0


@dataclass
class ApexState:
    equity: float
    peak_equity: float
    trailing_level: float
    liquidated: bool = False
    liquidation_reason: Optional[str] = None

    @staticmethod
    def init(spec: ApexSpec) -> "ApexState":
        eq = spec.starting_balance
        peak = eq
        trailing = eq - spec.trailing_dd
        return ApexState(equity=eq, peak_equity=peak, trailing_level=trailing)

    def _cap(self, spec: ApexSpec) -> float:
        if spec.trailing_mode == "cap_at_start_plus":
            return spec.starting_balance + spec.safety_net_extra
        return float("inf")

    def update_peak_with_favorable_unrealized(self, spec: ApexSpec, favorable_unrealized_usd: float) -> None:
        """
        favorable_unrealized_usd: meilleur latent possible à cet instant (ex: high favorable)
        """
        if self.liquidated:
            return

        candidate_peak = self.equity + favorable_unrealized_usd
        if candidate_peak > self.peak_equity:
            self.peak_equity = candidate_peak
            raw_trailing = self.peak_equity - spec.trailing_dd
            base = spec.starting_balance - spec.trailing_dd
            cap = self._cap(spec)
            self.trailing_level = max(base, min(raw_trailing, cap))

    def check_liquidation_with_worst_unrealized(self, worst_unrealized_usd: float, reason: str = "TRAILING_TOUCH") -> None:
        """
        worst_unrealized_usd: pire latent possible à cet instant (ex: low adverse)
        Si equity + worst_unrealized <= trailing => liquidation.
        """
        if self.liquidated:
            return
        if (self.equity + worst_unrealized_usd) <= self.trailing_level:
            self.liquidated = True
            self.liquidation_reason = reason
            # liquidation: on force l'equity au niveau trailing (hypothèse conservatrice)
            self.equity = self.trailing_level

    def apply_realized_pnl(self, pnl_usd: float) -> None:
        if self.liquidated:
            return
        self.equity += pnl_usd
        if self.equity > self.peak_equity:
            # si gain réalisé augmente l'equity au-delà du peak
            self.peak_equity = self.equity
            # trailing doit être recalculé par l'appelant via update_peak_with_favorable_unrealized(0)
            # pour centraliser la logique (cap, base etc.)


def daily_risk_allows_new_trade(daily: DailyRiskState, strat: StrategySpec, market: MarketSpec) -> bool:
    """
    Règles:
      - max N trades/jour
      - stop si 2 pertes consécutives
      - stop si PnL jour <= -1R
    """
    if daily.trades_taken >= strat.max_trades_per_day:
        return False
    if daily.consecutive_losses >= strat.max_consecutive_losses:
        return False

    R_usd = strat.sl_ticks * market.tick_value
    if daily.pnl_usd <= strat.daily_stop_r * R_usd:
        return False

    return True


def update_daily_after_trade(daily: DailyRiskState, pnl_usd: float) -> None:
    daily.trades_taken += 1
    daily.pnl_usd += pnl_usd
    if pnl_usd < 0:
        daily.consecutive_losses += 1
    else:
        daily.consecutive_losses = 0


def pnl_after_costs(pnl_usd_gross: float, strat: StrategySpec) -> float:
    """
    Coûts: slippage (déjà dans le fill) + commission RT
    Ici: on retire seulement la commission RT pour être explicite.
    """
    return pnl_usd_gross - strat.commission_roundturn_usd
